"""Meeting orchestrator: capture frames → in-RAM store → finalized transcript.

This is the spine that turns the accuracy core (``pipeline.finalize_channel``)
into a live meeting. It:

1. consumes ``AudioFrame`` objects from any :class:`CaptureProvider`,
2. accumulates each channel's PCM in a bounded in-RAM store (never disk),
3. on stop, runs the finalize pass **per channel** with that channel's known
   speaker count — the biggest diarization accuracy lever (PLAN.md §2) — then
4. interleaves the two channels' entries into one timeline, labelling the mic
   channel ``Local-N`` and the system channel ``Remote-N``.

The channel prior is free separation: diarization never has to tell local from
remote voices, only voices *within* a channel. The provider is swappable
(Swift helper on macOS, sounddevice on Linux, file replay for dev/test), so the
whole orchestrator is exercisable without native capture.

Hybrid-mode cross-channel dedup (room speaker bleed on the mic) is not done
here yet: its primary mitigation is AEC in the capture helper, which does not
exist yet, so there is nothing to dedup in the synthetic paths. Tracked for
when the macOS helper lands (PLAN.md §2 "Hybrid-mode caveats").
"""

from __future__ import annotations

import threading
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from stenograf.asr.base import ASRBackend
from stenograf.audio import to_float32
from stenograf.capture.base import (
    ORDER_TOLERANCE_SAMPLES,
    SAMPLE_RATE,
    AudioFrame,
    CaptureProvider,
    Channel,
)
from stenograf.config import Language, MeetingProfile
from stenograf.diarization.base import Diarizer
from stenograf.lid import detect_language
from stenograf.live import LiveDecoder, StreamingUpdate
from stenograf.pipeline import finalize_channel, relabel_speakers
from stenograf.transcript import Transcript, TranscriptEntry
from stenograf.vad import SileroVAD

_CHANNEL_LABEL = {Channel.MIC: "Local-{n}", Channel.SYSTEM: "Remote-{n}"}


class SessionStore:
    """Per-channel in-RAM PCM accumulation — int16, ~115 MB/hour/channel.

    Frames are placed by their session timestamp, not merely concatenated, so a
    gap between frames pads silence and both channels share one clock anchored
    at t=0. That shared clock is what lets the two channels' finalized entries
    interleave correctly. No audio is ever written to disk.

    Thread-safe for a single-writer/many-reader pattern (Phase 2 live pass): the
    capture thread :meth:`append`\\ s while the live worker :meth:`view`\\ s
    trailing windows. Each channel's chunk list is append-only, so any chunk once
    stored is immutable and never moves (prefix-immortal). Readers therefore only
    hold ``_lock`` long enough to snapshot the chunk references covering their
    window; the expensive concatenate runs outside the lock. ``_offsets`` mirrors
    ``_chunks`` with each chunk's start sample, so :meth:`view` bisects straight
    to the covering chunks — O(window), not O(whole buffer).
    """

    def __init__(self, channels: set[Channel]) -> None:
        self._chunks: dict[Channel, list[np.ndarray]] = {ch: [] for ch in channels}
        self._offsets: dict[Channel, list[int]] = {ch: [] for ch in channels}
        self._lengths: dict[Channel, int] = dict.fromkeys(channels, 0)
        self._lock = threading.Lock()

    def append(self, frame: AudioFrame) -> None:
        """Store a frame at its timestamp; frames must arrive in order per channel."""
        chunks = self._chunks.get(frame.channel)
        if chunks is None:
            return  # a channel we're not recording — ignore
        offset = round(frame.timestamp * SAMPLE_RATE)
        length = self._lengths[frame.channel]  # only append writes lengths, so this read is safe
        if offset < length - ORDER_TOLERANCE_SAMPLES:
            # A backward jump past jitter tolerance means the stream desynced;
            # appending here would silently misalign every later frame.
            raise ValueError(
                f"{frame.channel} frame went backwards "
                f"{(length - offset) / SAMPLE_RATE:.3f}s (timestamp {frame.timestamp:.3f}s "
                f"< buffered {length / SAMPLE_RATE:.3f}s); frames must arrive in order"
            )
        # Build the new chunks (silence pad for a gap, then the samples) outside
        # the lock so allocation never stalls a reader; a minor overlap (within
        # tolerance) just appends at the tail, keeping the clock monotonic.
        pad = np.zeros(offset - length, dtype=np.int16) if offset > length else None
        samples = np.asarray(frame.samples, dtype=np.int16)
        offsets = self._offsets[frame.channel]
        # One short critical section publishes the mutation atomically: chunks,
        # offsets, and length always agree when a reader observes them. Only
        # non-empty chunks are stored, so offsets stays strictly increasing.
        with self._lock:
            if pad is not None:
                offsets.append(length)
                chunks.append(pad)
                length = offset
            if len(samples):
                offsets.append(length)
                chunks.append(samples)
            self._lengths[frame.channel] = length + len(samples)

    def channels(self) -> list[Channel]:
        return list(self._chunks)  # keys are fixed at construction — no lock needed

    def samples(self, channel: Channel) -> np.ndarray:
        """The channel's full audio as mono 16 kHz float32 (empty if none)."""
        with self._lock:
            selected = list(self._chunks[channel])  # snapshot references, concat outside
        pcm = np.concatenate(selected) if selected else np.zeros(0, dtype=np.int16)
        return to_float32(pcm)

    def view(self, channel: Channel, start_s: float, end_s: float | None = None) -> np.ndarray:
        """A trailing ``[start_s, end_s)`` window as mono 16 kHz float32.

        ``end_s`` defaults to the current end of the buffer. Bounds are clamped
        to what exists, so a window that runs past the tail simply returns what
        is available (empty if the range is empty or already gone). Cost is
        O(window), not O(buffer) — this is the live pass's re-decode feed and is
        called every ~1–1.5 s, so it must never re-scan the whole session.
        """
        with self._lock:
            chunks = self._chunks[channel]
            offsets = self._offsets[channel]
            length = self._lengths[channel]
            start = max(0, min(round(start_s * SAMPLE_RATE), length))
            end = length if end_s is None else round(end_s * SAMPLE_RATE)
            end = max(start, min(end, length))
            if start >= end:
                return np.zeros(0, dtype=np.float32)
            lo = bisect_right(offsets, start) - 1  # chunk containing `start`
            hi = bisect_left(offsets, end)  # first chunk starting at/after `end`
            selected = chunks[lo:hi]
            base = offsets[lo]
        pcm = np.concatenate(selected)
        return to_float32(pcm[start - base : end - base])

    def duration(self, channel: Channel) -> float:
        with self._lock:
            return self._lengths[channel] / SAMPLE_RATE


@dataclass(frozen=True)
class ChannelPlan:
    """How one channel is transcribed: its speaker count and display label."""

    channel: Channel
    num_speakers: int | None  # None = estimate, 1 = single (no diarization), >1 = known
    label_template: str


def plan_channels(profile: MeetingProfile) -> list[ChannelPlan]:
    """Resolve which channels to record and each channel's speaker count.

    The mic is recorded unless there is explicitly no local speaker
    (``local_speakers == 0``, a listen-only session); an unknown local count
    (``None``) defaults to 1 — a single local user — because far-field
    local-count estimation is Phase 3 (PLAN.md §2). The system tap is recorded
    unless the meeting is in-room (``remote_speakers == 0``); an unknown remote
    count (``None``) means record it and estimate. ``MeetingProfile`` forbids
    both counts being 0, so at least one channel is always planned.
    """
    plans = []
    if profile.local_speakers != 0:
        local = 1 if profile.local_speakers is None else profile.local_speakers
        plans.append(ChannelPlan(Channel.MIC, local, _CHANNEL_LABEL[Channel.MIC]))
    if profile.remote_speakers != 0:
        plans.append(
            ChannelPlan(Channel.SYSTEM, profile.remote_speakers, _CHANNEL_LABEL[Channel.SYSTEM])
        )
    return plans


def interleave(entries: list[TranscriptEntry]) -> list[TranscriptEntry]:
    """Merge per-channel entries into one timeline, ordered by start time."""
    return sorted(entries, key=lambda e: (e.start, e.end))


OnUpdate = Callable[[Channel, StreamingUpdate], None]
"""Live-pass callback: a channel's newest committed words plus its provisional
grey tail. Task 5's ``LiveView`` implements it; until then only tests consume it."""


class AudioBus:
    """Event-driven hand-off from the capture thread to the single live worker.

    The capture thread :meth:`advance`\\ s each channel's watermark — the seconds
    of audio stored so far — and the worker :meth:`wait`\\ s on it, waking only
    when there is new audio (a ``Condition``, never a poll loop). Crucially,
    ``wait`` hands back the *latest* watermarks, so the worker reconciles straight
    to the current edge rather than replaying every intermediate step: if
    inference falls behind real time it simply decodes a longer catch-up window
    next, and no audio is ever lost because the samples themselves live in the
    :class:`SessionStore` (this is the backpressure). Closing the bus — capture is
    done — is the worker's cue to feed the final window, flush, and exit.
    """

    def __init__(self, channels: Iterable[Channel]) -> None:
        self._marks: dict[Channel, float] = {ch: 0.0 for ch in channels}
        self._closed = False
        self._cond = threading.Condition()

    def advance(self, channel: Channel, watermark: float) -> None:
        """Publish that ``channel`` now holds ``watermark`` seconds of audio."""
        with self._cond:
            if watermark > self._marks[channel]:
                self._marks[channel] = watermark
                self._cond.notify_all()

    def close(self) -> None:
        """Signal that no more audio is coming; wakes the worker one last time."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def wait(self, seen: dict[Channel, float]) -> tuple[dict[Channel, float], bool]:
        """Block until a watermark passes ``seen`` or the bus closes.

        ``seen`` is what the caller has already processed. Returns a snapshot of
        the current per-channel watermarks (the *latest*, for reconcile) and
        whether the bus is closed.
        """
        with self._cond:
            self._cond.wait_for(
                lambda: self._closed or any(m > seen.get(ch, 0.0) for ch, m in self._marks.items())
            )
            return dict(self._marks), self._closed


class CaptureLoop(threading.Thread):
    """Consumes provider frames into the store; never blocks on inference.

    The capture thread's only job is to keep audio moving: append each frame to
    the :class:`SessionStore`, tee it (optional recording), and advance the
    :class:`AudioBus` watermark. Inference lives on the :class:`LiveWorker`
    thread, so a slow decode can neither stall capture nor drop a frame. On
    ``max_seconds`` it stops the provider and exits; either way it closes the bus
    on the way out so the worker can drain and finish. A capture-side error (a
    stream desync) is stored and re-raised by the orchestrator after the threads
    join, matching the batch path.
    """

    def __init__(
        self,
        provider: CaptureProvider,
        store: SessionStore,
        bus: AudioBus,
        *,
        channels: list[Channel],
        on_frame: Callable[[AudioFrame], None] | None = None,
        max_seconds: float | None = None,
    ) -> None:
        super().__init__(name="capture-loop", daemon=True)
        self._provider = provider
        self._store = store
        self._bus = bus
        self._channels = channels
        self._on_frame = on_frame
        self._max_seconds = max_seconds
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            for frame in self._provider.frames():
                self._store.append(frame)
                if self._on_frame is not None:
                    self._on_frame(frame)
                self._bus.advance(frame.channel, self._store.duration(frame.channel))
                if self._max_seconds is not None:
                    captured = max(self._store.duration(ch) for ch in self._channels)
                    if captured >= self._max_seconds:
                        self._provider.stop()
                        break
        except Exception as exc:  # a desync (backward frame) etc. — surfaced on join
            self.error = exc
        finally:
            self._bus.close()


class LiveWorker(threading.Thread):
    """The single ASR inference thread — one :class:`LiveDecoder` per channel.

    Waits on the :class:`AudioBus`, and each wake feeds every channel the audio
    that arrived since it last looked (``store.view`` — O(window)), reconciled to
    the latest watermark. Being the *only* inference thread makes the live pass
    single-flight, so it never contends with the finalize pass for the one
    accelerator; ``inference_lock`` makes that guarantee explicit and is the
    extension point for a future overlapping finalize. On close it feeds the final
    window and :meth:`~stenograf.live.LiveDecoder.flush`\\ es each decoder to
    force-commit the tail.
    """

    def __init__(
        self,
        store: SessionStore,
        bus: AudioBus,
        decoders: dict[Channel, LiveDecoder],
        inference_lock: threading.Lock,
        *,
        channels: list[Channel],
        on_update: OnUpdate | None = None,
    ) -> None:
        super().__init__(name="live-worker", daemon=True)
        self._store = store
        self._bus = bus
        self._decoders = decoders
        self._inference_lock = inference_lock
        self._channels = channels
        self._on_update = on_update
        self.error: BaseException | None = None

    def run(self) -> None:
        seen: dict[Channel, float] = {ch: 0.0 for ch in self._channels}
        try:
            while True:
                marks, closed = self._bus.wait(seen)
                for ch in self._channels:
                    if marks[ch] > seen[ch]:
                        chunk = self._store.view(ch, seen[ch], marks[ch])
                        with self._inference_lock:
                            update = self._decoders[ch].feed(chunk, seen[ch])
                        seen[ch] = marks[ch]
                        self._emit(ch, update)
                if closed:
                    for ch in self._channels:
                        with self._inference_lock:
                            self._emit(ch, self._decoders[ch].flush())
                    return
        except Exception as exc:  # surfaced on join, like the capture thread
            self.error = exc

    def _emit(self, channel: Channel, update: StreamingUpdate) -> None:
        if self._on_update is not None and (update.committed or update.interim):
            self._on_update(channel, update)


def _join_until_done(thread: threading.Thread, poll: float = 0.1) -> None:
    """Join ``thread`` while staying responsive to Ctrl-C on the main thread.

    A bare ``Thread.join()`` can swallow signals on some platforms; joining in
    short slices lets a ``KeyboardInterrupt`` reach the main thread promptly so
    the meeting still finalizes what it captured.
    """
    while thread.is_alive():
        thread.join(poll)


class MeetingRecorder:
    """Drives a capture session and produces the merged, labelled transcript.

    Construct with the finalize backends (shared across channels — the diarizer
    is stateless per call and takes each channel's count as an argument), then
    call :meth:`run` with a provider. Backends are reused, so a wrong meeting
    parameter is corrected by re-running :meth:`finalize` on the same store.
    """

    def __init__(
        self,
        profile: MeetingProfile,
        *,
        asr: ASRBackend,
        vad: SileroVAD | None = None,
        diarizer: Diarizer | None = None,
        language: Language | None = None,
    ) -> None:
        self.profile = profile
        self.asr = asr
        self.vad = vad
        self.diarizer = diarizer
        self.language = language or profile.language

    def run(
        self,
        provider: CaptureProvider,
        *,
        on_frame: Callable[[AudioFrame], None] | None = None,
        on_status: Callable[[str], None] | None = None,
        on_checkpoint: Callable[[Transcript], None] | None = None,
        checkpoint_interval: float = 180.0,
        max_seconds: float | None = None,
        live: bool = False,
        on_update: OnUpdate | None = None,
    ) -> Transcript:
        """Capture until the provider stops (or Ctrl-C), then finalize.

        ``on_frame`` sees every stored frame (used by the audio tee); a
        ``KeyboardInterrupt`` ends capture gracefully rather than aborting, so
        an interrupted meeting still yields a transcript of what was captured.
        ``max_seconds`` stops capture automatically after that much audio.

        With ``live=True`` the meeting runs the streaming pass: capture on its own
        thread feeding a single :class:`LiveWorker` that drives a
        :class:`~stenograf.live.LiveDecoder` per channel and streams committed and
        interim words to ``on_update``. The heavy finalize still runs once on stop
        and replaces the whole live transcript. In the default batch mode there is
        no live view; instead, if ``on_checkpoint`` is given, the completed audio
        is re-finalized every ``checkpoint_interval`` seconds and handed to the
        callback — the crash-recovery safety net (PLAN.md §3), so a crash loses at
        most one interval of *text* and audio is still never persisted. Either
        way the authoritative transcript is the full finalize returned on stop.
        """
        plans = plan_channels(self.profile)
        store = SessionStore({p.channel for p in plans})
        if live:
            return self._run_live(
                provider,
                plans,
                store,
                on_frame=on_frame,
                on_status=on_status,
                on_update=on_update,
                max_seconds=max_seconds,
            )
        return self._run_batch(
            provider,
            plans,
            store,
            on_frame=on_frame,
            on_status=on_status,
            on_checkpoint=on_checkpoint,
            checkpoint_interval=checkpoint_interval,
            max_seconds=max_seconds,
        )

    def _run_batch(
        self,
        provider: CaptureProvider,
        plans: list[ChannelPlan],
        store: SessionStore,
        *,
        on_frame: Callable[[AudioFrame], None] | None,
        on_status: Callable[[str], None] | None,
        on_checkpoint: Callable[[Transcript], None] | None,
        checkpoint_interval: float,
        max_seconds: float | None,
    ) -> Transcript:
        """Single-threaded capture + periodic re-finalize checkpoint (no live view)."""
        checkpointing = on_checkpoint is not None and checkpoint_interval > 0
        next_checkpoint = checkpoint_interval
        provider.start({p.channel for p in plans})
        try:
            for frame in provider.frames():
                store.append(frame)
                if on_frame is not None:
                    on_frame(frame)
                captured = max(store.duration(p.channel) for p in plans)
                if checkpointing and captured >= next_checkpoint:
                    # Inline re-finalize of the whole captured-so-far. Fine for
                    # the file/replay providers; the live path (``_run_live``)
                    # instead keeps inference off the consume thread. Task 4
                    # replaces this with an Option-B committed-text flush.
                    on_checkpoint(self.finalize(store, plans))
                    while captured >= next_checkpoint:
                        next_checkpoint += checkpoint_interval
                if max_seconds is not None and captured >= max_seconds:
                    break
        except KeyboardInterrupt:
            if on_status is not None:
                on_status("interrupted — finalizing captured audio")
        finally:
            provider.stop()
        return self.finalize(store, plans, on_status=on_status)

    def _run_live(
        self,
        provider: CaptureProvider,
        plans: list[ChannelPlan],
        store: SessionStore,
        *,
        on_frame: Callable[[AudioFrame], None] | None,
        on_status: Callable[[str], None] | None,
        on_update: OnUpdate | None,
        max_seconds: float | None,
    ) -> Transcript:
        """Live pass: threaded capture + one inference worker, then finalize.

        Capture runs on :class:`CaptureLoop` (never blocked by inference); one
        :class:`LiveWorker` drives a :class:`~stenograf.live.LiveDecoder` per
        channel and streams updates to ``on_update``. On stop the worker is joined
        and the full finalize pass runs — it replaces the whole live transcript,
        so live compromises never reach the final output (PLAN.md §2).
        Checkpointing the live text to ``.partial`` is Task 4; here the live
        transcript is view-only and only the finalize result is persisted.
        """
        channels = [p.channel for p in plans]
        bus = AudioBus(channels)
        decoders = {
            ch: LiveDecoder(self.asr, vad=self.vad, language=self.language) for ch in channels
        }
        inference_lock = threading.Lock()
        worker = LiveWorker(
            store, bus, decoders, inference_lock, channels=channels, on_update=on_update
        )
        capture = CaptureLoop(
            provider, store, bus, channels=channels, on_frame=on_frame, max_seconds=max_seconds
        )

        provider.start(set(channels))
        worker.start()
        capture.start()
        try:
            _join_until_done(capture)
        except KeyboardInterrupt:
            if on_status is not None:
                on_status("interrupted — finalizing captured audio")
            provider.stop()
            capture.join()
        worker.join()
        provider.stop()  # idempotent — releases the device if capture ended on its own
        if capture.error is not None:
            raise capture.error
        # The live pass is provisional; if a decode failed, surface it but still
        # finalize — the finalize pass is the authoritative transcript regardless.
        if worker.error is not None and on_status is not None:
            on_status(f"live pass stopped early: {worker.error}")
        # Single-flight: the worker is already joined, but taking the same lock it
        # held documents (and future-proofs) that finalize never runs alongside a
        # live decode.
        with inference_lock:
            return self.finalize(store, plans, on_status=on_status)

    def finalize(
        self,
        store: SessionStore,
        plans: list[ChannelPlan] | None = None,
        *,
        on_status: Callable[[str], None] | None = None,
    ) -> Transcript:
        """Run the finalize pass on every stored channel and interleave them."""
        plans = plans or plan_channels(self.profile)
        entries: list[TranscriptEntry] = []
        for plan in plans:
            if plan.channel not in store.channels():
                continue
            if on_status is not None:
                on_status(f"finalizing {plan.channel} ({_speaker_note(plan.num_speakers)})")
            samples = store.samples(plan.channel)
            diarizer = None if plan.num_speakers == 1 else self.diarizer
            raw = finalize_channel(
                samples,
                asr=self.asr,
                language=self.language,
                vad=self.vad,
                diarizer=diarizer,
                num_speakers=plan.num_speakers,
            )
            entries.extend(relabel_speakers(raw, plan.label_template))
        interleaved = interleave(entries)
        language = self._resolve_language(interleaved, on_status=on_status)
        return Transcript(language=language, profile=self.profile, entries=interleaved)

    def _resolve_language(
        self,
        entries: list[TranscriptEntry],
        *,
        on_status: Callable[[str], None] | None = None,
    ) -> Language | None:
        """Fill the meeting language by LID over the transcript, at most once.

        An explicit user setting always wins. Otherwise detect from the
        finalized text and lock the result on ``self.language`` so later
        checkpoints stay consistent (PLAN.md §2 "auto-detect once … then lock").
        """
        if self.language is not None:
            return self.language
        detected = detect_language(" ".join(e.text for e in entries))
        if detected is not None:
            self.language = detected  # lock for the session
            if on_status is not None:
                on_status(f"detected language: {detected.value}")
        return detected


def _speaker_note(num_speakers: int | None) -> str:
    if num_speakers is None:
        return "estimating speakers"
    if num_speakers == 1:
        return "single speaker"
    return f"{num_speakers} speakers"
