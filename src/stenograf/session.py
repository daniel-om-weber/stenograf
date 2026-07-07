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

import contextlib
import signal
import threading
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

import numpy as np

from stenograf.asr.base import ASRBackend, Word
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
from stenograf.glossary import DEFAULT_THRESHOLD, apply_glossary
from stenograf.lid import detect_language
from stenograf.live import LiveDecoder, StreamingUpdate
from stenograf.pipeline import SpeakerResolver, finalize_channel, group_words, relabel_speakers
from stenograf.transcript import Transcript, TranscriptEntry
from stenograf.vad import SileroVAD
from stenograf.view import LiveView

_CHANNEL_LABEL = {Channel.MIC: "Local-{n}", Channel.SYSTEM: "Remote-{n}"}
# Channel-coarse labels for the crash checkpoints (live committed text or the
# batch tail finalize): the checkpoint is not diarized, so it can only say which
# channel spoke, not which speaker. The on-stop finalize replaces these with the
# diarized ``Local-N``/``Remote-M`` labels (PLAN.md §3 Option B).
_CHANNEL_COARSE = {Channel.MIC: "Local", Channel.SYSTEM: "Remote"}


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


@dataclass(frozen=True)
class SpeakerCount:
    """How many speakers a channel asked for vs how many the finalize found.

    ``requested`` is the plan's ``num_speakers`` (``None`` = estimated),
    ``detected`` the number of distinct speakers in the finalized transcript.
    Surfaced so the user can see an auto-detected count and, if it is wrong,
    re-run finalize with an explicit count (PLAN.md §5 Stage 3a)."""

    channel: Channel
    requested: int | None
    detected: int


def plan_channels(profile: MeetingProfile) -> list[ChannelPlan]:
    """Resolve which channels to record and each channel's speaker count.

    Every count follows the same resolution order as elsewhere — explicit
    setting > auto-detected > default — so an unknown count (``None``) means
    "record the channel and estimate", never a hard-coded value. The mic is
    recorded unless there is explicitly no local speaker (``local_speakers == 0``,
    a listen-only session); the system tap unless the meeting is in-room
    (``remote_speakers == 0``). Both channels estimate an unknown count: local
    estimation is far-field and weaker than remote (PLAN.md §2), so the finalize
    surfaces the detected count as editable. ``MeetingProfile`` forbids both
    counts being 0, so at least one channel is always planned.
    """
    plans = []
    if profile.local_speakers != 0:
        plans.append(
            ChannelPlan(Channel.MIC, profile.local_speakers, _CHANNEL_LABEL[Channel.MIC])
        )
    if profile.remote_speakers != 0:
        plans.append(
            ChannelPlan(Channel.SYSTEM, profile.remote_speakers, _CHANNEL_LABEL[Channel.SYSTEM])
        )
    return plans


def interleave(entries: list[TranscriptEntry]) -> list[TranscriptEntry]:
    """Merge per-channel entries into one timeline, ordered by start time."""
    return sorted(entries, key=lambda e: (e.start, e.end))


@contextlib.contextmanager
def _shield_interrupt() -> Iterator[None]:
    """Ignore SIGINT for the duration so an on-stop finalize runs to completion.

    Once capture has stopped, the finalize pass *is* the authoritative transcript
    and must not be lost to an impatient second Ctrl-C. Shielding SIGINT makes the
    finalize uninterruptible — it is bounded (seconds), and audio is already safe in
    RAM. Only the main thread can install signal handlers; off it (the TUI runs the
    meeting on a background thread, where Textual captures Ctrl-C anyway) this is a
    harmless no-op.
    """
    try:
        previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (ValueError, OSError):  # not on the main thread — cannot set a handler
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


OnUpdate = Callable[[Channel, StreamingUpdate], None]
"""Live-pass callback: a channel's newest committed words plus its provisional
grey tail. ``LiveView.update`` (view.py) has this signature, so a view wires
straight to the worker; tests pass a plain callback."""


class _CallbackView(LiveView):
    """Adapt the raw ``on_update``/``on_status`` callbacks to a :class:`LiveView`.

    The orchestrator drives a single sink internally (structured view events), so
    a caller passing plain callbacks — the tests, and the batch CLI's status echo
    — is wrapped in one of these. ``update``/``status`` forward to their callback;
    ``language``/``error`` fold onto ``on_status`` as text so a callback-only
    caller still sees them; ``finalizing``/``finalized`` have no string form and
    are dropped (only a rendering view cares about the finalize swap).
    """

    def __init__(
        self,
        on_update: OnUpdate | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self._on_update = on_update
        self._on_status = on_status

    def update(self, channel: Channel, update: StreamingUpdate) -> None:
        if self._on_update is not None:
            self._on_update(channel, update)

    def status(self, message: str) -> None:
        if self._on_status is not None:
            self._on_status(message)

    def language(self, language: Language) -> None:
        if self._on_status is not None:
            self._on_status(f"detected language: {language.value}")

    def error(self, message: str) -> None:
        if self._on_status is not None:
            self._on_status(message)


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

    Option B checkpointing (PLAN.md §3): every ``flush_interval`` seconds of
    processed audio the worker calls ``on_flush`` — a zero-inference hook that
    snapshots the decoders' already-committed text to ``.partial``. Doing it here,
    on the same thread that owns the decoders, needs no lock and never runs the
    accelerator; the reconcile means a backlog coalesces into a single flush.
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
        on_flush: Callable[[], None] | None = None,
        flush_interval: float = 0.0,
    ) -> None:
        super().__init__(name="live-worker", daemon=True)
        self._store = store
        self._bus = bus
        self._decoders = decoders
        self._inference_lock = inference_lock
        self._channels = channels
        self._on_update = on_update
        self._on_flush = on_flush
        self._flush_interval = flush_interval
        self.error: BaseException | None = None
        self.shed_seconds = 0.0  # audio skipped by load-shedding (observability + tests)

    def run(self) -> None:
        seen: dict[Channel, float] = {ch: 0.0 for ch in self._channels}
        flushing = self._on_flush is not None and self._flush_interval > 0
        next_flush = self._flush_interval
        try:
            while True:
                marks, closed = self._bus.wait(seen)
                for ch in self._channels:
                    if marks[ch] > seen[ch]:
                        start = self._shed_if_behind(ch, seen[ch], marks[ch])
                        chunk = self._store.view(ch, start, marks[ch])
                        with self._inference_lock:
                            update = self._decoders[ch].feed(chunk, start)
                        seen[ch] = marks[ch]
                        self._emit(ch, update)
                if flushing:
                    processed = max(seen.values())
                    if processed >= next_flush:
                        self._on_flush()  # type: ignore[misc]  # guarded by `flushing`
                        while processed >= next_flush:
                            next_flush += self._flush_interval
                if closed:
                    for ch in self._channels:
                        with self._inference_lock:
                            self._emit(ch, self._decoders[ch].flush())
                    return
        except Exception as exc:  # surfaced on join, like the capture thread
            self.error = exc

    def _shed_if_behind(self, channel: Channel, start: float, mark: float) -> float:
        """Drop the middle of an over-long backlog so a slow decode can't spiral.

        Normally the worker feeds every second of audio since it last looked. But
        if inference has fallen so far behind that the unprocessed backlog exceeds a
        full decode window, feeding it all at once would make that decode even
        larger — positive feedback that spirals below real time. Instead abandon the
        decoder's window and restart at the recent edge, feeding only the last
        ``window_cap`` seconds: the skipped span becomes a caption *gap* (the
        finalize pass fills it on stop), not an ever-growing decode. Returns the
        (possibly advanced) start second to feed from (PLAN.md §5, Task 0f)."""
        cap = self._decoders[channel].window_cap
        if mark - start <= cap:
            return start
        self.shed_seconds += (mark - cap) - start
        self._decoders[channel].drop_window()
        return mark - cap

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


class _TailCheckpointer(threading.Thread):
    """Batch (``--no-live``) crash checkpoint: tail-only finalize, off capture.

    Waits on the :class:`AudioBus` and, each time a channel accumulates another
    ``interval`` seconds, finalizes just that new tail (``store.view`` — O(window))
    and appends its entries to a running transcript flushed via ``on_checkpoint``.
    Each second of audio is finalized exactly once, so the whole run is O(audio),
    not the old whole-buffer re-finalize's O(n²); running on its own thread means a
    slow finalize never stalls capture, which on a live device would drop audio.

    The checkpoint is channel-coarse and un-diarized (``recorder._tail_entries``):
    diarizing each tail independently would renumber speakers every tail. The
    authoritative on-stop :meth:`MeetingRecorder.finalize` diarizes the whole
    buffer and supersedes it. On close the worker exits without finalizing the
    final sub-interval tail — a clean stop supersedes the checkpoint anyway, and a
    crash is defined to lose at most one interval of finalized text (PLAN.md §3).
    """

    def __init__(
        self,
        recorder: MeetingRecorder,
        store: SessionStore,
        plans: list[ChannelPlan],
        bus: AudioBus,
        on_checkpoint: Callable[[Transcript], None],
        interval: float,
    ) -> None:
        super().__init__(name="tail-checkpoint", daemon=True)
        self._recorder = recorder
        self._store = store
        self._plans = plans
        self._bus = bus
        self._on_checkpoint = on_checkpoint
        self._interval = interval
        self._entries: list[TranscriptEntry] = []
        self.error: BaseException | None = None

    def run(self) -> None:
        finalized = {p.channel: 0.0 for p in self._plans}
        next_cp = {p.channel: self._interval for p in self._plans}
        try:
            while True:
                marks, closed = self._bus.wait(finalized)
                flushed = False
                for plan in self._plans:
                    ch = plan.channel
                    if marks[ch] >= next_cp[ch]:
                        tail = self._recorder._tail_entries(
                            self._store, plan, finalized[ch], marks[ch]
                        )
                        self._entries.extend(tail)
                        finalized[ch] = marks[ch]
                        while marks[ch] >= next_cp[ch]:
                            next_cp[ch] += self._interval
                        flushed = True
                if flushed:
                    self._on_checkpoint(self._recorder._checkpoint_transcript(self._entries))
                if closed:
                    return
        except Exception as exc:  # surfaced on join, like the capture thread
            self.error = exc


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
        reid: SpeakerResolver | None = None,
        language: Language | None = None,
        glossary_threshold: float | None = None,
    ) -> None:
        self.profile = profile
        self.asr = asr
        self.vad = vad
        self.diarizer = diarizer
        self.reid = reid
        self.language = language or profile.language
        self.glossary_threshold = (
            DEFAULT_THRESHOLD if glossary_threshold is None else glossary_threshold
        )
        self.speaker_counts: list[SpeakerCount] = []
        """Per-channel requested-vs-detected speaker counts from the last
        :meth:`finalize`; the CLI reports estimated counts as editable."""

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
        view: LiveView | None = None,
    ) -> Transcript:
        """Capture until the provider stops (or Ctrl-C), then finalize.

        ``on_frame`` sees every stored frame (used by the audio tee); a
        ``KeyboardInterrupt`` ends capture gracefully rather than aborting, so
        an interrupted meeting still yields a transcript of what was captured.
        ``max_seconds`` stops capture automatically after that much audio.

        With ``live=True`` the meeting runs the streaming pass: capture on its own
        thread feeding a single :class:`LiveWorker` that drives a
        :class:`~stenograf.live.LiveDecoder` per channel and streams committed and
        interim words to the view. The heavy finalize still runs once on stop and
        replaces the whole live transcript.

        Events go to a single :class:`~stenograf.view.LiveView` sink: pass a
        concrete ``view`` (the CLI's TUI / plain view), or the legacy
        ``on_update``/``on_status`` callbacks, which are adapted to a view. The
        orchestrator emits the structured lifecycle events on it — ``status`` /
        ``language`` / ``finalizing`` / ``finalized`` / ``error`` — around the
        capture and finalize passes.

        Both modes checkpoint for crash recovery (PLAN.md §3 Option B), if
        ``on_checkpoint`` is given, coalesced to ``checkpoint_interval`` seconds of
        capture — but never any inference the mode does not already do. Live: the
        already-committed live text is flushed as-is (zero inference). Batch: only
        the *new* tail since the last checkpoint is finalized (O(audio), off the
        capture thread), not the whole buffer. Either way a crash loses at most one
        interval of text, audio is never persisted, and the authoritative
        transcript is the full finalize returned on stop.
        """
        plans = plan_channels(self.profile)
        store = SessionStore({p.channel for p in plans})
        sink = view if view is not None else _CallbackView(on_update, on_status)
        if live:
            return self._run_live(
                provider,
                plans,
                store,
                on_frame=on_frame,
                view=sink,
                on_checkpoint=on_checkpoint,
                checkpoint_interval=checkpoint_interval,
                max_seconds=max_seconds,
            )
        return self._run_batch(
            provider,
            plans,
            store,
            on_frame=on_frame,
            view=sink,
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
        view: LiveView,
        on_checkpoint: Callable[[Transcript], None] | None,
        checkpoint_interval: float,
        max_seconds: float | None,
    ) -> Transcript:
        """Consume-thread capture + a tail-only checkpoint thread (no live view).

        Capture stays on this thread (so a ``KeyboardInterrupt`` in the provider
        ends the meeting cleanly), but the crash checkpoint is a separate
        :class:`_TailCheckpointer` fed via an :class:`AudioBus`: it finalizes only
        the newest tail each interval, off this thread, so it neither stalls
        capture nor re-finalizes the whole buffer (PLAN.md §3 Option B).
        """
        channels = [p.channel for p in plans]
        checkpointing = on_checkpoint is not None and checkpoint_interval > 0
        bus = AudioBus(channels) if checkpointing else None
        checkpointer: _TailCheckpointer | None = None
        if bus is not None and on_checkpoint is not None:
            checkpointer = _TailCheckpointer(
                self, store, plans, bus, on_checkpoint, checkpoint_interval
            )
            checkpointer.start()
        provider.start(set(channels))
        try:
            for frame in provider.frames():
                store.append(frame)
                if on_frame is not None:
                    on_frame(frame)
                if bus is not None:
                    bus.advance(frame.channel, store.duration(frame.channel))
                if max_seconds is not None:
                    captured = max(store.duration(ch) for ch in channels)
                    if captured >= max_seconds:
                        break
        except KeyboardInterrupt:
            view.status("interrupted — finalizing captured audio")
        finally:
            provider.stop()
            if bus is not None:
                bus.close()  # wakes the checkpointer so it drains and exits
            if checkpointer is not None:
                checkpointer.join()
        if checkpointer is not None and checkpointer.error is not None:
            view.error(f"checkpoint stopped early: {checkpointer.error}")
        view.finalizing()
        # Capture has stopped; the finalize is authoritative. Shield it from a
        # second Ctrl-C so an impatient interrupt cannot discard the transcript.
        with _shield_interrupt():
            transcript = self.finalize(store, plans, view=view)
        view.finalized(transcript)
        return transcript

    def _run_live(
        self,
        provider: CaptureProvider,
        plans: list[ChannelPlan],
        store: SessionStore,
        *,
        on_frame: Callable[[AudioFrame], None] | None,
        view: LiveView,
        on_checkpoint: Callable[[Transcript], None] | None,
        checkpoint_interval: float,
        max_seconds: float | None,
    ) -> Transcript:
        """Live pass: threaded capture + one inference worker, then finalize.

        Capture runs on :class:`CaptureLoop` (never blocked by inference); one
        :class:`LiveWorker` drives a :class:`~stenograf.live.LiveDecoder` per
        channel and streams updates to ``on_update``. On stop the worker is joined
        and the full finalize pass runs — it replaces the whole live transcript,
        so live compromises never reach the final output (PLAN.md §2).

        Option B checkpointing (PLAN.md §3): the worker flushes the decoders'
        already-committed text to ``on_checkpoint`` every ``checkpoint_interval``
        seconds — pure file I/O, no extra inference, since the live pass already
        produced that text. Empty flushes (nothing committed yet) are skipped so a
        ``.partial`` only appears once there is text to recover.
        """
        channels = [p.channel for p in plans]
        bus = AudioBus(channels)
        decoders = {
            ch: LiveDecoder(self.asr, vad=self.vad, language=self.language) for ch in channels
        }
        inference_lock = threading.Lock()

        def flush_checkpoint() -> None:
            transcript = self._live_checkpoint(decoders)
            if transcript.entries:
                on_checkpoint(transcript)  # type: ignore[misc]  # None-guarded below

        checkpointing = on_checkpoint is not None and checkpoint_interval > 0
        worker = LiveWorker(
            store,
            bus,
            decoders,
            inference_lock,
            channels=channels,
            on_update=view.update,
            on_flush=flush_checkpoint if checkpointing else None,
            flush_interval=checkpoint_interval,
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
            view.status("interrupted — finalizing captured audio")
            provider.stop()
            capture.join()
        # Capture has stopped; from here the finalize is authoritative and must not
        # be lost to a second Ctrl-C. Shield SIGINT across the worker join and the
        # finalize (a no-op off the main thread, e.g. under the TUI).
        with _shield_interrupt():
            worker.join()
            provider.stop()  # idempotent — releases the device if capture ended on its own
            if capture.error is not None:
                raise capture.error
            # The live pass is provisional; if a decode failed, surface it but still
            # finalize — the finalize pass is the authoritative transcript regardless.
            if worker.error is not None:
                view.error(f"live pass stopped early: {worker.error}")
            view.finalizing()
            # Single-flight: the worker is already joined, but taking the same lock it
            # held documents (and future-proofs) that finalize never runs alongside a
            # live decode.
            with inference_lock:
                transcript = self.finalize(store, plans, view=view)
        view.finalized(transcript)
        return transcript

    def finalize(
        self,
        store: SessionStore,
        plans: list[ChannelPlan] | None = None,
        *,
        view: LiveView | None = None,
    ) -> Transcript:
        """Run the finalize pass on every stored channel and interleave them."""
        plans = plans or plan_channels(self.profile)
        view = view or _CallbackView()
        entries: list[TranscriptEntry] = []
        counts: list[SpeakerCount] = []
        for plan in plans:
            if plan.channel not in store.channels():
                continue
            view.status(f"finalizing {plan.channel} ({_speaker_note(plan.num_speakers)})")
            samples = store.samples(plan.channel)
            diarizer = None if plan.num_speakers == 1 else self.diarizer
            raw = self._finalize_channel_safe(samples, diarizer, plan, view)
            labeled = relabel_speakers(raw, plan.label_template)
            detected = len({e.speaker for e in labeled})
            counts.append(SpeakerCount(plan.channel, plan.num_speakers, detected))
            if plan.num_speakers is None:
                view.status(f"{plan.channel}: detected {detected} speaker(s)")
            entries.extend(labeled)
        self.speaker_counts = counts
        interleaved = interleave(entries)
        # Snap domain vocabulary / attendee names to canonical spelling on the
        # authoritative transcript only (checkpoints stay raw — PLAN.md §5 Task 2b).
        interleaved = apply_glossary(
            interleaved,
            glossary=self.profile.glossary,
            attendee_names=self.profile.attendee_names,
            threshold=self.glossary_threshold,
        )
        language = self._resolve_language(interleaved, view=view)
        return Transcript(language=language, profile=self.profile, entries=interleaved)

    def _finalize_channel_safe(
        self,
        samples,
        diarizer: Diarizer | None,
        plan: ChannelPlan,
        view: LiveView,
    ) -> list[TranscriptEntry]:
        """Finalize one channel, never letting its failure lose another channel.

        Diarization is the fragile step (a real backend can raise on unexpected
        audio, as the sherpa path is otherwise untested). On failure, retry
        without diarization so the channel's *text* still survives — attributed
        to a single speaker rather than dropped. If even the un-diarized pass
        fails, skip this channel and keep the rest of the meeting.
        """
        try:
            return finalize_channel(
                samples,
                asr=self.asr,
                language=self.language,
                vad=self.vad,
                diarizer=diarizer,
                num_speakers=plan.num_speakers,
                reid=self.reid,
            )
        except Exception as exc:  # noqa: BLE001 — resilience across channels is the point
            if diarizer is None:
                view.error(f"{plan.channel}: finalize failed ({exc}); skipping channel")
                return []
            view.error(
                f"{plan.channel}: diarization failed ({exc}); transcribing without speaker labels"
            )
            try:
                return finalize_channel(
                    samples,
                    asr=self.asr,
                    language=self.language,
                    vad=self.vad,
                    diarizer=None,
                    num_speakers=1,
                )
            except Exception as exc2:  # noqa: BLE001
                view.error(f"{plan.channel}: finalize failed ({exc2}); skipping channel")
                return []

    def _live_checkpoint(self, decoders: dict[Channel, LiveDecoder]) -> Transcript:
        """A crash checkpoint from the live pass's already-committed words.

        Zero inference: the words are read straight off each channel's decoder and
        grouped into entries under a channel-coarse label (the live pass has no
        diarization). The on-stop :meth:`finalize` replaces this entirely.
        """
        entries: list[TranscriptEntry] = []
        for channel, decoder in decoders.items():
            entries.extend(group_words(list(decoder.committed_words), _CHANNEL_COARSE[channel]))
        return self._checkpoint_transcript(entries)

    def _tail_entries(
        self, store: SessionStore, plan: ChannelPlan, start_s: float, end_s: float
    ) -> list[TranscriptEntry]:
        """Finalize one channel's ``[start_s, end_s)`` tail into coarse entries.

        The batch (``--no-live``) crash checkpoint: VAD + ASR over just the new
        tail (O(window)), no diarization, times shifted back onto the session
        clock and attributed to the channel-coarse label. Speaker identity is the
        on-stop finalize's job; here each tail is finalized exactly once.
        """
        view = store.view(plan.channel, start_s, end_s)
        raw = finalize_channel(
            view,
            asr=self.asr,
            language=self.language,
            vad=self.vad,
            diarizer=None,
            num_speakers=1,
        )
        label = _CHANNEL_COARSE[plan.channel]
        return [
            TranscriptEntry(
                label,
                e.text,
                e.start + start_s,
                e.end + start_s,
                e.provisional,
                words=tuple(
                    Word(w.text, w.start + start_s, w.end + start_s, w.confidence) for w in e.words
                ),
            )
            for e in raw
        ]

    def _checkpoint_transcript(self, entries: list[TranscriptEntry]) -> Transcript:
        """Wrap accumulated coarse checkpoint entries into an ordered transcript.

        Keeps ``self.language`` as-is (explicit setting or ``None``): a checkpoint
        never locks the auto-detected language — that happens once, in the on-stop
        :meth:`finalize`, over the authoritative text.
        """
        return Transcript(language=self.language, profile=self.profile, entries=interleave(entries))

    def _resolve_language(
        self,
        entries: list[TranscriptEntry],
        *,
        view: LiveView,
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
            view.language(detected)
        return detected


def _speaker_note(num_speakers: int | None) -> str:
    if num_speakers is None:
        return "estimating speakers"
    if num_speakers == 1:
        return "single speaker"
    return f"{num_speakers} speakers"
