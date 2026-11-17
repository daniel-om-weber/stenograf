"""Meeting orchestrator: capture frames → in-RAM store → finalized transcript.

This is the spine that turns the accuracy core (``pipeline.finalize_channel``)
into a live meeting. It:

1. consumes ``AudioFrame`` objects from any :class:`CaptureProvider`,
2. accumulates each channel's PCM in a bounded in-RAM store (never disk),
3. on stop, runs the finalize pass **per channel** with that channel's
   speaker count, if stated (one skips diarization, more caps it), then
4. interleaves the two channels' entries into one timeline, labelling the mic
   channel ``Local-N`` and the system channel ``Remote-N``.

The channel prior is free separation: diarization never has to tell local from
remote voices, only voices *within* a channel. The provider is swappable
(Swift helper on macOS, sounddevice on Linux, file replay for dev/test), so the
whole orchestrator is exercisable without native capture.

Speaker bleed (remote audio leaving the speakers and re-entering the mic) is an
echo-cancellation problem, handled upstream of this store against the system
channel as the far-end reference. AEC never removes all of it, so cross-channel
text dedup at merge time is the backstop.
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import os
import re
import signal
import sys
import threading
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from difflib import SequenceMatcher

import numpy as np

from stenograf.asr.base import ASRBackend, Segment, Word
from stenograf.audio import merge_voices, to_float32, to_int16
from stenograf.capture.base import (
    ORDER_TOLERANCE_SAMPLES,
    SAMPLE_RATE,
    AudioFrame,
    CaptureProvider,
    Channel,
)
from stenograf.config import (
    Language,
    MeetingProfile,
    ResolvedParameters,
    resolve_value,
)
from stenograf.diarization.base import DiarizationStream, Diarizer, SpeakerTurn
from stenograf.glossary import DEFAULT_THRESHOLD
from stenograf.live import StreamingUpdate, WindowedLiveDecoder
from stenograf.pipeline import (
    SpeakerResolver,
    assemble_transcript,
    decode_channel,
    finalize_channel,
    group_words,
    raw_label_map,
    relabel_speakers,
)
from stenograf.transcript import MutedSpan, Transcript, TranscriptEntry, speech_seconds
from stenograf.vad import SileroVAD
from stenograf.view import LiveView

_CHANNEL_LABEL = {Channel.MIC: "Local-{n}", Channel.SYSTEM: "Remote-{n}"}
# Channel-coarse labels for the crash checkpoint (the live pass's committed
# text): the checkpoint is not diarized, so it can only say which channel
# spoke, not which speaker. The on-stop finalize replaces these with the
# diarized ``Local-N``/``Remote-M`` labels.
_CHANNEL_COARSE = {Channel.MIC: "Local", Channel.SYSTEM: "Remote"}


class SessionStore:
    """Per-channel in-RAM PCM accumulation — int16, ~115 MB/hour/channel.

    Frames are placed by their session timestamp, not merely concatenated, so a
    gap between frames pads silence and both channels share one clock anchored
    at t=0. That shared clock is what lets the two channels' finalized entries
    interleave correctly. No audio is ever written to disk.

    Thread-safe for a single-writer/many-reader pattern (the live pass): the
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
        end = None if end_s is None else round(end_s * SAMPLE_RATE)
        return self.read(channel, round(start_s * SAMPLE_RATE), end)

    def read(self, channel: Channel, start: int, end: int | None = None) -> np.ndarray:
        """Samples ``[start, end)`` as mono 16 kHz float32 — :meth:`view` in
        sample indices, clamped the same way."""
        with self._lock:
            chunks = self._chunks[channel]
            offsets = self._offsets[channel]
            length = self._lengths[channel]
            start = max(0, min(start, length))
            end = length if end is None else max(start, min(end, length))
            if start >= end:
                return np.zeros(0, dtype=np.float32)
            lo = bisect_right(offsets, start) - 1  # chunk containing `start`
            hi = bisect_left(offsets, end)  # first chunk starting at/after `end`
            selected = chunks[lo:hi]
            base = offsets[lo]
        pcm = np.concatenate(selected)
        return to_float32(pcm[start - base : end - base])

    def num_samples(self, channel: Channel) -> int:
        with self._lock:
            return self._lengths[channel]

    def duration(self, channel: Channel) -> float:
        return self.num_samples(channel) / SAMPLE_RATE


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
    re-run finalize with an explicit count."""

    channel: Channel
    requested: int | None
    detected: int


@dataclass(frozen=True)
class MeetingResult:
    """Everything one :meth:`MeetingRecorder.run` / ``finalize`` call produced.

    The transcript is the artifact; the rest is the run's report, surfaced by
    the CLI (requested-vs-detected speaker counts, echo-backstop drops, how
    long the canceller ran without its reference). Returned rather than stored
    on the recorder, so the recorder carries configuration only and is
    reentrant — nothing bleeds from one meeting into the next.
    """

    transcript: Transcript
    speaker_counts: list[SpeakerCount]
    """Per-channel requested-vs-detected counts; the CLI reports estimated
    counts as editable."""
    dropped_echo_lines: int = 0
    """Mic lines the finalize dropped as echoed remote speech; the CLI folds
    this into its degraded-reference warning."""
    reference_gap_s: float | None = None
    """Seconds the canceller cancelled against silence because the far-end
    reference never arrived (see :func:`_reference_gap`); ``None`` = no
    canceller was observed."""
    speaker_embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    """Each speaker's voice embedding under the label the transcript shows
    (``Local-N``/``Remote-N`` or a matched profile name) — what a later "this
    speaker is …" correction enrolls from (``steno profiles assign``).
    Diarized channels contribute per-cluster embeddings; a solo channel
    contributes one over its transcribed speech. Empty when the meeting ran
    without its speaker machinery (the diarization switch off)."""
    voice_clips: dict[str, list[np.ndarray]] = field(default_factory=dict)
    """A few seconds of each speaker's voice (int16, 16 kHz mono) under the
    label the transcript shows, cut while the audio still exists — what the
    app's "Who spoke?" plays so a voice can be recognised by ear. In memory
    only: never written anywhere. Empty without the speaker machinery."""


CLIP_COUNT = 3
CLIP_SECONDS = 4.0
"""Voice samples per speaker, and the longest one: enough to recognise a
voice, short enough that holding them costs ~0.4 MB per speaker."""


def cut_voice_clips(
    entries: list[TranscriptEntry], read: Callable[[float, float], np.ndarray]
) -> list[np.ndarray]:
    """Up to :data:`CLIP_COUNT` samples of one speaker's voice, in meeting order.

    From their longest lines nobody spoke over (an overlapped line is flagged
    provisional — its audio carries two voices), falling back to any line;
    each sample is the middle :data:`CLIP_SECONDS` of its line. ``read(start,
    end)`` returns that span of the speaker's channel as float32."""
    clean = [e for e in entries if not e.provisional] or entries
    longest = sorted(clean, key=lambda e: e.start - e.end)[:CLIP_COUNT]
    chosen = sorted(longest, key=lambda e: e.start)
    clips = []
    for e in chosen:
        middle, half = (e.start + e.end) / 2, min(CLIP_SECONDS, e.end - e.start) / 2
        clip = read(max(0.0, middle - half), middle + half)
        if len(clip):
            clips.append(to_int16(clip))
    return clips


def plan_channels(profile: MeetingProfile) -> list[ChannelPlan]:
    """Resolve which channels to record and each channel's speaker count.

    Every count follows the same resolution order as elsewhere — explicit
    setting > auto-detected > default — so an unknown count (``None``) means
    "record the channel and estimate", never a hard-coded value. The mic is
    recorded unless there is explicitly no local speaker (``local_speakers == 0``,
    a listen-only session); the system tap unless the meeting is in-room
    (``remote_speakers == 0``). Both channels estimate an unknown count: local
    estimation is far-field and weaker than remote, so the finalize
    surfaces the detected count as editable. ``MeetingProfile`` forbids both
    counts being 0, so at least one channel is always planned.
    """
    plans = []
    if profile.local_speakers != 0:
        plans.append(ChannelPlan(Channel.MIC, profile.local_speakers, _CHANNEL_LABEL[Channel.MIC]))
    if profile.remote_speakers != 0:
        plans.append(
            ChannelPlan(Channel.SYSTEM, profile.remote_speakers, _CHANNEL_LABEL[Channel.SYSTEM])
        )
    return plans


def resolve_parameters(
    profile: MeetingProfile,
    *,
    language: Language | None,
    speaker_counts: list[SpeakerCount],
) -> ResolvedParameters:
    """Record how each meeting parameter was resolved for the finalized transcript.

    ``language`` is the finalize pass's resolved language (the explicit setting if
    given, else the LID result, else ``None``); ``speaker_counts`` are the
    per-channel requested-vs-detected counts. Explicit profile values are tagged
    ``explicit``, otherwise a detected value is ``detected``, otherwise the
    parameter is left at a ``default`` — see
    :func:`~stenograf.config.resolve_value`. Both channels are always recorded
    so an explicit ``0`` ("this channel is off") is captured too.
    """
    detected = {c.channel: c.detected for c in speaker_counts}
    return ResolvedParameters(
        language=resolve_value(profile.language, language),
        speakers={
            Channel.MIC.value: resolve_value(profile.local_speakers, detected.get(Channel.MIC)),
            Channel.SYSTEM.value: resolve_value(
                profile.remote_speakers, detected.get(Channel.SYSTEM)
            ),
        },
    )


def interleave(entries: list[TranscriptEntry]) -> list[TranscriptEntry]:
    """Merge per-channel entries into one timeline, ordered by start time."""
    return sorted(entries, key=lambda e: (e.start, e.end))


ECHO_COVERAGE = 0.8
"""Fraction of a mic line that must also appear in an overlapping remote line
before we call it echo. Measured on real leaked lines, echoes score 0.89-1.00 and
genuine local speech scores 0.27-0.60, so the threshold sits in a wide gap."""

ECHO_WINDOW_S = 2.0
"""How far apart the two lines' spans may sit and still describe the same moment.
The echo reaches the mic within ~25 ms, but the two channels are transcribed
independently, so their segment boundaries drift apart by a second or so."""

_ECHO_MIN_WORDS = 3
"""Below this, a match means nothing: the local speaker agreeing ("yeah", "right")
while the remote says the same word is a real utterance, not an echo."""

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def _words_of(text: str) -> list[str]:
    return _PUNCT.sub(" ", text.lower()).split()


def _covered_by(mic: str, system: str) -> float:
    """How much of the mic line reads as a contiguous slice of the remote line.

    Characters, not words: the two channels are decoded independently, so an echo
    returns with small ASR divergences. The remote's "we cannot say sorry" came
    back on the mic as "we can't not say sorry" — 0.57 by word coverage, which is
    indistinguishable from unrelated speech, but 0.89 by character coverage.
    Summing matching blocks rather than taking one longest run lets a match
    survive a substituted word in the middle of an otherwise identical line.

    Normalized by the *longer* of the mic line and the span its matches occupy in
    the remote line — not by the mic line alone. Almost any short utterance is a
    chance character-subsequence of a long enough remote monologue ("no I don't
    think so" scored 1.00 against a 56-word remote line and was destroyed), but
    those chance matches scatter across the monologue, while a real echo lands in
    one dense stretch. The span denominator kills the scattered case and leaves
    genuine echoes — including an echo of one sentence inside a longer remote
    line — untouched.
    """
    if not mic:
        return 0.0
    matcher = SequenceMatcher(None, mic, system, autojunk=False)
    blocks = [b for b in matcher.get_matching_blocks() if b.size]
    if not blocks:
        return 0.0
    matched = sum(b.size for b in blocks)
    span = blocks[-1].b + blocks[-1].size - blocks[0].b
    return matched / max(len(mic), span)


def drop_echo_duplicates(
    mic: list[TranscriptEntry],
    system: list[TranscriptEntry],
    *,
    coverage: float = ECHO_COVERAGE,
    window: float = ECHO_WINDOW_S,
) -> list[TranscriptEntry]:
    """Remove mic lines that are really the remote speakers, echoed back.

    The backstop to :mod:`stenograf.aec`. Cancellation measured 30 dB on this
    hardware, which leaves a residual that is quiet but still speech-shaped —
    quiet enough to ignore, loud enough for the ASR to decode into a duplicate
    ``Local-N`` line. Audio-domain gates cannot separate that residual from
    genuine double-talk (real speech correlates with the reference too), so the
    decision is made on text, where an echo is unmistakable: the same words, at
    the same moment, on both channels. The system-channel copy is authoritative.
    """
    kept = []
    for entry in mic:
        words = _words_of(entry.text)
        if len(words) < _ECHO_MIN_WORDS:
            kept.append(entry)
            continue
        text = " ".join(words)
        overlapping = (
            other
            for other in system
            if other.end >= entry.start - window and other.start <= entry.end + window
        )
        if any(
            _covered_by(text, " ".join(_words_of(other.text))) >= coverage for other in overlapping
        ):
            continue
        kept.append(entry)
    return kept


@contextlib.contextmanager
def _shield_interrupt() -> Iterator[None]:
    """Ignore SIGINT for the duration so an on-stop finalize runs to completion.

    Once capture has stopped, the finalize pass *is* the authoritative transcript
    and must not be lost to an impatient second Ctrl-C. Shielding SIGINT makes the
    finalize uninterruptible — it is bounded (seconds), and audio is already safe in
    RAM. Only the main thread can install signal handlers; off it (the Qt app runs the
    meeting on a worker thread, where no terminal delivers Ctrl-C anyway) this is a
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


@contextlib.contextmanager
def _closing(streams: dict[Channel, DiarizationStream]) -> Iterator[None]:
    """Release every diarization stream on the way out, however finalize ends."""
    try:
        yield
    finally:
        for stream in streams.values():
            stream.close()


OnUpdate = Callable[[Channel, StreamingUpdate], None]
"""Live-pass callback: a channel's newest committed words plus its provisional
grey tail. ``LiveView.update`` (view.py) has this signature, so a view wires
straight to the worker."""


LIVE_FLUSH_INTERVAL_S = 15.0
"""Default checkpoint cadence. A live checkpoint is zero-inference — it
snapshots the captions the live pass already committed, a few KB of atomic
file I/O — so it can afford to be tight (a crash loses seconds of text, not
minutes)."""


@dataclass(frozen=True)
class CheckpointConfig:
    """Crash-checkpoint wiring — live runs only (batch runs never checkpoint).

    ``write`` receives each coalesced checkpoint transcript (the CLI wires the
    ``.partial`` writer here); ``interval`` is the seconds of captured audio
    between flushes. An ``interval`` <= 0 disables checkpointing, as does not
    passing a config at all.
    """

    write: Callable[[Transcript], None]
    interval: float = LIVE_FLUSH_INTERVAL_S

    @property
    def enabled(self) -> bool:
        return self.interval > 0


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

    def marks(self) -> dict[Channel, float]:
        """The current per-channel watermarks, without waiting."""
        with self._cond:
            return dict(self._marks)

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


class MuteSwitch:
    """The user's per-channel mute, as spans on the session clock.

    A press is stamped with the time being *captured* at that instant (the
    provider's :meth:`~CaptureProvider.now`), and :meth:`apply` silences every
    sample inside a span whenever its frame is consumed. Stamping the consumer
    instead would fail at a meeting's start, where frames queue for seconds
    while the models load: a call muted and unmuted in that time would be
    consumed after the unmute, and kept.

    Applied after the echo canceller (at :func:`_consume_frame`), so the
    canceller keeps the real far-end reference while the system channel is
    muted, and the store, the audio tee and the live pass all get the same
    silence. Written from any thread (the GUI), read by the capture thread.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._spans: dict[Channel, list[tuple[float, float | None]]] = {}
        self._clock: Callable[[], float | None] = lambda: None
        self._consumed = 0.0  # fallback clock: the furthest frame end applied

    def bind(self, clock: Callable[[], float | None]) -> None:
        """Stamp presses on ``clock`` (the provider's ``now``) from here on."""
        self._clock = clock

    def set(self, channel: Channel, muted: bool) -> None:
        """Mute or unmute ``channel`` as of now; repeating a state is a no-op."""
        now = self._clock()
        now = self._consumed if now is None else now
        with self._lock:
            spans = self._spans.setdefault(channel, [])
            open_start = spans[-1][0] if spans and spans[-1][1] is None else None
            if muted == (open_start is not None):
                return
            if muted:
                # Never before the previous span's end, so spans stay ordered.
                floor = spans[-1][1] if spans else None
                spans.append((max(now, floor or 0.0), None))
            else:
                assert open_start is not None
                spans[-1] = (open_start, max(now, open_start))

    def muted(self, channel: Channel) -> bool:
        with self._lock:
            spans = self._spans.get(channel)
            return bool(spans) and spans[-1][1] is None

    def spans(self, until: float) -> list[MutedSpan]:
        """Every span, an open one closed at ``until``; empty spans dropped."""
        with self._lock:
            found = [
                MutedSpan(channel.value, start, max(start, until) if end is None else end)
                for channel, spans in self._spans.items()
                for start, end in spans
            ]
        return sorted((span for span in found if span.end > span.start), key=lambda m: m.start)

    def apply(self, frame: AudioFrame) -> AudioFrame:
        """``frame`` with the samples inside a span silenced (a copy if any)."""
        start = frame.timestamp
        end = start + len(frame.samples) / SAMPLE_RATE
        self._consumed = max(self._consumed, end)
        with self._lock:
            hits = [
                (s0, end if s1 is None else s1)
                for s0, s1 in self._spans.get(frame.channel, ())
                if s0 < end and (s1 is None or s1 > start)
            ]
        if not hits:
            return frame
        samples = np.array(frame.samples, dtype=np.int16, copy=True)
        for s0, s1 in hits:
            lo = max(0, round((s0 - start) * SAMPLE_RATE))
            hi = min(len(samples), round((s1 - start) * SAMPLE_RATE))
            samples[lo:hi] = 0
        return dc_replace(frame, samples=samples)


def _consume_frame(
    frame: AudioFrame,
    store: SessionStore,
    bus: AudioBus | None,
    on_frame: Callable[[AudioFrame], None] | None,
    channels: list[Channel],
    max_seconds: float | None,
    mute: MuteSwitch | None = None,
) -> bool:
    """Store one captured frame (tee + watermark); True once ``max_seconds`` is hit.

    The shared per-frame body of both capture loops. The loops themselves stay
    separate on purpose: batch keeps capture on the calling thread so a
    ``KeyboardInterrupt`` in the provider ends the meeting cleanly, while the
    live pass runs it on :class:`CaptureLoop`.
    """
    if mute is not None:
        frame = mute.apply(frame)
    store.append(frame)
    if on_frame is not None:
        on_frame(frame)
    if bus is not None:
        bus.advance(frame.channel, store.duration(frame.channel))
    return max_seconds is not None and max(store.duration(ch) for ch in channels) >= max_seconds


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
        mute: MuteSwitch | None = None,
    ) -> None:
        super().__init__(name="capture-loop", daemon=True)
        self._mute = mute
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
                if _consume_frame(
                    frame,
                    self._store,
                    self._bus,
                    self._on_frame,
                    self._channels,
                    self._max_seconds,
                    self._mute,
                ):
                    self._provider.stop()
                    break
        except Exception as exc:  # a desync (backward frame) etc. — surfaced on join
            self.error = exc
        finally:
            self._bus.close()


class LiveWorker(threading.Thread):
    """The single ASR inference thread — one decoder per channel.

    Waits on the :class:`AudioBus`, and each wake feeds every channel the audio
    that arrived since it last looked (``store.view`` — O(window)), reconciled to
    the latest watermark. Being the *only* inference thread makes the live pass
    single-flight, so it never contends with the finalize pass for the one
    accelerator; ``inference_lock`` makes that guarantee explicit and is the
    extension point for a future overlapping finalize. On close it feeds the final
    window and :meth:`~stenograf.live.WindowedLiveDecoder.flush`\\ es each decoder
    to force-commit the tail.

    Option B checkpointing: every ``flush_interval`` seconds of
    processed audio the worker calls ``on_flush`` — a zero-inference hook that
    snapshots the decoders' already-committed text to ``.partial``. Doing it here,
    on the same thread that owns the decoders, needs no lock and never runs the
    accelerator; the reconcile means a backlog coalesces into a single flush.
    """

    def __init__(
        self,
        store: SessionStore,
        bus: AudioBus,
        decoders: dict[Channel, WindowedLiveDecoder],
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
        # Per-channel shed audio: a channel with any shed has a live-caption gap,
        # so the finalize pass must re-decode it instead of reusing live words.
        self.shed_by_channel: dict[Channel, float] = dict.fromkeys(channels, 0.0)
        self._seen: dict[Channel, float] = dict.fromkeys(channels, 0.0)

    def backlog(self) -> float:
        """Seconds of stored audio not yet fed, on the furthest-behind channel
        (read from other threads: the diarization streamer yields on it)."""
        marks = self._bus.marks()
        return max((marks[ch] - self._seen[ch] for ch in self._channels), default=0.0)

    def run(self) -> None:
        seen = self._seen
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
                        # Same discipline as the main loop above: never hold the
                        # accelerator lock across the view callback.
                        with self._inference_lock:
                            update = self._decoders[ch].flush()
                        self._emit(ch, update)
                    # One last checkpoint now that every word is committed: the
                    # finalize (diarization included) runs next and can still die,
                    # and the periodic flush may be an interval behind. Zero
                    # inference — it only snapshots the decoders' committed text.
                    if flushing:
                        self._on_flush()  # type: ignore[misc]  # guarded by `flushing`
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
        (possibly advanced) start second to feed from."""
        cap = self._decoders[channel].window_cap
        if mark - start <= cap:
            return start
        shed = (mark - cap) - start
        self.shed_seconds += shed
        self.shed_by_channel[channel] = self.shed_by_channel.get(channel, 0.0) + shed
        self._decoders[channel].drop_window()
        return mark - cap

    def _emit(self, channel: Channel, update: StreamingUpdate) -> None:
        if self._on_update is not None and (update.committed or update.interim):
            self._on_update(channel, update)


STREAM_YIELD_BACKLOG_S = 2.0
"""The live captions come first: while the ASR worker has more than this much
stored audio it has not fed, the diarization streamer starts no step. A
healthy worker stays below it (it feeds each frame batch as it lands); one
heading toward its load-shed bound (a full decode window) gets every core."""

_STREAM_RETRY_S = 1.0
_STREAM_WAKE_MARGIN_S = 0.25


def _lower_thread_priority() -> None:
    """Run the calling thread, and any thread it starts, below everything
    interactive — best effort, a no-op where the platform refuses.

    macOS: background QoS, which the scheduler keeps on the efficiency cores;
    a one-thread Nemotron step there costs 0.2 J against 1.4 J on a P-core
    (M4 Max, 2026-11-17, ``eval/README.md``). Linux: SCHED_IDLE (runs only on
    otherwise idle CPU time). Windows: background mode (lowest CPU, I/O and
    memory priority). Neither of the last two is measured."""
    with contextlib.suppress(Exception):
        if sys.platform == "darwin":
            libc = ctypes.CDLL("/usr/lib/libSystem.dylib")
            libc.pthread_set_qos_class_self_np(0x09, 0)  # QOS_CLASS_BACKGROUND
        elif sys.platform.startswith("linux"):
            os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))  # 0 = this thread
        elif sys.platform == "win32":
            kernel32 = ctypes.WinDLL("kernel32")  # pyright: ignore[reportAttributeAccessIssue]
            kernel32.SetThreadPriority(kernel32.GetCurrentThread(), 0x00010000)


class DiarizationStreamer(threading.Thread):
    """Runs each diarized channel's :class:`DiarizationStream` while the
    meeting records, so the stop only diarizes the last half-minute.

    A thread of its own, not a task on :class:`LiveWorker`: the worker is the
    single *accelerator* thread and must never wait behind a CPU model, while
    this one runs a one-thread ONNX session at the lowest priority the
    platform offers. It never takes the inference lock, and it starts no step
    while the worker is behind (:data:`STREAM_YIELD_BACKLOG_S`) — so it cannot
    cause a load-shed — or once :meth:`halt` was called. Each step reads only
    audio already in the append-only :class:`SessionStore`, so what it
    computed stays valid however the meeting continues (mute zeroes frames
    before they are stored; a device switch only appends).

    It sleeps until the audio the next step needs should have arrived — about
    one wake per step, every 27 s per channel — rather than on the frame bus.
    A stream that raises is dropped (its error kept in :attr:`errors`); the
    stop then diarizes that channel from wherever the stream got to."""

    def __init__(
        self,
        store: SessionStore,
        streams: dict[Channel, DiarizationStream],
        *,
        backlog: Callable[[], float] = lambda: 0.0,
    ) -> None:
        super().__init__(name="diarization-stream", daemon=True)
        self._store = store
        self._streams = streams
        self._backlog = backlog
        self._halted = threading.Event()
        self.errors: dict[Channel, BaseException] = {}
        self.steps = 0

    def run(self) -> None:
        _lower_thread_priority()
        active = dict(self._streams)
        while active and not self._halted.is_set():
            if self._backlog() > STREAM_YIELD_BACKLOG_S:
                self._halted.wait(_STREAM_RETRY_S)
                continue
            ran = 0
            short: list[int] = []  # samples each idle channel still lacks
            for channel, stream in list(active.items()):
                if self._halted.is_set():
                    return
                try:
                    needed = stream.samples_needed()
                    have = self._store.num_samples(channel)
                    if needed is None:
                        del active[channel]
                    elif have < needed:
                        short.append(needed - have)
                    else:
                        read = functools.partial(self._store.read, channel)
                        ran += stream.advance(read, have, limit=1)
                except Exception as exc:  # noqa: BLE001 — the stop diarizes it instead
                    if not self._halted.is_set():
                        self.errors[channel] = exc
                    del active[channel]
            self.steps += ran
            if not ran and short:
                arrives = max(min(short) / SAMPLE_RATE, _STREAM_RETRY_S)
                self._halted.wait(arrives + _STREAM_WAKE_MARGIN_S)

    def halt(self) -> None:
        """Stop starting steps, abandon the one running, and wait for the thread."""
        self._halted.set()
        for stream in self._streams.values():
            stream.interrupt()
        if self.is_alive():
            self.join()


def _join_until_done(thread: threading.Thread, poll: float = 0.1) -> None:
    """Join ``thread`` while staying responsive to Ctrl-C on the main thread.

    A bare ``Thread.join()`` can swallow signals on some platforms; joining in
    short slices lets a ``KeyboardInterrupt`` reach the main thread promptly so
    the meeting still finalizes what it captured.
    """
    while thread.is_alive():
        thread.join(poll)


class _LiveCheckpointer:
    """Groups the live pass's committed words into checkpoint entries, tail-only.

    ``committed_words`` is append-only and time-ordered, and :func:`group_words`
    closes an entry on the silence gap to its successor — a gap later words can
    only confirm, never reopen. Entries once closed are therefore final: each
    flush regroups only the words from the newest (still-open) run on, keeping
    the flush stream O(new words) instead of regrouping the whole meeting every
    interval.
    """

    def __init__(self, wrap: Callable[[list[TranscriptEntry]], Transcript]) -> None:
        self._wrap = wrap
        self._closed: dict[Channel, list[TranscriptEntry]] = {}
        self._grouped: dict[Channel, int] = {}  # words already inside closed entries

    def transcript(self, decoders: dict[Channel, WindowedLiveDecoder]) -> Transcript:
        entries: list[TranscriptEntry] = []
        for channel, decoder in decoders.items():
            closed = self._closed.setdefault(channel, [])
            grouped = self._grouped.get(channel, 0)
            fresh = group_words(list(decoder.committed_words[grouped:]), _CHANNEL_COARSE[channel])
            for entry in fresh[:-1]:  # every run but the newest is final
                closed.append(entry)
                grouped += len(entry.words)
            self._grouped[channel] = grouped
            entries.extend(closed)
            if fresh:
                entries.append(fresh[-1])
        return self._wrap(entries)


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
        dedup_echo: bool = True,
    ) -> None:
        self.profile = profile
        self.asr = asr
        self.vad = vad
        self.diarizer = diarizer
        self.reid = reid
        self.reuse_live_finalize = True
        """Whether the on-stop finalize may reuse the window pass's decodes.
        ``--full-finalize`` clears it to force a from-scratch ASR pass at stop
        (an A/B and paranoia escape hatch). Reuse also self-disables per channel
        on any live load-shed, and entirely on a live-worker error."""
        self.stream_diarization = True
        """Whether a live meeting runs the diarizer's stream on each diarized
        channel while it records (:class:`DiarizationStreamer`), leaving the
        stop only the tail. The speakers are the same either way; clearing it
        (``--full-finalize``) moves all of the work to the stop."""
        self.dedup_echo = dedup_echo
        """Whether finalize may run :func:`drop_echo_duplicates` on the mic channel.
        The CLI ties this to ``--aec``: with cancellation off the user asked for
        the mic exactly as captured, and no transcript lines should vanish. Even
        when True the backstop only *arms* if the canceller reported losing its
        reference (finalize's ``reference_gap_s``): a healthy canceller leaks
        nothing the ASR can decode (measured across the AEC scenario
        matrix), so in the healthy case the backstop's one false-positive class —
        the local speaker verbatim-repeating the remote — can never fire."""
        self.language = language or profile.language
        """The *configured* meeting language (explicit setting or profile), or
        ``None`` = auto-detect per finalize. Never mutated; the resolved
        language rides on each result's transcript."""
        self.glossary_threshold = (
            DEFAULT_THRESHOLD if glossary_threshold is None else glossary_threshold
        )

    def run(
        self,
        provider: CaptureProvider,
        *,
        view: LiveView | None = None,
        on_frame: Callable[[AudioFrame], None] | None = None,
        checkpoint: CheckpointConfig | None = None,
        max_seconds: float | None = None,
        live: bool = False,
        provider_started: bool = False,
        mute: MuteSwitch | None = None,
    ) -> MeetingResult:
        """Capture until the provider stops (or Ctrl-C), then finalize.

        ``on_frame`` sees every stored frame (used by the audio tee); a
        ``KeyboardInterrupt`` ends capture gracefully rather than aborting, so
        an interrupted meeting still yields a transcript of what was captured.
        ``max_seconds`` stops capture automatically after that much audio.
        ``provider_started`` means the caller already called ``provider.start``
        (so recording could begin before its own slow setup — model loading —
        with the provider buffering frames meanwhile); the run then only
        consumes and stops the provider, never starting it a second time.
        ``mute`` silences what the user muted (:class:`MuteSwitch`); its spans
        ride on the finished transcript.

        With ``live=True`` the meeting runs the streaming pass: capture on its own
        thread feeding a single :class:`LiveWorker` that drives a
        :class:`~stenograf.live.WindowedLiveDecoder` per channel and streams
        committed words to the view. The heavy finalize still runs once on stop
        and replaces the whole live transcript.

        Events go to the single :class:`~stenograf.view.LiveView` sink (the
        CLI's TUI / plain view; the bare base class is the null view and the
        default). The orchestrator emits the structured lifecycle events on it
        — ``status`` / ``language`` / ``finalizing`` / ``finalized`` /
        ``error`` — around the capture and finalize passes.

        Only the live mode checkpoints for crash recovery, if a
        :class:`CheckpointConfig` is given, coalesced to its ``interval`` seconds
        of capture — and never any extra inference: the already-committed live
        text is flushed as-is, so a crash loses at most one interval of text.
        Batch runs are eval/debug runs and get no crash checkpoints. Audio is
        never persisted, and the authoritative transcript is the full finalize
        returned on stop (the :class:`MeetingResult` carries it plus the run's
        report).
        """
        plans = plan_channels(self.profile)
        store = SessionStore({p.channel for p in plans})
        sink = view if view is not None else LiveView()
        if live:
            return self._run_live(
                provider,
                plans,
                store,
                on_frame=on_frame,
                view=sink,
                checkpoint=checkpoint,
                max_seconds=max_seconds,
                provider_started=provider_started,
                mute=mute,
            )
        return self._run_batch(
            provider,
            plans,
            store,
            on_frame=on_frame,
            view=sink,
            max_seconds=max_seconds,
            provider_started=provider_started,
            mute=mute,
        )

    def _run_batch(
        self,
        provider: CaptureProvider,
        plans: list[ChannelPlan],
        store: SessionStore,
        *,
        on_frame: Callable[[AudioFrame], None] | None,
        view: LiveView,
        max_seconds: float | None,
        provider_started: bool = False,
        mute: MuteSwitch | None = None,
    ) -> MeetingResult:
        """Consume-thread capture, then finalize (no live view, no checkpoints).

        Capture stays on this thread so a ``KeyboardInterrupt`` in the provider
        ends the meeting cleanly.
        """
        channels = [p.channel for p in plans]
        if not provider_started:
            provider.start(set(channels))
        capture_error: BaseException | None = None
        try:
            for frame in provider.frames():
                if _consume_frame(frame, store, None, on_frame, channels, max_seconds, mute):
                    break
        except KeyboardInterrupt:
            view.status("interrupted — finalizing captured audio")
        except Exception as exc:  # noqa: BLE001 — a desync etc. must not lose the transcript
            # Every frame that arrived is already in the store, so finalize what was
            # captured instead of aborting; surface the error but keep the transcript.
            capture_error = exc
        finally:
            provider.stop()
        return self._finalize_and_publish(
            provider, store, plans, view, capture_error=capture_error, aux_error=None, mute=mute
        )

    def _run_live(
        self,
        provider: CaptureProvider,
        plans: list[ChannelPlan],
        store: SessionStore,
        *,
        on_frame: Callable[[AudioFrame], None] | None,
        view: LiveView,
        checkpoint: CheckpointConfig | None,
        max_seconds: float | None,
        provider_started: bool = False,
        mute: MuteSwitch | None = None,
    ) -> MeetingResult:
        """Live pass: threaded capture + one inference worker, then finalize.

        Capture runs on :class:`CaptureLoop` (never blocked by inference); one
        :class:`LiveWorker` drives a :class:`~stenograf.live.WindowedLiveDecoder`
        per channel and streams updates to the view. On stop the worker is joined
        and the full finalize pass runs — it replaces the whole live transcript,
        so live compromises never reach the final output.

        Option B checkpointing: the worker flushes the decoders'
        already-committed text to ``checkpoint.write`` every ``checkpoint.interval``
        seconds — pure file I/O, no extra inference, since the live pass already
        produced that text. Empty flushes (nothing committed yet) are skipped so a
        ``.partial`` only appears once there is text to recover. The worker flushes
        once more after its final decode, so the checkpoint holds every committed
        word before the finalize — which can still die (e.g. OOM in diarization) —
        begins.
        """
        channels = [p.channel for p in plans]
        bus = AudioBus(channels)
        vad = self.vad
        if vad is None:
            raise ValueError("the live pass needs a streaming VAD (vad.stream)")
        decoders = {
            ch: WindowedLiveDecoder(self.asr, vad=vad, language=self.language) for ch in channels
        }
        inference_lock = threading.Lock()

        checkpointer = _LiveCheckpointer(self.checkpoint_transcript)

        def flush_checkpoint() -> None:
            transcript = checkpointer.transcript(decoders)
            if transcript.entries:
                checkpoint.write(transcript)  # type: ignore[union-attr]  # None-guarded below

        checkpointing = checkpoint is not None and checkpoint.enabled
        worker = LiveWorker(
            store,
            bus,
            decoders,
            inference_lock,
            channels=channels,
            on_update=view.update,
            on_flush=flush_checkpoint if checkpointing else None,
            flush_interval=checkpoint.interval if checkpoint is not None else 0.0,
        )
        capture = CaptureLoop(
            provider,
            store,
            bus,
            channels=channels,
            on_frame=on_frame,
            max_seconds=max_seconds,
            mute=mute,
        )
        streams = self._open_streams(plans, view)
        streamer = (
            DiarizationStreamer(store, streams, backlog=worker.backlog) if streams else None
        )

        if not provider_started:
            provider.start(set(channels))
        worker.start()
        capture.start()
        if streamer is not None:
            streamer.start()
        try:
            _join_until_done(capture)
        except KeyboardInterrupt:
            view.status("interrupted — finalizing captured audio")
            provider.stop()
            capture.join()
        # Capture has stopped; from here the finalize is authoritative and must not
        # be lost to a second Ctrl-C. Shield SIGINT across the worker join and the
        # finalize (a no-op off the main thread, e.g. under the TUI).
        with _shield_interrupt(), _closing(streams):
            if streamer is not None:
                streamer.halt()
                for channel, exc in streamer.errors.items():
                    view.status(
                        f"{channel}: diarizing during the meeting stopped ({exc}); "
                        "the stop diarizes the rest"
                    )
            worker.join()
            provider.stop()  # idempotent — releases the device if capture ended on its own
            # The window pass produced finalize-identical decodes (same windows,
            # same model — eval/live.py); reuse them so finalize
            # skips its ASR pass. A load-shed channel has a caption gap and a
            # worker error leaves unknown coverage — both re-decode classically.
            live_words: dict[Channel, tuple[Word, ...]] | None = None
            if self.reuse_live_finalize and worker.error is None:
                live_words = {
                    ch: decoders[ch].committed_words
                    for ch in channels
                    if worker.shed_by_channel.get(ch, 0.0) == 0.0
                }
            # The live pass is provisional; if a decode failed, surface it but still
            # finalize — the finalize pass is the authoritative transcript regardless.
            aux_error = f"live pass stopped early: {worker.error}" if worker.error else None
            # Single-flight: the worker is already joined, but taking the same lock it
            # held documents (and future-proofs) that finalize never runs alongside a
            # live decode.
            return self._finalize_and_publish(
                provider,
                store,
                plans,
                view,
                capture_error=capture.error,
                aux_error=aux_error,
                live_words=live_words,
                inference_lock=inference_lock,
                mute=mute,
                streams=streams,
            )

    def _open_streams(
        self, plans: list[ChannelPlan], view: LiveView
    ) -> dict[Channel, DiarizationStream]:
        """A diarization stream per channel the stop would diarize — never one
        with a stated count of 1, which skips the diarizer. Opened here, on the
        meeting's thread, because opening loads the stream's model; a stream
        that fails to open leaves its channel to the stop."""
        if self.diarizer is None or not self.stream_diarization:
            return {}
        streams = {}
        for plan in plans:
            if plan.num_speakers == 1:
                continue
            try:
                stream = self.diarizer.stream()
            except Exception as exc:  # noqa: BLE001 — the stop diarizes it instead
                view.status(f"{plan.channel}: diarizing at the stop ({exc})")
                continue
            if stream is not None:
                streams[plan.channel] = stream
        return streams

    def _finalize_and_publish(
        self,
        provider: CaptureProvider,
        store: SessionStore,
        plans: list[ChannelPlan],
        view: LiveView,
        *,
        capture_error: BaseException | None,
        aux_error: str | None,
        live_words: dict[Channel, tuple[Word, ...]] | None = None,
        inference_lock: threading.Lock | None = None,
        mute: MuteSwitch | None = None,
        streams: dict[Channel, DiarizationStream] | None = None,
    ) -> MeetingResult:
        """The shared stop→finalize tail of both run modes.

        Capture has stopped when this runs; the finalize is authoritative, so it
        is shielded from a second Ctrl-C (nested shields are harmless — the live
        path already holds one across the worker join). A capture error (a
        stream desync, a device drop) is non-fatal: every frame that did arrive
        is already in the store, so surface the error but still finalize what
        was captured rather than discarding the whole meeting's transcript.
        ``aux_error`` is the mode's secondary-thread failure (checkpointer /
        live worker), likewise surfaced without losing the transcript.

        The one exception: capture failed AND nothing was ever captured. There
        is no transcript to protect, and publishing would write an empty one
        that looks like a successful meeting (the 2026-07-20 OBS-conflict
        silent failure) — so the meeting fails with the capture error instead.
        """
        if capture_error is not None and all(store.duration(ch) == 0.0 for ch in store.channels()):
            raise capture_error
        if capture_error is not None:
            view.error(f"capture stopped early: {capture_error}; finalizing captured audio")
        if aux_error is not None:
            view.error(aux_error)
        view.finalizing()
        with _shield_interrupt():
            lock = inference_lock if inference_lock is not None else contextlib.nullcontext()
            with lock:
                end = max((store.duration(ch) for ch in store.channels()), default=0.0)
                result = self.finalize(
                    store,
                    plans,
                    view=view,
                    live_words=live_words,
                    reference_gap_s=_reference_gap(provider),
                    muted=mute.spans(until=end) if mute is not None else None,
                    streams=streams,
                )
        view.finalized(result.transcript)
        return result

    def finalize(
        self,
        store: SessionStore,
        plans: list[ChannelPlan] | None = None,
        *,
        view: LiveView | None = None,
        live_words: dict[Channel, tuple[Word, ...]] | None = None,
        reference_gap_s: float | None = None,
        muted: list[MutedSpan] | None = None,
        streams: dict[Channel, DiarizationStream] | None = None,
    ) -> MeetingResult:
        """Run the finalize pass on every stored channel and interleave them.

        ``live_words`` are a channel's window-pass decodes (finalize-identical
        windows; see :class:`~stenograf.live.WindowedLiveDecoder`): a channel
        present in the dict skips the VAD+ASR re-decode and goes straight to
        diarization/merge. A channel missing from the dict (or the dict being
        None — direct calls, batch mode, ``--full-finalize``) re-decodes.

        Every channel's words are in before any diarizer runs, and the ASR is
        then released: the stop never decodes again, and its weights plus MLX's
        buffer cache are ~3.4 GB that diarization would otherwise stack on
        (2026-11-15, 1 h 45 min meeting: the stop's peak 10.2 → 6.2 GB,
        ``eval/README.md``). The backends load again on their next transcribe,
        so a recorder finalized twice still works.

        ``reference_gap_s`` is how long the canceller ran without its far-end
        reference (the run modes measure it via :func:`_reference_gap`); the
        echo-text backstop arms on it. ``None`` means unknown — no canceller
        was observed (direct calls, e.g. a file transcribe) — and finalize
        treats unknown conservatively: backstop armed.

        ``muted`` are the meeting's muted spans; they ride on the transcript.

        ``streams`` are the channels' diarization streams, advanced during the
        meeting (:class:`DiarizationStreamer`): a channel with one diarizes
        through :meth:`~stenograf.diarization.base.DiarizationStream.resumed`,
        which finishes what the stream left — the same turns as the diarizer
        from scratch, and from scratch is its fallback.
        """
        plans = plans or plan_channels(self.profile)
        view = view or LiveView()  # the bare base class is the null view
        by_channel: dict[Channel, list[TranscriptEntry]] = {}
        counts: list[SpeakerCount] = []
        voices: list[tuple[str, np.ndarray, float]] = []
        plans = [p for p in plans if p.channel in store.channels()]
        decoded: dict[Channel, list[Segment] | None] = {}
        for plan in plans:
            if live_words is not None and plan.channel in live_words:
                continue
            view.status(f"transcribing {plan.channel}")
            try:
                decoded[plan.channel] = decode_channel(
                    store.samples(plan.channel),
                    asr=self.asr,
                    language=self.language,
                    vad=self.vad,
                    on_progress=None,
                )
            except Exception as exc:  # noqa: BLE001 — resilience across channels is the point
                view.error(f"{plan.channel}: finalize failed ({exc}); skipping channel")
                decoded[plan.channel] = None
        self.asr.unload()
        for plan in plans:
            reused = live_words.get(plan.channel) if live_words is not None else None
            segments = decoded.get(plan.channel)
            if reused is None and segments is None:
                by_channel[plan.channel] = []
                counts.append(SpeakerCount(plan.channel, plan.num_speakers, 0))
                continue
            note = ", reusing live decodes" if reused is not None else ""
            view.status(f"finalizing {plan.channel} ({_speaker_note(plan.num_speakers)}{note})")
            stream = streams.get(plan.channel) if streams else None
            diarizer: Diarizer | None = self.diarizer
            if plan.num_speakers == 1:
                diarizer = None
            elif stream is not None:
                diarizer = stream.resumed()
            # A solo channel skips the diarizer (nothing to separate) but not
            # the voice: with the meeting's speaker machinery on it still gets
            # one embedding over its transcribed speech, so a 1:1 counterpart
            # is named and `steno profiles assign` has something to enroll from.
            solo_voice = plan.num_speakers == 1 and self.diarizer is not None
            # Audio is read only by an embedding/diarization pass over actual
            # speech; the plain single-speaker path regroups the words alone.
            # Don't concatenate a full-meeting buffer it would just discard.
            has_speech = bool(reused) or bool(segments)
            needs_audio = (diarizer is not None or solo_voice) and has_speech
            samples = store.samples(plan.channel) if needs_audio else np.zeros(0, np.float32)
            embedded: dict[str, np.ndarray] = {}
            raw = self._finalize_channel_safe(
                samples,
                diarizer,
                plan,
                view,
                reused_words=reused,
                segments=segments,
                on_embeddings=embedded.update,
            )
            if solo_voice and raw:
                raw, embedded = self._name_solo_channel(samples, raw, view, plan)
            labeled = relabel_speakers(raw, plan.label_template)
            mapping = raw_label_map(raw, plan.label_template)
            # Merged, not overwritten, across channels: one profile can match
            # a voice on both (a person who moved, an echo leak re-ID named).
            seconds = speech_seconds(labeled)
            voices += [
                (mapping.get(label, label), v, seconds.get(mapping.get(label, label), 0.0))
                for label, v in embedded.items()
            ]
            detected = len({e.speaker for e in labeled})
            counts.append(SpeakerCount(plan.channel, plan.num_speakers, detected))
            if plan.num_speakers is None:
                view.status(f"{plan.channel}: detected {detected} speaker(s)")
            by_channel[plan.channel] = labeled
        dropped = self._apply_echo_backstop(by_channel, view, reference_gap_s)
        clips: dict[str, list[np.ndarray]] = {}
        if self.diarizer is not None:  # the speaker machinery ran: voices are nameable
            for plan in plans:
                entries = by_channel.get(plan.channel, [])
                for label in dict.fromkeys(e.speaker for e in entries):
                    clips.setdefault(label, []).extend(
                        cut_voice_clips(
                            [e for e in entries if e.speaker == label],
                            lambda start, end, ch=plan.channel: store.view(ch, start, end),
                        )
                    )
        transcript = self._assemble_transcript(plans, by_channel, counts, view)
        transcript.muted = list(muted or ())
        return MeetingResult(
            transcript=transcript,
            speaker_counts=counts,
            dropped_echo_lines=dropped,
            reference_gap_s=reference_gap_s,
            speaker_embeddings=merge_voices(voices),
            voice_clips={label: found[:CLIP_COUNT] for label, found in clips.items() if found},
        )

    def _apply_echo_backstop(
        self,
        by_channel: dict[Channel, list[TranscriptEntry]],
        view: LiveView,
        reference_gap_s: float | None,
    ) -> int:
        """Drop mic lines that duplicate remote speech, if the backstop is armed.

        The text backstop for a canceller that lost its reference (a stalled or
        mis-clocked tap): echo then lands on the mic channel uncancelled, and
        the remote channel's own transcript identifies it. Armed only when the
        reference was degraded (or unobserved) — a healthy canceller leaks
        nothing decodeable, and never arming it then means a verbatim local
        repeat of remote speech can never be mistaken for echo. Mutates
        ``by_channel`` in place; returns the number of dropped mic lines.
        """
        mic, system = by_channel.get(Channel.MIC), by_channel.get(Channel.SYSTEM)
        armed = self.dedup_echo and (reference_gap_s is None or reference_gap_s > 0)
        if not (armed and mic and system):
            return 0
        kept = drop_echo_duplicates(mic, system)
        dropped = len(mic) - len(kept)
        if dropped:
            cause = (
                f" — the canceller ran {reference_gap_s:.1f}s without its reference"
                if reference_gap_s
                else ""
            )
            view.status(
                f"echo backstop: dropped {dropped} mic line(s) duplicating remote speech{cause}"
            )
        by_channel[Channel.MIC] = kept
        return dropped

    def _assemble_transcript(
        self,
        plans: list[ChannelPlan],
        by_channel: dict[Channel, list[TranscriptEntry]],
        counts: list[SpeakerCount],
        view: LiveView,
    ) -> Transcript:
        """Interleave the channels, then run the shared finalize post-steps
        (:func:`~stenograf.pipeline.assemble_transcript`): glossary snap,
        language resolve (an explicit setting wins, else LID over the
        finalized text), parameter provenance."""
        entries = [entry for plan in plans for entry in by_channel.get(plan.channel, [])]
        return assemble_transcript(
            interleave(entries),
            profile=self.profile,
            language=self.language,
            glossary_threshold=self.glossary_threshold,
            parameters_for=lambda resolved: resolve_parameters(
                self.profile, language=resolved, speaker_counts=counts
            ),
            on_language=view.language,
        )

    def _name_solo_channel(
        self,
        samples: np.ndarray,
        entries: list[TranscriptEntry],
        view: LiveView,
        plan: ChannelPlan,
    ) -> tuple[list[TranscriptEntry], dict[str, np.ndarray]]:
        """One voice embedding for a single-speaker channel, re-ID applied.

        The embedding covers the transcript's own speech spans (no
        segmentation — the whole channel is one stated speaker), and a profile
        match renames the channel exactly like a matched cluster. A failure
        costs the voice, never the text (mirrors ``_finalize_channel_safe``).
        """
        label = entries[0].speaker
        turns = [SpeakerTurn(label, e.start, e.end) for e in entries]
        try:
            assert self.diarizer is not None  # guarded by the caller
            embedding = self.diarizer.channel_embedding(samples, turns)
        except Exception as exc:  # noqa: BLE001 — the transcript must survive
            view.error(f"{plan.channel}: voice embedding failed ({exc}); speaker not matched")
            return entries, {}
        if embedding is None:
            return entries, {}
        names = self.reid.resolve({label: embedding}) if self.reid is not None else {}
        if label in names:
            name = names[label]
            return [dc_replace(e, speaker=name) for e in entries], {name: embedding}
        return entries, {label: embedding}

    def _finalize_channel_safe(
        self,
        samples: np.ndarray,
        diarizer: Diarizer | None,
        plan: ChannelPlan,
        view: LiveView,
        *,
        reused_words: tuple[Word, ...] | None = None,
        segments: list[Segment] | None = None,
        on_embeddings: Callable[[dict[str, np.ndarray]], None] | None = None,
    ) -> list[TranscriptEntry]:
        """Finalize one channel, never letting its failure lose another channel.

        Diarization is the fragile step (a real backend can raise on unexpected
        audio). On failure, retry
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
                precomputed_words=reused_words,
                precomputed_segments=segments,
                on_embeddings=on_embeddings,
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
                    precomputed_words=reused_words,
                    precomputed_segments=segments,
                )
            except Exception as exc2:  # noqa: BLE001
                view.error(f"{plan.channel}: finalize failed ({exc2}); skipping channel")
                return []

    def live_checkpoint(self, decoders: dict[Channel, WindowedLiveDecoder]) -> Transcript:
        """A crash checkpoint from the live pass's already-committed words.

        Zero inference: the words are read straight off each channel's decoder and
        grouped into entries under a channel-coarse label (the live pass has no
        diarization). The on-stop :meth:`finalize` replaces this entirely.

        One-shot form: a fresh :class:`_LiveCheckpointer` groups the full
        history. The live run keeps a single instance across flushes instead,
        so each flush regroups only the new tail.
        """
        return _LiveCheckpointer(self.checkpoint_transcript).transcript(decoders)

    def checkpoint_transcript(self, entries: list[TranscriptEntry]) -> Transcript:
        """Wrap accumulated coarse checkpoint entries into an ordered transcript.

        Carries the *configured* language (explicit setting or ``None``): a
        checkpoint never runs language detection — that happens once, in the
        on-stop :meth:`finalize`, over the authoritative text.
        """
        return Transcript(language=self.language, profile=self.profile, entries=interleave(entries))


def _reference_gap(provider: CaptureProvider) -> float | None:
    """How long the canceller ran without its far-end reference, in seconds.

    Read after capture stops, immediately before finalize, which arms the
    echo-text backstop on it. A provider without an enabled canceller (mic
    only, ``--no-aec`` baselines, plain test providers) has no gap to report.
    """
    canceller = getattr(provider, "canceller", None)
    if canceller is not None and canceller.enabled:
        return canceller.far_end_missing_ticks / 100
    return None


def _speaker_note(num_speakers: int | None) -> str:
    if num_speakers is None:
        return "estimating speakers"
    if num_speakers == 1:
        return "single speaker"
    return f"{num_speakers} speakers"
