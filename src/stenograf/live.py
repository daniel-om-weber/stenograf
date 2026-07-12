"""Live pass: a re-decode window with LocalAgreement-2 commits.

This is Phase 2, Task 1 (PLAN.md §5). It turns the batch ``ASRBackend`` into a
streaming captioner **without any new dependency and without parakeet-mlx's
incremental ``transcribe_stream`` API** — the Phase 2 spike measured that API as
garbage at small right-context and fragile otherwise (PLAN.md §2 "Live ASR").

Instead the decoder re-decodes a short trailing window over the model's full
``generate()`` path — at most once per ``decode_interval`` of audio, since a
re-decode mostly re-transcribes the same window — and commits text with a
LocalAgreement-2 policy: a word becomes final ("committed", shown black) only
once two consecutive window decodes agree on it. The unstable tail stays
provisional ("interim", shown grey) and is replaced on the next decode. Because
every decode uses the same full-attention path as the finalize pass, live
captions land at finalize-grade accuracy (~10% WER, spike-measured), and the
finalize pass still replaces the whole live transcript on stop.

Window management (PLAN.md §2):

- The window is ``left_context`` seconds of already-committed audio (context for
  the model) plus the uncommitted tail, capped at ``window_cap`` seconds. As
  words commit, the committed audio drops out of the window, so it stays short
  (~a handful of seconds during normal speech) rather than growing unbounded.
- Silero VAD gates the work: with no fresh speech beyond the commit point the
  decoder does no ASR at all (~0% accelerator in silence). At an utterance
  boundary (a pause the VAD closes) the tail is force-committed and the window
  resets, so context never bleeds across utterances.

The commit stream is **append-only**: a committed word is never rewritten
(monotonicity — one of the label-free acceptance metrics). The decoder composes
``ASRBackend.transcribe`` and is model-agnostic, but the live pass needs word
timestamps, so in practice it runs Parakeet (the committed default).

:class:`WindowedLiveDecoder` is the second, cheaper live pass (the product
default): finalize-identical windows, decoded once each. Both decoders satisfy
the :class:`StreamingDecoder` protocol the orchestrator drives.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from stenograf.asr.base import ASRBackend, Word
from stenograf.audio import SAMPLE_RATE, sample_index, to_float32
from stenograf.config import Language
from stenograf.vad import SileroVAD, SpeechSegment

_WORD_KEY = re.compile(r"\W+", re.UNICODE)


@dataclass(frozen=True)
class StreamingUpdate:
    """The result of feeding one chunk: what newly committed, plus the grey tail.

    ``committed`` are the words finalized by *this* feed (empty when the decode
    agreed on nothing new or was skipped in silence); the full committed
    transcript lives on the decoder. ``interim`` is the current provisional tail
    — the best guess for the audio that has not settled yet, shown grey.
    """

    committed: tuple[Word, ...]
    interim: str

    @property
    def committed_text(self) -> str:
        return " ".join(w.text for w in self.committed)


class StreamingDecoder(Protocol):
    """What the live pass needs from a per-channel decoder.

    ``LiveWorker`` (session.py) drives exactly this surface — feed each new
    chunk, force-commit the tail on close, abandon the window on a load-shed —
    and the orchestrator's checkpoint/reuse paths read ``committed_words``.
    :class:`LiveDecoder` and :class:`WindowedLiveDecoder` satisfy it
    structurally (they share a buffer, not an algorithm).
    """

    window_cap: float
    """Seconds of unprocessed backlog the decoder absorbs in one feed; past it
    the worker load-sheds (``LiveWorker._shed_if_behind``)."""

    def feed(self, samples: np.ndarray, t_offset: float) -> StreamingUpdate: ...

    def flush(self) -> StreamingUpdate: ...

    def drop_window(self) -> None: ...

    @property
    def committed_words(self) -> tuple[Word, ...]: ...


class _CaptionBuffer:
    """The retained audio window a live decoder feeds, scans, and slices.

    Owns the absolute-timestamped mono float32 samples, the silence padding
    across feed gaps, the backwards-feed guard, and the optional streaming VAD
    fed in lockstep (created per buffer origin when the VAD object supports
    ``stream``; ``None`` falls back to the caller's per-feed window scan).

    ``quantize_origin`` pins the origin to an exact integer sample index: the
    windowed pass slices its windows with ``sample_index()`` over absolute
    times, and reuse is byte-identical to the batch pass only if it extracts
    the very same samples — so that origin is tracked as an integer (float
    accumulation would drift by ±1 sample) and ``start`` is always derived
    from it by a single division. The LocalAgreement pass keeps the float
    origin its feeds arrived with.
    """

    def __init__(self, vad: SileroVAD | None) -> None:
        self._vad = vad
        self.samples = np.zeros(0, dtype=np.float32)
        self.start: float | None = None  # absolute time of samples[0]
        self.start_idx: int | None = None  # integer sample origin (windowed pass)
        self.vad_stream = None

    def __len__(self) -> int:
        return len(self.samples)

    def append(self, chunk: np.ndarray, t_offset: float, *, tolerance: float) -> None:
        """Add audio at absolute time ``t_offset``; pad gaps beyond ``tolerance``."""
        chunk = to_float32(np.asarray(chunk)).reshape(-1)
        if self.start is None:
            self.start = float(t_offset)
            self.samples = chunk.copy()
            stream = getattr(self._vad, "stream", None)
            self.vad_stream = stream(self.start) if stream is not None else None
            self.push_vad(chunk)
            return
        gap = t_offset - self.end()
        if gap < -tolerance:
            raise ValueError(
                f"feed went backwards {-gap:.3f}s (t_offset {t_offset:.3f}s "
                f"< buffered end {self.end():.3f}s); frames must arrive in order"
            )
        if gap > tolerance:  # a real gap since the last chunk → pad silence
            pad = np.zeros(round(gap * SAMPLE_RATE), np.float32)
            self.samples = np.concatenate([self.samples, pad])
            self.push_vad(pad)
        self.samples = np.concatenate([self.samples, chunk])
        self.push_vad(chunk)

    def push_vad(self, samples: np.ndarray) -> None:
        if self.vad_stream is not None:
            self.vad_stream.push(samples)

    def quantize_origin(self) -> None:
        """Pin the origin to the sample grid (idempotent; see the class docstring)."""
        if self.start_idx is None and self.start is not None:
            self.start_idx = round(self.start * SAMPLE_RATE)
            self.start = self.start_idx / SAMPLE_RATE

    def end(self) -> float:
        """The live edge. On the integer origin this reproduces the batch
        pass's ``len(samples) / SAMPLE_RATE`` float bit-for-bit — (origin +
        buffered) equals the total sample count, so one division suffices;
        the float branch accumulates like the feeds did."""
        if self.start_idx is not None:
            return (self.start_idx + len(self.samples)) / SAMPLE_RATE
        return (self.start or 0.0) + len(self.samples) / SAMPLE_RATE

    def trim_before(self, keep_from: float) -> None:
        """Drop audio older than ``keep_from``.

        On the integer origin, trims land on the same ``sample_index()`` grid
        the decode slices use, so a window's padded start is never trimmed past
        (truncation only rounds down, and ``keep_from`` is a lower bound on
        every future span start).
        """
        if self.start_idx is not None:
            keep_idx = max(self.start_idx, sample_index(keep_from))
            drop = keep_idx - self.start_idx
            if drop > 0:
                self.samples = self.samples[drop:]
                self.start_idx = keep_idx
                self.start = keep_idx / SAMPLE_RATE
            return
        drop = round((keep_from - (self.start or 0.0)) * SAMPLE_RATE)
        if drop > 0:
            self.samples = self.samples[drop:]
            self.start = (self.start or 0.0) + drop / SAMPLE_RATE

    def reset_to_preroll(self, pre_roll: float) -> None:
        """Keep only a short silence pre-roll of the buffer."""
        keep = round(pre_roll * SAMPLE_RATE)
        if len(self.samples) <= keep:
            return
        if self.start_idx is not None:
            self.start_idx += len(self.samples) - keep
            self.samples = self.samples[-keep:]
            self.start = self.start_idx / SAMPLE_RATE
        else:
            self.start = self.end() - keep / SAMPLE_RATE
            self.samples = self.samples[-keep:]

    def drop(self) -> None:
        """Abandon the whole buffer; the next append restarts at its own origin.

        No silence is padded across the skipped span. The VAD stream's sample
        clock can't jump, so it is discarded too and rebuilt on the next append.
        """
        self.samples = np.zeros(0, dtype=np.float32)
        self.start = None
        self.start_idx = None
        self.vad_stream = None


def _extend_committed(committed: list[Word], words: list[Word]) -> list[Word]:
    """Append ``words`` to the committed stream, enforcing non-decreasing starts.

    Re-decoding jitters word boundaries by a few ms, so a fresh window can
    place a boundary word a hair before the last committed word's start. Such
    a regressor is a re-emitted duplicate, never genuinely new text — dropping
    it keeps the committed stream strictly append-only (the monotonicity
    invariant; PLAN.md §5) with no visible loss. Returns the words kept.
    """
    kept: list[Word] = []
    last = committed[-1].start if committed else float("-inf")
    for word in words:
        if word.start + 1e-6 < last:
            continue
        committed.append(word)
        last = word.start
        kept.append(word)
    return kept


class LiveDecoder:
    """Streaming captioner over a batch ``ASRBackend`` (re-decode + LocalAgreement-2).

    Feed mono 16 kHz PCM as it arrives; each :meth:`feed` re-decodes the trailing
    window and returns the words that just became final. Call :meth:`flush` at the
    end of the stream (or an utterance) to force-commit the tail, and :meth:`reset`
    to drop the window at a long silence without committing.

    Times are absolute session seconds: ``feed`` is given the timestamp of its
    first sample, and committed/interim word times are on that same clock so they
    line up with the finalize pass and the transcript store.
    """

    def __init__(
        self,
        asr: ASRBackend,
        *,
        vad: SileroVAD | None = None,
        language: Language | None = None,
        left_context: float = 4.0,
        window_cap: float = 15.0,
        grey_zone: float = 2.0,
        endpoint_silence: float = 0.6,
        pre_roll: float = 0.25,
        match_tolerance: float = 0.15,
        decode_interval: float | None = 0.75,
    ) -> None:
        self._asr = asr
        self._vad = vad
        self._language = language
        self.left_context = left_context
        self.window_cap = window_cap
        # Words ending within grey_zone of the live edge are too fresh to commit
        # even if two decodes agree — they lack right-context and often change.
        self.grey_zone = grey_zone
        # Trailing silence (s) past the last speech that closes an utterance.
        self.endpoint_silence = endpoint_silence
        # Silence kept after a reset so a word starting right away is not clipped.
        self.pre_roll = pre_roll
        self.match_tolerance = match_tolerance
        # Minimum audio time between speculative window re-decodes. Frames arrive
        # ~5×/s and each decode re-runs the whole window, so this is the GPU duty
        # cycle; endpoint and overflow flushes bypass it (accuracy-critical).
        # None = utterance mode: no speculative decodes at all — captions land
        # once per utterance and each second of speech is decoded exactly once
        # (the efficiency floor; needs a VAD to see any commits before flush).
        self.decode_interval = decode_interval

        self._window = _CaptionBuffer(vad)
        self._last_decode_end = float("-inf")  # audio_end at the last decode
        self._committed: list[Word] = []
        # Previous decode's uncommitted tail; LocalAgreement-2 commits the prefix
        # this agrees on with the next decode.
        self._buffer: list[Word] = []
        # Count of ASR decodes performed — a CPU proxy (should stay 0 in silence).
        self.decodes = 0

    # -- public API --------------------------------------------------------

    def feed(self, samples: np.ndarray, t_offset: float) -> StreamingUpdate:
        """Add audio at absolute time ``t_offset`` and re-decode the window."""
        if len(samples):
            self._window.append(samples, t_offset, tolerance=self.match_tolerance)
        if len(self._window) == 0:
            return StreamingUpdate((), self._interim_text())

        audio_end = self._window.end()
        buf_start = self._window.start
        assert buf_start is not None  # non-empty window ⇒ append set it
        speech = self._speech()
        uncommitted_speech = True  # without a VAD, assume the tail is speech
        if speech is not None:
            committed_end = self._committed_end()
            last_speech = speech[-1].end if speech else buf_start
            uncommitted_speech = any(s.end > committed_end + self.match_tolerance for s in speech)
            # An utterance boundary: the VAD-closed tail has gone quiet. Force the
            # pending words out (with the pause as right-context) and reset.
            if audio_end - last_speech >= self.endpoint_silence and (
                uncommitted_speech or self._buffer
            ):
                return StreamingUpdate(tuple(self._finalize_utterance(speech)), "")
            if not uncommitted_speech:
                # Idle silence between utterances: everything is committed, so drop
                # the buffered audio down to a pre-roll and keep memory bounded.
                self._window.reset_to_preroll(self.pre_roll)
                return StreamingUpdate((), "")

        # Bound the window: past window_cap of unbroken speech, force the tail out
        # (in order) rather than growing the buffer or dropping un-transcribed
        # audio. In utterance mode this is the only mid-utterance decode, so it
        # must fire on uncommitted *audio* — a pending LocalAgreement tail may
        # never exist.
        keep_from = max(self._committed_end() - self.left_context, buf_start)
        if audio_end - keep_from > self.window_cap and (self._buffer or uncommitted_speech):
            return StreamingUpdate(tuple(self._finalize_utterance(speech)), "")

        # Throttle: a speculative re-decode within decode_interval of the last one
        # would mostly re-transcribe the same audio — skip it, keep the interim.
        # In utterance mode (interval None) speculative decodes never run; the
        # endpoint, overflow, and flush paths above do all the decoding.
        if self.decode_interval is None or audio_end - self._last_decode_end < self.decode_interval:
            return StreamingUpdate((), self._interim_text())

        words = self._decode()
        new = self._filter_new(words)
        committed = self._commit(new, audio_end)
        self._trim()
        return StreamingUpdate(tuple(committed), self._interim_text())

    def flush(self) -> StreamingUpdate:
        """Force-commit the pending tail (end of stream or utterance) and reset."""
        return StreamingUpdate(tuple(self._finalize_utterance(self._speech())), "")

    def reset(self) -> None:
        """Drop the window and pending tail at a long silence; keep committed text."""
        self._buffer = []
        self._window.reset_to_preroll(self.pre_roll)

    def drop_window(self) -> None:
        """Abandon the retained window and pending tail without committing them.

        A live-pass **load-shed**. Unlike :meth:`reset`, this keeps no pre-roll and
        clears the buffer origin, so the next :meth:`feed` restarts clean at its own
        ``t_offset`` — no silence is padded across the skipped span. Committed
        history is left intact and still monotonic; the abandoned audio becomes a
        caption gap the finalize pass fills on stop. The worker calls this when
        inference has fallen so far behind that feeding the whole backlog would
        spiral (PLAN.md §5, Task 0f)."""
        self._window.drop()
        self._buffer = []

    @property
    def committed_words(self) -> tuple[Word, ...]:
        return tuple(self._committed)

    @property
    def committed_text(self) -> str:
        return " ".join(w.text for w in self._committed)

    @property
    def interim(self) -> str:
        return self._interim_text()

    @property
    def buffered_seconds(self) -> float:
        """Seconds of audio currently retained (bounded by trims and resets)."""
        return len(self._window) / SAMPLE_RATE

    # -- windowing ---------------------------------------------------------

    def _committed_end(self) -> float:
        return self._committed[-1].end if self._committed else (self._window.start or 0.0)

    def _speech(self) -> list[SpeechSegment] | None:
        """VAD segments over the current buffer (absolute time), or None w/o a VAD."""
        if self._vad is None or len(self._window) == 0:
            return None
        start = self._window.start or 0.0
        if self._window.vad_stream is not None:
            return self._window.vad_stream.segments(start)
        return [
            SpeechSegment(s.start + start, s.end + start)
            for s in self._vad.speech_segments(self._window.samples)
        ]

    def _decode(self) -> list[Word]:
        """Re-decode the whole retained window; word times shifted to absolute."""
        self.decodes += 1
        self._last_decode_end = self._window.end()
        start = self._window.start or 0.0
        return [
            Word(w.text, w.start + start, w.end + start, w.confidence)
            for seg in self._asr.transcribe(self._window.samples, self._language)
            for w in seg.words
        ]

    def _trim(self) -> None:
        """Drop committed audio so the window stays short (committed audio drops out).

        Only audio older than ``left_context`` before the last committed word is
        dropped — never uncommitted audio. The window length is instead bounded by
        the overflow-flush in :meth:`feed`, which keeps the retained buffer within
        ``window_cap`` without ever discarding un-transcribed speech.
        """
        keep_from = max(self._committed_end() - self.left_context, self._window.start or 0.0)
        self._window.trim_before(keep_from)

    def _finalize_utterance(self, speech: list[SpeechSegment] | None) -> list[Word]:
        """Commit everything still pending (utterance done → no grey zone) and reset."""
        committed: list[Word] = []
        if len(self._window) > 0 and (speech is None or speech):
            committed = _extend_committed(self._committed, self._filter_new(self._decode()))
        self._buffer = []
        self._window.reset_to_preroll(self.pre_roll)
        return committed

    # -- LocalAgreement-2 --------------------------------------------------

    def _filter_new(self, words: list[Word]) -> list[Word]:
        """Drop words the window re-emitted from the already-committed region."""
        if not self._committed:
            return list(words)
        cutoff = self._committed[-1].end - self.match_tolerance
        new = [w for w in words if w.start > cutoff]
        # Timestamps drift between decodes, so also drop a leading run of words
        # that just repeats the committed tail by text (whisper_streaming's
        # n-gram cleanup — PLAN.md §1 SimulStreaming reference).
        max_n = min(len(self._committed), len(new), 5)
        for n in range(max_n, 0, -1):
            if [_key(w) for w in self._committed[-n:]] == [_key(w) for w in new[:n]]:
                return new[n:]
        return new

    def _commit(self, new: list[Word], audio_end: float) -> list[Word]:
        """Commit the LocalAgreement-2 stable prefix, holding back the grey zone."""
        agree = 0
        for nw, bw in zip(new, self._buffer, strict=False):
            if _key(nw) != _key(bw):
                break
            agree += 1
        horizon = audio_end - self.grey_zone
        count = 0
        while count < agree and new[count].end <= horizon:
            count += 1
        self._buffer = new[count:]
        return _extend_committed(self._committed, new[:count])

    def _interim_text(self) -> str:
        return " ".join(w.text for w in self._buffer)


class WindowedLiveDecoder:
    """Live pass that decodes exactly the windows the finalize pass would build.

    :func:`stenograf.vad.pack_windows` is a greedy left-to-right merge, so it
    runs online: completed VAD runs accumulate into the current window, which
    closes — and is decoded ONCE — when the next run cannot join it (budget
    ``max_window`` exceeded, or silence beyond ``max_gap``). Same windows, same
    deterministic ``generate()`` ⇒ the committed text equals a batch
    ``finalize_channel`` ASR pass on the same audio (modulo streaming-VAD
    boundary jitter, eval/live.py --mode window), so the on-stop finalize can
    reuse it and skip its own ASR pass entirely.

    Cost: each second of speech is decoded exactly once, in finalize-sized
    windows — the same total ASR work the finalize pass alone would do.
    Captions land a window at a time (up to ``max_window`` s of speech plus
    ``max_gap`` of silence behind the live edge); there is no interim text.
    Chosen as the product default because the live view runs in the background
    (efficiency outranks caption latency).

    Satisfies the same :class:`StreamingDecoder` surface as
    :class:`LiveDecoder` but shares no algorithm with it — LocalAgreement has
    no role here — so it composes the same :class:`_CaptionBuffer` rather than
    inheriting machinery it would have to disable. Requires a streaming-capable
    VAD (``vad.stream``) — windows are VAD-defined.
    """

    def __init__(
        self,
        asr: ASRBackend,
        *,
        vad: SileroVAD,
        language: Language | None = None,
        max_window: float = 30.0,
        max_gap: float = 5.0,
        pad: float = 0.15,
        silence_guard: float = 1.0,
    ) -> None:
        if not hasattr(vad, "stream"):
            raise TypeError("WindowedLiveDecoder needs a streaming VAD (vad.stream)")
        self._asr = asr
        self._language = language
        # Window packing policy — MUST match pack_windows (the finalize pass
        # reuses these decodes verbatim; a policy drift silently degrades it).
        self.max_window = max_window
        self.max_gap = max_gap
        self.pad = pad
        # Audio kept behind the live edge during silence, so a speech onset the
        # VAD reports a beat late (plus the window pad) is still in the buffer.
        self.silence_guard = silence_guard
        self.window_cap = max_window  # the load-shed bound (StreamingDecoder)
        self.pre_roll = 0.25  # silence kept after the end-of-stream reset
        self.match_tolerance = 0.15  # feed-gap jitter still treated as contiguous
        # The buffer runs on the integer origin (quantized on first append):
        # decode slices must be byte-identical to the batch pass's.
        self._window = _CaptionBuffer(vad)
        self._committed: list[Word] = []
        self._pending: list[SpeechSegment] = []  # closed runs of the open window
        self._decoded_to = 0.0  # padded end of the last decoded window
        self.decodes = 0

    def feed(self, samples: np.ndarray, t_offset: float) -> StreamingUpdate:
        """Add audio; decode (only) the windows that closed since the last feed."""
        if len(samples):
            self._window.append(samples, t_offset, tolerance=self.match_tolerance)
            self._window.quantize_origin()
        if len(self._window) == 0 or self._window.vad_stream is None:
            return StreamingUpdate((), "")
        committed: list[Word] = []
        for seg in self._window.vad_stream.take_completed():
            committed.extend(self._absorb(seg))
        open_seg = self._window.vad_stream.open_segment()
        if self._pending:
            # The window also closes once nothing can join it any more — exactly
            # when pack_windows would split: the next run (open now, or anywhere
            # in the future silence) starts more than max_gap after it, or the
            # open run has already grown past the shared budget. Waiting longer
            # only delays the caption; the packing cannot change.
            next_start = open_seg.start if open_seg is not None else self._window.end()
            if next_start - self._pending[-1].end > self.max_gap or (
                open_seg is not None and open_seg.end - self._pending[0].start > self.max_window
            ):
                committed.extend(self._decode_window())
        self._retain(open_seg)
        return StreamingUpdate(tuple(committed), "")

    def flush(self) -> StreamingUpdate:
        """End of stream: close the VAD like the batch scan does, pack, decode."""
        stream = self._window.vad_stream
        if stream is None or len(self._window) == 0:
            return StreamingUpdate((), "")
        committed: list[Word] = []
        finish = getattr(stream, "finish", None)
        if finish is not None:
            finish()  # remainder + detector flush → the open run completes
        for seg in stream.take_completed():
            committed.extend(self._absorb(seg))
        if finish is None:
            open_seg = stream.open_segment()
            if open_seg is not None:
                committed.extend(self._absorb(open_seg))
        if self._pending:
            committed.extend(self._decode_window())
        self._window.reset_to_preroll(self.pre_roll)
        return StreamingUpdate(tuple(committed), "")

    def drop_window(self) -> None:
        """Load-shed: abandon the buffer and pending window without committing.

        Committed history is left intact and still monotonic; the abandoned
        audio becomes a caption gap the finalize pass fills on stop (see
        :meth:`LiveDecoder.drop_window`).
        """
        self._window.drop()
        self._pending = []

    @property
    def committed_words(self) -> tuple[Word, ...]:
        return tuple(self._committed)

    @property
    def committed_text(self) -> str:
        return " ".join(w.text for w in self._committed)

    @property
    def buffered_seconds(self) -> float:
        """Seconds of audio currently retained (bounded by :meth:`_retain`)."""
        return len(self._window) / SAMPLE_RATE

    # -- online pack_windows -------------------------------------------------

    def _absorb(self, seg: SpeechSegment) -> list[Word]:
        """Add one speech run to the open window, closing it first if needed."""
        committed: list[Word] = []
        if seg.end - seg.start > self.max_window:
            # Oversized run (sherpa's max_speech_duration is a soft bound —
            # 31 s runs happen): replicate pack_windows' hard split exactly.
            # Each cut is its own window; the previous window never absorbs the
            # run, and only the last piece stays open for later runs to join.
            if self._pending:
                committed.extend(self._decode_window())
            cuts = np.arange(seg.start, seg.end, self.max_window)
            for i, cut in enumerate(cuts):
                self._pending = [
                    SpeechSegment(float(cut), float(min(cut + self.max_window, seg.end)))
                ]
                if i < len(cuts) - 1:
                    committed.extend(self._decode_window())
            return committed
        if self._pending and (
            seg.end - self._pending[0].start > self.max_window
            or seg.start - self._pending[-1].end > self.max_gap
        ):
            committed = self._decode_window()
        self._pending.append(seg)
        return committed

    def _decode_window(self) -> list[Word]:
        """Decode the open window over its padded span; commit every word.

        The span floats and their sample_index() conversion mirror pack_windows
        + finalize_channel operation for operation, so the extracted slice is
        byte-identical to the batch pass's — the reuse guarantee.
        """
        start, end = self._pending[0].start, self._pending[-1].end
        self._pending = []
        a = max(self._window.start or 0.0, start - self.pad, self._decoded_to)
        b = min(self._window.end(), end + self.pad)
        self._decoded_to = b
        self.decodes += 1
        origin = self._window.start_idx or 0
        lo = max(0, sample_index(a) - origin)
        hi = sample_index(b) - origin
        words = [
            Word(w.text, w.start + a, w.end + a, w.confidence)
            for seg in self._asr.transcribe(self._window.samples[lo:hi], self._language)
            for w in seg.words
        ]
        return _extend_committed(self._committed, words)

    def _retain(self, open_seg: SpeechSegment | None) -> None:
        """Trim decoded/silent audio; keep the open window (plus its pad)."""
        if self._pending:
            keep_from = self._pending[0].start - self.pad
        elif open_seg is not None:
            keep_from = open_seg.start - self.pad
        else:
            keep_from = self._window.end() - self.silence_guard
        if self._window.start_idx is None:
            return
        self._window.trim_before(keep_from)


def _key(word: Word) -> str:
    """Match key for LocalAgreement: case- and punctuation-insensitive."""
    stripped = _WORD_KEY.sub("", word.text.lower())
    return stripped or word.text.strip().lower()
