"""Live pass: finalize-identical windows, each decoded once.

:class:`WindowedLiveDecoder` turns the batch ``ASRBackend`` into a streaming
captioner without any new dependency: it packs completed VAD runs into exactly
the windows the finalize pass would build and decodes each window once over the
model's full ``generate()`` path, so live captions land at finalize-grade
accuracy and the on-stop finalize can reuse the decodes verbatim
(eval/live.py measures the parity). Silero VAD gates the work: in silence the
decoder does no ASR at all (~0% accelerator).

The commit stream is **append-only**: a committed word is never rewritten
(monotonicity — one of the label-free acceptance metrics). The decoder composes
``ASRBackend.transcribe`` and is model-agnostic, but the live pass needs word
timestamps, so in practice it runs Parakeet (the committed default).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stenograf.asr.base import ASRBackend, Word
from stenograf.audio import SAMPLE_RATE, sample_index, to_float32
from stenograf.config import Language
from stenograf.vad import (
    DECODE_CONTEXT_S,
    SileroVAD,
    SpeechSegment,
    claim_start,
    context_start,
)


@dataclass(frozen=True)
class StreamingUpdate:
    """The result of feeding one chunk: what newly committed, plus the grey tail.

    ``committed`` are the words finalized by *this* feed (empty when no window
    closed); the full committed transcript lives on the decoder. ``interim`` is
    a provisional tail shown grey — the window pass commits whole windows and
    always leaves it empty, but the view contract carries it.
    """

    committed: tuple[Word, ...]
    interim: str

    @property
    def committed_text(self) -> str:
        return " ".join(w.text for w in self.committed)


class _CaptionBuffer:
    """The retained audio window the live decoder feeds, scans, and slices.

    Owns the absolute-timestamped mono float32 samples, the silence padding
    across feed gaps, the backwards-feed guard, and the streaming VAD fed in
    lockstep (created per buffer origin when the VAD object supports
    ``stream``; ``None`` falls back to the caller's per-feed window scan).

    The origin is pinned to an exact integer sample index: the window pass
    slices its windows with ``sample_index()`` over absolute times, and reuse
    is byte-identical to the batch pass only if it extracts the very same
    samples — so the origin is tracked as an integer (float accumulation would
    drift by ±1 sample) and ``start`` is always derived from it by a single
    division.
    """

    def __init__(self, vad: SileroVAD | None) -> None:
        self._vad = vad
        self.samples = np.zeros(0, dtype=np.float32)
        self.start_idx: int | None = None  # integer sample origin; None = empty
        self.vad_stream = None

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def start(self) -> float | None:
        """Absolute time of ``samples[0]`` (``None`` before the first append)."""
        return None if self.start_idx is None else self.start_idx / SAMPLE_RATE

    def append(self, chunk: np.ndarray, t_offset: float, *, tolerance: float) -> None:
        """Add audio at absolute time ``t_offset``; pad gaps beyond ``tolerance``."""
        chunk = to_float32(np.asarray(chunk)).reshape(-1)
        if self.start_idx is None:
            self.start_idx = round(t_offset * SAMPLE_RATE)
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

    def end(self) -> float:
        """The live edge. Reproduces the batch pass's ``len(samples) /
        SAMPLE_RATE`` float bit-for-bit — (origin + buffered) equals the total
        sample count, so one division suffices."""
        if self.start_idx is None:
            return 0.0
        return (self.start_idx + len(self.samples)) / SAMPLE_RATE

    def trim_before(self, keep_from: float) -> None:
        """Drop audio older than ``keep_from``.

        Trims land on the same ``sample_index()`` grid the decode slices use,
        so a window's padded start is never trimmed past (truncation only
        rounds down, and ``keep_from`` is a lower bound on every future span
        start).
        """
        if self.start_idx is None:
            return
        keep_idx = max(self.start_idx, sample_index(keep_from))
        drop = keep_idx - self.start_idx
        if drop > 0:
            self.samples = self.samples[drop:]
            self.start_idx = keep_idx

    def reset_to_preroll(self, pre_roll: float) -> None:
        """Keep only a short silence pre-roll of the buffer."""
        keep = round(pre_roll * SAMPLE_RATE)
        if self.start_idx is None or len(self.samples) <= keep:
            return
        self.start_idx += len(self.samples) - keep
        self.samples = self.samples[-keep:]

    def drop(self) -> None:
        """Abandon the whole buffer; the next append restarts at its own origin.

        No silence is padded across the skipped span. The VAD stream's sample
        clock can't jump, so it is discarded too and rebuilt on the next append.
        """
        self.samples = np.zeros(0, dtype=np.float32)
        self.start_idx = None
        self.vad_stream = None


def _extend_committed(committed: list[Word], words: list[Word]) -> list[Word]:
    """Append ``words`` to the committed stream, enforcing non-decreasing starts.

    Re-decoding jitters word boundaries by a few ms, so a fresh window can
    place a boundary word a hair before the last committed word's start. Such
    a regressor is a re-emitted duplicate, never genuinely new text — dropping
    it keeps the committed stream strictly append-only (the monotonicity
    invariant) with no visible loss. Returns the words kept.
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


class WindowedLiveDecoder:
    """Live pass that decodes exactly the windows the finalize pass would build.

    :func:`stenograf.vad.pack_windows` is a greedy left-to-right merge, so it
    runs online: completed VAD runs accumulate into the current window, which
    closes — and is decoded ONCE — when the next run cannot join it (budget
    ``max_window`` exceeded, or silence beyond ``max_gap``). Same windows, same
    deterministic ``generate()`` ⇒ the committed text equals a batch
    ``finalize_channel`` ASR pass on the same audio (modulo streaming-VAD
    boundary jitter, eval/live.py), so the on-stop finalize can
    reuse it and skip its own ASR pass entirely.

    Cost: each second of speech is decoded exactly once, in finalize-sized
    windows — the same total ASR work the finalize pass alone would do (short
    windows additionally re-read their left context, mostly discarded again;
    see ``vad.context_start`` — the batch pass pays the same).
    Captions land a window at a time (up to ``max_window`` s of speech plus
    ``max_gap`` of silence behind the live edge); there is no interim text.
    Chosen as the product default because the live view runs in the background
    (efficiency outranks caption latency).

    Requires a streaming-capable VAD (``vad.stream``) — windows are
    VAD-defined.
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
        self.window_cap = max_window  # the load-shed bound (LiveWorker._shed_if_behind)
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
        audio becomes a caption gap the finalize pass fills on stop. The
        worker calls this when inference has fallen so far behind that feeding
        the whole backlog would spiral.
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
        """Decode the open window over its padded span; commit its words.

        The span floats and their sample_index() conversion mirror pack_windows
        + finalize_channel operation for operation, so the extracted slice is
        byte-identical to the batch pass's — the reuse guarantee. The slice
        reaches back into contiguous left context (``vad.context_start``, same
        function as the batch pass) and the window keeps the words it owns —
        its span plus a pre-roll, never behind ``_decoded_to``
        (``vad.claim_start``, same function again); the rest is dropped.
        """
        start, end = self._pending[0].start, self._pending[-1].end
        self._pending = []
        a = max(self._window.start or 0.0, start - self.pad, self._decoded_to)
        b = min(self._window.end(), end + self.pad)
        keep_from = claim_start(a, self._decoded_to)
        self._decoded_to = b
        ctx = max(self._window.start or 0.0, context_start(a, b))
        self.decodes += 1
        origin = self._window.start_idx or 0
        lo = max(0, sample_index(ctx) - origin)
        hi = sample_index(b) - origin
        words = [
            Word(w.text, w.start + ctx, w.end + ctx, w.confidence)
            for seg in self._asr.transcribe(self._window.samples[lo:hi], self._language)
            for w in seg.words
            if (w.start + w.end) / 2 + ctx >= keep_from
        ]
        return _extend_committed(self._committed, words)

    def _retain(self, open_seg: SpeechSegment | None) -> None:
        """Trim decoded/silent audio; keep the open window, its pad, and the
        decode context behind it.

        Every branch reaches ``DECODE_CONTEXT_S`` further back than the span it
        protects: whether the eventual window closes short is unknown until it
        closes, and if it does, ``_decode_window`` needs that context in the
        buffer to reproduce the batch slice (~1 MB of extra float32 per
        channel at 15 s). The silence branch keeps the guard's onset-latency
        margin on top, exactly as before.
        """
        if self._pending:
            keep_from = self._pending[0].start - self.pad - DECODE_CONTEXT_S
        elif open_seg is not None:
            keep_from = open_seg.start - self.pad - DECODE_CONTEXT_S
        else:
            keep_from = self._window.end() - self.silence_guard - DECODE_CONTEXT_S
        self._window.trim_before(keep_from)
