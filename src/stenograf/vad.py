"""Voice activity detection and ASR windowing.

The finalize pass never feeds raw sliding windows to the ASR model: Silero
VAD finds speech, and ``pack_windows`` merges adjacent speech runs into
windows of at most ~30 s that cut in silence wherever possible (PLAN.md §2).
Where a cut is forced into speech anyway (budget close, oversized hard
split) the window edge is flagged (:class:`Window`) and repaired at decode
time with overlap (:func:`decode_slice`) — never by moving the bounds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stenograf.audio import SAMPLE_RATE


@dataclass(frozen=True)
class SpeechSegment:
    start: float
    end: float


@dataclass(frozen=True)
class Window:
    """One packed ASR window: its padded span, cut classification, speech runs.

    ``cut_start``/``cut_end`` are ``None`` at a natural silence close, or the
    shared **cut time** where the window boundary was a forced mid-speech cut
    (budget close with speech resuming within :data:`CUT_LOOKAHEAD_S`, or the
    oversized hard split). Two windows meeting at a cut carry the *identical*
    value — ``cut_end`` of the earlier equals ``cut_start`` of the later — so
    the keep-rules on both sides (:func:`decode_slice`) are exact complements:
    every decoded word lands in exactly one window.

    ``speech`` is the window's own (unpadded) VAD runs — what the packer
    merged into it. It is the ground truth the decode is checked against: a
    kept-word stream leaving a multi-second stretch of a run uncovered means
    the decode *skipped* speech (:func:`speech_hole`), and it tells a
    mid-speech cut start (gap ≈ 0: hard split, ``max_speech`` continuation)
    from one at a real pause (:func:`decode_slice`).
    """

    start: float
    end: float
    cut_start: float | None = None
    cut_end: float | None = None
    speech: tuple[SpeechSegment, ...] = ()


def _drain_vad(vad, origin: float = 0.0) -> list[SpeechSegment]:
    """Pop sherpa's completed speech runs, converted to seconds on ``origin``'s clock.

    The one place sample counts become segment times — the batch scan and the
    live stream must agree on this conversion exactly."""
    segments: list[SpeechSegment] = []
    while not vad.empty():
        seg = vad.front
        start = origin + seg.start / SAMPLE_RATE
        segments.append(SpeechSegment(start, start + len(seg.samples) / SAMPLE_RATE))
        vad.pop()
    return segments


class SileroVAD:
    """Silero v5 via sherpa-onnx (ONNX/CPU on every platform)."""

    def __init__(
        self,
        model_path: Path,
        *,
        threshold: float = 0.5,
        min_silence: float = 0.5,
        min_speech: float = 0.25,
        max_speech: float = 28.0,
    ) -> None:
        # threshold 0.5: 0.4 was tried (2026-07-19, to recover quiet mic
        # interjections the drop check found missed) and REVERTED the same day.
        # On a busy channel the looser gate stops yielding the min_silence
        # pauses that close a run, runs slam into max_speech, and pack_windows
        # cuts windows mid-speech at the budget — where the greedy TDT decode
        # is knife-edge unstable at the window tail and loses real words
        # (re-transcribing en-0713: −260 words, whole Whisper-confirmed
        # sentences gone; eval/README.md "window-length study"). A window tail
        # in a natural pause loses nothing; a tail in speech loses text.
        # Recovering the interjections needs cut-overlap decoding, not a
        # looser gate.
        import sherpa_onnx

        self._config = sherpa_onnx.VadModelConfig(
            silero_vad=sherpa_onnx.SileroVadModelConfig(
                model=str(model_path),
                threshold=threshold,
                min_silence_duration=min_silence,
                min_speech_duration=min_speech,
                # Bounds a single speech run so every ASR window fits the
                # pack_windows budget even in gap-free speech.
                max_speech_duration=max_speech,
            ),
            sample_rate=SAMPLE_RATE,
        )

    def speech_segments(self, samples: np.ndarray) -> list[SpeechSegment]:
        """Detect speech runs in mono 16 kHz float32 PCM."""
        import sherpa_onnx

        vad = sherpa_onnx.VoiceActivityDetector(self._config, buffer_size_in_seconds=120)
        segments: list[SpeechSegment] = []
        chunk = self._config.silero_vad.window_size
        for offset in range(0, len(samples), chunk):
            vad.accept_waveform(samples[offset : offset + chunk])
            segments.extend(_drain_vad(vad))
        vad.flush()
        segments.extend(_drain_vad(vad))
        return segments

    def stream(self, origin: float) -> SileroVADStream:
        """A persistent incremental detector for the live pass.

        ``origin`` is the absolute session time of the first pushed sample;
        reported segments are on that clock.
        """
        return SileroVADStream(self._config, origin)


class SileroVADStream:
    """One long-lived Silero detector fed only new samples (the live pass's VAD).

    Constructing sherpa's detector costs ~25 ms and the live decoder asks for
    speech on every audio frame, so :meth:`SileroVAD.speech_segments` — a fresh
    detector re-scanning the whole retained window per call — dominated the
    session's CPU. This keeps one causal detector per channel: completed
    segments accumulate as they close, and the in-progress run comes from
    sherpa's ``current_segment``, so each call costs only the new audio.
    """

    def __init__(self, config, origin: float) -> None:
        import sherpa_onnx

        self._vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=120)
        self._window = config.silero_vad.window_size
        self._origin = origin
        self._fed = 0  # samples pushed through the detector (whole windows only)
        self._pending = np.zeros(0, dtype=np.float32)
        self._segments: list[SpeechSegment] = []

    def push(self, samples: np.ndarray) -> None:
        """Feed mono 16 kHz float32 PCM continuing the stream (any length)."""
        buf = np.concatenate([self._pending, samples]) if len(self._pending) else samples
        end = len(buf) - len(buf) % self._window
        for offset in range(0, end, self._window):
            self._vad.accept_waveform(buf[offset : offset + self._window])
        self._fed += end
        self._pending = buf[end:]
        self._drain()

    def segments(self, min_end: float) -> list[SpeechSegment]:
        """Speech runs (absolute time) ending after ``min_end``, open tail included.

        ``min_end`` is the decoder's retained-buffer start; it only moves
        forward, so segments are pruned as they fall out of the window.
        """
        self._segments = [s for s in self._segments if s.end > min_end]
        out = list(self._segments)
        open_seg = self.open_segment()
        if open_seg is not None and open_seg.end > min_end:
            out.append(open_seg)
        return out

    def take_completed(self) -> list[SpeechSegment]:
        """Consume and return the closed speech runs accumulated so far.

        The window-mode consumer: each run is taken exactly once, in order, so
        the caller can pack them incrementally. Don't mix with :meth:`segments`
        on the same stream — that accessor keeps (and re-reports) its runs.
        """
        self._drain()
        out = self._segments
        self._segments = []
        return out

    def open_segment(self) -> SpeechSegment | None:
        """The in-progress speech run up to the pushed edge, if inside one."""
        if not self._vad.is_speech_detected():
            return None
        start = self._origin + self._vad.current_segment.start / SAMPLE_RATE
        return SpeechSegment(start, self._origin + self._fed / SAMPLE_RATE)

    def finish(self) -> None:
        """End of stream: feed the sub-window remainder and flush the detector.

        This mirrors the batch scan's tail handling (partial final chunk +
        ``flush``), closing an in-progress run at the true last sample — the
        batch pass would otherwise see up to one window (32 ms) more trailing
        speech than the stream did. The run lands in :meth:`take_completed`;
        the stream accepts no more pushes afterwards.
        """
        if len(self._pending):
            self._vad.accept_waveform(self._pending)
            self._fed += len(self._pending)
            self._pending = np.zeros(0, dtype=np.float32)
        self._vad.flush()
        self._drain()

    def _drain(self) -> None:
        self._segments.extend(_drain_vad(self._vad, self._origin))


def cut_boundary(prev_end: float, next_start: float) -> float | None:
    """The shared cut time for a window boundary, or ``None`` if it is natural.

    A boundary is a **cut** when speech resumes within
    :data:`CUT_LOOKAHEAD_S` of the earlier window's (unpadded) end — exactly
    the budget-close and hard-split boundaries, since a gap close needs more
    than ``max_gap`` of silence. The cut time is the midpoint of the non-speech
    gap (the hard split's zero gap degenerates to the split time itself), so
    both adjoining windows derive the identical value from the same two floats
    — the batch packer from adjacent windows, the online packer at close time.
    """
    gap = next_start - prev_end
    return (prev_end + next_start) / 2 if gap <= CUT_LOOKAHEAD_S else None


def pack_windows(
    segments: list[SpeechSegment],
    total_duration: float,
    *,
    max_window: float = 30.0,
    max_gap: float = 5.0,
    pad: float = 0.15,
) -> list[Window]:
    """Merge speech segments into ASR windows of at most ``max_window`` s.

    Consecutive speech runs share a window while they fit and the silence
    between them stays within ``max_gap``; each window is padded slightly into
    the surrounding silence so VAD onset jitter never clips a word. Returned
    windows are disjoint and sorted, and each edge is classified as a natural
    close or a forced mid-speech cut (:class:`Window`, :func:`cut_boundary`) —
    the decode paths repair cuts with overlap (:func:`decode_slice`) instead
    of ever moving these bounds.

    The ``max_gap`` bound exists for the live window pass: it lets an online
    packer close a window ``max_gap`` after speech stops (nothing later can
    join it), so live windows equal this function's output and the finalize
    pass can reuse the live decodes verbatim.
    """
    # pad 0.15: widening to 0.3 was tried (2026-07-19; onsets were measurably
    # over-represented among errors) and REVERTED — shifting every window
    # boundary re-rolls the greedy TDT's knife-edge tail decode, and the wider
    # pad systematically lost: −302 words net vs the same code at 0.15 on one
    # 87-min meeting, whole Whisper-confirmed sentences gone (bisected with
    # context-carry held constant; eval/README.md "window-length study").
    # Onset clipping is real but must be fixed without moving window bounds.
    runs: list[list[SpeechSegment]] = []
    for seg in segments:
        # Oversized run (VAD's max_speech_duration should prevent this):
        # hard-split rather than hand the model an unbounded window.
        if seg.end - seg.start > max_window:
            for cut in np.arange(seg.start, seg.end, max_window):
                runs.append([SpeechSegment(float(cut), float(min(cut + max_window, seg.end)))])
            continue
        if (
            runs
            and seg.end - runs[-1][0].start <= max_window
            and seg.start - runs[-1][-1].end <= max_gap
        ):
            runs[-1].append(seg)
        else:
            runs.append([seg])

    # Cut classification runs on the unpadded bounds: a boundary is a cut iff
    # the next window's speech starts within CUT_LOOKAHEAD_S of this one's end.
    boundaries = [cut_boundary(runs[i][-1].end, runs[i + 1][0].start) for i in range(len(runs) - 1)]

    padded: list[Window] = []
    for i, window_runs in enumerate(runs):
        start = max(0.0, window_runs[0].start - pad)
        if padded:
            # Keep windows disjoint: hard-split neighbours touch, so the pad
            # must not reach back into the previous window.
            start = max(start, padded[-1].end)
        end = min(total_duration, window_runs[-1].end + pad)
        padded.append(
            Window(
                start,
                end,
                cut_start=boundaries[i - 1] if i > 0 else None,
                cut_end=boundaries[i] if i < len(boundaries) else None,
                speech=tuple(window_runs),
            )
        )
    return padded


DECODE_CONTEXT_S = 15.0
"""Contiguous left context (s) a short window's decode reads (:func:`context_start`)."""

SHORT_WINDOW_S = 8.0
"""Windows shorter than this decode with left context. The bound is measured:
below it, the same model with context added wins the pivot referee ~2.5:1
(23:9 under 3 s, 29:15 at 3–8 s); at or above it the effect is null (26:29)
— eval/context_ab.py, 2026-07-19."""

CUT_LOOKAHEAD_S = 1.0
"""A window edge with speech resuming within this is a forced mid-speech cut
(:func:`cut_boundary`). Gap closes need > ``max_gap`` of silence, so only the
budget close and the oversized hard split can produce such an edge."""

OVERHANG_S = 2.5
"""Extra audio (s) a cut-ended window's decode slice reads past its end.

The greedy TDT decode is knife-edge unstable at the very tail of its slice —
the same span decodes completely or drops ~10 trailing words on a
millisecond-level bound shift (eval/README.md "window-length study"). The
overhang moves the unstable tail past the cut, into audio whose words are
dropped again (their midpoint lies beyond ``cut_end``), so the kept text no
longer sits in the danger zone. Same move, mirrored, for a cut *start*."""

SPEECH_HOLE_S = 1.5
"""A stretch of a window's own VAD speech this long with no kept word over it
means the decode *skipped* speech. The greedy TDT occasionally skips several
seconds of a long slice — anywhere in its later half, not just the last
second (measured: an overhang-only variant lost whole sentences mid-window
that the pre-change slice kept, and vice versa; the overhang re-rolls the
knife-edge rather than removing it). Real inter-word gaps inside one VAD run
stay well under a second, so 1.5 s separates skips from prosody. A detected
hole triggers one re-decode over the pre-change slice (:func:`speech_hole`,
``pipeline._decode_one``) and the variant covering more speech wins —
deterministic, paid only on detected skips, and structurally bounded: the
fallback is byte-identical to what the pre-change code decoded."""


def speech_hole(covered: list[tuple[float, float]], speech: tuple[SpeechSegment, ...]) -> float:
    """The longest stretch of VAD speech left uncovered by the kept words.

    ``covered`` are the kept words' time spans, in order. Both decode paths
    must judge retries with this same function, or the batch and live passes
    would retry different windows and their decodes would diverge. Returns the
    longest single uncovered stretch (s); compare against
    :data:`SPEECH_HOLE_S` to detect a skip, and prefer the decode variant
    with the smaller value.
    """
    worst = 0.0
    for run in speech:
        edge = run.start
        for lo, hi in covered:
            if lo > run.end:
                break
            if lo > edge:
                worst = max(worst, min(lo, run.end) - edge)
            edge = max(edge, hi)
        worst = max(worst, run.end - edge)
    return worst


MID_SPEECH_GAP_S = 0.1
"""A cut start with less silence than this before the window's first run is a
true mid-speech start. The gap half-width (``speech[0].start - cut_start``)
is bimodal by construction: ~0 for hard splits and ``max_speech``
continuations, ≥ 0.25 for cuts at a real VAD pause (``min_silence`` = 0.5 s
puts at least 0.5 s between distinct runs)."""


def context_start(start: float, end: float, *, cut: bool = False) -> float:
    """Where the decode slice for the packed window ``[start, end)`` begins.

    A short, isolated window starves the model of acoustic context — the
    dominant accuracy loss on short utterances (eval/out/context-ab.md). Short
    windows therefore decode from up to ``DECODE_CONTEXT_S`` of *contiguous*
    preceding audio, silence included (splicing distant speech across the gap
    measured worse — the seam costs accuracy), and the caller drops the words
    whose midpoint falls before the window's keep bound. A long window
    starting truly **mid-speech** (``cut=True``: hard split or ``max_speech``
    continuation — NOT a budget cut at a real pause, whose padded start is
    already a clean onset) reaches back ``OVERHANG_S`` so it does not start
    mid-word. The reach is deliberately NOT the short-window context: the A/B
    study measured added left context on ≥8 s windows as a null (26:29), so
    on long windows it is cut repair, not accuracy priming — and every extra
    re-read both costs decode time and re-rolls the knife-edge greedy decode
    of the whole window (measured: a reach on every cut-started window lost
    real sentences the unshifted slice kept). Other long windows decode
    exactly their span. Window packing is untouched: this changes only what
    the model reads, never the window bounds.

    Both decode paths — ``pipeline._decode`` and
    ``WindowedLiveDecoder._decode_window`` — and the byte-identity test reach
    this one function (via :func:`decode_slice`); diverging from it silently
    breaks the finalize pass's reuse of live decodes.
    """
    if end - start < SHORT_WINDOW_S:
        return max(0.0, start - DECODE_CONTEXT_S)
    if cut:
        return max(0.0, start - OVERHANG_S)
    return start


def decode_slice(window: Window) -> tuple[float, float, float, float]:
    """``(slice_start, slice_end, keep_lo, keep_hi)`` for one packed window.

    The single rule both decode paths share: the model reads
    ``[slice_start, slice_end)`` (callers clamp ``slice_end`` to the audio
    they actually have) and keeps exactly the words whose midpoint falls in
    ``[keep_lo, keep_hi)``. A cut-ended window reads ``OVERHANG_S`` past its
    end; a window starting truly mid-speech reads a left overhang
    (:func:`context_start`). The keep bounds are the shared cut times, so
    adjoining windows' keep intervals tile the timeline — half-open on the
    same side everywhere, meaning a word with its midpoint exactly on a cut
    belongs to the *later* window (the one that decodes it with left overlap
    rather than in its unstable tail).
    """
    mid_speech = (
        window.cut_start is not None
        and bool(window.speech)
        and window.speech[0].start - window.cut_start < MID_SPEECH_GAP_S
    )
    ctx = context_start(window.start, window.end, cut=mid_speech)
    hi = window.end + OVERHANG_S if window.cut_end is not None else window.end
    keep_lo = window.start if window.cut_start is None else window.cut_start
    keep_hi = math.inf if window.cut_end is None else window.cut_end
    return ctx, hi, keep_lo, keep_hi


def bare_slice(window: Window) -> tuple[float, float]:
    """The pre-cut-overlap decode slice for this window — the retry fallback.

    Context for short windows only, no overhang, no mid-speech reach:
    byte-identical to what the code before cut-overlap decoding fed the model
    for the same window. When the overlap decode skips speech
    (:func:`speech_hole`), re-decoding this slice bounds the damage: the
    shipped text is then whichever of {overlap decode, pre-change decode}
    covers the window's speech better.
    """
    return context_start(window.start, window.end), window.end
