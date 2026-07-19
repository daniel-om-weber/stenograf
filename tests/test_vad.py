import wave
from pathlib import Path

import numpy as np
import pytest

from stenograf import models
from stenograf.audio import SAMPLE_RATE
from stenograf.vad import (
    DECODE_CONTEXT_S,
    MAX_DECODE_S,
    OVERHANG_S,
    SileroVAD,
    SpeechSegment,
    Window,
    decode_slice,
    pack_windows,
)

_EVAL_WAV = Path(__file__).resolve().parent.parent / "eval" / "audio" / "de-1.wav"


def seg(start: float, end: float) -> SpeechSegment:
    return SpeechSegment(start, end)


def test_empty_input():
    assert pack_windows([], total_duration=60.0) == []


def test_short_segments_share_a_window():
    windows = pack_windows([seg(1, 5), seg(6, 10), seg(12, 20)], total_duration=30.0)
    assert len(windows) == 1
    assert windows[0].start == 0.85  # 1.0 - pad
    assert windows[0].end == 20.15


def test_window_budget_starts_new_window():
    windows = pack_windows([seg(0, 20), seg(25, 45)], total_duration=60.0, max_window=30.0)
    assert len(windows) == 2
    # Second window starts at the second segment, not at a hard cut.
    assert windows[1].start == 25.0 - 0.15


def test_oversized_segment_is_hard_split():
    windows = pack_windows([seg(0, 70)], total_duration=70.0, max_window=30.0)
    assert len(windows) == 3
    assert windows[0].end >= 30.0
    assert windows[-1].end == 70.0
    # Contiguous coverage: padding merged the touching splits.
    for prev, nxt in zip(windows, windows[1:], strict=False):
        assert nxt.start <= prev.end


def test_padding_clamped_to_audio_bounds():
    windows = pack_windows([seg(0.0, 29.9)], total_duration=30.0)
    assert windows == [Window(0.0, 30.0)]


def test_long_silence_starts_a_new_window():
    # max_gap: a run further than max_gap behind the window's last speech opens
    # a new window even though the span would fit the budget — this is what lets
    # the live window pass close (and decode) a window max_gap after speech
    # stops, guaranteeing its windows equal this function's.
    windows = pack_windows([seg(0, 5), seg(12, 20)], total_duration=60.0, max_gap=5.0)
    assert len(windows) == 2
    windows = pack_windows([seg(0, 5), seg(9, 20)], total_duration=60.0, max_gap=5.0)
    assert len(windows) == 1


class TestCutClassification:
    """Window edges: natural silence closes vs forced mid-speech cuts."""

    def test_gap_close_is_natural(self):
        windows = pack_windows([seg(0, 5), seg(12, 20)], total_duration=60.0, max_gap=5.0)
        assert [(w.cut_start, w.cut_end) for w in windows] == [(None, None), (None, None)]

    def test_budget_close_with_speech_resuming_is_a_cut(self):
        # [0,20] closes on budget (31 > 30); speech resumes 0.5 s later → both
        # edges carry the identical cut time, the midpoint of the silence gap.
        windows = pack_windows([seg(0, 20), seg(20.5, 31)], total_duration=60.0, max_window=30.0)
        assert len(windows) == 2
        assert windows[0].cut_end == windows[1].cut_start == 20.25
        assert windows[0].cut_start is None and windows[1].cut_end is None

    def test_budget_close_into_a_real_pause_is_natural(self):
        # Same budget close, but 2 s of silence follows — the tail sits in a
        # real pause, where the decode is stable; no cut.
        windows = pack_windows([seg(0, 20), seg(22, 33)], total_duration=60.0, max_window=30.0)
        assert windows[0].cut_end is None and windows[1].cut_start is None

    def test_hard_split_edges_are_cuts_at_the_split_times(self):
        windows = pack_windows([seg(0, 70)], total_duration=70.0, max_window=30.0)
        assert [w.cut_start for w in windows] == [None, 30.0, 60.0]
        assert [w.cut_end for w in windows] == [30.0, 60.0, None]


class TestDecodeSlice:
    def test_natural_window_decodes_exactly_its_span(self):
        ctx, hi, keep_lo, keep_hi = decode_slice(Window(10.0, 30.0))
        assert (ctx, hi, keep_lo) == (10.0, 30.0, 10.0)
        assert keep_hi == float("inf")

    def test_short_window_reads_left_context(self):
        ctx, hi, keep_lo, keep_hi = decode_slice(Window(20.0, 25.0))
        assert ctx == 20.0 - DECODE_CONTEXT_S
        assert (hi, keep_lo) == (25.0, 20.0)

    def test_cut_ended_window_reads_the_overhang_and_clips_at_the_cut(self):
        ctx, hi, keep_lo, keep_hi = decode_slice(Window(0.0, 30.0, cut_end=30.2))
        assert (ctx, keep_lo) == (0.0, 0.0)
        assert hi == 30.0 + OVERHANG_S
        assert keep_hi == 30.2

    def test_cut_started_long_window_reads_context_within_the_decode_budget(self):
        # A full-budget window cut at both ends still fits MAX_DECODE_S: the
        # left reach shrinks to the room the overhang leaves.
        win = Window(30.0, 60.0, cut_start=29.9, cut_end=60.1)
        ctx, hi, keep_lo, keep_hi = decode_slice(win)
        assert ctx == 30.0 - (MAX_DECODE_S - OVERHANG_S - 30.0)
        assert hi - ctx <= MAX_DECODE_S
        assert (keep_lo, keep_hi) == (29.9, 60.1)
        # A shorter cut-started window gets up to the full measured context.
        ctx, _, _, _ = decode_slice(Window(30.0, 45.0, cut_start=29.9))
        assert ctx == 30.0 - DECODE_CONTEXT_S

    def test_keep_intervals_are_exact_complements_at_a_cut(self):
        windows = pack_windows([seg(0, 20), seg(20.5, 31)], total_duration=60.0, max_window=30.0)
        _, _, _, hi_left = decode_slice(windows[0])
        _, _, lo_right, _ = decode_slice(windows[1])
        assert hi_left == lo_right  # keep is [lo, hi): no word duplicates or falls through


@pytest.mark.skipif(
    models.cached_path(models.SILERO_VAD) is None or not _EVAL_WAV.exists(),
    reason="needs the cached silero model and the eval audio",
)
def test_stream_matches_batch_scan_on_real_speech():
    # The live pass swaps the fresh-detector-per-call window scan for one
    # persistent stream fed incrementally; both must find the same speech.
    with wave.open(str(_EVAL_WAV)) as w:
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if w.getnchannels() == 2:
            raw = raw[::2]
    audio = raw[: 30 * SAMPLE_RATE].astype(np.float32) / 32768.0

    vad = SileroVAD(models.cached_path(models.SILERO_VAD))
    batch = vad.speech_segments(audio)
    assert batch, "the eval clip should contain speech"

    stream = vad.stream(origin=0.0)
    for offset in range(0, len(audio), SAMPLE_RATE // 5):  # ~200 ms live frames
        stream.push(audio[offset : offset + SAMPLE_RATE // 5])
    streamed = stream.segments(min_end=0.0)

    assert len(streamed) == len(batch)
    for b, s in zip(batch, streamed, strict=True):
        assert abs(b.start - s.start) < 0.15
        # The trailing segment may still be open in the stream (no flush), so
        # its end can only lag the batch scan's flushed end.
        assert abs(b.end - s.end) < 0.15 or (s is streamed[-1] and s.end <= b.end)

    # Segments are reported on the stream's absolute clock.
    shifted = vad.stream(origin=100.0)
    for offset in range(0, len(audio), SAMPLE_RATE // 5):
        shifted.push(audio[offset : offset + SAMPLE_RATE // 5])
    for s, t in zip(streamed, shifted.segments(min_end=0.0), strict=True):
        assert abs((t.start - s.start) - 100.0) < 1e-6
