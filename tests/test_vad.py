import wave
from pathlib import Path

import numpy as np
import pytest

from stenograf import models
from stenograf.audio import SAMPLE_RATE
from stenograf.vad import (
    DECODE_CONTEXT_S,
    PREROLL_S,
    SileroVAD,
    SpeechSegment,
    claim_start,
    context_start,
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
    start, end = windows[0]
    assert start == 0.85  # 1.0 - pad
    assert end == 20.15


def test_window_budget_starts_new_window():
    windows = pack_windows([seg(0, 20), seg(25, 45)], total_duration=60.0, max_window=30.0)
    assert len(windows) == 2
    # Second window starts at the second segment, not at a hard cut.
    assert windows[1][0] == 25.0 - 0.15


def test_oversized_segment_is_hard_split():
    windows = pack_windows([seg(0, 70)], total_duration=70.0, max_window=30.0)
    assert len(windows) == 3
    assert windows[0][1] >= 30.0
    assert windows[-1][1] == 70.0
    # Contiguous coverage: padding merged the touching splits.
    for (_, prev_end), (next_start, _) in zip(windows, windows[1:], strict=False):
        assert next_start <= prev_end


def test_padding_clamped_to_audio_bounds():
    windows = pack_windows([seg(0.0, 29.9)], total_duration=30.0)
    assert windows == [(0.0, 30.0)]


def test_long_silence_starts_a_new_window():
    # max_gap: a run further than max_gap behind the window's last speech opens
    # a new window even though the span would fit the budget — this is what lets
    # the live window pass close (and decode) a window max_gap after speech
    # stops, guaranteeing its windows equal this function's.
    windows = pack_windows([seg(0, 5), seg(12, 20)], total_duration=60.0, max_gap=5.0)
    assert len(windows) == 2
    windows = pack_windows([seg(0, 5), seg(9, 20)], total_duration=60.0, max_gap=5.0)
    assert len(windows) == 1


class TestDecodeSlice:
    """context_start/claim_start: what a window reads, and what it may keep."""

    def test_only_short_windows_read_left_context(self):
        assert context_start(40.0, 43.0) == 40.0 - DECODE_CONTEXT_S
        # A long window decodes exactly its span, so its claim below has no
        # audio to reach into and can never fire — measured, not incidental.
        assert context_start(40.0, 60.0) == 40.0

    def test_context_never_reaches_before_the_recording(self):
        assert context_start(1.0, 2.0) == 0.0

    def test_a_window_claims_a_preroll_behind_its_start(self):
        # The onset the VAD reported late: the word is inside the decode slice
        # either way, and this is what decides whether anyone emits it.
        assert claim_start(40.0, previous_end=10.0) == 40.0 - PREROLL_S

    def test_the_claim_never_reaches_into_the_previous_window(self):
        # Windows that touch (a hard split mid-speech, or a pad that ran into
        # its neighbour) get no pre-roll at all — that audio is already owned.
        assert claim_start(40.0, previous_end=40.0) == 40.0
        assert claim_start(40.0, previous_end=39.9) == 39.9
        # First window of the channel: nothing before it, so the floor is 0.
        assert claim_start(0.2, previous_end=0.0) == 0.0

    def test_over_real_packing_only_touching_windows_lose_their_preroll(self):
        # A 70 s unbroken run hard-splits into touching windows; the run after
        # the long silence stands alone. Over pack_windows' own spans that is
        # the difference between a window that may reach back and one that
        # may not.
        windows = pack_windows([seg(0, 70), seg(90, 95)], total_duration=120.0, max_window=30.0)
        claims = []
        previous_end = 0.0
        for start, end in windows:
            claims.append(claim_start(start, previous_end))
            previous_end = end
        assert claims[0] == 0.0  # first window: nothing before it
        assert claims[1] == windows[0][1] and claims[2] == windows[1][1]  # touching
        assert claims[3] == windows[3][0] - PREROLL_S  # isolated: full pre-roll
        assert claims[3] > windows[2][1]  # …and still clear of the window before


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
