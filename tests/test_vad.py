from stenograf.vad import SpeechSegment, pack_windows


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
