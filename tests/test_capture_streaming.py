"""SessionClock unit tests — the one home of the timestamp-anchoring formula.

The provider suites (test_capture_linux.py / test_capture_windows.py) cover
the same invariants end to end through their transports; these pin the
arithmetic itself with a scripted clock and no threads.
"""

from __future__ import annotations

import math

import pytest

from stenograf.capture.base import SAMPLE_RATE, Channel
from stenograf.capture.streaming import SessionClock

FRAME = SAMPLE_RATE // 5  # 200 ms


class ScriptedClock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class TestSessionClock:
    def test_first_frame_anchors_at_arrival_minus_duration(self):
        clock = ScriptedClock()
        session = SessionClock(clock=clock)
        session.start()
        clock.t = 0.25  # the frame finished arriving 0.25 s in, carrying 0.2 s
        assert session.stamp(Channel.MIC, FRAME) == pytest.approx(0.05)

    def test_anchor_clamps_to_session_start(self):
        # A first frame delivered faster than real time (buffered startup)
        # must not stamp before t=0.
        clock = ScriptedClock()
        session = SessionClock(clock=clock)
        session.start()
        clock.t = 0.1  # 0.2 s of audio arrived only 0.1 s in
        assert session.stamp(Channel.MIC, FRAME) == 0.0

    def test_stamps_derive_from_sample_count_not_arrival(self):
        clock = ScriptedClock()
        session = SessionClock(clock=clock)
        session.start()
        stamps = []
        for arrival in (0.2, 0.6, 0.65, 1.1):  # jittered delivery
            clock.t = arrival
            stamps.append(session.stamp(Channel.MIC, FRAME))
        assert stamps == pytest.approx([0.0, 0.2, 0.4, 0.6])

    def test_channels_anchor_independently_on_one_clock(self):
        clock = ScriptedClock()
        session = SessionClock(clock=clock)
        session.start()
        clock.t = 0.2
        mic = session.stamp(Channel.MIC, FRAME)
        clock.t = 0.5  # the system stream opened later
        system = session.stamp(Channel.SYSTEM, FRAME)
        assert (mic, system) == pytest.approx((0.0, 0.3))
        clock.t = 0.7
        assert session.stamp(Channel.MIC, FRAME) == pytest.approx(0.2)
        assert session.stamp(Channel.SYSTEM, FRAME) == pytest.approx(0.5)

    def test_reanchors_forward_past_tolerance(self):
        # An under-filled silence gap (WASAPI loopback): the second frame
        # arrives 1.5 s after the first but carries only 0.2 s of audio.
        clock = ScriptedClock()
        session = SessionClock(clock=clock, reanchor_tolerance_s=0.5)
        session.start()
        clock.t = 0.2
        assert session.stamp(Channel.SYSTEM, FRAME) == pytest.approx(0.0)
        clock.t = 1.7
        assert session.stamp(Channel.SYSTEM, FRAME) == pytest.approx(1.5)
        clock.t = 1.9  # and the derived clock continues from the new anchor
        assert session.stamp(Channel.SYSTEM, FRAME) == pytest.approx(1.7)

    def test_lag_within_tolerance_keeps_the_derived_clock(self):
        clock = ScriptedClock()
        session = SessionClock(clock=clock, reanchor_tolerance_s=0.5)
        session.start()
        clock.t = 0.2
        session.stamp(Channel.MIC, FRAME)
        clock.t = 0.85  # 0.45 s behind — jitter, not a silence gap
        assert session.stamp(Channel.MIC, FRAME) == pytest.approx(0.2)

    def test_infinite_tolerance_never_reanchors(self):
        clock = ScriptedClock()
        session = SessionClock(clock=clock, reanchor_tolerance_s=math.inf)
        session.start()
        clock.t = 0.2
        session.stamp(Channel.MIC, FRAME)
        clock.t = 60.0  # an hour of lag would still derive from samples
        assert session.stamp(Channel.MIC, FRAME) == pytest.approx(0.2)

    def test_start_resets_channel_state(self):
        clock = ScriptedClock()
        session = SessionClock(clock=clock)
        session.start()
        clock.t = 0.4
        session.stamp(Channel.MIC, FRAME)
        session.start()  # a new capture session on the same provider
        clock.t = 0.6
        assert session.stamp(Channel.MIC, FRAME) == pytest.approx(0.0)
        assert session.started
