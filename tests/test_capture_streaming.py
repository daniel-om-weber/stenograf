"""Unit tests for the shared queue-streaming machinery.

SessionClock is the one home of the timestamp-anchoring formula: the provider
suites (test_capture_linux.py / test_capture_windows.py) cover the same
invariants end to end through their transports, while these pin the arithmetic
itself with a scripted clock. TestPumpSentinel pins the other base-class
contract — the end-of-channel sentinel that terminates ``frames()``.
"""

from __future__ import annotations

import pytest

from stenograf.capture.base import SAMPLE_RATE, Channel
from stenograf.capture.streaming import QueueStreamingProvider, SessionClock

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

    def test_a_late_frame_never_moves_the_channels_anchor(self):
        # The forward re-anchor lived here for WASAPI loopback, whose silence
        # gaps were wall-clock estimates; that transport is gone and the
        # sample count is now the only authority, however late a frame is.
        clock = ScriptedClock()
        session = SessionClock(clock=clock)
        session.start()
        clock.t = 0.2
        assert session.stamp(Channel.SYSTEM, FRAME) == pytest.approx(0.0)
        clock.t = 60.0  # a minute of delivery lag
        assert session.stamp(Channel.SYSTEM, FRAME) == pytest.approx(0.2)

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


class ExplodingTeardownProvider(QueueStreamingProvider[None]):
    """Its stream ends at once, and tearing the siblings down always raises."""

    _thread_prefix = "boom"

    def _open_channel(self, channel: Channel) -> None:
        return None

    def _pump(self, channel: Channel, transport: None) -> None:
        return  # end of stream, immediately

    def _stop_transport(self) -> None:
        raise RuntimeError("teardown blew up")


class TestPumpSentinel:
    def test_sentinel_survives_a_teardown_that_raises(self):
        # A pump that ends tears its siblings down, and the sentinel it
        # enqueues on the way out is the only way frames() learns the channel
        # is done. A teardown that raises must not swallow it — the Windows
        # pump used to die on "cannot join thread before it is started" here,
        # which left frames() blocked on an empty queue, every pump dead, for
        # the rest of the meeting.
        provider = ExplodingTeardownProvider()
        provider._clock.start()
        with pytest.raises(RuntimeError, match="teardown blew up"):
            provider._run_pump(Channel.MIC, None)
        assert provider._queue.get_nowait() is Channel.MIC
