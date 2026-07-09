"""Echo cancellation: does it remove the speakers, keep the speaker, and fail safe?"""

from __future__ import annotations

import numpy as np
import pytest

from stenograf.aec import TICK_SAMPLES, EchoCanceller, EchoCancellingProvider
from stenograf.capture.base import SAMPLE_RATE, AudioFrame, CaptureProvider, Channel

BOTH = {Channel.MIC, Channel.SYSTEM}
SECONDS = 6.0
ECHO_DELAY = 25 * SAMPLE_RATE // 1000
MIC_CHUNK = 1600  # 100 ms, like AVAudioEngine's buffers
SYS_CHUNK = 160  # 10 ms, like the process tap's


def _speech(seconds: float, seed: int) -> np.ndarray:
    """Broadband, amplitude-modulated noise: speech-like enough for an adaptive filter."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2.7 * t + seed)
    carrier = rng.normal(0, 0.25, n) + 0.3 * np.sin(2 * np.pi * 220 * t)
    return (carrier * envelope).astype(np.float32)


def _echo_of(far: np.ndarray, gain: float = 0.4) -> np.ndarray:
    """Delay, attenuate, soft-clip — laptop speakers are nonlinear."""
    echo = np.zeros_like(far)
    echo[ECHO_DELAY:] = far[:-ECHO_DELAY] * gain
    return np.tanh(echo * 1.6) / 1.6


def _i16(x: np.ndarray) -> np.ndarray:
    return np.clip(x * 32768.0, -32768, 32767).astype(np.int16)


def _energy_db(x: np.ndarray) -> float:
    return 10 * np.log10(float(np.mean(x.astype(np.float64) ** 2)) + 1e-12)


def _interleave(
    mic: np.ndarray, system: np.ndarray | None, *, mic_ts: float = 0.05
) -> list[AudioFrame]:
    """Frames in arrival order: the tap runs ahead of the mic, as it does live."""
    frames: list[tuple[float, int, AudioFrame]] = []
    if system is not None:
        for i in range(0, system.size - SYS_CHUNK + 1, SYS_CHUNK):
            ts = i / SAMPLE_RATE
            frames.append((ts, 0, AudioFrame(Channel.SYSTEM, ts, system[i : i + SYS_CHUNK])))
    for i in range(0, mic.size - MIC_CHUNK + 1, MIC_CHUNK):
        ts = mic_ts + i / SAMPLE_RATE
        frames.append((ts, 1, AudioFrame(Channel.MIC, ts, mic[i : i + MIC_CHUNK])))
    frames.sort(key=lambda f: (f[0], f[1]))
    return [f for _, _, f in frames]


def _run(
    frames: list[AudioFrame], channels: set[Channel] = BOTH
) -> tuple[np.ndarray, EchoCanceller]:
    aec = EchoCanceller(channels)
    out: list[np.ndarray] = []
    for frame in frames:
        for produced in aec.process(frame):
            if produced.channel is Channel.MIC:
                out.append(produced.samples)
    for produced in aec.drain():
        if produced.channel is Channel.MIC:
            out.append(produced.samples)
    return (np.concatenate(out) if out else np.zeros(0, np.int16)), aec


def _tail(x: np.ndarray, seconds: float = 2.0) -> np.ndarray:
    """Skip AEC3's convergence; judge it once the filter has adapted."""
    return x[-int(seconds * SAMPLE_RATE) :]


class TestCancellation:
    def test_removes_speaker_bleed_from_the_mic(self) -> None:
        far = _speech(SECONDS, seed=1)
        near = _echo_of(far)
        out, _ = _run(_interleave(_i16(near), _i16(far)))

        erle = _energy_db(_tail(_i16(near)[: out.size])) - _energy_db(_tail(out))
        assert erle > 12, f"only {erle:.1f} dB of echo removed"

    def test_keeps_your_voice_when_nothing_is_playing(self) -> None:
        local = _speech(SECONDS, seed=2)
        silence = np.zeros_like(local)
        out, _ = _run(_interleave(_i16(local), _i16(silence)))

        loss = _energy_db(_tail(_i16(local)[: out.size])) - _energy_db(_tail(out))
        assert abs(loss) < 2.0, f"near-end speech changed by {loss:.1f} dB with no echo present"

    def test_keeps_your_voice_during_double_talk(self) -> None:
        far = _speech(SECONDS, seed=3)
        local = _speech(SECONDS, seed=4)
        echo_only, _ = _run(_interleave(_i16(_echo_of(far)), _i16(far)))
        both, _ = _run(_interleave(_i16(_echo_of(far) + local), _i16(far)))

        # Speaking over the echo must survive far above what the echo alone leaves.
        margin = _energy_db(_tail(both)) - _energy_db(_tail(echo_only))
        assert margin > 10, f"local speech only {margin:.1f} dB above the residual echo"


class TestPassThrough:
    def test_system_channel_is_untouched(self) -> None:
        far = _speech(1.0, seed=5)
        aec = EchoCanceller(BOTH)
        frame = AudioFrame(Channel.SYSTEM, 0.0, _i16(far)[:SYS_CHUNK])
        (out,) = aec.process(frame)
        assert out is frame

    def test_bypasses_entirely_without_a_system_channel(self) -> None:
        aec = EchoCanceller({Channel.MIC})
        assert not aec.enabled
        frame = AudioFrame(Channel.MIC, 0.0, np.ones(MIC_CHUNK, np.int16))
        assert aec.process(frame) == [frame]

    def test_mic_timeline_is_preserved(self) -> None:
        far = _speech(2.0, seed=6)
        frames = _interleave(_i16(_echo_of(far)), _i16(far))
        aec = EchoCanceller(BOTH)
        produced = [p for f in frames for p in aec.process(f) if p.channel is Channel.MIC]
        produced += [p for p in aec.drain() if p.channel is Channel.MIC]

        assert produced[0].timestamp == pytest.approx(0.05, abs=1e-6)
        for a, b in zip(produced, produced[1:], strict=False):
            assert b.timestamp == pytest.approx(a.timestamp + a.duration, abs=1e-6)


class TestFailureModes:
    def test_a_stalled_reference_does_not_stall_the_mic(self) -> None:
        """A tap that dies mid-meeting must degrade to 'no cancellation', not 'no captions'."""
        far = _speech(SECONDS, seed=7)
        near = _i16(_echo_of(far))
        # The tap delivers only the first 200 ms, then goes silent forever.
        system = _i16(far)[: SYS_CHUNK * 20]
        frames = _interleave(near, system)

        out, aec = _run(frames)
        assert aec.far_end_missing_ticks > 0
        assert out.size > near.size // 2, "mic frames stopped flowing when the tap stalled"

    def test_a_gap_in_the_mic_is_padded_not_shifted(self) -> None:
        aec = EchoCanceller(BOTH)
        for i in range(0, SAMPLE_RATE, SYS_CHUNK):  # 1 s of reference
            ts = i / SAMPLE_RATE
            aec.process(AudioFrame(Channel.SYSTEM, ts, np.zeros(SYS_CHUNK, np.int16)))

        aec.process(AudioFrame(Channel.MIC, 0.0, np.ones(MIC_CHUNK, np.int16)))
        # 100 ms of mic audio was dropped by the device; the next frame jumps.
        produced = aec.process(AudioFrame(Channel.MIC, 0.2, np.ones(MIC_CHUNK, np.int16)))

        total = sum(p.samples.size for p in produced)
        assert total == MIC_CHUNK * 2, "the hole was not filled, so the timeline shifted"

    def test_a_backwards_frame_is_an_error(self) -> None:
        aec = EchoCanceller(BOTH)
        aec.process(AudioFrame(Channel.MIC, 1.0, np.ones(MIC_CHUNK, np.int16)))
        with pytest.raises(ValueError, match="desynced"):
            aec.process(AudioFrame(Channel.MIC, 0.0, np.ones(MIC_CHUNK, np.int16)))


class _FakeProvider(CaptureProvider):
    def __init__(self, frames: list[AudioFrame]) -> None:
        self._frames = frames
        self.started: set[Channel] | None = None
        self.stopped = False

    def start(self, channels: set[Channel]) -> None:
        self.started = channels

    def frames(self):
        yield from self._frames

    def stop(self) -> None:
        self.stopped = True


class TestProvider:
    def test_wraps_the_inner_provider(self) -> None:
        far = _speech(1.0, seed=8)
        inner = _FakeProvider(_interleave(_i16(_echo_of(far)), _i16(far)))
        provider = EchoCancellingProvider(inner)
        provider.start(BOTH)
        out = list(provider.frames())
        provider.stop()

        assert inner.started == BOTH
        assert inner.stopped
        assert {f.channel for f in out} == BOTH
        mic = np.concatenate([f.samples for f in out if f.channel is Channel.MIC])
        assert mic.size % TICK_SAMPLES == 0
