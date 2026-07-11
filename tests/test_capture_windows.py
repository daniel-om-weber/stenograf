"""WindowsCaptureProvider unit tests — a fake soundcard module, no audio hardware.

The fake mimics the surface the provider uses (default_microphone /
default_speaker / get_microphone / recorder / record) and lets each test
script the delivered blocks; the re-anchor test drives the injected session
clock. Mirrors test_capture_linux.py.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pytest

from stenograf.capture.base import SAMPLE_RATE, Channel
from stenograf.capture.windows import (
    CaptureUnavailableError,
    WindowsCaptureProvider,
    default_devices,
)

ONE = 1 / 32767  # float32 amplitude that lands on int16 value 1
TWO = 2 / 32767


class FakeRecorder:
    """Yields one scripted block per record() call, then dies like a stream.

    ``script`` entries are float amplitudes (a full block of that value), the
    string "forever" (zeros until the provider stops asking, with a real sleep
    so the pump loop isn't a busy spin), or an Exception to raise. ``advance``
    moves a fake clock by the paired amount before each delivery.
    """

    def __init__(self, script, clock=None, advance=None):
        self._script = list(script)
        self._clock = clock
        self._advance = list(advance) if advance else []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def record(self, numframes):
        if self._advance and self._clock is not None:
            self._clock.t += self._advance.pop(0)
        if not self._script:
            raise RuntimeError("stream ended")
        step = self._script[0]
        if step == "forever":
            time.sleep(0.005)
            return np.zeros((numframes, 2), dtype=np.float32)
        self._script.pop(0)
        if isinstance(step, Exception):
            raise step
        return np.full((numframes, 2), step, dtype=np.float32)


class FakeDevice:
    def __init__(self, name, make_recorder):
        self.name = name
        self.id = f"{{{name}}}"
        self._make_recorder = make_recorder

    def recorder(self, samplerate):
        assert samplerate == SAMPLE_RATE  # server-side SRC to the wire rate
        return self._make_recorder()


class FakeSoundcard:
    """The module surface capture/windows.py touches, with call tracking."""

    def __init__(self, mic=None, loopback=None):
        self._mic = mic
        self._loopback = loopback
        self.resolved: list[str] = []

    def default_microphone(self):
        self.resolved.append("mic")
        if self._mic is None:
            raise RuntimeError("no default input device")
        return self._mic

    def default_speaker(self):
        self.resolved.append("speaker")
        if self._loopback is None:
            raise RuntimeError("no default output device")
        return self._loopback

    def get_microphone(self, id, include_loopback=False):
        assert include_loopback and id == self._loopback.id
        return self._loopback


def fake_backend(mic_script=(ONE, ONE), system_script=(TWO, TWO), clock=None, advance=None):
    mic = FakeDevice("Fake Mic", lambda: FakeRecorder(mic_script, clock, advance))
    loopback = FakeDevice("Fake Speakers", lambda: FakeRecorder(system_script, clock, advance))
    return FakeSoundcard(mic=mic, loopback=loopback)


class TestWindowsCaptureProvider:
    def test_reads_both_channels_until_stream_end(self):
        provider = WindowsCaptureProvider(backend=fake_backend(), frame_ms=50)
        provider.start({Channel.MIC, Channel.SYSTEM})
        frames = list(provider.frames())
        provider.stop()

        assert {f.channel for f in frames} == {Channel.MIC, Channel.SYSTEM}
        mic = [f for f in frames if f.channel is Channel.MIC]
        system = [f for f in frames if f.channel is Channel.SYSTEM]
        # two scripted blocks per channel, channel-distinct content
        assert len(mic) == 2 and len(system) == 2
        assert all(np.all(f.samples == 1) for f in mic)
        assert all(np.all(f.samples == 2) for f in system)
        assert all(f.samples.dtype == np.int16 for f in frames)

    def test_timestamps_are_sample_derived_per_channel(self):
        provider = WindowsCaptureProvider(backend=fake_backend(mic_script=[ONE] * 5), frame_ms=50)
        provider.start({Channel.MIC})
        frames = list(provider.frames())
        provider.stop()
        # Consecutive frames sit exactly one frame apart on the session clock —
        # derived from the sample count, not from arrival jitter.
        deltas = np.diff([f.timestamp for f in frames])
        assert np.allclose(deltas, len(frames[0].samples) / SAMPLE_RATE)
        assert frames[0].timestamp < 0.5  # anchored near session start

    def test_reanchors_after_underfilled_loopback_silence(self):
        class FakeClock:
            t = 0.0

            def __call__(self):
                return self.t

        clock = FakeClock()
        # Frame 2 arrives 1.5 s after frame 1 but carries only 0.2 s of audio —
        # a silence gap soundcard's zero-fill under-estimated. The pump must
        # re-anchor to arrival-derived time instead of stamping it at 0.2 s.
        backend = fake_backend(system_script=[TWO, TWO, TWO], clock=clock, advance=[0.2, 1.5, 0.2])
        provider = WindowsCaptureProvider(backend=backend, clock=clock, frame_ms=200)
        provider.start({Channel.SYSTEM})
        frames = list(provider.frames())
        provider.stop()
        stamps = [f.timestamp for f in frames]
        assert stamps == pytest.approx([0.0, 1.5, 1.7])

    def test_only_requested_channel_is_started(self):
        backend = fake_backend()
        provider = WindowsCaptureProvider(backend=backend, frame_ms=50)
        provider.start({Channel.MIC})  # in-room: no system capture
        channels = {f.channel for f in provider.frames()}
        provider.stop()
        assert channels == {Channel.MIC}
        assert "speaker" not in backend.resolved  # loopback never opened

    def test_stop_ends_streams(self):
        backend = fake_backend(mic_script=["forever"], system_script=["forever"])
        provider = WindowsCaptureProvider(backend=backend, frame_ms=50)
        provider.start({Channel.MIC, Channel.SYSTEM})
        iterator = provider.frames()
        assert next(iterator).samples.size  # streaming
        provider.stop()
        provider.stop()  # idempotent
        for _ in iterator:  # drains to the end — pumps saw the stop event
            pass
        assert all(not t.is_alive() for t in provider._threads.values())

    def test_one_stream_dying_ends_the_whole_capture(self):
        # The system stream runs forever; the mic stream dies after one block.
        # The provider must tear the survivor down too, so the meeting ends
        # visibly (and finalizes) instead of silently continuing half-captured.
        backend = fake_backend(mic_script=[ONE], system_script=["forever"])
        provider = WindowsCaptureProvider(backend=backend, frame_ms=50)
        provider.start({Channel.MIC, Channel.SYSTEM})
        frames = list(provider.frames())  # must terminate on its own
        assert any(f.channel is Channel.MIC for f in frames)
        assert all(not t.is_alive() for t in provider._threads.values())

    def test_frames_before_start_raises(self):
        provider = WindowsCaptureProvider(backend=fake_backend())
        with pytest.raises(RuntimeError):
            next(provider.frames())

    def test_missing_soundcard_fails_at_construction(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "soundcard", None)
        with pytest.raises(CaptureUnavailableError, match="soundcard"):
            WindowsCaptureProvider()


class TestDefaultDevices:
    def _patch_backend(self, monkeypatch, backend):
        import stenograf.capture.windows as windows

        monkeypatch.setattr(windows, "_import_soundcard", lambda: backend)

    def test_resolves_mic_and_loopback_names(self, monkeypatch):
        self._patch_backend(monkeypatch, fake_backend())
        devices = default_devices({Channel.MIC, Channel.SYSTEM})
        assert devices == {
            Channel.MIC: "Fake Mic",
            Channel.SYSTEM: "Fake Speakers (loopback)",
        }

    def test_mic_only_never_asks_for_a_speaker(self, monkeypatch):
        backend = fake_backend()
        self._patch_backend(monkeypatch, backend)
        default_devices({Channel.MIC})  # in-room mode: a speakerless box must pass
        assert backend.resolved == ["mic"]

    def test_no_default_microphone_is_a_capture_error(self, monkeypatch):
        self._patch_backend(monkeypatch, FakeSoundcard(mic=None))
        with pytest.raises(CaptureUnavailableError, match="microphone"):
            default_devices({Channel.MIC})

    def test_no_default_output_is_a_capture_error(self, monkeypatch):
        self._patch_backend(monkeypatch, FakeSoundcard(mic=None, loopback=None))
        with pytest.raises(CaptureUnavailableError, match="output"):
            default_devices({Channel.SYSTEM})
