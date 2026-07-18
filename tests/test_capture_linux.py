import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from stenograf.capture.base import SAMPLE_RATE, Channel
from stenograf.capture.linux import (
    CaptureUnavailableError,
    LinuxCaptureProvider,
    default_devices,
)

FAKE = [sys.executable, str(Path(__file__).parent / "fake_parec.py")]


class TestLinuxCaptureProvider:
    def test_reads_both_channels_until_eof(self):
        provider = LinuxCaptureProvider(command=FAKE)
        provider.start({Channel.MIC, Channel.SYSTEM})
        frames = list(provider.frames())
        provider.stop()

        assert {f.channel for f in frames} == {Channel.MIC, Channel.SYSTEM}
        mic = [f for f in frames if f.channel is Channel.MIC]
        system = [f for f in frames if f.channel is Channel.SYSTEM]
        # 0.4 s at 200 ms frames = 2 frames per channel, channel-distinct content
        assert len(mic) == 2 and len(system) == 2
        assert all(np.all(f.samples == 1) for f in mic)
        assert all(np.all(f.samples == 2) for f in system)
        assert all(f.samples.dtype == np.int16 for f in frames)

    def test_timestamps_are_sample_derived_per_channel(self):
        provider = LinuxCaptureProvider(command=FAKE)
        provider.start({Channel.MIC})
        frames = list(provider.frames())
        provider.stop()
        # Consecutive frames sit exactly one frame apart on the session clock —
        # derived from the sample count, not from arrival jitter.
        deltas = np.diff([f.timestamp for f in frames])
        assert np.allclose(deltas, len(frames[0].samples) / SAMPLE_RATE)
        assert frames[0].timestamp < 0.5  # anchored near session start

    def test_anchor_is_arrival_minus_frame_duration(self):
        # Deterministic anchor test (the Windows suite's equivalent drives an
        # injected clock too): the session clock is read once at start() and
        # once per frame, and fake_parec's 0.4 s burst is exactly two 200 ms
        # frames, so a scripted clock pins every timestamp exactly. The first
        # frame anchors at arrival minus its own duration (0.25 - 0.2); the
        # second sits one frame later regardless of its arrival reading.
        ticks = iter([100.0, 100.25, 100.45])
        provider = LinuxCaptureProvider(command=FAKE, clock=lambda: next(ticks))
        provider.start({Channel.MIC})
        frames = list(provider.frames())
        provider.stop()
        assert [f.timestamp for f in frames] == pytest.approx([0.05, 0.25])

    def test_only_requested_channel_is_started(self):
        provider = LinuxCaptureProvider(command=FAKE)
        provider.start({Channel.MIC})  # in-room: no system capture
        channels = {f.channel for f in provider.frames()}
        provider.stop()
        assert channels == {Channel.MIC}

    def test_stop_terminates_running_streams(self, monkeypatch):
        monkeypatch.setenv("FAKE_PAREC_SECONDS", "-1")
        provider = LinuxCaptureProvider(command=FAKE)
        provider.start({Channel.MIC, Channel.SYSTEM})
        iterator = provider.frames()
        assert next(iterator).samples.size  # streaming
        provider.stop()
        provider.stop()  # idempotent
        assert provider._procs == {}
        for _ in iterator:  # drains to the end — readers saw EOF
            pass

    def test_one_stream_dying_ends_the_whole_capture(self, monkeypatch):
        # The system stream runs forever; the mic stream dies after 0.2 s. The
        # provider must tear the survivor down too, so the meeting ends visibly
        # (and finalizes) instead of silently continuing half-captured.
        monkeypatch.setenv("FAKE_PAREC_MIC_SECONDS", "0.2")
        monkeypatch.setenv("FAKE_PAREC_SYSTEM_SECONDS", "-1")
        provider = LinuxCaptureProvider(command=FAKE)
        provider.start({Channel.MIC, Channel.SYSTEM})
        frames = list(provider.frames())  # must terminate on its own
        assert any(f.channel is Channel.MIC for f in frames)
        assert provider._procs == {}

    def test_on_log_keeps_parec_stderr_off_the_terminal(self, monkeypatch, capfd):
        # The TUI path: server messages must reach the sink per line, never the
        # real stderr, where they would be painted over the Textual screen.
        monkeypatch.setenv("FAKE_PAREC_CHATTER", "1")
        lines = []
        provider = LinuxCaptureProvider(command=FAKE, on_log=lines.append)
        provider.start({Channel.MIC, Channel.SYSTEM})
        list(provider.frames())
        provider.stop()
        assert sorted(lines) == [
            "fake-parec: Connection failure: @DEFAULT_MONITOR@",
            "fake-parec: Connection failure: @DEFAULT_SOURCE@",
        ]
        assert capfd.readouterr().err == ""

    def test_parec_stderr_is_inherited_by_default(self, monkeypatch, capfd):
        # The plain CLI keeps today's behaviour: no sink, server messages land
        # on the terminal's stderr where they have always been visible.
        monkeypatch.setenv("FAKE_PAREC_CHATTER", "1")
        provider = LinuxCaptureProvider(command=FAKE)
        provider.start({Channel.MIC})
        list(provider.frames())
        provider.stop()
        assert "fake-parec: Connection failure" in capfd.readouterr().err

    def test_frames_before_start_raises(self):
        provider = LinuxCaptureProvider(command=FAKE)
        with pytest.raises(RuntimeError):
            next(provider.frames())

    def test_missing_parec_fails_at_construction(self, monkeypatch):
        import stenograf.capture.linux as linux

        monkeypatch.setattr(linux.shutil, "which", lambda name: None)
        with pytest.raises(CaptureUnavailableError, match="parec"):
            LinuxCaptureProvider()


class TestDefaultDevices:
    def _run_result(self, stdout="", returncode=0, stderr=""):
        return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)

    def test_resolves_source_and_sink_monitor(self, monkeypatch):
        import stenograf.capture.linux as linux

        answers = {"get-default-source": "rnnoise_source", "get-default-sink": "alsa_out"}

        def fake_run(argv, **kwargs):
            return self._run_result(stdout=answers[argv[1]] + "\n")

        monkeypatch.setattr(linux.subprocess, "run", fake_run)
        devices = default_devices({Channel.MIC, Channel.SYSTEM})
        assert devices == {
            Channel.MIC: "rnnoise_source",
            Channel.SYSTEM: "alsa_out.monitor",
        }

    def test_mic_only_never_asks_for_a_sink(self, monkeypatch):
        import stenograf.capture.linux as linux

        asked = []

        def fake_run(argv, **kwargs):
            asked.append(argv[1])
            return self._run_result(stdout="mic\n")

        monkeypatch.setattr(linux.subprocess, "run", fake_run)
        default_devices({Channel.MIC})  # in-room mode: a sinkless box must pass
        assert asked == ["get-default-source"]

    def test_missing_pactl_is_a_capture_error(self, monkeypatch):
        import stenograf.capture.linux as linux

        def fake_run(argv, **kwargs):
            raise FileNotFoundError("pactl")

        monkeypatch.setattr(linux.subprocess, "run", fake_run)
        with pytest.raises(CaptureUnavailableError, match="pactl"):
            default_devices({Channel.MIC})

    def test_unreachable_server_is_a_capture_error(self, monkeypatch):
        import stenograf.capture.linux as linux

        monkeypatch.setattr(
            linux.subprocess,
            "run",
            lambda argv, **kwargs: self._run_result(returncode=1, stderr="Connection refused"),
        )
        with pytest.raises(CaptureUnavailableError, match="Connection refused"):
            default_devices({Channel.SYSTEM})

    def test_no_default_device_is_a_capture_error(self, monkeypatch):
        import stenograf.capture.linux as linux

        monkeypatch.setattr(
            linux.subprocess, "run", lambda argv, **kwargs: self._run_result(stdout="\n")
        )
        with pytest.raises(CaptureUnavailableError, match="microphone"):
            default_devices({Channel.MIC})
