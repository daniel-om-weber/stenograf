import io
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

from stenograf.capture.base import Channel
from stenograf.capture.macos import (
    HelperNotFoundError,
    MacOSCaptureProvider,
    find_helper,
    read_frame,
)

FAKE = [sys.executable, str(Path(__file__).parent / "fake_stenocap.py")]
_HEADER = struct.Struct("<BdI")


def make_frame(code: int, timestamp: float, samples: list[int]) -> bytes:
    body = struct.pack(f"<{len(samples)}h", *samples)
    return _HEADER.pack(code, timestamp, len(samples)) + body


class TestReadFrame:
    def test_parses_a_frame(self):
        stream = io.BytesIO(make_frame(1, 1.5, [1, 2, 3]))
        frame = read_frame(stream)
        assert frame.channel is Channel.SYSTEM
        assert frame.timestamp == 1.5
        assert frame.samples.tolist() == [1, 2, 3]
        assert frame.samples.dtype == np.int16

    def test_returns_none_at_clean_eof(self):
        assert read_frame(io.BytesIO(b"")) is None

    def test_returns_none_on_truncated_payload(self):
        # Header promises 3 samples but only 1 is present → treated as EOF.
        truncated = _HEADER.pack(0, 0.0, 3) + struct.pack("<h", 9)
        assert read_frame(io.BytesIO(truncated)) is None

    def test_rejects_a_bad_channel_code(self):
        with pytest.raises(ValueError):
            read_frame(io.BytesIO(_HEADER.pack(7, 0.0, 0)))

    def test_rejects_an_absurd_sample_count(self):
        with pytest.raises(ValueError):
            read_frame(io.BytesIO(_HEADER.pack(0, 0.0, 10_000_000)))


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="MacOSCaptureProvider.stop() delivers SIGINT, which Windows cannot send",
)
class TestMacOSCaptureProvider:
    def test_reads_both_channels_until_eof(self):
        provider = MacOSCaptureProvider(command=FAKE)
        provider.start({Channel.MIC, Channel.SYSTEM})
        frames = list(provider.frames())
        provider.stop()

        assert {f.channel for f in frames} == {Channel.MIC, Channel.SYSTEM}
        mic = [f for f in frames if f.channel is Channel.MIC]
        assert len(mic) == 3
        assert [f.timestamp for f in mic] == [0.0, 0.1, 0.2]

    def test_only_requested_channel_is_started(self):
        provider = MacOSCaptureProvider(command=FAKE)
        provider.start({Channel.MIC})  # in-room: no system tap
        channels = {f.channel for f in provider.frames()}
        provider.stop()
        assert channels == {Channel.MIC}

    def test_stop_terminates_a_running_helper(self):
        provider = MacOSCaptureProvider(command=[*FAKE, "--forever"])
        provider.start({Channel.MIC})
        first = next(provider.frames())
        assert first.channel is Channel.MIC
        provider.stop()
        assert provider._proc is None  # torn down

    def test_a_stalled_consumer_never_blocks_the_helper(self):
        # The regression behind two production bugs (ebf660a, 7dd1510): the
        # consumer stalls, the 64 KB pipe fills, the helper blocks in write()
        # and Core Audio kills the tap permanently. The drain thread must absorb
        # the stream regardless of the consumer, so a helper with far more
        # output than the pipe holds can exit before frames() is ever read.
        provider = MacOSCaptureProvider(command=[*FAKE, "--frames", "200"])
        provider.start({Channel.MIC})
        assert provider._proc.wait(timeout=10) == 0  # a blocked write would time out
        frames = list(provider.frames())
        provider.stop()
        assert len(frames) == 200  # buffered while the consumer stalled, none dropped

    def test_stream_desync_raises_in_the_consumer(self):
        # The drain thread hits the malformed header; the error must surface in
        # frames(), not die silently on the drain thread.
        provider = MacOSCaptureProvider(command=[*FAKE, "--malformed"])
        provider.start({Channel.MIC})
        with pytest.raises(ValueError, match="malformed"):
            list(provider.frames())
        provider.stop()

    def test_on_log_keeps_helper_stderr_off_the_terminal(self, capfd):
        # The TUI path: helper chatter must reach the sink line-by-line and
        # never the real stderr, where it would be painted over the Textual
        # screen (the "device format at start / stopped at Ctrl-C" bug).
        lines = []
        provider = MacOSCaptureProvider(command=[*FAKE, "--chatter"], on_log=lines.append)
        provider.start({Channel.MIC})
        list(provider.frames())
        provider.stop()  # joins the relay: the final lines are in by now
        assert "fake-stenocap: mic format: 48000.0 Hz, 1 ch" in lines
        assert "fake-stenocap: stopped" in lines
        assert capfd.readouterr().err == ""

    def test_helper_stderr_is_inherited_by_default(self, capfd):
        # The plain CLI keeps today's behaviour: no sink, chatter lands on the
        # terminal's stderr where capture errors have always been visible.
        provider = MacOSCaptureProvider(command=[*FAKE, "--chatter"])
        provider.start({Channel.MIC})
        list(provider.frames())
        provider.stop()
        assert "fake-stenocap: mic format" in capfd.readouterr().err

    def test_a_raising_sink_does_not_break_capture(self):
        def bad_sink(line: str) -> None:
            raise RuntimeError("sink is broken")

        provider = MacOSCaptureProvider(command=[*FAKE, "--chatter"], on_log=bad_sink)
        provider.start({Channel.MIC})
        frames = list(provider.frames())
        provider.stop()
        assert len(frames) == 3  # audio unaffected by the sink's failures

    def test_stop_does_not_close_the_pipe_under_a_paused_reader(self):
        # stop() may fire (max_seconds, TUI quit) while the consumer sits between
        # yields; the read that resumes afterwards must end at clean EOF, not
        # raise "read of closed file" because stop() closed the pipe object.
        provider = MacOSCaptureProvider(command=[*FAKE, "--forever"])
        provider.start({Channel.MIC})
        frames = provider.frames()
        next(frames)  # reader is now paused mid-iteration
        provider.stop()
        for _ in frames:  # drain to EOF — must not raise
            pass


class TestFindHelper:
    def test_env_override_wins(self, monkeypatch, tmp_path):
        target = tmp_path / "stenocap"
        target.write_text("")
        monkeypatch.setenv("STENOGRAF_CAPTURE_HELPER", str(target))
        assert find_helper() == target

    def test_packaged_binary_regains_executable_bit(self, monkeypatch, tmp_path):
        # A wheel-installed binary that lost its executable bit (some install
        # paths drop it) must come back executable, not fail later with EACCES.
        import stenograf.capture.macos as macos

        monkeypatch.delenv("STENOGRAF_CAPTURE_HELPER", raising=False)
        packaged = tmp_path / "bin" / "stenocap"
        packaged.parent.mkdir()
        packaged.write_bytes(b"\x00")
        packaged.chmod(0o644)
        monkeypatch.setattr(macos.resources, "files", lambda package: tmp_path)
        found = find_helper()
        assert found == packaged
        assert os.access(found, os.X_OK)

    def test_raises_when_absent(self, monkeypatch):
        monkeypatch.delenv("STENOGRAF_CAPTURE_HELPER", raising=False)
        # Point the package-resource and dev-tree lookups at nothing by faking
        # __file__ location is overkill; instead assert the error type when the
        # binary is genuinely missing is exercised via a fresh temp env.
        import stenograf.capture.macos as macos

        monkeypatch.setattr(macos, "HELPER_NAME", "stenocap-does-not-exist")
        with pytest.raises(HelperNotFoundError):
            find_helper()
