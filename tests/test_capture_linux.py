"""LinuxCaptureProvider unit tests — the fake helper, no audio hardware.

The transport is tested once, against macOS (test_capture_macos.py), and the
stdin-EOF stop gesture once, against Windows (test_capture_windows.py):
LinuxCaptureProvider shares both, because one Rust helper serves Windows and
Linux and it stops on stdin EOF on each. What is tested here is the wiring
only Linux has: the provider subclass really stops its helper the stdin way,
and the device preflight passes the helper's names through undecorated (a
monitor's name already says ``….monitor``; Windows is the platform that needs
a "(loopback)" suffix).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from stenograf.capture.base import CaptureUnavailableError, Channel
from stenograf.capture.linux import LinuxCaptureProvider, default_devices

FAKE = [sys.executable, str(Path(__file__).parent / "fake_stenocap.py")]


class TestStopGesture:
    def test_stop_ends_a_helper_that_only_watches_stdin(self):
        provider = LinuxCaptureProvider(command=[*FAKE, "--forever", "--stop-on-stdin"])
        provider.start({Channel.MIC, Channel.SYSTEM})
        frames = provider.frames()
        assert next(frames) is not None  # capture is live before we stop it
        provider.stop()
        assert list(frames) is not None  # the iterator finishes rather than hanging
        assert provider._proc is None


class TestDefaultDevices:
    def _helper(self, monkeypatch, *extra):
        import stenograf.capture.helper as helper

        argv = [*FAKE, *extra]
        monkeypatch.setattr(helper, "find_helper", lambda: argv[0])
        # find_helper returns one path; the fake needs its script argument too,
        # so the run itself is what gets patched for these tests.
        real_run = subprocess.run
        monkeypatch.setattr(
            helper.subprocess,
            "run",
            lambda command, **kw: real_run([*argv, *command[1:]], **kw),
        )

    def test_names_pass_through_undecorated(self, monkeypatch):
        self._helper(monkeypatch)
        devices = default_devices({Channel.MIC, Channel.SYSTEM})
        assert devices == {
            Channel.MIC: "Fake mic device",
            Channel.SYSTEM: "Fake system device",  # no "(loopback)" here
        }

    def test_mic_only_never_asks_for_a_sink(self, monkeypatch):
        # In-room mode: a box with no output device at all must still pass.
        self._helper(monkeypatch)
        assert default_devices({Channel.MIC}) == {Channel.MIC: "Fake mic device"}

    def test_a_helper_that_cannot_resolve_a_device_is_a_capture_error(self, monkeypatch):
        self._helper(monkeypatch, "--die-at-start")
        with pytest.raises(CaptureUnavailableError) as caught:
            default_devices({Channel.SYSTEM})
        # The helper's own reason, without its log prefixes.
        assert "FATAL" not in str(caught.value)
        assert "stenocap:" not in str(caught.value)

    def test_a_missing_helper_is_a_capture_error(self, monkeypatch):
        import stenograf.capture.helper as helper

        monkeypatch.setattr(helper, "find_helper", lambda: "no-such-helper-binary")
        with pytest.raises(CaptureUnavailableError, match="could not be run"):
            default_devices({Channel.MIC})
