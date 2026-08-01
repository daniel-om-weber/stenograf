"""WindowsCaptureProvider unit tests — the fake helper, no audio hardware.

The transport itself is tested once, against macOS, in test_capture_macos.py:
both providers are the same :class:`HelperCaptureProvider`. What is tested here
is what Windows does differently — it stops its helper by closing stdin rather
than signalling it — and what only Windows has: the device preflight it runs
through the helper, and the privacy consent store it reads before capture.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from stenograf.capture.base import CaptureUnavailableError, Channel
from stenograf.capture.windows import (
    WindowsCaptureProvider,
    default_devices,
)

FAKE = [sys.executable, str(Path(__file__).parent / "fake_stenocap.py")]


class TestStopGesture:
    """Windows has no signal a parent can aim at one child, so stop() closes the
    helper's stdin. A helper that only watches stdin — which the real one is —
    never exits if the provider forgets to open that pipe."""

    def test_stop_ends_a_helper_that_only_watches_stdin(self):
        provider = WindowsCaptureProvider(command=[*FAKE, "--forever", "--stop-on-stdin"])
        provider.start({Channel.MIC, Channel.SYSTEM})
        frames = provider.frames()
        assert next(frames) is not None  # capture is live before we stop it
        provider.stop()
        assert list(frames) is not None  # the iterator finishes rather than hanging
        assert provider._proc is None

    def test_the_helper_is_spawned_with_a_stdin_pipe_to_close(self):
        provider = WindowsCaptureProvider(command=[*FAKE, "--forever", "--stop-on-stdin"])
        provider.start({Channel.MIC})
        assert provider._proc is not None
        assert provider._proc.stdin is not None
        provider.stop()

    def test_no_signal_is_sent(self, monkeypatch):
        # send_signal on Windows can only mean CTRL_C_EVENT/CTRL_BREAK_EVENT,
        # which reach a whole process group — including this test runner.
        sent = []
        monkeypatch.setattr(
            subprocess.Popen, "send_signal", lambda self, sig: sent.append(sig), raising=True
        )
        provider = WindowsCaptureProvider(command=[*FAKE, "--forever", "--stop-on-stdin"])
        provider.start({Channel.MIC})
        next(provider.frames())
        provider.stop()
        assert sent == []

    def test_stop_before_start_is_a_no_op(self):
        WindowsCaptureProvider(command=FAKE).stop()


class TestDefaultDevices:
    def _unblocked(self, monkeypatch):
        # Hermetic: don't let the host machine's real privacy settings leak in.
        import stenograf.capture.windows as windows

        monkeypatch.setattr(windows, "mic_access_blocked", lambda: None)

    def _helper(self, monkeypatch, *extra):
        import stenograf.capture.windows as windows

        argv = [*FAKE, *extra]
        monkeypatch.setattr(windows, "find_helper", lambda: argv[0])
        # find_helper returns one path; the fake needs its script argument too,
        # so the run itself is what gets patched for these tests.
        real_run = subprocess.run
        monkeypatch.setattr(
            windows.subprocess,
            "run",
            lambda command, **kw: real_run([*argv, *command[1:]], **kw),
        )

    def test_names_what_each_channel_would_record(self, monkeypatch):
        self._unblocked(monkeypatch)
        self._helper(monkeypatch)
        devices = default_devices({Channel.MIC, Channel.SYSTEM})
        assert devices == {
            Channel.MIC: "Fake mic device",
            Channel.SYSTEM: "Fake system device (loopback)",
        }

    def test_mic_only_never_asks_about_an_output(self, monkeypatch):
        # In-room mode: a box with no output device at all must still pass.
        self._unblocked(monkeypatch)
        self._helper(monkeypatch)
        assert default_devices({Channel.MIC}) == {Channel.MIC: "Fake mic device"}

    def test_a_helper_that_cannot_resolve_a_device_is_a_capture_error(self, monkeypatch):
        self._unblocked(monkeypatch)
        self._helper(monkeypatch, "--die-at-start")
        with pytest.raises(CaptureUnavailableError, match="no default system device"):
            default_devices({Channel.SYSTEM})

    def test_the_helpers_own_reason_survives_without_its_prefixes(self, monkeypatch):
        self._unblocked(monkeypatch)
        self._helper(monkeypatch, "--die-at-start")
        with pytest.raises(CaptureUnavailableError) as caught:
            default_devices({Channel.SYSTEM})
        assert "FATAL" not in str(caught.value)
        assert "stenocap:" not in str(caught.value)

    def test_unparseable_output_is_a_capture_error(self, monkeypatch):
        import stenograf.capture.windows as windows

        self._unblocked(monkeypatch)
        monkeypatch.setattr(windows, "find_helper", lambda: "helper")
        monkeypatch.setattr(
            windows.subprocess,
            "run",
            lambda *a, **kw: subprocess.CompletedProcess(a, 0, b"not json", b""),
        )
        with pytest.raises(CaptureUnavailableError, match="did not report its devices"):
            default_devices({Channel.MIC})

    def test_a_missing_helper_is_a_capture_error(self, monkeypatch):
        import stenograf.capture.windows as windows

        self._unblocked(monkeypatch)
        monkeypatch.setattr(windows, "find_helper", lambda: "no-such-helper-binary")
        with pytest.raises(CaptureUnavailableError, match="could not be run"):
            default_devices({Channel.MIC})

    def test_denied_mic_privacy_fails_before_the_helper_runs(self, monkeypatch):
        import stenograf.capture.windows as windows

        monkeypatch.setattr(windows, "mic_access_blocked", lambda: "privacy settings say no")
        monkeypatch.setattr(
            windows, "find_helper", lambda: pytest.fail("the helper must not be spawned")
        )
        with pytest.raises(CaptureUnavailableError, match="privacy settings"):
            default_devices({Channel.MIC, Channel.SYSTEM})

    def test_denied_mic_privacy_does_not_gate_loopback(self, monkeypatch):
        # System audio is not privacy-gated; a remote-only capture must pass.
        import stenograf.capture.windows as windows

        monkeypatch.setattr(windows, "mic_access_blocked", lambda: "privacy settings say no")
        self._helper(monkeypatch)
        devices = default_devices({Channel.SYSTEM})
        assert devices == {Channel.SYSTEM: "Fake system device (loopback)"}

    def test_channels_the_caller_did_not_ask_for_are_dropped(self, monkeypatch):
        """The helper answering more than it was asked must not widen the preflight."""
        import stenograf.capture.windows as windows

        self._unblocked(monkeypatch)
        monkeypatch.setattr(windows, "find_helper", lambda: "helper")
        both = json.dumps({"mic": "M", "system": "S"}).encode()
        monkeypatch.setattr(
            windows.subprocess,
            "run",
            lambda *a, **kw: subprocess.CompletedProcess(a, 0, both, b""),
        )
        assert default_devices({Channel.MIC}) == {Channel.MIC: "M"}


class TestMicAccessBlocked:
    def _patch_consent(self, monkeypatch, values):
        """values maps (subkey, machine) -> 'allow' | 'deny'; absent means None."""
        import stenograf.capture.windows as windows

        monkeypatch.setattr(
            windows,
            "_consent_value",
            lambda subkey, *, machine: values.get((subkey, machine)),
        )

    def test_missing_keys_mean_allowed(self, monkeypatch):
        from stenograf.capture.windows import mic_access_blocked

        self._patch_consent(monkeypatch, {})
        assert mic_access_blocked() is None

    def test_explicit_allow_everywhere(self, monkeypatch):
        from stenograf.capture.windows import mic_access_blocked

        self._patch_consent(
            monkeypatch,
            {("", False): "allow", ("NonPackaged", False): "allow", ("", True): "allow"},
        )
        assert mic_access_blocked() is None

    @pytest.mark.parametrize(
        "denied_key,expected",
        [
            (("", False), "turned off in Windows"),
            (("NonPackaged", False), "desktop apps"),
            (("", True), "machine-wide"),
        ],
    )
    def test_any_deny_names_the_toggle_and_the_fix(self, monkeypatch, denied_key, expected):
        from stenograf.capture.windows import mic_access_blocked

        self._patch_consent(monkeypatch, {denied_key: "deny"})
        blocked = mic_access_blocked()
        assert blocked is not None
        assert expected in blocked
        assert "Privacy & security" in blocked  # points at the settings page
