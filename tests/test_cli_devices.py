"""`steno devices` — the microphones a meeting can record from.

The command exists because a device id is the one thing a user cannot guess:
Windows endpoint ids are GUIDs, and macOS UIDs are driver strings.
"""

from click.testing import CliRunner
from conftest import write_settings

from stenograf import cli


class TestDevicesCommand:
    """`steno devices` — the only place a mic_device id can be read off."""

    @staticmethod
    def _inputs(monkeypatch, devices, configured=None):
        from stenograf.capture import helper

        monkeypatch.setattr(helper, "list_input_devices", lambda: devices)
        if configured is not None:
            write_settings(f'[capture]\nmic_device = "{configured}"\n')

    def test_lists_ids_names_and_the_default(self, monkeypatch):
        from stenograf.capture.helper import InputDevice

        self._inputs(
            monkeypatch,
            [
                InputDevice(id="builtin", name="MacBook Pro Microphone", is_default=True),
                InputDevice(id="usb-1", name="Yeti Stereo Microphone", is_default=False),
            ],
        )
        result = CliRunner().invoke(cli.main, ["devices"])

        assert result.exit_code == 0, result.output
        assert "builtin" in result.output and "MacBook Pro Microphone" in result.output
        assert "usb-1" in result.output and "Yeti Stereo Microphone" in result.output
        assert "(system default)" in result.output.split("usb-1")[0]  # marked on the right one
        assert "settings.toml" in result.output  # how to make a choice standing

    def test_a_configured_device_that_is_absent_is_called_out(self, monkeypatch):
        from stenograf.capture.helper import InputDevice

        self._inputs(
            monkeypatch,
            [InputDevice(id="builtin", name="MacBook Pro Microphone", is_default=True)],
            configured="usb-1",
        )
        result = CliRunner().invoke(cli.main, ["devices"])

        assert result.exit_code == 0, result.output
        assert "not connected, meetings will stop" in result.output

    def test_two_devices_of_one_name_are_reported_as_ambiguous_not_missing(self, monkeypatch):
        # "plug it in" would be the wrong instruction: it is plugged in twice,
        # and the helper refuses to guess between them.
        from stenograf.capture.helper import InputDevice

        self._inputs(
            monkeypatch,
            [
                InputDevice(id="usb-1", name="Yeti", is_default=True),
                InputDevice(id="usb-2", name="Yeti", is_default=False),
            ],
            configured="Yeti",
        )
        result = CliRunner().invoke(cli.main, ["devices"])

        assert "matches 2 connected devices (usb-1, usb-2)" in result.output

    def test_the_word_default_is_not_a_missing_device(self, monkeypatch):
        # `mic_device = "default"` means "follow the system default"; reporting
        # it as absent would send the user hunting for a device called default.
        from stenograf.capture.helper import InputDevice

        self._inputs(
            monkeypatch,
            [InputDevice(id="builtin", name="Built-in", is_default=True)],
            configured="default",
        )
        result = CliRunner().invoke(cli.main, ["devices"])

        assert "meetings will stop" not in result.output
        assert "follows the system default" in result.output

    def test_a_broken_capture_stack_is_a_clean_error(self, monkeypatch):
        from stenograf.capture import helper
        from stenograf.capture.base import CaptureUnavailableError

        def boom():
            raise CaptureUnavailableError("capture helper 'stenocap' not found")

        monkeypatch.setattr(helper, "list_input_devices", boom)
        result = CliRunner().invoke(cli.main, ["devices"])

        assert result.exit_code != 0
        assert "stenocap" in result.output
        assert "Traceback" not in result.output
