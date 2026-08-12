import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import TOOL_TIMEOUT_S, write_settings

from stenograf import doctor

# Several tests here call run_checks()/_desktop_app_check(), which spawn the
# real Qt probe — a subprocess the library gives 180 s of its own (doctor.py).
# A per-test cap under that budget would fire first, and since a tripped cap
# exits the process, a slow-but-legal probe would cost the run every other
# test's result plus doctor's own "the probe did not finish" message. So this
# file's cap sits above the library's number rather than under the suite's.
pytestmark = pytest.mark.timeout(200)


@pytest.fixture(autouse=True)
def _no_ollama_probe(monkeypatch):
    """``run_checks()``'s notes check probes the configured backend — on the
    platforms that default to Ollama that is a real HTTP request to localhost
    (5 s timeout), so a dev with a live Ollama would get a different run than
    CI. Tests that want a reachable Ollama patch ``is_available`` back."""
    from stenograf.notes.ollama import OllamaBackend

    monkeypatch.setattr(OllamaBackend, "is_available", lambda self: False)


def test_run_checks_includes_python_and_asr():
    names = {c.name for c in doctor.run_checks()}
    assert "Python" in names
    assert "ASR backend" in names
    python = next(c for c in doctor.run_checks() if c.name == "Python")
    assert python.ok  # we require >=3.12 and run under it


def test_desktop_app_check_runs_and_is_reported():
    # The window the app opens, built by the Qt this machine actually has —
    # the check exists because a Qt that imports can still fail to load its
    # plugins, which nothing short of loading them can tell. Optional: a
    # headless machine runs every subcommand without a window.
    check = doctor._desktop_app_check()
    assert check.ok, check.detail
    assert check.optional
    assert "Desktop app" in {c.name for c in doctor.run_checks()}


def test_desktop_app_check_compiles_every_screen(tmp_path):
    """Not just the window: each screen is a file of its own, reached at
    runtime, and a plugin only one of them imports is exactly how a broken Qt
    hides from a check that compiles the entry point alone."""
    from stenograf.gui import app as gui_app

    tree = tmp_path / "qml"
    shutil.copytree(gui_app.QML_DIR, tree)
    (tree / "Notes.qml").write_text("import QtQuick\nNoSuchTypeAnywhere {}\n")
    # The probe runs in its own interpreter, so the tree it reads is swapped in
    # its source rather than by patching this process.
    probe = doctor._QML_PROBE.replace(
        "from stenograf.gui.app import QML_DIR", f"QML_DIR = Path({str(tree)!r})"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=TOOL_TIMEOUT_S,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode != 0
    assert "NoSuchTypeAnywhere" in result.stderr


@pytest.mark.parametrize(
    ("stderr", "expected"),
    [
        # An aborted Qt: the actionable sentence is not the last line, which
        # only lists what was available instead.
        (
            'qt.qpa.plugin: Could not find the Qt platform plugin "xcb"\n'
            "This application failed to start because no Qt platform plugin "
            "could be initialized.\n\nAvailable platform plugins are: offscreen.\n",
            "This application failed to start",
        ),
        # A dlopen failure: the exception is followed by indented continuation
        # lines naming every path tried, and there can be dozens.
        (
            "Traceback (most recent call last):\n"
            + "".join(f'  File "<string>", line {n}, in <module>\n    x\n' for n in range(8))
            + "ImportError: dlopen(QtQml.abi3.so): Library not loaded: @rpath/QtQml\n"
            + "".join(f"    tried: '/opt/{n}/QtQml' (no such file)\n" for n in range(30)),
            "ImportError: dlopen",
        ),
    ],
)
def test_desktop_app_check_quotes_the_actionable_line(monkeypatch, stderr, expected):
    """The report is the only place a user ever sees why the app will not open,
    so the wrong line here costs them the cause entirely."""
    failed = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=stderr)
    monkeypatch.setattr(doctor.subprocess, "run", lambda *a, **kw: failed)

    check = doctor._desktop_app_check()

    assert not check.ok
    assert expected in check.detail
    assert "uv tool install --force stenograf" in check.detail
    assert len(check.detail) < 400  # a report line, not a dyld dump


def test_desktop_app_check_survives_a_probe_that_cannot_run(monkeypatch):
    def refuse(*args, **kwargs):
        raise OSError("no such interpreter")

    monkeypatch.setattr(doctor.subprocess, "run", refuse)
    check = doctor._desktop_app_check()
    assert not check.ok
    assert "no such interpreter" in check.detail


def test_desktop_app_check_does_not_quote_the_probe_at_a_timeout(monkeypatch):
    # TimeoutExpired stringifies its command, and the command is the whole
    # probe source — a screenful of Python in the middle of the report.
    def stall(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=[sys.executable, "-c", doctor._QML_PROBE], timeout=180)

    monkeypatch.setattr(doctor.subprocess, "run", stall)
    check = doctor._desktop_app_check()
    assert not check.ok
    assert "import sys" not in check.detail
    assert len(check.detail) < 100


def test_capture_helper_check_reports_found(monkeypatch, tmp_path):
    from stenograf.capture import helper as capture_helper

    helper = tmp_path / "stenocap"
    helper.write_bytes(b"\x00")
    helper.chmod(0o755)
    monkeypatch.setattr(capture_helper, "find_helper", lambda: helper)
    monkeypatch.setattr(doctor, "_codesign_valid", lambda path: (True, ""))
    check = doctor._capture_helper_check()
    assert check.ok
    assert str(helper) in check.detail
    assert "permission" in check.detail  # first-run permission guidance surfaced


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no executable bit")
def test_capture_helper_check_rejects_non_executable(monkeypatch, tmp_path):
    from stenograf.capture import helper as capture_helper

    helper = tmp_path / "stenocap"
    helper.write_bytes(b"\x00")
    helper.chmod(0o644)
    monkeypatch.setattr(capture_helper, "find_helper", lambda: helper)
    check = doctor._capture_helper_check()
    assert not check.ok
    assert "executable" in check.detail


def test_capture_helper_check_rejects_bad_signature(monkeypatch, tmp_path):
    from stenograf.capture import helper as capture_helper

    helper = tmp_path / "stenocap"
    helper.write_bytes(b"\x00")
    helper.chmod(0o755)
    monkeypatch.setattr(capture_helper, "find_helper", lambda: helper)
    monkeypatch.setattr(doctor, "_codesign_valid", lambda path: (False, "code has no signature"))
    check = doctor._capture_helper_check()
    assert not check.ok
    assert "signature" in check.detail
    assert "build.sh" in check.detail  # points at the fix


def test_codesign_valid_against_real_codesign(tmp_path):
    # An arbitrary file is not validly signed; the real codesign must say so.
    unsigned = tmp_path / "not-a-binary"
    unsigned.write_bytes(b"\x00" * 16)
    ok, why = doctor._codesign_valid(unsigned)
    assert not ok
    assert why  # carries codesign's reason (or its absence off-macOS)


def test_capture_helper_check_reports_missing(monkeypatch):
    from stenograf.capture import helper as capture_helper

    def boom():
        raise capture_helper.HelperNotFoundError("helper not found: build it")

    monkeypatch.setattr(capture_helper, "find_helper", boom)
    check = doctor._capture_helper_check()
    assert not check.ok
    assert "build it" in check.detail


def test_asr_check_present_and_absent(monkeypatch):
    # Pin the backend under test: the built-in default is capability-based,
    # so the bare default names a different backend per machine.
    monkeypatch.setenv("STENOGRAF_ASR_BACKEND", "parakeet")
    monkeypatch.setattr(doctor, "installed", lambda module: True)
    assert doctor._asr_check().ok
    monkeypatch.setattr(doctor, "installed", lambda module: False)
    absent = doctor._asr_check()
    assert not absent.ok
    assert "parakeet-mlx" in absent.detail
    assert "parakeet_mlx" in absent.detail  # the missing modules are named


def test_unsupported_platform_check_is_optional(monkeypatch):
    monkeypatch.setattr(doctor.sys, "platform", "freebsd14")
    platform_check = next(c for c in doctor.run_checks() if c.name == "Platform")
    assert not platform_check.ok
    assert platform_check.optional  # transcribe is supported; only live capture is missing
    assert "transcribe" in platform_check.detail


def test_linux_capture_check_names_the_devices(monkeypatch):
    from stenograf.capture import helper as capture_helper
    from stenograf.capture.base import Channel

    monkeypatch.setattr(capture_helper, "find_helper", lambda: Path("stenocap"))
    monkeypatch.setattr(
        capture_helper,
        "query_devices",
        lambda channels, mic_device=None: {Channel.MIC: "mymic", Channel.SYSTEM: "mysink.monitor"},
    )
    monkeypatch.setattr(doctor.sys, "platform", "linux")
    check = next(c for c in doctor.run_checks() if c.name == "Capture")
    assert check.ok
    assert not check.optional  # live capture is first-class on Linux
    assert "mic ← mymic" in check.detail
    assert "system ← mysink.monitor" in check.detail


def test_linux_capture_check_fails_without_the_helper(monkeypatch):
    from stenograf.capture import helper as capture_helper

    def boom():
        raise capture_helper.HelperNotFoundError("capture helper 'stenocap' not found")

    monkeypatch.setattr(capture_helper, "find_helper", boom)
    check = doctor._linux_capture_check()
    assert not check.ok
    assert "stenocap" in check.detail


def test_windows_capture_check_names_the_devices(monkeypatch):
    from stenograf.capture import helper as capture_helper
    from stenograf.capture import windows
    from stenograf.capture.base import Channel

    # The provider resolves the binary through the helper module's own
    # find_helper, not the name windows.py imported — patching the import site
    # leaves the check hitting the real lookup, which only succeeds on a machine
    # that happens to have a helper built.
    monkeypatch.setattr(capture_helper, "find_helper", lambda: Path("stenocap.exe"))
    monkeypatch.setattr(
        windows,
        "default_devices",
        lambda channels, mic_device=None: {
            Channel.MIC: "Headset Mic",
            Channel.SYSTEM: "Speakers (loopback)",
        },
    )
    monkeypatch.setattr(doctor.sys, "platform", "win32")
    check = next(c for c in doctor.run_checks() if c.name == "Capture")
    assert check.ok
    assert not check.optional  # live capture is first-class on Windows
    assert "mic ← Headset Mic" in check.detail
    assert "system ← Speakers (loopback)" in check.detail


def test_windows_capture_check_fails_without_the_helper(monkeypatch):
    from stenograf.capture import helper as capture_helper

    def boom():
        raise capture_helper.HelperNotFoundError("capture helper 'stenocap.exe' not found")

    monkeypatch.setattr(capture_helper, "find_helper", boom)
    monkeypatch.setattr(doctor.sys, "platform", "win32")
    check = doctor._windows_capture_check()
    assert not check.ok
    assert "stenocap.exe" in check.detail


def test_macos_version_check_parses_and_compares(monkeypatch):
    monkeypatch.setattr(platform, "mac_ver", lambda: ("14.4", ("", "", ""), ""))
    assert doctor._macos_version_check().ok
    monkeypatch.setattr(platform, "mac_ver", lambda: ("14.3", ("", "", ""), ""))
    assert not doctor._macos_version_check().ok
    monkeypatch.setattr(platform, "mac_ver", lambda: ("", ("", "", ""), ""))
    assert not doctor._macos_version_check().ok  # unparseable → not ok


def test_ffmpeg_check_reports_the_bundled_binary(monkeypatch):
    from stenograf import audio

    monkeypatch.setattr(audio, "ffmpeg_exe", lambda: "/bundled/ffmpeg")
    check = doctor._ffmpeg_check()
    assert check.ok
    assert "/bundled/ffmpeg" in check.detail

    def boom():
        raise RuntimeError("no binary for this platform")

    monkeypatch.setattr(audio, "ffmpeg_exe", boom)
    check = doctor._ffmpeg_check()
    assert not check.ok
    assert "IMAGEIO_FFMPEG_EXE" in check.detail  # the escape hatch is named


def test_models_check_reflects_cache(monkeypatch):
    from stenograf import assets

    monkeypatch.setattr(assets, "cached_path", lambda asset: Path("/cached"))
    assert doctor._models_check().ok
    monkeypatch.setattr(assets, "cached_path", lambda asset: None)
    missing = doctor._models_check()
    assert not missing.ok
    assert "pending" in missing.detail


def test_notes_check_ollama_down_is_optional_not_ok(monkeypatch, tmp_path):
    monkeypatch.setenv("STENOGRAF_NOTES_BACKEND", "ollama")  # force the branch under test
    from stenograf.notes.ollama import OllamaBackend

    monkeypatch.setattr(OllamaBackend, "is_available", lambda self: False)
    check = doctor._notes_check()
    assert not check.ok
    assert check.optional  # an absent backend must not fail the doctor gate
    assert "ollama serve" in check.detail


def test_notes_check_ollama_up_but_model_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("STENOGRAF_NOTES_BACKEND", "ollama")  # force the branch under test
    from stenograf.notes.ollama import OllamaBackend

    monkeypatch.setattr(OllamaBackend, "is_available", lambda self: True)
    monkeypatch.setattr(OllamaBackend, "installed_models", lambda self: ["llama3:8b"])
    check = doctor._notes_check()
    assert not check.ok
    assert check.optional
    assert "ollama pull" in check.detail


def test_notes_check_mlx_reports_cache_state(monkeypatch, tmp_path):
    monkeypatch.setenv("STENOGRAF_NOTES_BACKEND", "mlx")
    from stenograf.notes.mlx import MlxBackend

    monkeypatch.setattr(MlxBackend, "is_available", lambda self: True)
    monkeypatch.setattr(MlxBackend, "weights_cached", lambda self: False)
    check = doctor._notes_check()
    assert check.ok
    assert "downloads on first" in check.detail

    monkeypatch.setattr(MlxBackend, "is_available", lambda self: False)
    check = doctor._notes_check()
    assert not check.ok
    assert check.optional
    assert "mlx-lm" in check.detail


def test_notes_check_command_backend_reports_path_presence(monkeypatch, tmp_path):
    # Forward slashes: a raw Windows path in a TOML basic string is invalid (\U…).
    executable = sys.executable.replace("\\", "/")
    write_settings(f'[notes]\nbackend = "command"\ncommand = ["{executable}", "-c", "pass"]\n')
    check = doctor._notes_check()
    assert check.ok, check.detail

    write_settings('[notes]\nbackend = "command"\ncommand = ["no-such-notes-binary"]\n')
    check = doctor._notes_check()
    assert not check.ok
    assert check.optional
    assert "PATH" in check.detail


def test_notes_check_unconfigured_command_backend_is_optional(monkeypatch, tmp_path):
    write_settings('[notes]\nbackend = "command"\n')
    check = doctor._notes_check()
    assert not check.ok
    assert check.optional
    assert "settings.toml" in check.detail


def test_doctor_exit_gate_ignores_optional_failures(monkeypatch):
    from click.testing import CliRunner

    from stenograf import cli

    ok = doctor.Check(name="A", ok=True, detail="fine")
    opt = doctor.Check(name="B", ok=False, detail="absent", optional=True)
    hard = doctor.Check(name="C", ok=False, detail="broken")

    monkeypatch.setattr(cli.doctor, "run_checks", lambda: [ok, opt])
    assert CliRunner().invoke(cli.main, ["doctor"]).exit_code == 0

    monkeypatch.setattr(cli.doctor, "run_checks", lambda: [ok, opt, hard])
    assert CliRunner().invoke(cli.main, ["doctor"]).exit_code == 1


def test_preset_notes_checks_cover_what_the_standing_check_misses(tmp_path, monkeypatch):
    # doctor greens the standing backend; a preset selecting another one used
    # to get a green doctor and a failed notes run after a real meeting.
    from stenograf import doctor

    write_settings(
        """
[meetings.agentic.notes]
backend = "command"
command = ["definitely-not-a-real-binary-xyz"]

[meetings.plain]
title = "No notes overlay"
"""
    )

    checks = {check.name: check for check in doctor._preset_checks()}

    assert "Preset 'plain' notes (optional)" not in checks  # nothing to check
    check = checks["Preset 'agentic' notes (optional)"]
    assert not check.ok
    assert check.optional  # notes are opt-in; a broken preset must not fail doctor
    assert "definitely-not-a-real-binary-xyz" in check.detail
    assert "PATH" in check.detail


def test_doctor_command_runs_and_prints_checks():
    from click.testing import CliRunner

    from stenograf import cli

    result = CliRunner().invoke(cli.main, ["doctor"])
    # Exit code is environment-dependent (0 all-ok, 1 if e.g. models uncached);
    # what matters is it ran and printed the check table without crashing.
    assert result.exit_code in (0, 1)
    assert "Python" in result.output
    assert "ASR backend" in result.output


def test_the_capture_check_threads_the_configured_microphone(monkeypatch, tmp_path):
    """The pin must reach the preflight, or the check blesses a run that fails."""
    from stenograf.capture import helper as capture_helper
    from stenograf.capture.base import Channel

    (tmp_path / "settings.toml").write_text('[capture]\nmic_device = "usb-1"\n', encoding="utf-8")
    monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path))
    monkeypatch.setattr(capture_helper, "find_helper", lambda: Path("stenocap"))
    asked = {}

    def query(channels, mic_device=None):
        asked["mic_device"] = mic_device
        return {Channel.MIC: "Desk Mic", Channel.SYSTEM: "sink.monitor"}

    monkeypatch.setattr(capture_helper, "query_devices", query)
    monkeypatch.setattr(doctor.sys, "platform", "linux")

    check = doctor._linux_capture_check()
    assert check.ok
    assert asked["mic_device"] == "usb-1"


def test_a_configured_microphone_that_is_absent_fails_the_doctor(monkeypatch, tmp_path):
    from stenograf.capture.helper import InputDevice

    (tmp_path / "settings.toml").write_text(
        '[capture]\nmic_device = "studio-mic"\n', encoding="utf-8"
    )
    monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path))
    from stenograf.capture import helper as capture_helper

    monkeypatch.setattr(
        capture_helper,
        "list_input_devices",
        lambda: [InputDevice(id="builtin", name="Built-in", is_default=True)],
    )

    ok, detail = doctor._pinned_mic_detail("studio-mic")
    assert not ok
    assert "not connected" in detail and "steno devices" in detail


def test_the_word_default_is_not_treated_as_a_device(monkeypatch, tmp_path):
    # `mic_device = "default"` means "follow the system default"; the doctor
    # must not hunt for a device by that name and then declare it missing.
    (tmp_path / "settings.toml").write_text('[capture]\nmic_device = "default"\n', encoding="utf-8")
    monkeypatch.setenv("STENOGRAF_DATA", str(tmp_path))

    assert doctor._configured_mic_device() is None
