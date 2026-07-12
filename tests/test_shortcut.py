"""The desktop launcher ``steno setup`` drops, and the entries it relies on.

Phase 7 Task 6: the launcher embeds the absolute interpreter and runs
``-m stenograf`` — a double-clicked shortcut gets a login-shell PATH that may
lack uv's shim directory, so ``steno`` by name is never good enough.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from stenograf import shortcut

REPO_ROOT = Path(__file__).parent.parent
WINDOWS = sys.platform == "win32"  # the real host, before any monkeypatching


def _home(monkeypatch, path: Path) -> None:
    """Redirect the home dir: POSIX Path.home() reads HOME, Windows USERPROFILE."""
    monkeypatch.setenv("HOME", str(path))
    monkeypatch.setenv("USERPROFILE", str(path))


def test_macos_shortcut_is_an_executable_command_file(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    _home(monkeypatch, tmp_path)

    target = shortcut.install_shortcut()

    assert target == tmp_path / "Desktop" / "Stenograf.command"
    if not WINDOWS:  # Windows stat reports no exec bits
        assert target.stat().st_mode & 0o111  # double-click needs the exec bit
    content = target.read_text()
    assert content.startswith("#!/bin/sh")
    assert f'exec "{sys.executable}" -m stenograf' in content


def test_linux_shortcut_is_a_terminal_desktop_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))

    target = shortcut.install_shortcut()

    assert target == tmp_path / "xdg" / "applications" / "stenograf.desktop"
    content = target.read_text()
    assert "Terminal=true" in content  # the TUI needs a real terminal
    assert f'Exec="{sys.executable}" -m stenograf' in content


def test_linux_shortcut_defaults_to_local_share(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    _home(monkeypatch, tmp_path)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)

    target = shortcut.install_shortcut()

    assert target == tmp_path / ".local" / "share" / "applications" / "stenograf.desktop"


def test_reinstall_overwrites_and_self_heals(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    _home(monkeypatch, tmp_path)

    first = shortcut.install_shortcut()
    first.write_text("#!/bin/sh\nexec /stale/interpreter -m stenograf\n")
    second = shortcut.install_shortcut()

    assert second == first
    assert sys.executable in second.read_text()


def test_windows_shortcut_is_a_cmd_wrapper(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    # Bypass the registry lookup: on POSIX hosts winreg doesn't exist, and on
    # a real Windows host it would point at the user's actual Desktop.
    monkeypatch.setattr(shortcut, "_windows_desktop", lambda: tmp_path / "Desktop")

    target = shortcut.install_shortcut()

    assert target == tmp_path / "Desktop" / "Stenograf.cmd"
    content = target.read_text()
    assert content.startswith("@echo off")
    assert f'"{sys.executable}" -m stenograf' in content
    assert "pause" in content  # a crash must not vanish with the console window


@pytest.mark.skipif(not WINDOWS, reason="reads the real User Shell Folders registry key")
def test_windows_desktop_is_the_shell_folder():
    # Redirected (OneDrive) or not, the shell key names an absolute, existing
    # dir with no unexpanded %VARS% left in it.
    desktop = shortcut._windows_desktop()
    assert desktop.is_absolute()
    assert "%" not in str(desktop)
    assert desktop.is_dir()


def test_unsupported_platform_installs_nothing(tmp_path, monkeypatch):
    # `steno transcribe` works anywhere Python does; setup must not fail there.
    monkeypatch.setattr(sys, "platform", "freebsd14")
    _home(monkeypatch, tmp_path)

    assert shortcut.install_shortcut() is None
    assert not list(tmp_path.rglob("*"))


def test_python_m_stenograf_is_a_working_entry():
    # The shortcut's actual invocation — keep `python -m stenograf` alive.
    result = subprocess.run(
        [sys.executable, "-m", "stenograf", "--version"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "stenograf" in result.stdout


@pytest.mark.skipif(WINDOWS, reason="POSIX installer: exec bits and sh don't exist here")
def test_install_script_parses_and_is_executable():
    script = REPO_ROOT / "install.sh"
    assert script.stat().st_mode & 0o111
    subprocess.run(["sh", "-n", str(script)], check=True)
    content = script.read_text()
    assert "tool install --upgrade stenograf" in content
    assert "setup" in content  # the script must end in `steno setup`


@pytest.mark.skipif(not WINDOWS, reason="Windows installer: PowerShell parsing is win32-only")
def test_install_ps1_parses():
    script = REPO_ROOT / "install.ps1"
    # Parse-only: [ScriptBlock]::Create raises on a syntax error, runs nothing.
    check = f"[ScriptBlock]::Create((Get-Content -Raw '{script}')) | Out-Null"
    subprocess.run(["powershell", "-NoProfile", "-Command", check], check=True)
    content = script.read_text()
    assert "tool install --upgrade stenograf" in content
    assert "setup" in content  # the script must end in `steno setup`
