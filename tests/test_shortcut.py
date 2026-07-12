"""The desktop launcher ``steno setup`` drops, and the entries it relies on.

Phase 7 Task 6: the launcher embeds the absolute interpreter and runs
``-m stenograf`` — a double-clicked shortcut gets a login-shell PATH that may
lack uv's shim directory, so ``steno`` by name is never good enough.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from stenograf import shortcut

REPO_ROOT = Path(__file__).parent.parent


def test_macos_shortcut_is_an_executable_command_file(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("HOME", str(tmp_path))

    target = shortcut.install_shortcut()

    assert target == tmp_path / "Desktop" / "Stenograf.command"
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
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)

    target = shortcut.install_shortcut()

    assert target == tmp_path / ".local" / "share" / "applications" / "stenograf.desktop"


def test_reinstall_overwrites_and_self_heals(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("HOME", str(tmp_path))

    first = shortcut.install_shortcut()
    first.write_text("#!/bin/sh\nexec /stale/interpreter -m stenograf\n")
    second = shortcut.install_shortcut()

    assert second == first
    assert sys.executable in second.read_text()


def test_unsupported_platform_installs_nothing(tmp_path, monkeypatch):
    # Windows gets its .lnk with Phase 6; until then setup must not fail there.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("HOME", str(tmp_path))

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


def test_install_script_parses_and_is_executable():
    script = REPO_ROOT / "install.sh"
    assert script.stat().st_mode & 0o111
    subprocess.run(["sh", "-n", str(script)], check=True)
    content = script.read_text()
    assert "tool install --upgrade stenograf" in content
    assert "setup" in content  # the script must end in `steno setup`
