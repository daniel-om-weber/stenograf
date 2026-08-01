"""The desktop launchers ``steno setup`` drops, and the entries they rely on.

Three rules are worth the tests. The launchers embed the absolute
interpreter and run ``-m stenograf`` (Phase 7 Task 6) — a double-clicked
shortcut gets a login-shell PATH that may lack uv's shim directory, so ``steno``
by name is never good enough. ``Stenograf.app`` (Phase 8 step 5) is copied
byte for byte and never generated, because macOS pins the app's microphone
grant to the bundle's exact contents. And a pre-flip terminal launcher is
retired or converted in place on every setup, never left beside the app as a
second Stenograf that opens a dead TUI.
"""

from __future__ import annotations

import hashlib
import plistlib
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from stenograf import shortcut

REPO_ROOT = Path(__file__).parent.parent
WINDOWS = sys.platform == "win32"  # the real host, before any monkeypatching
MACOS = sys.platform == "darwin"

BUNDLE_FINGERPRINT = "69f0436e2e1b774402c95f2e8d808e60d8912b3442c1cb02b5eb9f78cfca8a42"
"""sha256 over every file in the committed bundle, path and bytes.

The one constant in this file that is a *decision*, not an observation. TCC
stores the app's microphone and system-audio grants against the cdhash of
``Contents/MacOS/Stenograf`` — no identifier, no anchor (PLAN.md Phase 8 step 2)
— and the Info.plist and the icon are sealed into that hash. So changing
anything in the bundle, including rebuilding the same source, makes every
machine that already granted access prompt again, with no way to migrate the
old grant. Matching cdhashes as built: arm64 ``ea4e7966…``, x86_64 ``1e0f106b…``.

Changed once, on 2026-07-25, to replace the icon before the bundle had ever
shipped in a tagged release — so the re-prompt it cost reached nobody but this
machine. That window is closed now.
"""


def _home(monkeypatch, path: Path) -> None:
    """Redirect the home dir: POSIX Path.home() reads HOME, Windows USERPROFILE."""
    monkeypatch.setenv("HOME", str(path))
    monkeypatch.setenv("USERPROFILE", str(path))


def _macos(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    _home(monkeypatch, tmp_path)


def _linux(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))


def _windows(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    # Bypass the registry lookups: on POSIX hosts winreg doesn't exist, and on
    # a real Windows host they would point at the user's actual Desktop and
    # Start menu — which no test may write into.
    monkeypatch.setattr(shortcut, "_windows_desktop", lambda: tmp_path / "Desktop")
    monkeypatch.setattr(shortcut, "_windows_programs", lambda: tmp_path / "Programs")


def _fingerprint(root: Path) -> str:
    # Sort the posix strings, never the Path objects: WindowsPath compares
    # case-insensitively, which puts _CodeSignature first there and last
    # everywhere else — the same bytes then hash differently per platform, and
    # BUNDLE_FINGERPRINT is a constant about the bundle, not about the runner.
    digest = hashlib.sha256()
    files = {p.relative_to(root).as_posix(): p for p in root.rglob("*") if p.is_file()}
    for relative in sorted(files):
        digest.update(relative.encode())
        digest.update(files[relative].read_bytes())
    return digest.hexdigest()


# -- the macOS app bundle ----------------------------------------------------


def test_the_committed_bundle_is_frozen():
    # Read BUNDLE_FINGERPRINT's docstring before touching this.
    assert _fingerprint(shortcut.BUNDLE_TEMPLATE) == BUNDLE_FINGERPRINT


def test_the_bundle_has_the_shape_macos_requires():
    contents = shortcut.BUNDLE_TEMPLATE / "Contents"
    info = plistlib.loads((contents / "Info.plist").read_bytes())
    assert info["CFBundleIdentifier"] == "dev.stenograf.app"  # the TCC client name
    assert info["CFBundleExecutable"] == "Stenograf"
    # Both usage strings: the capture helper's prompt is attributed to this app.
    assert "microphone" in info["NSMicrophoneUsageDescription"]
    assert info["NSAudioCaptureUsageDescription"]
    # Menu-bar mode (step 6) is a runtime activation-policy call in the spawned
    # child, measured 2026-07-25 — an LSUIElement here would only stop `open`
    # from bringing the window forward, and could never be added later anyway.
    assert "LSUIElement" not in info

    executable = contents / "MacOS" / "Stenograf"
    # A Mach-O, never a script: an interpreted main executable makes the
    # interpreter — which lives outside the bundle — the process TCC sees.
    assert executable.read_bytes()[:4] == b"\xca\xfe\xba\xbe"  # universal binary
    assert (contents / "Resources" / "Stenograf.icns").is_file()
    assert (contents / "_CodeSignature" / "CodeResources").is_file()


def test_macos_installs_the_app_bundle(tmp_path, monkeypatch):
    _macos(monkeypatch, tmp_path)

    target = shortcut.install_shortcut()

    assert target == tmp_path / "Applications" / "Stenograf.app"
    # Byte for byte, including the signature: the copy is the same app to TCC.
    assert _fingerprint(target) == BUNDLE_FINGERPRINT
    if not WINDOWS:  # Windows stat reports no exec bits
        assert (target / "Contents" / "MacOS" / "Stenograf").stat().st_mode & 0o111


def test_the_app_is_pointed_at_this_installation_from_outside_the_bundle(tmp_path, monkeypatch):
    _macos(monkeypatch, tmp_path)

    target = shortcut.install_shortcut()

    lines = shortcut.launch_target_path().read_text(encoding="utf-8").splitlines()
    command = [line for line in lines if line and not line.startswith("#")]
    assert command[0].endswith(("steno", "python", "python3", "python.exe", "python3.exe"))
    assert command[-1] == "--gui"  # the app has no terminal to host the TUI in
    # Nothing machine-specific may be written *into* the bundle: that is what
    # would move the cdhash on every setup.
    assert str(tmp_path) not in (target / "Contents" / "Info.plist").read_text(encoding="utf-8")


def test_reinstall_refreshes_the_target_but_never_the_bundle(tmp_path, monkeypatch):
    _macos(monkeypatch, tmp_path)
    target = shortcut.install_shortcut()
    executable = target / "Contents" / "MacOS" / "Stenograf"
    before = executable.stat().st_ino
    shortcut.launch_target_path().write_text("/stale/steno\n", encoding="utf-8")

    again = shortcut.install_shortcut()

    assert again == target
    assert executable.stat().st_ino == before  # an up-to-date bundle is left alone
    assert "/stale/steno" not in shortcut.launch_target_path().read_text(encoding="utf-8")


def test_a_damaged_bundle_is_replaced(tmp_path, monkeypatch):
    _macos(monkeypatch, tmp_path)
    target = shortcut.install_shortcut()
    executable = target / "Contents" / "MacOS" / "Stenograf"
    executable.write_bytes(b"truncated")

    shortcut.install_shortcut()

    assert _fingerprint(target) == BUNDLE_FINGERPRINT


def test_the_app_retires_the_pre_flip_command_file(tmp_path, monkeypatch):
    # A pre-flip install wrote ~/Desktop/Stenograf.command, which opened the
    # retired TUI; leaving it beside the app is a launcher that opens nothing.
    _macos(monkeypatch, tmp_path)
    legacy = tmp_path / "Desktop" / "Stenograf.command"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        "#!/bin/sh\n# Stenograf launcher — regenerated by `steno setup`.\n"
        'exec "/old/python" -m stenograf\n',
        encoding="utf-8",
    )

    shortcut.install_shortcut()

    # Two launchers with different behaviour is the confusing outcome.
    assert not legacy.exists()


def test_a_foreign_command_file_is_left_alone(tmp_path, monkeypatch):
    _macos(monkeypatch, tmp_path)
    desktop = tmp_path / "Desktop"
    desktop.mkdir()
    (desktop / "Stenograf.command").write_text("#!/bin/sh\n# mine, not yours\n", encoding="utf-8")

    shortcut.install_shortcut()

    assert (desktop / "Stenograf.command").exists()  # it is the user's Desktop


def test_a_non_utf8_command_file_does_not_crash_setup(tmp_path, monkeypatch):
    # read_text raises UnicodeDecodeError (a ValueError, not an OSError) on
    # arbitrary bytes; a foreign binary-ish file must be skipped, not fatal.
    _macos(monkeypatch, tmp_path)
    desktop = tmp_path / "Desktop"
    desktop.mkdir()
    (desktop / "Stenograf.command").write_bytes(b"#!/bin/sh\n\xff\xfe not text\n")

    shortcut.install_shortcut()

    assert (desktop / "Stenograf.command").exists()  # left alone, like any foreign file


@pytest.mark.skipif(not MACOS, reason="codesign only exists on macOS")
def test_the_installed_copy_still_satisfies_its_signature(tmp_path, monkeypatch):
    # The whole distribution model: the seal travels with the files, so a bundle
    # written by shutil is the same code object as the one that was signed here.
    _macos(monkeypatch, tmp_path)

    target = shortcut.install_shortcut()

    subprocess.run(["codesign", "--verify", "--strict", str(target)], check=True, timeout=60)


# -- the app launchers off macOS ---------------------------------------------


def test_linux_entry_opens_the_app(tmp_path, monkeypatch):
    _linux(monkeypatch, tmp_path)

    target = shortcut.install_shortcut()

    assert target == tmp_path / "xdg" / "applications" / f"{shortcut.DESKTOP_FILE_NAME}.desktop"
    content = target.read_text()
    assert f'Exec="{sys.executable}" -m stenograf --gui' in content
    assert "Terminal=false" in content  # a window must not get a terminal behind it
    # The X11 half of the handshake, and the one that is easy to get wrong: Qt
    # builds WM_CLASS from the *application name*, not from the app_id (measured
    # under XWayland 2026-07-25: "steno", "Stenograf"), so the entry has to
    # declare the capitalized class rather than lean on a shell lowercasing it.
    assert f"StartupWMClass={shortcut.APPLICATION_NAME}" in content
    assert "StartupWMClass=stenograf" not in content
    # KDE's half of the single-instance rule (gui.app.claim_single_instance is
    # the portable half); ignored elsewhere.
    assert "SingleMainWindow=true" in content
    assert "StartupNotify=true" in content


def test_the_linux_entry_names_an_installed_icon(tmp_path, monkeypatch):
    # An absolute Icon= into site-packages works until a reinstall moves the
    # venv, and gives a notification daemon a file where a name would let it
    # pick a size. hicolor is the theme every other theme falls back to.
    _linux(monkeypatch, tmp_path)

    target = shortcut.install_shortcut()

    icon = tmp_path / "xdg" / "icons" / "hicolor" / "512x512" / "apps" / "stenograf.png"
    assert icon.read_bytes() == shortcut.ICON.read_bytes()
    assert f"Icon={shortcut.DESKTOP_FILE_NAME}\n" in target.read_text()


def test_the_linux_entry_lists_the_app_once(tmp_path, monkeypatch):
    # Two main categories make the app appear twice in the menu, which is what
    # the old AudioVideo;Audio;Utility; did.
    _linux(monkeypatch, tmp_path)

    target = shortcut.install_shortcut()

    assert "Categories=AudioVideo;Audio;Recorder;" in target.read_text()


@pytest.mark.skipif(
    shutil.which("desktop-file-validate") is None,
    reason="desktop-file-utils is not installed here",
)
def test_the_linux_entry_validates(tmp_path, monkeypatch):
    # The whole file against the spec's own validator — the only check that
    # catches a key we invented or a category that does not exist.
    _linux(monkeypatch, tmp_path)

    target = shortcut.install_shortcut()
    assert target is not None

    result = subprocess.run(
        ["desktop-file-validate", str(target)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0 and not result.stdout, result.stdout + result.stderr


def test_a_pre_flip_linux_entry_converts_in_place(tmp_path, monkeypatch):
    # The pre-flip entry opened the retired TUI in a terminal; setup rewrites
    # the same file rather than leaving two menu entries named Stenograf.
    _linux(monkeypatch, tmp_path)
    apps = tmp_path / "xdg" / "applications"
    apps.mkdir(parents=True)
    legacy = apps / f"{shortcut.DESKTOP_FILE_NAME}.desktop"
    legacy.write_text(
        "[Desktop Entry]\nType=Application\nName=Stenograf\n"
        'Exec="/old/python" -m stenograf\nTerminal=true\n',
        encoding="utf-8",
    )

    target = shortcut.install_shortcut()

    assert target == legacy  # one menu entry named Stenograf, never two
    content = target.read_text()
    assert "Terminal=false" in content and "--gui" in content


def test_the_windows_icon_ships_every_size_the_shell_asks_for():
    # An .ico is a *directory* of independent images and Windows picks the entry
    # nearest the size it wants, so one 256 px image would be downsampled into
    # mush at 16 px — which is the size the taskbar and the title bar use.
    # Parsed rather than trusted: the file is committed, and nothing else reads
    # it until a user is looking at their Start menu.
    header = shortcut.WINDOWS_ICON.read_bytes()
    reserved, kind, count = struct.unpack("<HHH", header[:6])
    assert (reserved, kind) == (0, 1)  # an icon directory, not a cursor
    sizes = set()
    for index in range(count):
        width, height = header[6 + 16 * index], header[7 + 16 * index]
        assert width == height  # square, or the shell letterboxes it
        sizes.add(width or 256)  # one byte per dimension, so the format spells 256 as 0
    assert sizes == {16, 24, 32, 48, 64, 128, 256}


@pytest.mark.skipif(not WINDOWS, reason="only the Windows shell can write a shell link")
def test_windows_installs_shell_links_that_carry_the_app_identity(tmp_path, monkeypatch):
    from stenograf import winlink

    _windows(monkeypatch, tmp_path)

    target = shortcut.install_shortcut()

    # The Start-menu entry is what setup names: it is the pinnable one, and the
    # one Windows wants before it will put our name on a toast.
    assert target == tmp_path / "Programs" / "Stenograf.lnk"
    desktop = tmp_path / "Desktop" / "Stenograf.lnk"
    assert desktop.is_file()
    for path in (target, desktop):
        link = winlink.read_shortcut(path)
        # pythonw.exe, so there is no console window behind the app at all —
        # the flash the `start` wrapper could only shorten.
        assert link.target == shortcut._windowed_python()
        assert link.arguments == "-m stenograf --gui"
        assert link.icon == str(shortcut.WINDOWS_ICON)
        # The whole reason this is COM and not a `WScript.Shell` one-liner: the
        # taskbar matches a window to its launcher by this string and nothing
        # else, and the detour cannot reach the property store that holds it.
        assert link.app_id == shortcut.APP_USER_MODEL_ID


@pytest.mark.skipif(not WINDOWS, reason="only the Windows shell can write a shell link")
def test_the_links_retire_the_pre_flip_batch_file(tmp_path, monkeypatch):
    # A pre-flip install without the extra wrote a TUI batch launcher; the
    # shell links replace it or two Stenografs sit on the Desktop.
    _windows(monkeypatch, tmp_path)
    legacy = tmp_path / "Desktop" / "Stenograf.cmd"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        "@echo off\nrem Stenograf launcher - regenerated by `steno setup`.\n"
        '"C:\\old\\python.exe" -m stenograf\nif errorlevel 1 pause\n',
        encoding="ascii",
    )

    link = shortcut.install_shortcut()

    assert link is not None and link.is_file()
    assert not legacy.exists()  # one launcher named Stenograf, never two


@pytest.mark.skipif(not WINDOWS, reason="only the Windows shell can write a shell link")
def test_retiring_links_leaves_a_foreign_shell_link_alone(tmp_path, monkeypatch):
    # Same rule as the macOS command file: a shortcut we did not write is the
    # user's. Ours are identified by the app id, which nothing else declares.
    # The retire runs on the COM-refusal fallback path, so it is unit-tested.
    from stenograf import winlink

    _windows(monkeypatch, tmp_path)
    theirs = tmp_path / "Desktop" / "Stenograf.lnk"
    winlink.write_shortcut(theirs, target=sys.executable, description="mine, not yours")
    ours = tmp_path / "Programs" / "Stenograf.lnk"
    ours.parent.mkdir(parents=True)
    winlink.write_shortcut(
        ours, target=sys.executable, description="ours", app_id=shortcut.APP_USER_MODEL_ID
    )

    shortcut._retire_windows_links()

    assert theirs.is_file()  # no app id: the user's, left alone
    assert not ours.exists()  # our identity: retired with the links


def test_the_windows_app_launcher_prefers_the_console_less_interpreter(tmp_path, monkeypatch):
    # A plain python.exe would put a console window behind the app for the
    # whole meeting; pythonw.exe is the same interpreter without one.
    (tmp_path / "python.exe").touch()
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python.exe"))
    assert shortcut._windowed_python() == str(tmp_path / "python.exe")  # none beside it yet

    (tmp_path / "pythonw.exe").touch()

    assert shortcut._windowed_python() == str(tmp_path / "pythonw.exe")


# -- fallbacks and self-healing ----------------------------------------------


def test_linux_shortcut_defaults_to_local_share(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    _home(monkeypatch, tmp_path)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)

    target = shortcut.install_shortcut()

    assert target == tmp_path / ".local" / "share" / "applications" / "stenograf.desktop"


def test_reinstall_overwrites_and_self_heals(tmp_path, monkeypatch):
    # A reinstall can move the venv; re-running setup must embed the new
    # interpreter rather than leave the entry pointing into a deleted one.
    _linux(monkeypatch, tmp_path)

    first = shortcut.install_shortcut()
    assert first is not None
    first.write_text('[Desktop Entry]\nExec="/stale/interpreter" -m stenograf --gui\n')
    second = shortcut.install_shortcut()

    assert second == first
    assert sys.executable in second.read_text()


def test_the_windows_batch_fallback_starts_the_app(tmp_path, monkeypatch):
    # The COM-refusal fallback: a Desktop .cmd that `start`s the app and
    # exits, so no console window lives behind the meeting.
    monkeypatch.setattr(shortcut, "_windows_desktop", lambda: tmp_path / "Desktop")

    target = shortcut._install_cmd_file()

    assert target == tmp_path / "Desktop" / "Stenograf.cmd"
    content = target.read_text()
    assert content.startswith("@echo off")
    assert "-m stenograf --gui" in content
    assert content.count("start ") == 1  # hand over and exit, no lingering console


@pytest.mark.skipif(not WINDOWS, reason="reads the real User Shell Folders registry key")
@pytest.mark.parametrize("folder", ["_windows_desktop", "_windows_programs"])
def test_windows_shell_folders_resolve(folder):
    # Redirected (OneDrive backs up the Desktop on a large share of Windows 11
    # machines) or not, the shell key names an absolute, existing dir with no
    # unexpanded %VARS% left in it.
    resolved = getattr(shortcut, folder)()
    assert resolved.is_absolute()
    assert "%" not in str(resolved)
    assert resolved.is_dir()


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
