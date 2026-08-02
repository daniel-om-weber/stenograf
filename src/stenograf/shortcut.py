"""The double-clickable launcher ``steno setup`` drops on each platform.

After the one-time setup nobody has to type a command again. Every launcher
opens the desktop app — Qt has been a base dependency since the GUI became
the default (Phase 8 step 7); the pre-flip terminal launchers survive only as
retirement targets (:func:`_retire_command_file`, :func:`_retire_cmd_file`),
so an upgrade never leaves a second "Stenograf" behind that opens a dead TUI:

- **macOS**: ``~/Applications/Stenograf.app`` (Phase 8 step 5) — a real app
  with an icon, copied verbatim from the frozen bundle in ``assets/``. See
  below; it is the only launcher with rules attached.
- **Linux**: an application-menu entry (a file *on* the GNOME desktop would
  need a manual "allow launching" step, the menu entry doesn't) running
  ``--gui``, plus the app icon copied into the user's icon theme, so the
  entry names an icon instead of pointing into site-packages
  (:func:`_install_icon`).
- **Windows**: two ``Stenograf.lnk`` shell links, in the Start Menu and on
  the Desktop, written through COM (:mod:`stenograf.winlink`) so both can
  carry the app's icon *and* its :data:`APP_USER_MODEL_ID` — the string the
  taskbar matches a window against. They target ``pythonw.exe`` directly, so
  there is no console to flash. A COM refusal is a clear setup error, not a
  degraded launcher: a link without the id would bind the user's taskbar
  choices to the interpreter path instead of the app.

The launchers embed the absolute interpreter and run ``-m stenograf``: a
double-clicked launcher gets a login-shell PATH that may lack uv's shim
directory. Re-running ``steno setup`` regenerates them, self-healing the
embedded path after a reinstall; the Linux entry and the Windows launchers
are rewritten in place, never duplicated.

**The app bundle is the exception to that self-healing, and it matters.** macOS
stores the microphone and system-audio grants against the cdhash of
``Contents/MacOS/Stenograf`` — no identifier, no anchor (measured, PLAN.md Phase
8 step 2) — so a bundle rewritten with a machine-specific path inside it would
be a different app to TCC on every install, and every change to it silently
revokes the grant of everyone who already answered the prompt. The bundle is
therefore copied byte-for-byte and never generated: the part that varies, the
command to launch, goes into :func:`launch_target_path` *outside* the bundle,
which the stub reads at every launch.

A locally *written* file carries no quarantine flag, so all of this opens
without Gatekeeper friction — unlike anything downloaded from a browser, which
macOS now hard-blocks unless it is notarized (tested on Tahoe; that is why
there is no downloadable double-click artifact yet).
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from stenograf import ASSETS

BUNDLE_TEMPLATE = ASSETS / "Stenograf.app"
ICON = ASSETS / "icon.png"

WINDOWS_ICON = ASSETS / "icon.ico"
"""The multi-size icon the Windows shell links point at (16…256 px).

Named as a path into the installed package rather than copied out to a stable
location, which is the opposite of what :func:`_install_icon` does on Linux —
and for a reason that only holds here. Linux copies because a ``.desktop``
entry's ``Icon=`` is read forever by whatever draws it; a ``.lnk``'s icon path
is read by ``steno setup``'s own successor, since every upgrade path
(``install.ps1``, the manual ``uv tool install --upgrade``) ends in ``steno
setup`` and rewrites both links. Rendered by ``native/appbundle/render_ico.py``
from the same ``icon.svg`` as everything else.
"""

DESKTOP_FILE_NAME = "stenograf"
"""Basename of the Linux desktop entry, the app_id the window reports, and the
name its icon is installed under.

A desktop shell matches a running window back to the entry that launched it by
app_id (Wayland) or ``WM_CLASS`` (X11); without a match the window loses the
entry's icon and appears as a second, nameless "Stenograf" in the taskbar. Qt
takes the Wayland app_id from ``QGuiApplication.setDesktopFileName``, which
``gui/app.py`` calls with *this* constant — shared rather than spelled twice,
since two strings that must be equal will otherwise drift. Measured on KDE
Plasma 6.7.3 (Wayland) on 2026-07-25: KWin reports ``resourceClass=stenograf``
and ``desktopFileName=stenograf`` for the running window, and the task manager
shows one entry carrying the app's icon.

Lowercase, like every app_id and every icon name. **X11 does not use this
constant at all**: Qt builds ``WM_CLASS`` from ``argv[0]``'s basename and
:data:`APPLICATION_NAME`, i.e. ``"steno", "Stenograf"`` — or
``"__main__.py", "Stenograf"`` when started as ``python -m stenograf``, which is
what the entry does. Only the *class* half is stable, so ``StartupWMClass``
below is written with that (measured under XWayland 2026-07-25, where KWin
reports the window as ``resourceClass=Stenograf``) rather than with the app_id,
which used to match only through the shells' case-insensitive fallbacks.
"""

APPLICATION_NAME = "Stenograf"
"""``QGuiApplication.applicationName``, set from here by ``gui/app.py``.

It is the X11 ``WM_CLASS`` *class* (hence ``StartupWMClass`` below), the tray
item's title, and the name notifications are attributed to. Shared for the same
reason as :data:`DESKTOP_FILE_NAME`: the entry declares a string the window has
to carry, and the two may not drift.
"""

APP_USER_MODEL_ID = "dev.stenograf.app"
"""The Windows identity, claimed by ``gui/app.py`` and written into every ``.lnk``.

Windows' spelling of what :data:`DESKTOP_FILE_NAME` does on Wayland: the shell
matches a running window back to the shortcut that launched it by
AppUserModelID and by nothing else — that match is what makes taskbar grouping,
pinning, and a toast attributed to *Stenograf* rather than to ``pythonw.exe``
work at all. Qt sets no id of its own, so the process claims one explicitly and
:func:`~stenograf.winlink.write_shortcut` stamps the same string into the
shortcut's property store. Shared for the reason the other two constants are:
the halves must be equal or the match fails silently, which is the worst way for
it to fail.

``dev.stenograf.app`` is also the macOS bundle identifier — one identity per
app, spelled the same wherever a platform asks for one.
"""

_LAUNCH_TARGET = """\
# What Stenograf.app runs, one argument per line (blank lines and #-comments
# are ignored). Rewritten by `steno setup`.
#
# It lives out here because the app bundle itself must never change: macOS pins
# the app's microphone grant to the bundle's exact contents, so anything
# machine-specific — this path — has to stay outside it.
{command}
"""

# `Icon=` is a themed *name*, not a path: _install_icon puts the PNG where every
# desktop and every notification daemon can find it at whatever size it wants.
# Categories: the old `AudioVideo;Audio;Utility;` drew "contains more than one
# main category; application might appear more than once in the application
# menu" from `desktop-file-validate`, because Utility is a second, unrelated main
# category. `Recorder` says the same thing without being one — it is an
# additional category, valid only alongside AudioVideo, and Audio is AudioVideo's
# own subtype rather than a rival. The set below validates clean.
# StartupWMClass/StartupNotify/SingleMainWindow are window matching, so they
# belong to an app entry (see DESKTOP_FILE_NAME). SingleMainWindow is KDE's
# half of the single-instance rule that `gui.app.claim_single_instance`
# enforces for everyone — it makes a second click on the launcher activate the
# running window instead of asking us to start.
_LINUX_DESKTOP = """\
[Desktop Entry]
# Stenograf launcher — regenerated by `steno setup`.
Type=Application
Name=Stenograf
Comment=Meeting transcription
Exec="{python}" -m stenograf --gui
Icon={icon}
Terminal=false
Categories=AudioVideo;Audio;Recorder;
StartupWMClass={wm_class}
StartupNotify=true
SingleMainWindow=true
"""

def install_shortcut() -> Path | None:
    """Write this platform's app launcher and return its path (``None`` if headless).

    On macOS this also writes :func:`launch_target_path`, which is what points
    the app at *this* installation of ``steno``.
    """
    if sys.platform == "darwin":
        return _install_app()
    if sys.platform.startswith("linux"):
        return _install_desktop_entry()
    if sys.platform == "win32":
        return _install_windows_links()
    return None


def launch_target_path() -> Path:
    """The file naming the command ``Stenograf.app`` launches.

    Deliberately not under :func:`~stenograf.paths.data_dir`, whose location
    ``$STENOGRAF_DATA`` can move: the app's launcher stub is started by launchd
    with no user environment, reads one hard-coded path and has no way to learn
    about an override. The two must agree, so both are fixed.
    """
    return Path.home() / "Library" / "Application Support" / "stenograf" / "launch-target"


def _install_desktop_entry() -> Path:
    """The Linux application-menu entry for the app.

    Rewritten in place on every setup, so a pre-flip entry that pointed at the
    retired TUI (``Terminal=true``, no ``--gui``) converts into the app entry
    instead of leaving a second Stenograf in the menu.
    """
    data_home = Path(os.environ.get("XDG_DATA_HOME", "~/.local/share")).expanduser()
    apps = data_home / "applications"
    apps.mkdir(parents=True, exist_ok=True)
    target = apps / f"{DESKTOP_FILE_NAME}.desktop"
    _install_icon(data_home)
    content = _LINUX_DESKTOP.format(
        python=sys.executable,
        icon=DESKTOP_FILE_NAME,
        wm_class=APPLICATION_NAME,
    )
    target.write_text(content, encoding="utf-8")
    return target


def _install_icon(data_home: Path) -> Path:
    """Put the app icon into the user's icon theme, under :data:`DESKTOP_FILE_NAME`.

    So the entry can name its icon instead of pointing at one. The old absolute
    path into site-packages worked, but it aimed inside a venv that a reinstall
    can move, and it handed the notification daemon a single file where a name
    lets every consumer pick the size it wants. hicolor is the fallback theme
    every icon theme inherits from, so one 512×512 PNG under it is found by all
    of them — and by the ``.desktop`` file, whatever theme the user runs.
    """
    target = data_home / "icons" / "hicolor" / "512x512" / "apps" / f"{DESKTOP_FILE_NAME}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ICON, target)
    return target


def _windows_link_paths() -> list[Path]:
    """Where the app's shell links go — the Start Menu first, then the Desktop.

    The Start Menu entry is the one that carries weight: it is what the search
    box finds, what can be pinned to Start or the taskbar, and what Windows
    insists a desktop app have before it will attribute a toast to that app
    rather than to the interpreter. The Desktop copy is there because that is
    where this project's launcher has always been, and where `steno setup` has
    been telling people to look.
    """
    return [_windows_programs() / "Stenograf.lnk", _windows_desktop() / "Stenograf.lnk"]


def _install_windows_links() -> Path:
    """Write both shell links, retire the batch file, and return the Start-menu one.

    Raises ``OSError`` when COM refuses (policy, a locked-down profile); the
    caller reports it. The links carry two things a batch file could not — an
    icon, and the :data:`APP_USER_MODEL_ID` that lets the taskbar recognize
    its own window — so there is no degraded fallback to fall to.
    """
    # The ignore is the type checker's platform, not a real doubt: `winlink`
    # guards its whole body on `sys.platform == "win32"`, and pyright runs on
    # macOS (the only place the mlx deps resolve), where that makes the module's
    # contents unreachable and therefore invisible. The import stays inside the
    # win32-only call path at runtime.
    from stenograf.winlink import write_shortcut  # pyright: ignore[reportAttributeAccessIssue]

    paths = _windows_link_paths()
    for path in paths:
        write_shortcut(
            path,
            target=_windowed_python(),
            arguments="-m stenograf --gui",
            # Home, so anything the app ever resolves relatively lands somewhere
            # the user owns rather than inside the venv Explorer would default to.
            working_directory=Path.home(),
            description="Meeting transcription",
            icon=WINDOWS_ICON,
            app_id=APP_USER_MODEL_ID,
        )
    _retire_cmd_file()
    return paths[0]


def _retire_cmd_file() -> None:
    """Remove the batch launcher the shell links replace — but only if we wrote it.

    Same rule as :func:`_retire_command_file` on macOS, for the same reason: two
    launchers named Stenograf that behave differently is the confusing outcome,
    and anything that does not look like our own generated file is the user's.
    """
    legacy = _windows_desktop() / "Stenograf.cmd"
    try:
        content = legacy.read_text(encoding="ascii")
    except (OSError, ValueError):  # missing, unreadable, or not the ASCII we write
        return
    if content.startswith("@echo off") and "-m stenograf" in content:
        legacy.unlink(missing_ok=True)


def _windowed_python() -> str:
    """``pythonw.exe`` beside this interpreter — the console-less twin.

    Every CPython install and every venv ships both, so the fallback only covers
    a stripped or embedded interpreter, where a console window that stays open
    beats a launcher that starts nothing.
    """
    windowed = Path(sys.executable).with_name("pythonw.exe")
    return str(windowed) if windowed.is_file() else sys.executable


def _install_app() -> Path:
    """Put the frozen bundle in ``~/Applications`` and point it at this install.

    ``~/Applications`` rather than the Desktop: it is where a user-owned app
    belongs, and Spotlight and Launchpad index it. The grant does not care —
    TCC keys on the bundle's contents, not its path — so the app can be moved
    or dragged to the Dock afterwards.
    """
    target = Path.home() / "Applications" / "Stenograf.app"
    target.parent.mkdir(parents=True, exist_ok=True)

    path = launch_target_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_LAUNCH_TARGET.format(command="\n".join(_launch_command())), encoding="utf-8")

    if not _bundle_is_current(target):
        _replace_bundle(target)
    _retire_command_file()
    return target


def _launch_command() -> list[str]:
    """The argv the app should run: this installation's ``steno``, with ``--gui``.

    ``sys.executable``'s own directory first — a machine can hold several
    installs (a uv tool, a dev checkout), and setup should point the app at the
    one it is being run from rather than at whatever PATH resolves to.

    ``--gui`` is not optional here: bare ``steno`` gates on an interactive
    terminal, which a launchd-started stub never has. The flag is also baked
    into the frozen stub as its fallback, so the CLI has to keep accepting it
    forever, flip or no flip.
    """
    console_script = Path(sys.executable).parent / "steno"
    if console_script.is_file():
        return [str(console_script), "--gui"]
    if (found := shutil.which("steno")) is not None:
        return [found, "--gui"]
    return [sys.executable, "-m", "stenograf", "--gui"]


def _bundle_is_current(target: Path) -> bool:
    """Is the installed bundle already the one we ship, byte for byte?

    Only the two sealed files are compared — the executable carries the cdhash,
    and the Info.plist and icon are hashed into it — which is enough to tell an
    up-to-date install from a stale one without walking the tree.
    """
    for relative in ("Contents/MacOS/Stenograf", "Contents/Info.plist"):
        installed = target / relative
        if not installed.is_file():
            return False
        if installed.read_bytes() != (BUNDLE_TEMPLATE / relative).read_bytes():
            return False
    return True


def _replace_bundle(target: Path) -> None:
    """Copy the template into place, staging it first.

    Staged and renamed rather than written over: a copy interrupted halfway
    would otherwise leave an app that launches into an error dialog. The one
    thing done to the copy is a ``chmod`` — wheels do not always preserve the
    executable bit, and file modes are not part of the signature, so this
    cannot disturb the cdhash.
    """
    staging = target.with_name(target.name + ".new")
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(BUNDLE_TEMPLATE, staging)
    (staging / "Contents" / "MacOS" / "Stenograf").chmod(0o755)
    if target.exists():
        shutil.rmtree(target)
    staging.rename(target)


def _command_file_path() -> Path:
    return Path.home() / "Desktop" / "Stenograf.command"


def _retire_command_file() -> None:
    """Remove the pre-app Desktop shortcut the bundle replaces — if we wrote it.

    ``Stenograf.command`` was the pre-flip macOS launcher (a Terminal.app shell
    script that opened the retired TUI); a pre-flip install upgrading past the
    flip would otherwise keep it as a second launcher that opens a dead TUI.
    Anything that does not look like our own generated file is left alone; it
    is the user's Desktop.
    """
    legacy = _command_file_path()
    try:
        content = legacy.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return
    if content.startswith("#!/bin/sh") and "-m stenograf" in content:
        legacy.unlink(missing_ok=True)


def _windows_desktop() -> Path:
    """The user's real Desktop folder, honoring shell-folder redirection.

    OneDrive's folder backup relocates the Desktop (``~/OneDrive/Desktop``) on
    a large share of Windows 11 machines, leaving ``~/Desktop`` an invisible
    decoy — the shell's User Shell Folders key knows where it really is.
    """
    return _windows_shell_folder("Desktop", Path.home() / "Desktop")


def _windows_programs() -> Path:
    """The user's Start Menu → Programs folder, i.e. where an app is listed.

    Per-user, not the machine-wide ``%ProgramData%`` copy: everything `steno
    setup` writes belongs to the account that ran it, and writing under
    ProgramData would need elevation. Redirected far less often than the
    Desktop, but read through the same key for the same reason.
    """
    default = Path.home() / "AppData" / "Roaming" / "Microsoft" / "Windows" / "Start Menu"
    return _windows_shell_folder("Programs", default / "Programs")


def _windows_shell_folder(name: str, fallback: Path) -> Path:
    """One entry from the shell's User Shell Folders key, or ``fallback``."""
    if sys.platform != "win32":  # also lets the type checker use win32 stubs
        return fallback
    import winreg

    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders",
        ) as key:
            value, _ = winreg.QueryValueEx(key, name)
    except OSError:
        return fallback
    if not isinstance(value, str) or not value:
        return fallback
    # REG_EXPAND_SZ arrives unexpanded (typically "%USERPROFILE%\Desktop").
    return Path(os.path.expandvars(value))
