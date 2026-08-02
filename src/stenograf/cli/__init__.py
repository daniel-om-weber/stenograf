"""Command-line interface: ``stenograf`` / ``steno``.

A package of one module per command (plus ``run`` for the flag+settings
resolution the commands share and ``format`` for the human-facing rendering
helpers); this ``__init__`` only assembles the click group. Domain logic
lives in the library — each command body resolves its inputs and makes
library calls.
"""

from __future__ import annotations

import contextlib
import os
import signal
import sys

import click

from stenograf import __version__

# format and run carry no commands but are bound here so every cli submodule
# is reachable as an attribute of the package (tests patch through them).
from stenograf.cli import (  # noqa: F401
    doctor_cmd,
    format,
    notes,
    profiles,
    run,
    settings_cmd,
    start,
    transcribe,
)


@click.group(invoke_without_command=True)
@click.option(
    # This flag has to keep working forever, including after the GUI becomes
    # the default: `Stenograf.app`'s launcher stub is a frozen binary that
    # cannot be changed without revoking every user's microphone grant, and
    # `--gui` is compiled into it as the fallback argv.
    "--gui",
    is_flag=True,
    help="Open the desktop app (what bare `steno` in a terminal does anyway).",
)
@click.option(
    "--tray",
    is_flag=True,
    help="With --gui: start in the menu bar with no window (needs a system tray).",
)
@click.version_option(__version__, prog_name="stenograf")
@click.pass_context
def main(ctx: click.Context, gui: bool, tray: bool) -> None:
    """Accuracy-first local meeting transcription. Audio never touches disk.

    Run without a subcommand in a terminal to open the desktop app, or use
    the subcommands below for a line-oriented pipeline.
    """
    # Windows pipes/redirects default to the legacy code page (cp1252), and a
    # single ✓/← in our output would then crash click.echo with a
    # UnicodeEncodeError. Degrade unencodable glyphs to "?" instead; the
    # interactive console is unaffected (it is UTF-16 under the hood), as are
    # the output files (written encoding="utf-8" throughout).
    for stream in (sys.stdout, sys.stderr):
        if sys.platform == "win32" and hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    # SIGTERM must end a run the way Ctrl-C does — finalize, then leave — not
    # by killing the process around the finalize: a Linux session-manager
    # logout IS a SIGTERM, and the app-bundle stub forwards one at macOS
    # logout. Installed for every subcommand; the Qt path below replaces it
    # with its own event-loop-aware handler. Left alone when something that
    # embeds us already set a handler.
    if signal.getsignal(signal.SIGTERM) is signal.SIG_DFL:
        with contextlib.suppress(ValueError, OSError):  # not on the main thread
            signal.signal(signal.SIGTERM, _sigterm_becomes_interrupt)

    if ctx.invoked_subcommand is not None:
        if gui or tray:  # both are about the entry, not about any one command
            raise click.UsageError("--gui opens the desktop app; it takes no subcommand")
        return
    if tray and not gui:
        raise click.UsageError("--tray is a mode of the desktop app; pass --gui as well")

    # Bare `steno` in an interactive terminal with a display opens the desktop
    # app; everywhere else it prints help. Both gates are
    # load-bearing: the TTY gate keeps the cron / launchd / pipe / `make` / CI
    # class of invocation on help text exactly as before the flip, and the
    # display gate keeps SSH and consoles without a display server off a Qt
    # launch that could not open a window. A Qt platform-plugin init failure is
    # a C++ qFatal → abort() that no Python except can catch (measured
    # 2026-07-30), so the residue — interactive terminal, display env present,
    # display actually broken, e.g. a stale DISPLAY in tmux — aborts with Qt's
    # own two-line stderr; a pre-flight probe would cost a subprocess PySide6
    # import on every launch and is not worth that residue. `steno --gui`
    # skips both gates: it is a window, and it is how the app icon starts the
    # tool.
    if gui or (_interactive_terminal() and _display_available()):
        from stenograf.gui import run_gui

        run_gui(tray=tray)
    else:
        click.echo(ctx.get_help())


def _sigterm_becomes_interrupt(signum: int, frame: object) -> None:
    """SIGTERM raises KeyboardInterrupt, so every Ctrl-C path handles it too.

    The session's interrupt story then covers both signals: capture ends
    gracefully, the finalize runs, notes are skipped with their notice. While
    the finalize shield is up (:func:`stenograf.session._shield_interrupt`
    sets SIGINT to SIG_IGN), SIGTERM is honored the same way — raising there
    would break the one stretch that must not be interrupted, and the shield
    is bounded (seconds)."""
    if signal.getsignal(signal.SIGINT) is signal.SIG_IGN:
        return
    raise KeyboardInterrupt


def _interactive_terminal() -> bool:
    """Both ends of the session are a TTY (patchable seam)."""
    return sys.stdout.isatty() and sys.stdin.isatty()


def _display_available() -> bool:
    """A display server is plausibly reachable (patchable seam).

    Linux: a Wayland or X11 socket is advertised in the environment. macOS:
    ask the window server itself (:func:`_macos_aqua_session`) — the SSH env
    heuristic below false-positives wherever the ``SSH_*`` variables are lost,
    and both losses are ordinary: a tmux server started locally does not carry
    an attaching SSH client's variables into new windows, and ``sudo`` scrubs
    them (both measured 2026-07-30). Windows (and the macOS fallback when the
    probe cannot run): a local session always has a window server, so the
    question inverts — is this a *remote* shell? SSH marks its sessions with
    SSH_CONNECTION/SSH_TTY, and an SSH login cannot open a window on the
    sitting user's display.
    """
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if sys.platform == "darwin":
        aqua = _macos_aqua_session()
        if aqua is not None:
            return aqua
    return not (os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"))


def _macos_aqua_session() -> bool | None:
    """Whether this process belongs to a window-server session (patchable seam).

    ``CGSessionCopyCurrentDictionary()`` returns NULL exactly when the calling
    process has no Aqua session — an SSH login, however its environment was
    laundered on the way here. This is the authoritative answer the env
    heuristic approximates, and it matters because a Qt launch without a
    window server is an uncatchable C++ abort. One in-process ctypes call: no
    subprocess, no Qt import. ``None`` when the probe itself is unavailable,
    and the caller falls back to the SSH_* heuristic.
    """
    try:
        import ctypes
        import ctypes.util

        path = ctypes.util.find_library("CoreGraphics")
        if path is None:
            return None
        coregraphics = ctypes.CDLL(path)
        coregraphics.CGSessionCopyCurrentDictionary.restype = ctypes.c_void_p
        coregraphics.CGSessionCopyCurrentDictionary.argtypes = []
        session = coregraphics.CGSessionCopyCurrentDictionary()
        if not session:
            return False
        corefoundation = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
        corefoundation.CFRelease.argtypes = [ctypes.c_void_p]
        corefoundation.CFRelease(session)
        return True
    except Exception:  # noqa: BLE001 — a probe failure must not decide dispatch
        return None


main.add_command(start.start)
main.add_command(transcribe.transcribe)
main.add_command(doctor_cmd.doctor)
main.add_command(doctor_cmd.setup)
main.add_command(profiles.profiles)
main.add_command(settings_cmd.settings_group)
main.add_command(settings_cmd.presets_command)
main.add_command(notes.notes_command)
