"""Command-line interface: ``stenograf`` / ``steno``.

A package of one module per command (plus ``run`` for the flag+settings
resolution the commands share and ``format`` for the human-facing rendering
helpers); this ``__init__`` only assembles the click group. Domain logic
lives in the library — each command body resolves its inputs and makes
library calls.
"""

from __future__ import annotations

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
    # `--gui` is compiled into it as the fallback argv (PLAN.md Phase 8 step 5).
    "--gui",
    is_flag=True,
    help="Open the native desktop app instead of the terminal launcher (needs stenograf[gui]).",
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

    Run without a subcommand in a terminal to open the interactive launcher,
    or with --gui to open the desktop app.
    """
    # Windows pipes/redirects default to the legacy code page (cp1252), and a
    # single ✓/← in our output would then crash click.echo with a
    # UnicodeEncodeError. Degrade unencodable glyphs to "?" instead; the
    # interactive console is unaffected (it is UTF-16 under the hood), as are
    # the output files (written encoding="utf-8" throughout).
    for stream in (sys.stdout, sys.stderr):
        if sys.platform == "win32" and hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    if ctx.invoked_subcommand is not None:
        if gui or tray:  # both are about the entry, not about any one command
            raise click.UsageError("--gui opens the desktop app; it takes no subcommand")
        return
    if tray and not gui:
        raise click.UsageError("--tray is a mode of the desktop app; pass --gui as well")

    # Bare `steno` in a terminal opens the launcher (Phase 7); in a
    # pipe or script it prints help instead — Textual needs a real TTY, and a
    # script author hitting this by accident wants the usage text, not an app.
    # `steno --gui` opens the desktop app (Phase 8) and needs no TTY at all: it
    # is a window, and it is how a Dock/desktop shortcut will start the tool.
    if gui:
        from stenograf.gui import run_gui

        run_gui(tray=tray)
    elif _interactive_terminal():
        from stenograf.ui import run_launcher

        run_launcher()
    else:
        click.echo(ctx.get_help())


def _interactive_terminal() -> bool:
    """Both ends of the session are a TTY (patchable seam, like `_stdout_is_tty`)."""
    return sys.stdout.isatty() and sys.stdin.isatty()


main.add_command(start.start)
main.add_command(transcribe.transcribe)
main.add_command(doctor_cmd.doctor)
main.add_command(doctor_cmd.setup)
main.add_command(profiles.profiles)
main.add_command(settings_cmd.settings_group)
main.add_command(settings_cmd.presets_command)
main.add_command(notes.notes_command)
