"""Command-line interface: ``stenograf`` / ``steno``.

A package of one module per command (plus ``run`` for the flag+settings
resolution the commands share and ``format`` for the human-facing rendering
helpers); this ``__init__`` only assembles the click group. Domain logic
lives in the library — each command body resolves its inputs and makes
library calls (PLAN-CLEANUP.md C7).
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


@click.group()
@click.version_option(__version__, prog_name="stenograf")
def main() -> None:
    """Accuracy-first local meeting transcription. Audio never touches disk."""
    # Windows pipes/redirects default to the legacy code page (cp1252), and a
    # single ✓/← in our output would then crash click.echo with a
    # UnicodeEncodeError. Degrade unencodable glyphs to "?" instead; the
    # interactive console is unaffected (it is UTF-16 under the hood), as are
    # the output files (written encoding="utf-8" throughout).
    for stream in (sys.stdout, sys.stderr):
        if sys.platform == "win32" and hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")


main.add_command(start.start)
main.add_command(transcribe.transcribe)
main.add_command(doctor_cmd.doctor)
main.add_command(doctor_cmd.setup)
main.add_command(profiles.profiles)
main.add_command(settings_cmd.settings_group)
main.add_command(notes.notes_command)
