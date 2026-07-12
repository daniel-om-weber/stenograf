"""The ``steno`` launcher — a mouse-driven Textual home for the pipeline.

Phase 7 (PLAN.md §5): bare ``steno`` opens a button-based app so people who
don't live in a terminal can start a meeting, transcribe a recording, and
generate notes without remembering subcommands. One module per screen,
mirroring ``cli/``'s one-module-per-command layout; every screen is a thin
client of the library (the C7 rule) — domain logic never lives here.

This ``__init__`` stays dependency-light: the CLI imports it to reach
:func:`run_launcher`, and the heavy import (textual, via ``ui.app``) happens
only inside that call — pipes and scripts never pay for it.
"""

from __future__ import annotations


def run_launcher() -> None:
    """Run the launcher app until the user quits it (blocking)."""
    from stenograf.ui.app import StenografApp

    StenografApp().run()
