"""The native desktop app — ``steno --gui`` (Phase 8).

The second button-driven entry to the pipeline, next
to the Textual launcher: a real Qt Quick window for people who don't live in a
terminal, with the same information architecture (home menu → one screen per
workflow) and the same library underneath. The Textual launcher stays the
default until this reaches parity; both run the shared workflows in
:mod:`stenograf.flow`, so they cannot drift.

This ``__init__`` stays dependency-light — the CLI imports it to reach
:func:`run_gui`, and Qt is imported only inside that call, so a terminal
session never pays for it (and an install without the ``gui`` extra still runs
everything else).
"""

from __future__ import annotations


def run_gui(*, tray: bool = False) -> None:
    """Open the desktop app and block until it quits.

    With ``tray`` it starts in the menu bar with no window (Phase 8 step 6).

    Qt is an optional dependency while the GUI is opt-in, so a missing PySide6
    is a plain instruction rather than an ImportError traceback."""
    import importlib.util

    import click

    if importlib.util.find_spec("PySide6") is None:
        raise click.ClickException(
            "the desktop app needs Qt, which is not installed. Add it with:\n"
            "  uv tool install --force 'stenograf[gui]'\n"
            "(or `pip install 'stenograf[gui]'`). The terminal launcher — bare "
            "`steno` — needs nothing extra."
        )
    from stenograf.gui.app import run

    code = run(tray=tray)
    if code:
        raise SystemExit(code)
