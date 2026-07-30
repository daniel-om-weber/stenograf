"""The native desktop app — bare ``steno`` in a terminal, ``steno --gui``,
or the app icon (Phase 8).

The button-driven entry to the pipeline: a real Qt Quick window (home menu →
one screen per workflow) over the same library the CLI subcommands use — the
shared workflows live in :mod:`stenograf.flow`, so the two cannot drift.

This ``__init__`` stays dependency-light — the CLI imports it to reach
:func:`run_gui`, and Qt is imported only inside that call, so a terminal
session running subcommands never pays for it.
"""

from __future__ import annotations


def run_gui(*, tray: bool = False) -> None:
    """Open the desktop app and block until it quits.

    With ``tray`` it starts in the menu bar with no window (Phase 8 step 6).

    PySide6 is a base dependency since the default flip, so missing Qt means a
    broken or pre-flip install — still a plain instruction rather than an
    ImportError traceback, because the frozen ``Stenograf.app`` stub can reach
    this path on such an install."""
    import importlib.util

    import click

    if importlib.util.find_spec("PySide6") is None:
        raise click.ClickException(
            "the desktop app needs Qt (PySide6), which is missing from this "
            "install. Reinstall stenograf to get it:\n"
            "  uv tool install --force stenograf\n"
            "(or `pip install --upgrade --force-reinstall stenograf`). The "
            "CLI subcommands — `steno start`, `steno transcribe`, … — work "
            "without it."
        )
    from stenograf.gui.app import run

    code = run(tray=tray)
    if code:
        raise SystemExit(code)
