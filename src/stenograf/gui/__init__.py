"""The native desktop app — bare ``steno`` in a terminal, ``steno --gui``,
or the app icon.

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

    With ``tray`` it starts in the menu bar with no window.

    PySide6 is a base dependency since the default flip, so an ImportError here
    means a broken or pre-flip install — that includes Qt *present* but not
    loadable (a missing libEGL on a minimal Linux box, an arch-mismatched
    wheel), which is why this guards the real import rather than a
    ``find_spec`` presence check. Still a plain instruction rather than a
    traceback, because the frozen ``Stenograf.app`` stub can reach this path
    on such an install."""
    import click

    try:
        from stenograf.gui.app import run
    except ImportError as exc:
        raise click.ClickException(
            "the desktop app needs Qt (PySide6), which this install cannot "
            "load. Reinstall stenograf:\n"
            "  uv tool install --force stenograf\n"
            "(or `pip install --upgrade --force-reinstall stenograf`). The "
            "CLI subcommands — `steno start`, `steno transcribe`, … — work "
            f"without it.\nThe import failed with: {exc}"
        ) from exc

    code = run(tray=tray)
    if code:
        raise SystemExit(code)
