"""The launcher application shell.

Phase 7, Task 1 (PLAN.md §5). One long-lived :class:`StenografApp`; each
workflow is a :class:`~textual.screen.Screen` pushed onto its stack (Home is
the default). The app owns the minimal-redraw budget for every screen: the
frame cap is pinned by the :mod:`stenograf.ui._fps` import below (which must
precede the textual imports) and animations are disabled on mount — same
budget the live-caption view established in Phase 2.
"""

from __future__ import annotations

import stenograf.ui._fps  # noqa: F401  — must precede the textual imports (frame cap)

# isort: split

from textual.app import App
from textual.screen import Screen

from stenograf.ui.home import HomeScreen


class StenografApp(App[None]):
    """Screen-stack shell: composes nothing itself, just mounts Home."""

    TITLE = "stenograf"

    def get_default_screen(self) -> Screen:
        return HomeScreen()

    def on_mount(self) -> None:
        self.animation_level = "none"  # minimal redraw: no CSS/scroll animations
