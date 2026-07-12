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
    """Screen-stack shell: mounts Home, or a caller-supplied root screen.

    ``initial`` is the second entry mode (PLAN.md §5, Phase 7): ``steno start``
    runs this same app with the meeting screen as its root, so the CLI and the
    launcher share one codepath. A root screen sits at the bottom of the stack
    and cannot be popped — dismissing it exits the app (``MeetingScreen._leave``).
    """

    TITLE = "stenograf"

    def __init__(self, initial: Screen | None = None) -> None:
        super().__init__()
        self._initial = initial

    def get_default_screen(self) -> Screen:
        return self._initial if self._initial is not None else HomeScreen()

    def on_mount(self) -> None:
        self.animation_level = "none"  # minimal redraw: no CSS/scroll animations
