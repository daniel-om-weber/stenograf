"""Pin Textual's frame cap low — import this before anything else textual.

The minimal-redraw budget (PLAN.md §5, Phase 2 Task 6) caps Textual at 15 fps.
The cap is import-order-sensitive: textual reads ``TEXTUAL_FPS`` once, at
import, into ``constants.MAX_FPS`` and ``Screen.UPDATE_PERIOD`` — so the env
var must be set before textual is imported. Both textual entry points
(:mod:`stenograf.ui.meeting`, the live view; :mod:`stenograf.ui.app`, the
launcher) import this module first, keeping the sensitive dance in one place.

Importing it late is still safe: the re-pin below overwrites the baked
constants regardless of who imported textual first. ``UPDATE_PERIOD`` (the
screen-refresh interval) is read only when an app first builds its update
timer, so assigning it here — before any app runs — reliably bounds the
redraw rate. A ``TEXTUAL_FPS`` the user already set is honoured (setdefault).
"""

from __future__ import annotations

import os

os.environ.setdefault("TEXTUAL_FPS", "15")

import textual.constants  # noqa: E402
import textual.screen  # noqa: E402

FPS = int(os.environ["TEXTUAL_FPS"])
textual.constants.MAX_FPS = FPS
textual.screen.UPDATE_PERIOD = 1 / FPS
