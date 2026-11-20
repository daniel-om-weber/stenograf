"""Starting a console program without opening a window for it on Windows."""

from __future__ import annotations

import os
import subprocess
import sys


def no_window() -> int:
    """``creationflags`` that keep a console program from opening a window.

    A process without a console of its own — the app, started as
    ``pythonw.exe`` — makes Windows create a new one for every console program
    it starts, and show it: a Windows Terminal window flashed up at every app
    start for the capture helper's device query (measured 2026-11-20, Windows
    11). A process with a console (the CLI in a terminal) passes it on to its
    children, which show nothing, and keeps doing so. ``os.name`` is asked as
    well because tests pretend to be Windows through ``sys.platform`` alone.
    """
    if sys.platform == "win32" and os.name == "nt":
        import ctypes

        if not ctypes.windll.kernel32.GetConsoleWindow():
            return subprocess.CREATE_NO_WINDOW
    return 0
