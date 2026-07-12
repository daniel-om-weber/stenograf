"""``python -m stenograf`` — same entry as the ``steno`` console script.

The desktop launchers (:mod:`stenograf.shortcut`) run ``"<absolute python>"
-m stenograf`` instead of ``steno``: a double-clicked launcher gets a fresh
login-shell environment whose PATH may not include uv's shim directory.
"""

from stenograf.cli import main

if __name__ == "__main__":
    main()
