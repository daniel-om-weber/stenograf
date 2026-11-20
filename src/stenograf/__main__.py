"""``python -m stenograf`` — same entry as the ``steno`` console script.

The desktop launchers (:mod:`stenograf.shortcut`) run ``"<absolute python>"
-m stenograf`` instead of ``steno``: a double-clicked launcher gets a fresh
login-shell environment whose PATH may not include uv's shim directory.
"""

import os
import sys

# On Windows those launchers and the updater's restart run ``pythonw.exe``,
# whose sys.stdout and sys.stderr are None. huggingface_hub draws a progress bar
# on stderr while it downloads the speech model, and with None there it raised
# AttributeError (hub 2.4, tqdm 4.70) or deadlocked (hub 0.36, tqdm 4.68) —
# measured 2026-11-20 with sys.stderr set to None on macOS — so a first meeting
# after an update could never load.
for _stream in ("stdout", "stderr"):
    if getattr(sys, _stream) is None:
        setattr(sys, _stream, open(os.devnull, "w", encoding="utf-8"))  # noqa: SIM115 — lives as long as the process

from stenograf.cli import main  # noqa: E402 — after the streams exist

if __name__ == "__main__":
    main()
