"""The platform directories stenograf reads and writes, in one stdlib-only module.

Three distinct locations, one function each — the distinction is load-bearing:

- :func:`cache_dir`: re-downloadable model assets (safe to delete);
- :func:`data_dir`: precious user data — speaker voiceprints, settings.toml;
- :func:`documents_dir`: the user's *visible* documents folder, where the
  output home (``Meetings``) lives — transcripts and notes are user documents,
  not app state, and the filesystem is the index.

Kept free of any non-stdlib import so settings.py (and anything else that
must load before heavy dependencies) can resolve its file location without
dragging numpy or a backend along.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

_XDG_DOCUMENTS_DIR = re.compile(r'^\s*XDG_DOCUMENTS_DIR\s*=\s*"(.+)"', re.MULTILINE)
"""One line of ``user-dirs.dirs``. The file documents its own format as
``XDG_xxx_DIR="$HOME/yyy"`` or ``XDG_xxx_DIR="/yyy"``, and says no other form is
supported — so the quotes are guaranteed and ``$HOME`` is the only variable."""


def cache_dir() -> Path:
    """Directory for re-downloadable model assets (``$STENOGRAF_CACHE`` overrides)."""
    if override := os.environ.get("STENOGRAF_CACHE"):
        return Path(override).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "stenograf"
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
        return Path(local) / "stenograf" / "cache"
    xdg = os.environ.get("XDG_CACHE_HOME", "~/.cache")
    return Path(xdg).expanduser() / "stenograf"


def data_dir() -> Path:
    """Directory for precious user data (speaker profiles, settings), distinct
    from the model cache: ``$STENOGRAF_DATA`` if set, else the platform data
    dir (``%APPDATA%`` on Windows — added with Phase 6, before any Windows
    release, so no pre-existing installs need migrating)."""
    if override := os.environ.get("STENOGRAF_DATA"):
        return Path(override).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "stenograf"
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))
        return Path(appdata) / "stenograf"
    xdg = os.environ.get("XDG_DATA_HOME", "~/.local/share")
    return Path(xdg).expanduser() / "stenograf"


def documents_dir() -> Path:
    """The user's documents folder, by the name their file manager uses.

    ``~/Documents`` on macOS and Windows. On Linux the folder is *localised*:
    a German desktop calls it ``~/Dokumente``, and writing to ``~/Documents``
    there creates a second documents tree that the file manager's sidebar does
    not list — which breaks the promise that the filesystem is the index. The
    real name is recorded in ``$XDG_CONFIG_HOME/user-dirs.dirs``.

    That file is read directly rather than through ``xdg-user-dir``: the binary
    belongs to the xdg-user-dirs package and is not installed everywhere, while
    the file's own format — one quoted assignment per line, ``$HOME`` the only
    variable it uses — is a regex. When the file is absent,
    unreadable, or has the folder disabled (an entry pointing at ``$HOME``
    itself), ``~/Documents`` stands — which is also xdg-user-dirs' own default
    for an English desktop.
    """
    if sys.platform.startswith("linux") and (configured := _xdg_documents_dir()) is not None:
        return configured
    return Path.home() / "Documents"


def _xdg_documents_dir() -> Path | None:
    """``XDG_DOCUMENTS_DIR`` from ``user-dirs.dirs``, or ``None`` if it says nothing.

    The environment is deliberately not consulted: only some desktops export
    these variables into the session, so the file is the one answer that is the
    same however the app was started — from a shell, from the desktop entry, or
    from a login item."""
    config = Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser()
    try:
        text = (config / "user-dirs.dirs").read_text(encoding="utf-8")
    except OSError:
        return None
    values = _XDG_DOCUMENTS_DIR.findall(text)
    if not values:
        return None
    # Last assignment wins: the file is sourced as shell, so a duplicated key
    # would leave the later value in the environment.
    path = Path(os.path.expandvars(values[-1]))
    if not path.is_absolute() or path == Path.home():
        return None  # $HOME means "this folder is disabled" in the XDG spec
    return path
