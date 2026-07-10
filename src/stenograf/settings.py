"""User settings — ``data_dir()/settings.toml``.

Machine-specific configuration that must NOT live in a :class:`MeetingProfile`
(profiles serialize into every transcript; a local command line or vault path
would leak into shared files). First consumer is the ``[notes]`` table
(PLAN.md §5 Stage D5)::

    [notes]
    backend = "command"            # "ollama" (default) or "command"
    model = "claude-opus-4-8"      # provenance label; the Ollama model to run
    command = ["claude", "-p", "Summarize the meeting transcript on stdin."]
    timeout_s = 600
    instructions = "~/notes-style.md"   # appended to the built-in system prompt

    [notes.export]
    dir = "~/Obsidian/Meetings"    # combined-note export target (unset = off)

Precedence everywhere a value is consumed: CLI flag > environment variable
(``STENOGRAF_NOTES_BACKEND``, ``STENOGRAF_NOTES_MODEL``, ``OLLAMA_HOST``) >
this file > built-in default. A missing file is simply all defaults.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from stenograf.profiles import data_dir


class SettingsError(Exception):
    """settings.toml exists but cannot be used; the message names the file."""


@dataclass(frozen=True)
class NotesSettings:
    backend: str | None = None
    model: str | None = None
    command: tuple[str, ...] = ()
    timeout_s: float | None = None
    instructions: Path | None = None
    ollama_url: str | None = None
    export_dir: Path | None = None


@dataclass(frozen=True)
class Settings:
    notes: NotesSettings = field(default_factory=NotesSettings)


def settings_path() -> Path:
    return data_dir() / "settings.toml"


def load_settings(path: Path | None = None) -> Settings:
    """Read ``settings.toml`` (or ``path``); a missing file is all defaults.

    Malformed TOML, a wrong-typed value, or an unknown ``backend`` raise one
    :class:`SettingsError` naming the file — settings problems must never
    surface as a traceback deep inside a meeting run."""
    path = path or settings_path()
    if not path.exists():
        return Settings()
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise SettingsError(f"cannot read {path}: {exc}") from exc
    try:
        return Settings(notes=_notes_from_table(data.get("notes", {}), path))
    except (TypeError, ValueError) as exc:
        raise SettingsError(f"invalid settings in {path}: {exc}") from exc


def _notes_from_table(table: dict, path: Path) -> NotesSettings:
    if not isinstance(table, dict):
        raise ValueError("[notes] must be a table")
    backend = _opt_str(table, "backend")
    if backend is not None:
        from stenograf.notes.backend import available_backends

        if backend not in available_backends():
            raise SettingsError(
                f"invalid settings in {path}: unknown notes backend {backend!r} "
                f"(choose from {', '.join(available_backends())})"
            )
    command = table.get("command", ())
    if isinstance(command, str) or not all(isinstance(c, str) for c in command):
        raise ValueError("notes.command must be an array of strings (argv)")
    timeout = table.get("timeout_s")
    if timeout is not None and not isinstance(timeout, int | float):
        raise ValueError("notes.timeout_s must be a number")
    export = table.get("export", {})
    if not isinstance(export, dict):
        raise ValueError("[notes.export] must be a table")
    return NotesSettings(
        backend=backend,
        model=_opt_str(table, "model"),
        command=tuple(command),
        timeout_s=float(timeout) if timeout is not None else None,
        instructions=_opt_path(table, "instructions"),
        ollama_url=_opt_str(table, "ollama_url"),
        export_dir=_opt_path(export, "dir"),
    )


def _opt_str(table: dict, key: str) -> str | None:
    value = table.get(key)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"notes.{key} must be a string")
    return value


def _opt_path(table: dict, key: str) -> Path | None:
    value = _opt_str(table, key)
    return Path(value).expanduser() if value is not None else None
