"""``steno settings`` — inspect and edit the settings.toml defaults."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import click

from stenograf.cli.format import _fmt_setting
from stenograf.output import atomic_write_text
from stenograf.transcript import DEFAULT_FORMATS

if TYPE_CHECKING:
    from pathlib import Path


@click.group("settings")
def settings_group() -> None:
    """Inspect and edit the settings.toml defaults."""


@settings_group.command("show")
def settings_show() -> None:
    """Print the effective configuration and where each value comes from.

    Sources: an environment override, settings.toml, or the built-in default.
    (CLI flags outrank all three but are per-run, so they never appear here.)
    """
    from stenograf.settings import SettingsError, load_settings, settings_path

    path = settings_path()
    suffix = "" if path.exists() else " (not present — all defaults)"
    click.echo(f"settings: {path}{suffix}")
    try:
        settings = load_settings()
    except SettingsError as exc:
        raise click.ClickException(f"{exc} — fix it with `steno settings edit`") from exc
    for table, rows in _settings_rows(settings):
        click.echo(f"\n[{table}]")
        width = max(len(key) for key, _, _ in rows)
        for key, value, source in rows:
            click.echo(f"  {key:<{width}} = {value}  ({source})")


@settings_group.command("edit")
def settings_edit() -> None:
    """Open settings.toml in $EDITOR and validate it on save.

    A missing file is first created from a fully commented template, so every
    available key is in front of you. Validation failures keep your edits —
    rerun to fix them.
    """
    from stenograf.settings import SettingsError, load_settings

    path, created = _ensure_settings_file()
    if created:
        click.echo(f"created {path}")
    click.edit(filename=str(path))
    try:
        load_settings(path)
    except SettingsError as exc:
        raise click.ClickException(
            f"{exc}\nyour edits are saved — run `steno settings edit` again to fix them"
        ) from exc
    click.echo(f"{path} OK")


def _ensure_settings_file() -> tuple[Path, bool]:
    """settings.toml's path, created from the commented template when missing.

    Shared by ``settings edit`` and the launcher's Settings screen — the
    template (every key present, commented out) is the editing surface both
    hand to $EDITOR. Returns ``(path, created)``."""
    from stenograf.settings import SETTINGS_TEMPLATE, settings_path

    path = settings_path()
    if path.exists():
        return path, False
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, SETTINGS_TEMPLATE)
    return path, True


def _settings_rows(settings) -> list[tuple[str, list[tuple[str, str, str]]]]:
    """``(table, [(key, value, source), …])`` rows behind ``settings show``.

    Values are TOML-flavored so a line can be pasted into the file; defaults
    that aren't literal values (an unset optional, a per-backend choice) read
    as a parenthesized description instead."""
    from stenograf.asr.registry import default_backend_name as asr_default
    from stenograf.glossary import DEFAULT_THRESHOLD as GLOSSARY_THRESHOLD
    from stenograf.notes.backend import default_backend_name as notes_default
    from stenograf.notes.backend import settings_defaults as notes_defaults
    from stenograf.notes.ollama import DEFAULT_URL
    from stenograf.output import default_output_home
    from stenograf.profiles import DEFAULT_THRESHOLD as REID_THRESHOLD
    from stenograf.profiles import default_store_path

    # Per-backend notes defaults resolve against the *effective* backend, so the
    # display matches what a notes run would actually use; keys the backend has
    # no say over get a which-backend placeholder.
    notes_backend = notes_default(settings.notes.backend)
    per_backend = notes_defaults(notes_backend)

    # (table, key, file value, effective default, env override) — one row each.
    descriptors = [
        ("transcript", "formats", settings.transcript.formats, DEFAULT_FORMATS, None),
        ("vocab", "glossary_file", settings.vocab.glossary_file, "(none)", None),
        ("vocab", "attendees", settings.vocab.attendees, "(none)", None),
        (
            "vocab",
            "glossary_threshold",
            settings.vocab.glossary_threshold,
            GLOSSARY_THRESHOLD,
            None,
        ),
        ("output", "dir", settings.output.dir, default_output_home(), None),
        ("speakers", "diarization", settings.speakers.diarization, True, None),
        ("speakers", "reid_threshold", settings.speakers.reid_threshold, REID_THRESHOLD, None),
        ("speakers", "profile_store", settings.speakers.profile_store, default_store_path(), None),
        ("asr", "backend", settings.asr.backend, asr_default(), "STENOGRAF_ASR_BACKEND"),
        ("asr", "provider", settings.asr.provider, "cpu", "STENOGRAF_ASR_PROVIDER"),
        ("notes", "backend", settings.notes.backend, notes_backend, "STENOGRAF_NOTES_BACKEND"),
        (
            "notes",
            "model",
            settings.notes.model,
            per_backend.get("model", "(provenance label — none)"),
            "STENOGRAF_NOTES_MODEL",
        ),
        ("notes", "command", settings.notes.command, "(none)", None),
        (
            "notes",
            "timeout_s",
            settings.notes.timeout_s,
            per_backend.get("timeout_s", "(command backend only)"),
            None,
        ),
        ("notes", "instructions", settings.notes.instructions, "(none)", None),
        ("notes", "ollama_url", settings.notes.ollama_url, DEFAULT_URL, "OLLAMA_HOST"),
        (
            "notes",
            "max_input_chars",
            settings.notes.max_input_chars,
            per_backend["max_input_chars"],
            None,
        ),
        (
            "notes",
            "thinking",
            settings.notes.thinking,
            per_backend.get("thinking", "(mlx backend only)"),
            None,
        ),
        ("notes.export", "dir", settings.notes.export_dir, "(off)", None),
    ]

    def pick(file_value, default, env_var: str | None) -> tuple[str, str]:
        if env_var and (env_value := os.environ.get(env_var)):
            return _fmt_setting(env_value), f"${env_var}"
        if file_value is not None and file_value != ():
            return _fmt_setting(file_value), "settings.toml"
        return _fmt_setting(default), "default"

    tables: dict[str, list[tuple[str, str, str]]] = {}
    for table, key, file_value, default, env_var in descriptors:
        tables.setdefault(table, []).append((key, *pick(file_value, default, env_var)))
    return list(tables.items())
