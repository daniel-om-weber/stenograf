"""``steno notes`` — LLM meeting notes, plus the ``--notes`` tail the other
commands share.

Generation itself lives in the library (:mod:`stenograf.notes.run`); this
module owns the flags, the echoes, and the CLI-only Ctrl-C choreography."""

from __future__ import annotations

import contextlib
import signal
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import click

from stenograf.cli.run import _library_errors
from stenograf.notes import run as notes_run
from stenograf.transcript import Transcript


def _echo_progress(message: str) -> None:
    """The CLI's notes progress line (the library stays silent on its own)."""
    click.echo(f"notes: {message}")


@click.command("notes")
@click.argument(
    "meeting", required=False, type=click.Path(exists=True, path_type=Path), metavar="[PATH]"
)
@click.option(
    "--last",
    "last",
    is_flag=True,
    help="Use the newest meeting folder in the output home ([output] dir in "
    "settings.toml, else Meetings/ in your documents folder) instead of naming "
    "a PATH.",
)
@click.option(
    "--backend",
    "backend_name",
    default=None,
    metavar="NAME",
    help="Notes backend: mlx (local, in-process), ollama (local server), or command "
    "(any CLI, e.g. claude) [default: settings.toml, else mlx where installed, else ollama].",
)
@click.option(
    "--model",
    default=None,
    help="Model to use (HF repo id for mlx, Ollama model tag for ollama; "
    "a provenance label for command backends).",
)
@click.option("--ollama-url", default=None, metavar="URL", help="Ollama server URL.")
@click.option(
    "--preset",
    "preset",
    default=None,
    metavar="NAME",
    help="Meeting preset ([meetings.NAME] in settings.toml): its notes setup, "
    "protocol template and instructions apply to this regeneration. "
    "`steno presets` lists them; explicit flags still win.",
)
@click.option(
    "--instructions",
    "instructions_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    metavar="FILE",
    help="Style/structure instructions appended to the notes prompt for this run "
    "[default: [notes] instructions in settings.toml].",
)
@click.option(
    "--export-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Also export one combined markdown note (frontmatter + summary + transcript) "
    "here — e.g. an Obsidian vault folder [default: [notes.export] dir in settings.toml].",
)
@click.option(
    "--no-export",
    is_flag=True,
    help="Skip the combined-note export even when settings.toml configures a dir.",
)
@_library_errors
def notes_command(
    meeting: Path | None,
    last: bool,
    backend_name: str | None,
    model: str | None,
    ollama_url: str | None,
    preset: str | None,
    instructions_file: Path | None,
    export_dir: Path | None,
    no_export: bool,
) -> None:
    """Generate LLM meeting notes (summary, decisions, action items).

    PATH is a meeting folder (its transcript.json is used) or a transcript
    JSON file; --last picks the newest meeting folder in the output home
    instead. Notes are written as a sibling .notes.md file; the meeting
    profile's glossary and attendees steer the prompt. Configure the backend
    in settings.toml under [notes].
    """
    import json as json_mod

    from stenograf.notes import NotesBackendError
    from stenograf.output import created_at_from_dir_name
    from stenograf.settings import SettingsError
    from stenograf.transcript import UnsupportedTranscriptVersion

    notes_settings = None
    if preset is not None:
        from stenograf.settings import apply_meeting_preset, load_settings

        # An unknown preset name is a UsageError via the _library_errors boundary.
        overlaid, preset_obj = apply_meeting_preset(load_settings(), preset)
        click.echo(f"preset: {preset}")  # echo-on-use: the typo mitigation
        notes_settings = overlaid.notes
        # Explicit, so a preset backend beats STENOGRAF_NOTES_BACKEND (a
        # CLI-level choice; --backend still beats the preset).
        backend_name = backend_name or preset_obj.notes.backend

    path = _resolve_notes_target(meeting, last)
    try:
        transcript = Transcript.from_json(path.read_text(encoding="utf-8"))
    except (json_mod.JSONDecodeError, UnsupportedTranscriptVersion, KeyError) as exc:
        raise click.ClickException(f"{path} is not a readable transcript JSON: {exc}") from exc
    out_dir, basename = path.parent, path.stem
    # A date-named meeting folder carries the start time in its name; anything
    # else (a loose transcript file) falls back to the file's mtime.
    created_at = created_at_from_dir_name(out_dir.name) or datetime.fromtimestamp(
        path.stat().st_mtime
    )

    try:
        written, notes = notes_run.generate_and_write_notes(
            transcript,
            out_dir,
            basename,
            created_at=created_at,
            backend_name=backend_name,
            model=model,
            ollama_url=ollama_url,
            instructions_file=instructions_file,
            export_dir=export_dir,
            no_export=no_export,
            notes_settings=notes_settings,
            on_progress=_echo_progress,
        )
    except (NotesBackendError, SettingsError, ValueError, OSError) as exc:
        # The documented failure modes become clean CLI errors; anything else
        # is a bug and must propagate as a traceback, not masquerade as one.
        raise click.ClickException(str(exc)) from exc

    _echo_warnings(notes)
    click.echo(f"wrote {', '.join(str(p) for p in written)}")


def _echo_warnings(notes) -> None:
    """Validation warnings, yellow, one line each — they are also recorded in
    the note's own provenance footer, so the screen is the courtesy copy."""
    if notes.provenance is not None:
        for warning in notes.provenance.warnings:
            click.secho(f"notes warning: {warning}", fg="yellow")


def _resolve_notes_target(meeting: Path | None, last: bool) -> Path:
    """The transcript JSON a ``steno notes`` invocation names.

    Exactly one of PATH and ``--last`` must be given. A directory PATH means
    its ``transcript.json``; ``--last`` scans the output home for the newest
    finished meeting folder (by name — the name encodes the start time)."""
    from stenograf.output import TRANSCRIPT_STEM, default_output_home, latest_meeting_dir
    from stenograf.settings import load_settings

    if last and meeting is not None:
        raise click.UsageError("give either a PATH or --last, not both")
    if last:
        home = load_settings().output.dir or default_output_home()
        newest = latest_meeting_dir(home)
        if newest is None:
            raise click.ClickException(
                f"no finished meeting found in {home} — run `steno start` first, "
                "or name a transcript path"
            )
        click.echo(f"meeting: {newest}")  # say which one --last picked
        return newest / f"{TRANSCRIPT_STEM}.json"
    if meeting is None:
        raise click.UsageError("name a meeting folder or transcript.json, or use --last")
    if meeting.is_dir():
        target = meeting / f"{TRANSCRIPT_STEM}.json"
        if not target.is_file():
            raise click.ClickException(f"{meeting} holds no {TRANSCRIPT_STEM}.json")
        return target
    return meeting


def _notes_after_run(
    transcript: Transcript,
    out_dir: Path,
    basename: str,
    *,
    created_at: datetime,
    notes_settings=None,
    backend_name: str | None = None,
    model: str | None = None,
    instructions_file: Path | None = None,
) -> None:
    """The opt-in ``--notes`` step after a transcript is safely written.

    Non-fatal by contract: the transcript already stands, so
    any notes failure warns and returns — rerun later with ``steno notes``."""
    try:
        with _second_interrupt_skips():
            written, notes = notes_run.generate_and_write_notes(
                transcript,
                out_dir,
                basename,
                created_at=created_at,
                notes_settings=notes_settings,
                backend_name=backend_name,
                model=model,
                instructions_file=instructions_file,
                on_progress=_echo_progress,
            )
    except KeyboardInterrupt:
        click.secho(
            f"notes skipped — the transcript is safe; `steno notes {out_dir}` regenerates them",
            fg="yellow",
            err=True,
        )
        return
    except Exception as exc:
        click.secho(f"notes failed: {exc}", fg="yellow")
        click.secho(f"  the transcript is safe — retry with `steno notes {out_dir}`", fg="yellow")
        return
    _echo_warnings(notes)
    click.echo(f"notes: wrote {', '.join(str(p) for p in written)}")


@contextlib.contextmanager
def _second_interrupt_skips() -> Iterator[None]:
    """First Ctrl-C during the notes tail warns and keeps going; the second skips.

    The retired TUI joined the notes tail on quit, so a stray Ctrl-C never
    killed a half-written notes run; the plain CLI runs it on the main thread,
    where the default handler would. But an unconditional shield would trap
    the user behind an agentic ``[notes] command`` backend for up to its
    ``timeout_s`` — the same reason the Qt app quits *around* a notes run. So:
    announce once, and let a second press raise into the caller's
    KeyboardInterrupt arm. A no-op off the main thread, where handlers cannot
    be installed.
    """
    interrupted = False

    def handler(signum: int, frame: object) -> None:
        nonlocal interrupted
        if interrupted:
            raise KeyboardInterrupt
        interrupted = True
        click.secho("finishing notes — Ctrl-C again to skip them", fg="yellow", err=True)

    try:
        previous = signal.signal(signal.SIGINT, handler)
    except (ValueError, OSError):  # not on the main thread
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)
