"""``steno notes`` — regenerate LLM meeting notes for a finished meeting.

Generation lives in the library (:mod:`stenograf.notes.run`, which is also
the ``--notes`` tail the other commands share); this module owns the flags
and the echoes."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click

from stenograf.cli.run import _library_errors
from stenograf.notes import run as notes_run

if TYPE_CHECKING:
    from stenograf.notes import MeetingNotes


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
    from stenograf.notes import NotesBackendError
    from stenograf.output import load_transcript
    from stenograf.settings import SettingsError

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

    target = _resolve_notes_target(meeting, last)
    try:
        transcript, path, created_at = load_transcript(target)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    out_dir, basename = path.parent, path.stem

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


def _echo_warnings(notes: MeetingNotes) -> None:
    """Validation warnings, yellow, one line each — they are also recorded in
    the note's own provenance footer, so the screen is the courtesy copy."""
    if notes.provenance is not None:
        for warning in notes.provenance.warnings:
            click.secho(f"notes warning: {warning}", fg="yellow")


def _resolve_notes_target(meeting: Path | None, last: bool) -> Path:
    """The meeting folder (or transcript JSON) a ``steno notes`` invocation names.

    Exactly one of PATH and ``--last`` must be given; ``--last`` scans the
    output home for the newest finished meeting folder (by name — the name
    encodes the start time). Resolving the target to its transcript is
    :func:`~stenograf.output.load_transcript`'s job, not this one's."""
    from stenograf.output import latest_meeting_dir, output_home
    from stenograf.settings import load_settings

    if last and meeting is not None:
        raise click.UsageError("give either a PATH or --last, not both")
    if last:
        home = output_home(load_settings())
        newest = latest_meeting_dir(home)
        if newest is None:
            raise click.ClickException(
                f"no finished meeting found in {home} — run `steno start` first, "
                "or name a transcript path"
            )
        click.echo(f"meeting: {newest}")  # say which one --last picked
        return newest
    if meeting is None:
        raise click.UsageError("name a meeting folder or transcript.json, or use --last")
    return meeting


