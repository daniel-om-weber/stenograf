"""``steno notes`` — LLM meeting notes, plus the ``--notes`` tail the other
commands share."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import click

from stenograf.output import atomic_write_text
from stenograf.transcript import Transcript

if TYPE_CHECKING:
    from stenograf.view import LiveView


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

        try:
            overlaid, preset_obj = apply_meeting_preset(load_settings(), preset)
        except SettingsError as exc:
            raise click.UsageError(str(exc)) from exc
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
        written, notes = _generate_and_write_notes(
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
    from stenograf.cli.run import _cli_settings
    from stenograf.output import TRANSCRIPT_STEM, default_output_home, latest_meeting_dir

    if last and meeting is not None:
        raise click.UsageError("give either a PATH or --last, not both")
    if last:
        home = _cli_settings().output.dir or default_output_home()
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


def _generate_and_write_notes(
    transcript: Transcript,
    out_dir: Path,
    basename: str,
    *,
    created_at: datetime,
    backend_name: str | None = None,
    model: str | None = None,
    ollama_url: str | None = None,
    instructions_file: Path | None = None,
    export_dir: Path | None = None,
    no_export: bool = False,
    notes_settings=None,
    on_progress: Callable[[str], None] | None = None,
):
    """Generate notes and write ``<basename>.notes.md`` (plus the combined-note
    export when a target dir is configured). Returns ``(written_paths,
    notes)``; raises typed errors, writing nothing, on failure. Validation
    warnings ride on ``notes.provenance.warnings`` — the callers surface them.

    ``notes_settings`` is the ``[notes]`` table a command already loaded at its
    start (so a ``--notes`` run uses the values in force when the meeting began);
    ``None`` loads it here (the standalone ``steno notes`` path). ``on_progress``
    overrides the click-echoed progress line — the launcher routes it to the
    meeting screen's header, where an echo would corrupt the raw-mode terminal."""
    import dataclasses

    from stenograf.notes import create_backend
    from stenograf.notes.export import export_note
    from stenograf.notes.generate import generate_notes

    if notes_settings is None:
        from stenograf.settings import load_settings

        notes_settings = load_settings().notes
    settings = notes_settings
    if backend_name and settings.backend and backend_name != settings.backend:
        # [notes] model in settings.toml was written for the configured
        # backend and must not ride along to an explicitly chosen other one
        # (--model below still wins).
        settings = dataclasses.replace(settings, backend=backend_name, model=None)
    if model or ollama_url:
        settings = dataclasses.replace(
            settings,
            model=model or settings.model,
            ollama_url=ollama_url or settings.ollama_url,
        )
    backend = create_backend(backend_name, settings)
    position = getattr(backend, "set_position", None)
    if position is not None:
        # The agentic contract (command backend only): the meeting folder is
        # the working directory, the parent — where the meetings live — rides
        # in the environment. Fetching context is the command's job.
        position(out_dir, out_dir.parent)
    instructions = None
    # A per-run --instructions file replaces the standing one, it does not
    # append: the flag exists to try a different style, not to stack two.
    instructions_path = instructions_file or settings.instructions
    if instructions_path is not None:
        instructions = instructions_path.read_text(encoding="utf-8")
    from stenograf.notes.template import DEFAULT_TEMPLATE

    # [notes] template (or a preset's) is the protocol layout; its headings
    # are what validation matches the response against.
    template = DEFAULT_TEMPLATE
    if settings.template is not None:
        template = settings.template.read_text(encoding="utf-8")

    notes = generate_notes(
        transcript,
        backend,
        instructions=instructions,
        on_progress=on_progress or (lambda message: click.echo(f"notes: {message}")),
        template=template,
        fallback_title=f"{created_at:%Y-%m-%d}",
    )

    md_path = out_dir / f"{basename}.notes.md"
    atomic_write_text(md_path, notes.to_markdown())
    # A regenerated meeting must not keep a stale JSON (the pre-markdown
    # format) that disagrees with its markdown.
    (out_dir / f"{basename}.notes.json").unlink(missing_ok=True)
    written = [md_path]

    target = None if no_export else (export_dir or settings.export_dir)
    if target is not None:
        written.append(export_note(transcript, notes, target, created_at=created_at))
    return written, notes


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
        written, notes = _generate_and_write_notes(
            transcript,
            out_dir,
            basename,
            created_at=created_at,
            notes_settings=notes_settings,
            backend_name=backend_name,
            model=model,
            instructions_file=instructions_file,
        )
    except Exception as exc:
        click.secho(f"notes failed: {exc}", fg="yellow")
        click.secho(f"  the transcript is safe — retry with `steno notes {out_dir}`", fg="yellow")
        return
    _echo_warnings(notes)
    click.echo(f"notes: wrote {', '.join(str(p) for p in written)}")


def _generate_notes(
    view: LiveView,
    transcript: Transcript,
    out_dir: Path,
    basename: str,
    *,
    created_at: datetime,
    notes_settings,
    backend_name: str | None = None,
    model: str | None = None,
    instructions_file: Path | None = None,
) -> bool:
    """The ``--notes`` tail for a run behind a live TUI: non-fatal, progress via
    the view.

    Same contract as :func:`_notes_after_run` — the transcript
    is already on disk, so a notes failure warns and returns ``False``; rerun
    later with ``steno notes``. Shared by the launcher's meeting flow and the
    ``steno start`` TUI shape, so both generate notes *while the meeting screen
    is still up* instead of after the user quits it. Generation goes through
    the shared entry point, which owns the MLX thread-affinity guard.
    """
    view.status("generating notes…")
    try:
        _written, notes = _generate_and_write_notes(
            transcript,
            out_dir,
            basename,
            created_at=created_at,
            notes_settings=notes_settings,
            backend_name=backend_name,
            model=model,
            instructions_file=instructions_file,
            on_progress=lambda message: view.status(f"notes: {message}"),
        )
    except Exception as exc:  # noqa: BLE001 — non-fatal by contract
        view.error(f"notes failed: {exc} — the transcript is safe; retry with `steno notes`")
        return False
    if notes.provenance is not None and notes.provenance.warnings:
        # A courtesy flash only — the durable copy is the note's own footer,
        # because the done screen's next status line overwrites this one.
        view.error(f"notes warning: {'; '.join(notes.provenance.warnings)}")
    return True
