"""Flag+settings resolution and the command tail shared by ``start`` and
``transcribe`` (plus ``notes``' settings loading)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import click

from stenograf.transcript import DEFAULT_FORMATS, FORMATS, Transcript

if TYPE_CHECKING:
    from datetime import datetime

    from stenograf.settings import Settings


def _cli_settings():
    """Load settings.toml once, at command start, as a clean CLI error.

    Every ``start``/``transcribe``/``notes`` invocation resolves its defaults
    from here — and loading up front means a broken file fails *before* an
    hour of capture, not when the finalize (or notes) step first reads it."""
    from stenograf.settings import SettingsError, load_settings

    try:
        return load_settings()
    except SettingsError as exc:
        raise click.ClickException(str(exc)) from exc


def _resolve_formats(spec: str | None, settings) -> list[str]:
    """``--format`` > ``[transcript] formats`` > the built-in default."""
    if spec is not None:
        return _parse_formats(spec)
    return list(settings.transcript.formats or DEFAULT_FORMATS)


@dataclass(frozen=True)
class _RunConfig:
    """Flag+settings resolution shared by ``start`` and ``transcribe``.

    One place applies the standard order (flag beats settings.toml beats
    built-in default) to everything both commands consume. The settings-derived
    ``reid_store`` feeds re-ID loading only — never the MeetingProfile, which
    serializes into every transcript, and the settings file's whole point is
    keeping machine-local paths out of shared files. An explicit
    ``--profile-store`` is recorded on the profile by the caller, as before."""

    settings: Settings
    write_formats: list[str]
    glossary_terms: tuple[str, ...]
    attendee_names: tuple[str, ...]
    glossary_threshold: float | None
    reid_threshold: float | None
    reid_store: Path | None


def _resolve_run_config(
    *,
    formats: str | None,
    glossary: tuple[str, ...],
    glossary_file: Path | None,
    attendee: tuple[str, ...],
    glossary_threshold: float | None,
    reid_threshold: float | None,
    profile_store: Path | None,
) -> _RunConfig:
    settings = _cli_settings()
    glossary_terms, attendee_names = _collect_terms(
        glossary, glossary_file, attendee, vocab=settings.vocab
    )
    return _RunConfig(
        settings=settings,
        write_formats=_resolve_formats(formats, settings),
        glossary_terms=glossary_terms,
        attendee_names=attendee_names,
        glossary_threshold=(
            settings.vocab.glossary_threshold if glossary_threshold is None else glossary_threshold
        ),
        reid_threshold=(
            settings.speakers.reid_threshold if reid_threshold is None else reid_threshold
        ),
        reid_store=profile_store or settings.speakers.profile_store,
    )


def _finish_run(
    transcript: Transcript,
    out_dir: Path,
    basename: str,
    *,
    created_at: datetime,
    settings,
    notes_flag: bool,
    print_markdown: bool,
) -> None:
    """The tail both commands share: optional notes, optional stdout print."""
    from stenograf.cli.notes import _notes_after_run

    if notes_flag:
        _notes_after_run(
            transcript,
            out_dir,
            basename,
            created_at=created_at,
            notes_settings=settings.notes,
        )
    if print_markdown:
        click.echo()
        click.echo(transcript.to_markdown(), nl=False)


def _parse_formats(spec: str) -> list[str]:
    """Parse a ``--format`` value (comma-separated) into an ordered, de-duped list."""
    formats: list[str] = []
    for name in spec.split(","):
        name = name.strip().lower()
        if not name or name in formats:
            continue
        if name not in FORMATS:
            raise click.BadParameter(
                f"unknown format {name!r}; choose from {', '.join(FORMATS)}",
                param_hint="--format",
            )
        formats.append(name)
    if not formats:
        raise click.BadParameter("no formats given", param_hint="--format")
    return formats


def _apply_no_diarization(
    enabled: bool, local_speakers: int | None, remote_speakers: int | None
) -> tuple[int | None, int | None]:
    """Coerce the per-channel speaker counts to 1 for ``--no-diarization``.

    A count of 1 is the pipeline's existing diarizer-free path: the model is
    never loaded and every word on the channel lands on one label. An explicit
    0 (channel off) is preserved; a count above 1 contradicts the flag.
    """
    if not enabled:
        return local_speakers, remote_speakers
    if (local_speakers or 0) > 1 or (remote_speakers or 0) > 1:
        raise click.UsageError("--no-diarization conflicts with a speaker count above 1")
    return (0 if local_speakers == 0 else 1, 0 if remote_speakers == 0 else 1)


def _vocab_options(func: Callable) -> Callable:
    """Shared glossary/attendee/re-ID-store options for ``start`` and ``transcribe``.

    The finalize pass has no decode-time biasing (Parakeet), so these drive the
    deterministic text post-correction in ``stenograf.glossary`` (PLAN.md Task 2b).
    """
    for option in reversed(
        (
            click.option(
                "--glossary",
                multiple=True,
                metavar="TERMS",
                help="Domain term(s) to snap the transcript to; repeatable and comma-separated.",
            ),
            click.option(
                "--glossary-file",
                type=click.Path(exists=True, dir_okay=False, path_type=Path),
                default=None,
                help="File of glossary terms, one per line (# comments and blank lines ignored).",
            ),
            click.option(
                "--attendee",
                multiple=True,
                metavar="NAMES",
                help="Attendee name(s) to correct (also token-by-token); repeatable + comma-list.",
            ),
            click.option(
                "--glossary-threshold",
                type=click.FloatRange(0, 1),
                default=None,
                help="Similarity 0–1 required to correct a term "
                "[default: [vocab] glossary_threshold in settings.toml, else 0.82].",
            ),
            click.option(
                "--profile-store",
                type=click.Path(dir_okay=False, path_type=Path),
                default=None,
                help="Use this re-ID profile store instead of the default location "
                "([speakers] profile_store in settings.toml also sets this).",
            ),
        )
    ):
        func = option(func)
    return func


def _collect_terms(
    glossary: tuple[str, ...],
    glossary_file: Path | None,
    attendee: tuple[str, ...],
    *,
    vocab=None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Gather glossary terms (inline + file) and attendee names from the options.

    ``vocab`` (the ``[vocab]`` settings table) is the standing baseline: its
    glossary file and attendees come first and per-run ``--glossary``/
    ``--glossary-file``/``--attendee`` values *merge* on top — configuring a
    vocabulary must never make the flags stop working, or vice versa. Inline
    values may each be comma-separated; a file is one term per line. Both lists
    are de-duplicated preserving first-seen order.
    """
    terms: list[str] = []
    names: list[str] = []
    if vocab is not None:
        if vocab.glossary_file is not None:
            terms.extend(_read_glossary_lines(vocab.glossary_file, source="[vocab] glossary_file"))
        names.extend(vocab.attendees)
    for value in glossary:
        terms.extend(part.strip() for part in value.split(",") if part.strip())
    if glossary_file is not None:
        terms.extend(_read_glossary_lines(glossary_file))
    for value in attendee:
        names.extend(part.strip() for part in value.split(",") if part.strip())
    return tuple(dict.fromkeys(terms)), tuple(dict.fromkeys(names))


def _read_glossary_lines(path: Path, *, source: str | None = None) -> list[str]:
    """Terms from a glossary file (# comments and blank lines ignored).

    ``source`` names the setting that configured the path — the CLI flag
    validates existence itself (``exists=True``), but a stale path in
    settings.toml must say where it came from, not just fail to open."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        where = f" ({source} in settings.toml)" if source else ""
        raise click.ClickException(f"cannot read glossary file {path}{where}: {exc}") from exc
    terms = []
    for raw_line in raw.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            terms.append(line)
    return terms


def _prepare_output(
    out: Path | None, created_at: datetime, settings, *, force: bool = False
) -> tuple[Path, str, Path]:
    """Resolve the directory this run's files land in.

    Returns ``(out_dir, basename, audio_default)``. By default the meeting gets
    a fresh date-named folder under the visible output home (``[output] dir``
    in settings.toml, else ``~/Documents/Meetings``); ``--out`` uses that path
    itself as the meeting's folder. Either way the files inside are plainly
    named — ``transcript.{fmt}``, ``audio.wav`` (PLAN.md §5 Stage C1).

    File names inside a meeting folder are fixed, so pointing ``--out`` at a
    folder that already holds a transcript would silently replace that meeting;
    refuse unless ``--force`` says overwriting is the point (a re-run over the
    same recording). The default path allocates a fresh name and cannot collide;
    ``.partial`` checkpoints don't count — resuming after a crash must not
    demand ``--force``."""
    from stenograf.output import (
        AUDIO_NAME,
        TRANSCRIPT_STEM,
        allocate_meeting_dir,
        default_output_home,
    )

    if out is not None:
        if not force:
            existing = next(
                (
                    f"{TRANSCRIPT_STEM}.{ext}"
                    for ext in FORMATS
                    if (out / f"{TRANSCRIPT_STEM}.{ext}").exists()
                ),
                None,
            )
            if existing is not None:
                raise click.ClickException(
                    f"{out} already holds {existing} — pass --force to overwrite "
                    "this meeting's files, or drop --out for a fresh folder"
                )
        out_dir = out
    else:
        out_dir = allocate_meeting_dir(settings.output.dir or default_output_home(), created_at)
    return out_dir, TRANSCRIPT_STEM, out_dir / AUDIO_NAME
