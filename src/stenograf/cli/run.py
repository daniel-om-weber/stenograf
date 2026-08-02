"""Flag+settings resolution, the command tail shared by ``start`` and
``transcribe``, and the boundary that maps library errors to CLI errors."""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import click

from stenograf.capture.base import CaptureHelperError, CaptureUnavailableError
from stenograf.loaders import BackendUnavailableError
from stenograf.settings import (
    SettingsError,
    UnknownPresetError,
    apply_meeting_preset,
    load_settings,
)
from stenograf.transcript import DEFAULT_FORMATS, FORMATS, Transcript
from stenograf.vocab import collect_terms

if TYPE_CHECKING:
    from datetime import datetime

    from stenograf.diarization.base import Diarizer
    from stenograf.profiles import SpeakerReID
    from stenograf.settings import MeetingPreset, Settings

def _library_errors[T](func: Callable[..., T]) -> Callable[..., T]:
    """The command boundary: the library's typed failures become clean CLI errors.

    The library raises its own types — :class:`SettingsError` (a broken
    settings.toml, a stale glossary path), :class:`BackendUnavailableError`,
    :class:`CaptureUnavailableError`/:class:`CaptureHelperError`,
    :class:`FileExistsError` from :func:`~stenograf.output.prepare_output` —
    and every command maps them here, once, instead of per call site.
    :class:`UnknownPresetError` is bad command-line input, hence a UsageError.
    Anything else is a bug and keeps its traceback."""

    @functools.wraps(func)
    def wrapper(*args: object, **kwargs: object) -> T:
        try:
            return func(*args, **kwargs)
        except UnknownPresetError as exc:
            raise click.UsageError(str(exc)) from exc
        except (
            SettingsError,
            BackendUnavailableError,
            CaptureUnavailableError,
            CaptureHelperError,
            FileExistsError,
        ) as exc:
            raise click.ClickException(str(exc)) from exc

    return wrapper


def _resolve_formats(spec: str | None, settings: Settings) -> list[str]:
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
    preset: MeetingPreset | None = None
    """The applied ``--preset``, for the values that never enter ``Settings``:
    ``title``/``language`` are form defaults a typed value beats, and the
    preset's notes backend must be passed *explicitly* to ``create_backend``
    (a CLI-level choice beats ``STENOGRAF_NOTES_BACKEND``, which would
    otherwise win over the overlaid table)."""


def _resolve_run_config(
    *,
    formats: str | None,
    glossary: tuple[str, ...],
    glossary_file: Path | None,
    attendee: tuple[str, ...],
    glossary_threshold: float | None,
    reid_threshold: float | None,
    profile_store: Path | None,
    preset: str | None = None,
) -> _RunConfig:
    # Loading up front means a broken settings.toml fails *before* an hour of
    # capture, not when the finalize (or notes) step first reads it; the
    # _library_errors boundary turns it into a clean CLI error.
    settings = load_settings()
    preset_obj = None
    if preset is not None:
        settings, preset_obj = apply_meeting_preset(settings, preset)
        click.echo(f"preset: {preset}")  # echo-on-use: the typo mitigation
    # The preset must enter HERE, before collect_terms — its [vocab] feeds
    # decode-time biasing, not just the notes prompt.
    glossary_terms, attendee_names = collect_terms(
        glossary,
        glossary_file,
        attendee,
        vocab=settings.vocab,
        extra_vocab=preset_obj.vocab if preset_obj is not None else None,
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
        preset=preset_obj,
    )


def _notes_enabled(notes_flag: bool | None, settings: Settings) -> bool:
    """Whether this run generates notes: ``--notes`` asked for them, or — with
    no flag either way — ``[notes] auto`` is on; ``--no-notes`` skips them even
    then. One seam, so ``start``'s in-TUI notes and :func:`_finish_run` can
    never disagree."""
    return notes_flag if notes_flag is not None else settings.notes.auto is True


def _finish_run(
    transcript: Transcript,
    out_dir: Path,
    basename: str,
    *,
    created_at: datetime,
    settings: Settings,
    notes_flag: bool | None,
    print_markdown: bool,
    notes_backend: str | None = None,
    notes_model: str | None = None,
    notes_instructions: Path | None = None,
) -> None:
    """The tail both commands share: optional notes, optional stdout print.

    Notes run per :func:`_notes_enabled`; the per-run trio overrides the
    ``[notes]`` snapshot for this run only."""
    from stenograf.cli.notes import _notes_after_run

    if _notes_enabled(notes_flag, settings):
        _notes_after_run(
            transcript,
            out_dir,
            basename,
            created_at=created_at,
            notes_settings=settings.notes,
            backend_name=notes_backend,
            model=notes_model,
            instructions_file=notes_instructions,
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


def _resolve_diarization(flag: bool | None, setting: bool | None, *counts: int | None) -> bool:
    """Whether diarization runs this run: flag > explicit count > settings > off.

    The per-run ``--diarization/--no-diarization`` flag wins outright (an
    explicit ``--no-diarization`` beside a count above 1 stays the
    :func:`_apply_no_diarization` UsageError). Without a flag, an explicit
    speaker count above 1 is itself a request to diarize, so it beats the
    settings value; otherwise that value decides, and the built-in default
    is off — each channel is one speaker and the diarizer is never loaded.
    """
    if flag is not None:
        return flag
    if any((count or 0) > 1 for count in counts):
        return True
    return setting if setting is not None else False


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

    These terms drive both vocabulary layers: they boost the decoder as it transcribes
    (``stenograf.asr.biasing``) and then snap the near-misses it still got wrong
    (``stenograf.glossary``).
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
                "[default: [vocab] glossary_threshold in settings.toml, else 0.95].",
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


def _reid_format_options(func: Callable) -> Callable:
    """Shared re-ID and ``--format`` options for ``start`` and ``transcribe``."""
    for option in reversed(
        (
            click.option(
                "--reid/--no-reid",
                "use_reid",
                default=True,
                help="Relabel diarized speakers to saved profile names when their voice matches "
                "(cross-meeting re-identification). No effect without enrolled profiles.",
            ),
            click.option(
                "--reid-threshold",
                type=click.FloatRange(0, 1),
                default=None,
                help="Cosine similarity required to match a saved profile "
                "[default: [speakers] reid_threshold in settings.toml, else 0.5].",
            ),
            click.option(
                "--format",
                "formats",
                default=None,
                metavar="LIST",
                help="Comma-separated transcript formats to write: md, json, txt, srt, vtt "
                "[default: [transcript] formats in settings.toml, else md,json,txt]. txt is "
                "plain prose without speakers or timestamps; srt/vtt re-flow speaker turns "
                "into subtitle cues.",
            ),
        )
    ):
        func = option(func)
    return func


def _notes_options(func: Callable) -> Callable:
    """Shared post-transcript options for ``start`` and ``transcribe``.

    The per-run notes trio (backend/model/instructions) exists so one meeting
    can use a different notes setup — say, an agentic CLI with a
    protocol-style file — without editing settings.toml. They only steer the
    notes step of *this* run; ``steno notes`` has its own equivalents."""
    for option in reversed(
        (
            click.option(
                "--notes/--no-notes",
                "notes_flag",
                default=None,
                help="After the transcript is written, generate LLM meeting notes "
                "(summary, decisions, action items) with the backend configured in "
                "settings.toml. Non-fatal: a notes failure never loses the transcript "
                "[default: [notes] auto in settings.toml, else off].",
            ),
            click.option(
                "--notes-backend",
                "notes_backend",
                default=None,
                metavar="NAME",
                help="Notes backend for this run: mlx (local, in-process), ollama "
                "(local server), or command (any CLI, e.g. claude) "
                "[default: settings.toml, else mlx where installed, else ollama].",
            ),
            click.option(
                "--notes-model",
                "notes_model",
                default=None,
                metavar="NAME",
                help="Notes model for this run (HF repo id for mlx, Ollama model tag "
                "for ollama; a provenance label for command backends).",
            ),
            click.option(
                "--instructions",
                "notes_instructions",
                type=click.Path(exists=True, dir_okay=False, path_type=Path),
                default=None,
                metavar="FILE",
                help="Style/structure instructions appended to the notes prompt for "
                "this run [default: [notes] instructions in settings.toml].",
            ),
            click.option(
                "--print", "print_markdown", is_flag=True, help="Also print the transcript."
            ),
        )
    ):
        func = option(func)
    return func


def _load_reid(
    diarizer: Diarizer | None, *, enabled: bool, threshold: float | None, store: Path | None
) -> SpeakerReID | None:
    """Load the re-ID matcher when diarization runs, echoing its profile count."""
    from stenograf import loaders

    if diarizer is None:
        return None
    reid = loaders.load_reid(enabled=enabled, threshold=threshold, store_path=store)
    if reid is not None:
        click.echo(f"re-ID: {len(reid.store.for_model(reid.model))} profile(s) active")
    return reid


def _echo_glossary(terms: tuple[str, ...], names: tuple[str, ...]) -> None:
    if terms or names:
        click.echo(f"glossary: {len(terms)} term(s), {len(names)} name(s)")
