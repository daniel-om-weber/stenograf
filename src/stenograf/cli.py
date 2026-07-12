"""Command-line interface: ``stenograf`` / ``steno``."""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    import numpy as np

    from stenograf.settings import Settings

from stenograf import __version__, loaders
from stenograf.config import Language, MeetingProfile
from stenograf.doctor import run_checks
from stenograf.output import atomic_write_text
from stenograf.transcript import DEFAULT_FORMATS, FORMATS, Transcript

# Sentinel for --record-audio given without a value (write next to the transcript).
_RECORD_DEFAULT = "\0default"
# Crash checkpoints render these (no subtitles — pointless for a partial
# transcript). _cleanup_checkpoints must remove exactly this set.
_CHECKPOINT_FORMATS = ("md", "json", "txt")

# Settable speaker-count ranges, kept in sync with the --local/--remote and
# --speakers IntRange bounds. The unconstrained diarizer can *detect* more (or, on
# silence, zero) speakers than the user can set, so the "lock the detected count"
# hint is clamped to these — never suggesting an out-of-range or nonsensical re-run.
_MEETING_MAX_SPEAKERS = 8
_FILE_MAX_SPEAKERS = 16

# --flush-interval defaults, sized to what one checkpoint costs per mode.
_LIVE_FLUSH_INTERVAL_S = 15.0
_BATCH_FLUSH_INTERVAL_S = 180.0


def _resolve_flush_interval(value: float | None, *, live: bool) -> float:
    """The ``--flush-interval`` default is sized to what a checkpoint costs.

    A live checkpoint is zero-inference — it snapshots the captions the live
    pass already committed, a few KB of atomic file I/O — so it can afford to
    be tight (a crash loses seconds of text, not minutes). A batch
    (``--no-live``) checkpoint runs VAD+ASR over the new tail, so it stays
    sparse to keep that mode's near-zero-power promise. An explicit value
    (including 0 = disabled) wins in both modes.
    """
    if value is not None:
        return value
    return _LIVE_FLUSH_INTERVAL_S if live else _BATCH_FLUSH_INTERVAL_S


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


@click.group()
@click.version_option(__version__, prog_name="stenograf")
def main() -> None:
    """Accuracy-first local meeting transcription. Audio never touches disk."""
    # Windows pipes/redirects default to the legacy code page (cp1252), and a
    # single ✓/← in our output would then crash click.echo with a
    # UnicodeEncodeError. Degrade unencodable glyphs to "?" instead; the
    # interactive console is unaffected (it is UTF-16 under the hood), as are
    # the output files (written encoding="utf-8" throughout).
    for stream in (sys.stdout, sys.stderr):
        if sys.platform == "win32" and hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")


@main.command()
@click.option(
    "--lang",
    type=click.Choice([lang.value for lang in Language]),
    default=None,
    help="Meeting language; omit to auto-detect from the first speech.",
)
@click.option(
    "--local",
    "local_speakers",
    type=click.IntRange(0, _MEETING_MAX_SPEAKERS),
    default=None,
    help="Number of speakers in the room; omit to auto-detect.",
)
@click.option(
    "--remote",
    "remote_speakers",
    type=click.IntRange(0, _MEETING_MAX_SPEAKERS),
    default=None,
    help="Number of remote speakers; 0 = in-room meeting without system audio.",
)
@click.option(
    "--no-diarization",
    "no_diarization",
    is_flag=True,
    help="Skip speaker diarization: the diarizer model is never loaded and each "
    "captured channel is attributed to a single speaker (Local-1/Remote-1). "
    "Conflicts with a --local/--remote count above 1.",
)
@click.option(
    "--replay",
    "replay",
    default=None,
    metavar="MIC[,SYSTEM]",
    help="Dev: replay audio file(s) as the mic (and optional system) channel "
    "instead of live capture. Exercises the full finalize pipeline without the "
    "native capture helper.",
)
@click.option(
    "--out",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Use this directory as the meeting's folder instead of creating a "
    "date-named one under the output home ([output] dir in settings.toml, "
    "else ~/Documents/Meetings). Refuses a directory that already holds a "
    "transcript unless --force.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Let --out overwrite an existing transcript (e.g. re-processing the same meeting).",
)
@click.option(
    "--title",
    default=None,
    metavar="TEXT",
    help="A human-readable title for this meeting (recorded in the transcript "
    "and used by the notes prompt and the combined-note export).",
)
@click.option(
    "--record-audio",
    "record_audio",
    is_flag=False,
    flag_value=_RECORD_DEFAULT,
    default=None,
    metavar="[PATH]",
    help="Also save the raw captured audio to a WAV (mic left, system right). "
    "Off by default — audio otherwise never touches disk. Give a PATH or omit it "
    "to write <transcript>.wav.",
)
@click.option(
    "--flush-interval",
    "--checkpoint-interval",
    "flush_interval",
    type=click.FloatRange(0),
    default=None,
    metavar="SECONDS",
    help="Flush a <transcript>.partial crash checkpoint every N seconds of capture "
    "(live: the already-committed captions, zero extra inference — default 15; "
    "batch: only the newest tail is finalized, off the capture thread — "
    "default 180); 0 disables it.",
)
@click.option(
    "--max-seconds",
    type=click.FloatRange(0, min_open=True),
    default=None,
    metavar="SECONDS",
    help="Stop capture automatically after this many seconds [default: until Ctrl-C].",
)
@click.option(
    "--live/--no-live",
    default=True,
    help="Stream live captions while the meeting runs (the on-stop finalize still "
    "replaces them). --no-live captures silently and only finalizes on stop.",
)
@click.option(
    "--plain",
    is_flag=True,
    help="Force the plain line-by-line caption stream instead of the full-screen "
    "TUI (also the automatic choice when stdout is not a terminal).",
)
@click.option(
    "--aec/--no-aec",
    "use_aec",
    default=True,
    help="Cancel speaker bleed out of the mic, using the system channel as the "
    "reference. Only applies when both channels are captured; harmless on "
    "headphones. Disable to capture the mic exactly as the device hears it — "
    "this also disables the cross-channel echo dedup at merge time, so no mic "
    "line is ever dropped.",
)
@click.option(
    "--aec-dump",
    "aec_dump",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    metavar="DIR",
    help="Write the echo canceller's mic/lpb/enh WAV triple to DIR for offline "
    "scoring (eval/aec_score.py). Writes meeting audio to disk, like "
    "--record-audio. With --no-aec the triple records the uncancelled baseline.",
)
@click.option(
    "--reid/--no-reid",
    "use_reid",
    default=True,
    help="Relabel diarized speakers to saved profile names when their voice matches "
    "(cross-meeting re-identification). No effect without enrolled profiles.",
)
@click.option(
    "--reid-threshold",
    type=click.FloatRange(0, 1),
    default=None,
    help="Cosine similarity required to match a saved profile "
    "[default: [speakers] reid_threshold in settings.toml, else 0.5].",
)
@click.option(
    "--format",
    "formats",
    default=None,
    metavar="LIST",
    help="Comma-separated transcript formats to write: md, json, txt, srt, vtt "
    "[default: [transcript] formats in settings.toml, else md,json,txt]. txt is "
    "plain prose without speakers or timestamps; srt/vtt re-flow speaker turns "
    "into subtitle cues.",
)
@_vocab_options
@click.option(
    "--full-finalize",
    is_flag=True,
    help="Re-transcribe everything at stop instead of reusing the live window "
    "pass's decodes. The live pass already decodes the exact windows the "
    "finalize pass would (so reuse is the default); this forces the "
    "from-scratch ASR pass for A/B comparison or paranoia.",
)
@click.option(
    "--notes",
    "notes_flag",
    is_flag=True,
    help="After the transcript is written, generate LLM meeting notes "
    "(summary, decisions, action items) with the backend configured in "
    "settings.toml. Non-fatal: a notes failure never loses the transcript.",
)
@click.option("--print", "print_markdown", is_flag=True, help="Also print the transcript.")
def start(
    lang: str | None,
    local_speakers: int | None,
    remote_speakers: int | None,
    no_diarization: bool,
    replay: str | None,
    out: Path | None,
    force: bool,
    title: str | None,
    record_audio: str | None,
    flush_interval: float | None,
    max_seconds: float | None,
    live: bool,
    plain: bool,
    use_aec: bool,
    aec_dump: Path | None,
    use_reid: bool,
    reid_threshold: float | None,
    formats: str | None,
    glossary: tuple[str, ...],
    glossary_file: Path | None,
    attendee: tuple[str, ...],
    glossary_threshold: float | None,
    profile_store: Path | None,
    full_finalize: bool,
    notes_flag: bool,
    print_markdown: bool,
) -> None:
    """Start transcribing a meeting (capture → finalize on stop)."""
    from stenograf.session import MeetingRecorder, plan_channels

    cfg = _resolve_run_config(
        formats=formats,
        glossary=glossary,
        glossary_file=glossary_file,
        attendee=attendee,
        glossary_threshold=glossary_threshold,
        reid_threshold=reid_threshold,
        profile_store=profile_store,
    )
    settings, write_formats = cfg.settings, cfg.write_formats
    glossary_terms, attendee_names = cfg.glossary_terms, cfg.attendee_names
    glossary_threshold, reid_threshold = cfg.glossary_threshold, cfg.reid_threshold
    reid_store = cfg.reid_store

    local_speakers, remote_speakers = _apply_no_diarization(
        no_diarization, local_speakers, remote_speakers
    )
    try:
        profile = MeetingProfile(
            language=Language(lang) if lang else None,
            local_speakers=local_speakers,
            remote_speakers=remote_speakers,
            glossary=glossary_terms,
            attendee_names=attendee_names,
            speaker_profile_store=profile_store,
            title=title,
        )
    except ValueError as exc:  # e.g. --local 0 --remote 0 — report cleanly, not a traceback
        raise click.ClickException(str(exc)) from exc
    mode = profile.mode.value if profile.mode else "auto"
    click.echo(f"profile: language={profile.language or 'auto'} mode={mode}")

    plans = plan_channels(profile)
    # Pace file replay to wall-clock only when it feeds the live pass, so
    # `--replay` demonstrates captions at meeting cadence; batch just dumps it.
    provider = loaders.make_provider(replay, plans, paced=live, aec=use_aec, aec_dump=aec_dump)
    if aec_dump is not None:
        from stenograf.aec import EchoCancellingProvider

        if isinstance(provider, EchoCancellingProvider):
            click.secho(
                f"● AEC DUMP to {aec_dump} — mic/lpb/enh audio is being written to disk",
                fg="red",
                bold=True,
            )
        else:
            click.secho(
                "--aec-dump ignored: it needs both the mic and the system channel",
                fg="yellow",
            )

    # Every meeting gets its own date-named folder in the visible output home
    # (or --out as the folder), holding transcript.{md,json,…} + optional
    # audio.wav — self-describing files, no index (PLAN.md §5 Stage C).
    created_at = datetime.now()
    out_dir, basename, audio_default = _prepare_output(out, created_at, settings, force=force)

    started = time.monotonic()
    asr, vad, diarizer = loaders.load_backends(
        need_diarizer=any(p.num_speakers != 1 for p in plans),
        asr_backend=settings.asr.backend,
        asr_provider=settings.asr.provider,
    )
    reid = (
        loaders.load_reid(enabled=use_reid, threshold=reid_threshold, store_path=reid_store)
        if diarizer is not None
        else None
    )
    if reid is not None:
        click.echo(f"re-ID: {len(reid.store.for_model(reid.model))} profile(s) active")
    if glossary_terms or attendee_names:
        click.echo(f"glossary: {len(glossary_terms)} term(s), {len(attendee_names)} name(s)")
    recorder = MeetingRecorder(
        profile,
        asr=asr,
        vad=vad,
        diarizer=diarizer,
        reid=reid,
        language=profile.language,
        glossary_threshold=glossary_threshold,
        dedup_echo=use_aec,
    )
    recorder.reuse_live_finalize = not full_finalize

    tee = _make_tee(record_audio, audio_default, plans)
    flush_interval = _resolve_flush_interval(flush_interval, live=live)

    def _persist_files(transcript: Transcript) -> list[Path]:
        """Write the transcript files and drop the ``.partial`` checkpoint."""
        paths = _write_transcript(transcript, out_dir, basename, write_formats)
        _cleanup_checkpoints(out_dir, basename)
        return paths

    persist = _PersistOnce(_persist_files)

    # The full-screen TUI owns the terminal, so it can only run on a real TTY and
    # unless the user forced the plain stream (or turned live off entirely).
    use_tui = live and not plain and _stdout_is_tty()
    channels = ", ".join(p.channel.value for p in plans)
    if not use_tui:  # the TUI header shows REC / elapsed instead of this hint
        stop_hint = f"stops after {max_seconds:g}s" if max_seconds else "press Ctrl-C to stop"
        click.echo(f"capturing: {channels} ({stop_hint} and transcribe)")
    if len(plans) > 1:
        state = "on" if use_aec else "off"
        click.echo(f"echo cancellation: {state} (mic cancelled against system audio)")
    try:
        transcript = _run_meeting(
            recorder,
            provider,
            live=live,
            use_tui=use_tui,
            profile=profile,
            on_frame=tee.add if tee else None,
            out_dir=out_dir,
            basename=basename,
            flush_interval=flush_interval,
            max_seconds=max_seconds,
            persist=persist,
        )
    finally:
        if tee is not None:
            tee.close()
            click.echo(f"recorded audio: {tee.path}")

    # The canceller counts every 10 ms mic tick it had to cancel against silence
    # because the system reference never arrived. A stalled tap degrades to "no
    # cancellation" by design — but silently, so say how much of the meeting ran
    # unprotected, and whether the armed text backstop had to clean up after it.
    canceller = getattr(provider, "canceller", None)
    if canceller is not None and canceller.far_end_missing_ticks > 0:
        if recorder.dropped_echo_lines:
            backstop = (
                f"; the text backstop removed {recorder.dropped_echo_lines} mic "
                "line(s) that duplicated remote speech"
            )
        else:
            backstop = "; review Local lines in those spans for leaked remote speech"
        click.secho(
            f"echo cancellation ran without its reference for "
            f"{canceller.far_end_missing_ticks / 100:.1f}s — the system-audio tap "
            f"stalled{backstop}",
            fg="yellow",
        )

    if transcript is None:
        # Defensive: a live view exited without producing a transcript. There is
        # nothing authoritative to write; leave any .partial checkpoint in place
        # for recovery rather than deleting it or crashing on None.
        raise click.ClickException(
            "meeting ended before a transcript was produced; any .partial checkpoint is kept"
        )

    # Usually already persisted at the ``finalized`` event (the TUI path writes
    # while the app still shows the "done" screen); this is the no-op replay
    # then, and the write for the plain/batch paths — or the retry if the
    # event-time write failed, surfacing the error as a normal CLI error here.
    paths = persist(transcript)
    elapsed = time.monotonic() - started
    _report_speaker_counts(recorder.speaker_counts)
    click.echo(f"wrote {', '.join(p.name for p in paths)} → {out_dir} ({elapsed:.1f}s)")
    _finish_run(
        transcript,
        out_dir,
        basename,
        created_at=created_at,
        settings=settings,
        notes_flag=notes_flag,
        print_markdown=print_markdown,
    )


def _describe_channel(channel) -> tuple[str, str]:
    """The human name and CLI flag for a channel's speaker count."""
    from stenograf.capture.base import Channel

    return ("local", "--local") if channel is Channel.MIC else ("remote", "--remote")


def _report_speaker_counts(counts) -> None:
    """Print per-channel speaker counts, flagging estimated ones as editable.

    Explicit counts are echoed as given; an auto-detected count shows what the
    finalize found and the exact flag to lock or correct it by re-running over
    the retained/recorded audio (PLAN.md §5 Stage 3a — a wrong estimate is never
    fatal, just re-run finalize)."""
    if not counts:
        click.echo("speakers: none found")
        return
    parts, corrections = [], []
    capped = False
    for count in counts:
        name, flag = _describe_channel(count.channel)
        if count.requested is None:
            parts.append(f"{count.detected} {name} (detected)")
            hint = _lock_hint(count.detected, _MEETING_MAX_SPEAKERS)
            if hint is not None:  # None → nothing to lock (a silent channel, 0 found)
                value, was_capped = hint
                corrections.append(f"{flag} {value}")
                capped = capped or was_capped
        else:
            parts.append(f"{count.requested} {name} (given)")
    click.echo("speakers: " + ", ".join(parts))
    if corrections:
        note = f" (estimate exceeded the {_MEETING_MAX_SPEAKERS}-speaker max)" if capped else ""
        click.echo(f"  estimated — re-run with {' '.join(corrections)} to lock or correct{note}")


def _lock_hint(detected: int, max_settable: int) -> tuple[int, bool] | None:
    """The value to suggest for locking an estimated count, clamped to the settable
    range, or ``None`` when there is nothing sensible to lock.

    Returns ``(value, capped)``: ``value`` is ``detected`` clamped into
    ``[1, max_settable]`` and ``capped`` flags that the raw estimate exceeded that
    range (an over-cluster artifact of unconstrained clustering — the displayed
    count stays the raw estimate; only the suggested lock value is capped).
    ``None`` when no speaker was found (``detected < 1``), so a silent channel never
    produces a nonsensical ``--local 0`` hint (PLAN.md §5 Phase 3→4 audit)."""
    if detected < 1:
        return None
    if detected > max_settable:
        return max_settable, True
    return detected, False


def _run_meeting(
    recorder,
    provider,
    *,
    live: bool,
    use_tui: bool,
    profile: MeetingProfile,
    on_frame,
    out_dir: Path,
    basename: str,
    flush_interval: float,
    max_seconds: float | None,
    persist: Callable[[Transcript], object] | None = None,
) -> Transcript:
    """Run the capture session through the right live view and return the transcript.

    Three shapes behind one call:

    - **TUI** (live, on a TTY, not ``--plain``): the Textual view runs the app on
      this thread while the meeting runs on a background thread; its quit binding
      crosses to ``provider.stop`` to end capture. Checkpoints are written silently
      (the TUI owns the screen). ``persist`` is wired into the view's ``finalized``
      event, so the transcript reaches disk while the app still shows the "done"
      screen — only the TUI has a gap between finalize and return worth closing;
      the other two shapes return immediately and the caller persists then.
    - **Plain live** (live, no TTY or ``--plain``): the meeting runs on this thread
      and streams committed captions to stdout; checkpoints written silently.
    - **Batch** (``--no-live``): no live pass; status and checkpoint notices echo
      as before.
    """
    if use_tui:
        from stenograf.tui import TextualLiveView

        view = TextualLiveView(
            profile, language=profile.language, stop=provider.stop, persist=persist
        )
        return view.serve(
            lambda: recorder.run(
                provider,
                live=True,
                view=view,
                on_frame=on_frame,
                on_checkpoint=_checkpoint_writer(out_dir, basename),
                checkpoint_interval=flush_interval,
                max_seconds=max_seconds,
            )
        )
    if live:
        from stenograf.view import PlainLiveView

        with PlainLiveView() as view:
            return recorder.run(
                provider,
                live=True,
                view=view,
                on_frame=on_frame,
                on_checkpoint=_checkpoint_writer(out_dir, basename),
                checkpoint_interval=flush_interval,
                max_seconds=max_seconds,
            )
    return recorder.run(
        provider,
        on_frame=on_frame,
        on_status=lambda msg: click.echo(f"  {msg}"),
        on_checkpoint=_checkpoint_writer(
            out_dir, basename, announce=lambda m: click.echo(f"  {m}")
        ),
        checkpoint_interval=flush_interval,
        max_seconds=max_seconds,
    )


def _stdout_is_tty() -> bool:
    """Whether stdout is an interactive terminal (a seam so the view choice is testable)."""
    return sys.stdout.isatty()


class _PersistOnce:
    """Persist the finalized transcript exactly once, wherever that fires first.

    The TUI wires this into the ``finalized`` event — the moment the
    authoritative transcript exists, while the app still sits on the "done"
    screen — so a crash or force-quit before the user presses ``q`` no longer
    loses the meeting. Every path also calls it after the meeting returns;
    that second call is a no-op returning the already-written paths. A failure
    at the event leaves ``paths`` unset, so the exit-path call retries and a
    raise there surfaces as a normal CLI error. Calls are sequential (the
    meeting thread is joined before the CLI tail runs), so no lock is needed.
    """

    def __init__(self, write: Callable[[Transcript], list[Path]]) -> None:
        self._write = write
        self.paths: list[Path] | None = None

    def __call__(self, transcript: Transcript) -> list[Path]:
        if self.paths is None:
            self.paths = self._write(transcript)
        return self.paths


def _checkpoint_writer(
    out_dir: Path, basename: str, announce: Callable[[str], None] | None = None
) -> Callable[[Transcript], None]:
    """Build the ``on_checkpoint`` sink that writes the ``.partial`` crash file.

    Live views keep the caption stream clean (``announce=None`` → write silently);
    the batch path narrates each write, as it always has. The final transcript
    supersedes these files, which ``_cleanup_checkpoints`` then removes.
    """

    def on_checkpoint(transcript: Transcript) -> None:
        md = _write_transcript(transcript, out_dir, f"{basename}.partial", _CHECKPOINT_FORMATS)[0]
        if announce is not None:
            announce(f"checkpoint: {md.name} ({len(transcript.entries)} entries)")

    return on_checkpoint


def _make_tee(record_audio: str | None, default_path: Path, plans):
    """Create the audio tee if --record-audio was given, with a loud banner.

    ``default_path`` is where a bare ``--record-audio`` (no value) writes — the
    meeting folder's ``audio.wav``; an explicit ``--record-audio PATH``
    overrides it.
    """
    if record_audio is None:
        return None
    from stenograf.recording import WavTee

    path = default_path if record_audio == _RECORD_DEFAULT else Path(record_audio)
    path.parent.mkdir(parents=True, exist_ok=True)
    tee = WavTee(path, {p.channel for p in plans})
    click.secho(
        f"● RECORDING AUDIO to {path} — raw audio is being written to disk",
        fg="red",
        bold=True,
    )
    return tee


def _resolve_split_channels(
    audio_file: Path, mode: str
) -> tuple[tuple[np.ndarray, np.ndarray] | None, float | None]:
    """Decide mixed vs per-channel transcription for ``steno transcribe``.

    Returns ``(pcms, correlation)``: ``pcms`` is the ``(left, right)`` float32
    pair when the file should be transcribed as two voice channels, ``None``
    for the classic mixed stream. ``correlation`` is the envelope correlation
    whenever ``auto`` examined a 2-channel file (for the CLI to explain its
    decision), ``None`` when no decision was needed or the split was forced.
    """
    from stenograf.audio import (
        audio_channel_count,
        channels_look_independent,
        load_audio_channels,
    )

    count = audio_channel_count(audio_file)
    if mode == "split" and count != 2:
        raise click.ClickException(
            f"--channels split needs 2-channel audio; {audio_file.name} has {count} channel(s)"
        )
    if count != 2 or mode == "mix":
        return None, None
    left, right = load_audio_channels(audio_file)
    if mode == "split":
        return (left, right), None
    independent, correlation = channels_look_independent(left, right)
    return ((left, right) if independent else None), correlation


def _transcribe_split_channels(
    left: np.ndarray,
    right: np.ndarray,
    *,
    profile: MeetingProfile,
    use_reid: bool,
    reid_threshold: float | None,
    glossary_threshold: float | None,
    asr_backend: str | None = None,
    asr_provider: str | None = None,
    profile_store: Path | None = None,
):
    """Transcribe two voice channels through the meeting finalize.

    This is the exact pipeline a live meeting runs on stop — per-channel ASR
    and diarization with the channel's speaker count, cross-channel echo-text
    dedup (armed conservatively: the recording's canceller state is unknown),
    glossary, one interleaved Local-N/Remote-N transcript — just fed from a
    file instead of a capture session. Returns ``(transcript, recorder,
    elapsed)``; the recorder carries the per-channel speaker counts for
    reporting, ``elapsed`` the processing seconds (clocked after model load,
    so a first-run weight download never masquerades as transcription speed).
    """
    from stenograf.audio import to_int16
    from stenograf.capture.base import AudioFrame, Channel
    from stenograf.session import MeetingRecorder, SessionStore, plan_channels
    from stenograf.view import LiveView

    class _StatusEcho(LiveView):
        def status(self, message: str) -> None:
            click.echo(message)

        def error(self, message: str) -> None:
            click.echo(f"warning: {message}", err=True)

    plans = plan_channels(profile)
    asr, vad, diarizer = loaders.load_backends(
        need_diarizer=any(p.num_speakers != 1 for p in plans),
        asr_backend=asr_backend,
        asr_provider=asr_provider,
    )
    reid = (
        loaders.load_reid(
            enabled=use_reid,
            threshold=reid_threshold,
            store_path=profile_store or profile.speaker_profile_store,
        )
        if diarizer is not None
        else None
    )
    if reid is not None:
        click.echo(f"re-ID: {len(reid.store.for_model(reid.model))} profile(s) active")
    recorder = MeetingRecorder(
        profile,
        asr=asr,
        vad=vad,
        diarizer=diarizer,
        reid=reid,
        glossary_threshold=glossary_threshold,
    )
    store = SessionStore({Channel.MIC, Channel.SYSTEM})
    store.append(AudioFrame(Channel.MIC, 0.0, to_int16(left)))
    store.append(AudioFrame(Channel.SYSTEM, 0.0, to_int16(right)))
    started = time.monotonic()
    transcript = recorder.finalize(store, plans, view=_StatusEcho())
    return transcript, recorder, time.monotonic() - started


@main.command()
@click.argument("audio_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--lang",
    type=click.Choice([lang.value for lang in Language]),
    default=None,
    help="Audio language (transcript metadata; the default ASR model is "
    "multilingual). Omit to auto-detect from the transcript.",
)
@click.option(
    "--speakers",
    type=click.IntRange(1, _FILE_MAX_SPEAKERS),
    default=None,
    help="Known speaker count (the biggest diarization accuracy lever); "
    "1 skips diarization, omit to estimate. Mixed single stream only — "
    "with split voice channels give --local/--remote instead.",
)
@click.option(
    "--channels",
    "channels_mode",
    type=click.Choice(["auto", "mix", "split"]),
    default="auto",
    show_default=True,
    help="How to treat 2-channel audio. Two separate voice feeds (a "
    "--record-audio tee: mic left / system right; a dual-channel call "
    "recording) are transcribed per channel through the meeting pipeline — "
    "auto detects them by their independent activity; a stereo image of one "
    "room is downmixed to mono as before. mix/split force either way.",
)
@click.option(
    "--local",
    "local_speakers",
    type=click.IntRange(0, _MEETING_MAX_SPEAKERS),
    default=None,
    help="Split channels: number of speakers on the left/local channel; omit to auto-detect.",
)
@click.option(
    "--remote",
    "remote_speakers",
    type=click.IntRange(0, _MEETING_MAX_SPEAKERS),
    default=None,
    help="Split channels: number of speakers on the right/remote channel; omit to auto-detect.",
)
@click.option(
    "--no-diarization",
    "no_diarization",
    is_flag=True,
    help="Skip speaker diarization: the diarizer model is never loaded and each "
    "voice channel (or the mixed stream) is attributed to a single speaker. "
    "Conflicts with a speaker count above 1.",
)
@click.option(
    "--out",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Use this directory as the transcription's folder instead of creating "
    "a date-named one under the output home ([output] dir in settings.toml, "
    "else ~/Documents/Meetings). Refuses a directory that already holds a "
    "transcript unless --force.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Let --out overwrite an existing transcript (e.g. re-transcribing the "
    "same meeting with a corrected speaker count).",
)
@click.option(
    "--title",
    default=None,
    metavar="TEXT",
    help="A human-readable title for this transcription (recorded in the "
    "transcript and used by the notes prompt and the combined-note export).",
)
@click.option(
    "--reid/--no-reid",
    "use_reid",
    default=True,
    help="Relabel diarized speakers to saved profile names when their voice matches "
    "(cross-meeting re-identification). No effect without enrolled profiles.",
)
@click.option(
    "--reid-threshold",
    type=click.FloatRange(0, 1),
    default=None,
    help="Cosine similarity required to match a saved profile "
    "[default: [speakers] reid_threshold in settings.toml, else 0.5].",
)
@click.option(
    "--format",
    "formats",
    default=None,
    metavar="LIST",
    help="Comma-separated transcript formats to write: md, json, txt, srt, vtt "
    "[default: [transcript] formats in settings.toml, else md,json,txt]. txt is "
    "plain prose without speakers or timestamps; srt/vtt re-flow speaker turns "
    "into subtitle cues.",
)
@_vocab_options
@click.option(
    "--notes",
    "notes_flag",
    is_flag=True,
    help="After the transcript is written, generate LLM meeting notes "
    "(summary, decisions, action items) with the backend configured in "
    "settings.toml. Non-fatal: a notes failure never loses the transcript.",
)
@click.option("--print", "print_markdown", is_flag=True, help="Also print the transcript.")
def transcribe(
    audio_file: Path,
    lang: str | None,
    speakers: int | None,
    channels_mode: str,
    local_speakers: int | None,
    remote_speakers: int | None,
    no_diarization: bool,
    out: Path | None,
    force: bool,
    title: str | None,
    use_reid: bool,
    reid_threshold: float | None,
    formats: str | None,
    glossary: tuple[str, ...],
    glossary_file: Path | None,
    attendee: tuple[str, ...],
    glossary_threshold: float | None,
    profile_store: Path | None,
    notes_flag: bool,
    print_markdown: bool,
) -> None:
    """Transcribe an audio/video file (batch finalize pass).

    A 2-channel recording whose channels are separate voice feeds — a
    `--record-audio` tee (mic left, system right) or a dual-channel call
    recording — is detected and transcribed per channel through the meeting
    pipeline (Local/Remote labels, per-channel diarization); ordinary stereo
    is downmixed to mono as before. See --channels to force either way.

    Runs the same finalize pipeline a live meeting runs on stop, and writes
    the transcript into its own date-named folder under the output home
    ([output] dir in settings.toml, else ~/Documents/Meetings). Use --out to
    name the folder yourself; --format also emits srt/vtt subtitles.
    """
    from stenograf.audio import SAMPLE_RATE, load_audio

    cfg = _resolve_run_config(
        formats=formats,
        glossary=glossary,
        glossary_file=glossary_file,
        attendee=attendee,
        glossary_threshold=glossary_threshold,
        reid_threshold=reid_threshold,
        profile_store=profile_store,
    )
    settings, write_formats = cfg.settings, cfg.write_formats
    glossary_terms, attendee_names = cfg.glossary_terms, cfg.attendee_names
    glossary_threshold, reid_threshold = cfg.glossary_threshold, cfg.reid_threshold
    reid_store = cfg.reid_store
    given_language = Language(lang) if lang else None
    language = given_language

    # Resolve (and overwrite-guard) the output folder before the transcription
    # work, so a refusal costs nothing — not minutes of ASR.
    created_at = datetime.now()
    out_dir, basename, _ = _prepare_output(out, created_at, settings, force=force)

    try:
        split_pcms, correlation = _resolve_split_channels(audio_file, channels_mode)
    except RuntimeError as exc:  # unreadable input (ffmpeg could not decode it)
        raise click.ClickException(str(exc)) from exc
    if split_pcms is not None and speakers is not None:
        raise click.ClickException(
            "--speakers applies to one mixed stream; with split voice channels "
            "give --local/--remote (or force --channels mix)"
        )
    if split_pcms is None and (local_speakers is not None or remote_speakers is not None):
        raise click.ClickException(
            "--local/--remote apply to split voice channels only; this run "
            "transcribes one mixed stream (--channels split to force splitting)"
        )
    if no_diarization:
        if (speakers or 0) > 1:
            raise click.UsageError("--no-diarization conflicts with a speaker count above 1")
        if split_pcms is None:
            speakers = 1
        else:
            local_speakers, remote_speakers = _apply_no_diarization(
                True, local_speakers, remote_speakers
            )

    recorder = None  # bound by the split-channel branch, which reports its counts
    if split_pcms is not None:
        duration = len(split_pcms[0]) / SAMPLE_RATE
        reason = (
            f"independent activity, envelope correlation {correlation:.2f}"
            if correlation is not None
            else "--channels split"
        )
        click.echo(f"audio: {audio_file.name} ({_fmt_duration(duration)}, 2 voice channels)")
        click.echo(
            f"  {reason} — transcribing per channel: left → Local, right → Remote"
            + ("; --channels mix to downmix" if correlation is not None else "")
        )
        if glossary_terms or attendee_names:
            click.echo(f"glossary: {len(glossary_terms)} term(s), {len(attendee_names)} name(s)")
        profile = MeetingProfile(
            language=given_language,
            local_speakers=local_speakers,
            remote_speakers=remote_speakers,
            glossary=glossary_terms,
            attendee_names=attendee_names,
            speaker_profile_store=profile_store,
            title=title,
        )
        transcript, recorder, elapsed = _transcribe_split_channels(
            *split_pcms,
            profile=profile,
            use_reid=use_reid,
            reid_threshold=reid_threshold,
            glossary_threshold=glossary_threshold,
            asr_backend=settings.asr.backend,
            asr_provider=settings.asr.provider,
            profile_store=reid_store,
        )
    else:
        from stenograf.pipeline import finalize_file

        try:
            samples = load_audio(audio_file)
        except RuntimeError as exc:  # unreadable input (ffmpeg could not decode it)
            raise click.ClickException(str(exc)) from exc
        duration = len(samples) / SAMPLE_RATE
        click.echo(f"audio: {audio_file.name} ({_fmt_duration(duration)})")
        if correlation is not None:  # auto looked at 2 channels and declined
            click.echo(
                f"  2 channels carry one stereo image (envelope correlation {correlation:.2f})"
                " — downmixed to mono; --channels split to treat them as separate voices"
            )

        asr, vad, diarizer = loaders.load_backends(
            need_diarizer=speakers != 1,
            asr_backend=settings.asr.backend,
            asr_provider=settings.asr.provider,
        )
        started = time.monotonic()  # post-load: the speed stat must not count a model download
        reid = (
            loaders.load_reid(enabled=use_reid, threshold=reid_threshold, store_path=reid_store)
            if diarizer is not None
            else None
        )
        if reid is not None:
            click.echo(f"re-ID: {len(reid.store.for_model(reid.model))} profile(s) active")
        if glossary_terms or attendee_names:
            click.echo(f"glossary: {len(glossary_terms)} term(s), {len(attendee_names)} name(s)")

        def progress(stage: str, done: int, total: int) -> None:
            if stage == "asr" and done == 0:
                click.echo(f"transcribing {total} windows")
            elif stage == "diarization":
                click.echo(f"diarizing ({speakers or 'estimating'} speakers)")

        # The settings-derived store path stays off this profile too (see above);
        # the library assembles the whole transcript, the CLI only reports.
        transcript = finalize_file(
            samples,
            profile=MeetingProfile(
                language=given_language,
                glossary=glossary_terms,
                attendee_names=attendee_names,
                speaker_profile_store=profile_store,
                title=title,
            ),
            asr=asr,
            vad=vad,
            diarizer=diarizer,
            num_speakers=speakers,
            reid=reid,
            glossary_threshold=glossary_threshold,
            on_progress=progress,
        )
        elapsed = time.monotonic() - started

    entries = transcript.entries
    language = transcript.language
    if given_language is None and language is not None:
        click.echo(f"language: detected {language.value}")

    paths = _write_transcript(transcript, out_dir, basename, write_formats)
    speed = duration / elapsed if elapsed else 0.0
    if recorder is not None:
        _report_speaker_counts(recorder.speaker_counts)
    elif speakers is None:
        found = len({e.speaker for e in entries})
        click.echo(f"speakers: {found} detected")
        hint = _lock_hint(found, _FILE_MAX_SPEAKERS)
        if hint is not None:  # None → no speech found, nothing to lock
            value, over = hint
            note = f" (estimate over the {_FILE_MAX_SPEAKERS}-speaker max)" if over else ""
            click.echo(
                f"  estimated — re-run with --speakers {value} to lock or correct the count{note}"
            )
    else:
        click.echo(f"speakers: {speakers} given")
    click.echo(
        f"wrote {', '.join(p.name for p in paths)} → {out_dir} "
        f"({elapsed:.1f}s, {speed:.1f}x realtime)"
    )
    _finish_run(
        transcript,
        out_dir,
        basename,
        created_at=created_at,
        settings=settings,
        notes_flag=notes_flag,
        print_markdown=print_markdown,
    )


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


def _write_transcript(
    transcript: Transcript,
    out_dir: Path,
    basename: str,
    formats: tuple[str, ...] | list[str] = DEFAULT_FORMATS,
) -> list[Path]:
    """Write the transcript in each requested format; returns the written paths.

    ``basename`` is the full file stem (extension excluded) — ``transcript`` for
    a meeting folder, or e.g. ``transcript.partial`` for a crash checkpoint.
    Markdown + JSON + plain text are the default (the only files stenograf
    emits unless the user asks for subtitles); SRT/VTT are opt-in via ``--format``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = out_dir / f"{basename}.{fmt}"
        atomic_write_text(path, getattr(transcript, FORMATS[fmt])())
        paths.append(path)
    return paths


def _cleanup_checkpoints(out_dir: Path, basename: str) -> None:
    """Remove the crash-recovery checkpoints once the final transcript is written."""
    for fmt in _CHECKPOINT_FORMATS:
        (out_dir / f"{basename}.partial.{fmt}").unlink(missing_ok=True)


def _fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


@main.command()
def doctor() -> None:
    """Check this machine's readiness (permissions, OS version, models)."""
    checks = run_checks()
    for check in checks:
        if check.ok:
            symbol = click.style("✓", fg="green")
        elif check.optional:  # reported, but doesn't fail the run — opt-in feature
            symbol = click.style("○", fg="yellow")
        else:
            symbol = click.style("✗", fg="red")
        click.echo(f" {symbol} {check.name}: {check.detail}")
    if not all(check.ok or check.optional for check in checks):
        raise SystemExit(1)


@main.command()
@click.option(
    "--models-only",
    is_flag=True,
    help="Skip the permission prompts and only download the models (headless machines, CI).",
)
def setup(models_only: bool) -> None:
    """One-time setup: permission prompts, then model downloads.

    Launches the capture helper so macOS shows both permission prompts (mic +
    system audio) now instead of at the start of your first meeting — nothing
    is recorded — then downloads every model the first meeting would otherwise
    stop to fetch. macOS scopes the grant to the app the helper was launched
    from, so re-run this from each terminal app (or IDE) you will run meetings
    from; the models are cached machine-wide.
    """
    if not models_only and sys.platform == "darwin":
        _grant_capture_permissions()  # only macOS gates capture behind TCC prompts

    # Permissions first (they need the user at the keyboard), then the long
    # unattended part: everything a first meeting would otherwise stop to fetch.
    try:
        loaders.prefetch_models()
    except Exception as exc:
        raise click.ClickException(
            f"model download failed: {exc} — re-run `steno setup`, or let the models "
            "download on first use."
        ) from exc
    click.echo(click.style("✓", fg="green") + " setup complete.")


def _grant_capture_permissions() -> None:
    if sys.platform != "darwin":
        raise click.ClickException(
            "the permission prompts are macOS-only — use `steno setup --models-only` here"
        )
    from stenograf.capture.base import Channel
    from stenograf.capture.macos import HelperNotFoundError, MacOSCaptureProvider

    try:
        provider = MacOSCaptureProvider()
    except HelperNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo("Starting the capture helper — answer the macOS permission prompts if they appear.")
    provider.start({Channel.MIC, Channel.SYSTEM})
    got_mic = False
    try:
        # The helper requests mic permission, then creates the system-audio tap
        # (the second TCC prompt), then starts the mic engine — so one mic frame
        # proves both grants. The system channel is silent while nothing plays,
        # so it cannot be the signal. On a denial the helper exits with the
        # reason on stderr and the frame stream ends without a mic frame.
        for frame in provider.frames():
            if frame.channel is Channel.MIC:
                got_mic = True
                break
    finally:
        provider.stop()
    if not got_mic:
        raise click.ClickException(
            "the capture helper exited before delivering audio — a permission was denied "
            "(see the message above). Re-enable it under System Settings → Privacy & "
            "Security → Microphone / Screen & System Audio Recording, then re-run "
            "`steno setup`."
        )
    click.echo(click.style("✓", fg="green") + " microphone and system-audio access granted.")
    click.echo("  The grant is per launching app — a different terminal or IDE prompts again.")


@main.group()
def profiles() -> None:
    """Manage saved speaker voiceprints for cross-meeting re-identification.

    Enroll a voice once and every later meeting relabels that speaker
    automatically (``steno start``/``transcribe`` unless ``--no-reid``).
    """


@profiles.command("list")
def profiles_list() -> None:
    """List enrolled speaker profiles."""
    from stenograf import models
    from stenograf.profiles import ProfileStore, default_store_path

    store = ProfileStore.load()
    all_profiles = store.profiles()
    if not all_profiles:
        click.echo(f"no speaker profiles yet ({default_store_path()})")
        click.echo("enroll one with: steno profiles enroll NAME sample.wav")
        return
    active_model = models.SPEAKER_EMBEDDING.name
    click.echo(f"speaker profiles ({default_store_path()}):")
    for p in sorted(all_profiles, key=lambda p: (p.embedding_model, p.name.lower())):
        noun = "sample" if p.samples == 1 else "samples"
        # A profile made under a different embedding model can never match a
        # cluster from the current one — flag it so the count is not misleading.
        tag = "" if p.embedding_model == active_model else "  [inactive: other embedding model]"
        click.echo(f"  {p.name}  ({p.samples} {noun}){tag}")


@profiles.command("enroll")
@click.argument("name")
@click.argument("audio_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--speakers",
    type=click.IntRange(1, 16),
    default=1,
    show_default=True,
    help="How many speakers are in the clip; it is diarized into this many and one "
    "cluster is enrolled.",
)
@click.option(
    "--speaker",
    "cluster",
    default=None,
    metavar="S<n>",
    help="Which diarized cluster to enroll when the clip has several speakers "
    "(re-run without it to see the choices). Ignored for a single-speaker clip.",
)
@click.option(
    "--reinforce",
    is_flag=True,
    help="Fold this sample into an existing profile's voiceprint instead of creating a new one.",
)
def profiles_enroll(
    name: str, audio_file: Path, speakers: int, cluster: str | None, reinforce: bool
) -> None:
    """Enroll speaker NAME from a voice sample in AUDIO_FILE.

    Give a short clip in which NAME is the only speaker (the default), or a
    multi-speaker recording (e.g. a meeting saved with ``--record-audio``) plus
    ``--speakers N`` and ``--speaker S<n>`` to enroll one person from it. The
    voiceprint is computed exactly the way meetings embed their clusters, so
    future meetings relabel this speaker automatically.
    """
    from stenograf import models
    from stenograf.audio import load_audio
    from stenograf.profiles import ProfileStore

    samples = load_audio(audio_file)
    diarizer = loaders.load_diarizer()
    result = diarizer.diarize_with_embeddings(samples, num_speakers=speakers)
    if not result.embeddings:
        raise click.ClickException(
            f"no embeddable speech found in {audio_file.name}; is it silent or too short?"
        )
    embedding = _choose_cluster(result.embeddings, result.turns, cluster)

    model = models.SPEAKER_EMBEDDING.name
    store = ProfileStore.load()
    existing = store.get(name, model)
    if reinforce:
        if existing is None:
            raise click.ClickException(
                f"no profile named {name!r} to reinforce; drop --reinforce to create it."
            )
        updated = store.reinforce(existing, embedding)
        store.save()
        click.echo(f"reinforced {name!r} ({updated.samples} samples)")
        return
    if existing is not None:
        raise click.ClickException(
            f"a profile named {name!r} already exists; use --reinforce to add this sample "
            "to it, or remove it first with `steno profiles remove`."
        )
    store.enroll(name, embedding, model)
    store.save()
    click.echo(f"enrolled {name!r} from {audio_file.name}")


def _choose_cluster(embeddings, turns, cluster: str | None):
    """Pick one cluster's embedding, or raise a helpful error when it is ambiguous."""
    if cluster is not None:
        if cluster not in embeddings:
            available = ", ".join(sorted(embeddings)) or "none"
            raise click.ClickException(
                f"no cluster {cluster!r} in the clip; available: {available}"
            )
        return embeddings[cluster]
    if len(embeddings) == 1:
        return next(iter(embeddings.values()))
    durations: dict[str, float] = {}
    for turn in turns:
        durations[turn.speaker] = durations.get(turn.speaker, 0.0) + (turn.end - turn.start)
    listing = "\n".join(f"  {c}  ({durations.get(c, 0.0):.1f}s speech)" for c in sorted(embeddings))
    raise click.ClickException(
        "the clip has several speakers; re-run with --speaker to pick one:\n" + listing
    )


@profiles.command("rename")
@click.argument("old")
@click.argument("new")
def profiles_rename(old: str, new: str) -> None:
    """Rename speaker profile OLD to NEW."""
    from stenograf import models
    from stenograf.profiles import ProfileStore

    store = ProfileStore.load()
    profile = store.get(old, models.SPEAKER_EMBEDDING.name)
    if profile is None:
        raise click.ClickException(f"no profile named {old!r}")
    try:
        store.rename(profile, new)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    store.save()
    click.echo(f"renamed {old!r} → {new!r}")


@profiles.command("remove")
@click.argument("name")
@click.option("--yes", is_flag=True, help="Skip the confirmation prompt.")
def profiles_remove(name: str, yes: bool) -> None:
    """Delete speaker profile NAME."""
    from stenograf import models
    from stenograf.profiles import ProfileStore

    store = ProfileStore.load()
    profile = store.get(name, models.SPEAKER_EMBEDDING.name)
    if profile is None:
        raise click.ClickException(f"no profile named {name!r}")
    if not yes:
        click.confirm(f"delete speaker profile {name!r}?", abort=True)
    store.remove(profile)
    store.save()
    click.echo(f"removed {name!r}")


@main.group("settings")
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
    from stenograf.settings import SETTINGS_TEMPLATE, SettingsError, load_settings, settings_path

    path = settings_path()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, SETTINGS_TEMPLATE)
        click.echo(f"created {path}")
    click.edit(filename=str(path))
    try:
        load_settings(path)
    except SettingsError as exc:
        raise click.ClickException(
            f"{exc}\nyour edits are saved — run `steno settings edit` again to fix them"
        ) from exc
    click.echo(f"{path} OK")


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


def _fmt_setting(value) -> str:
    """One effective value, TOML-flavored (bools lowercase, arrays bracketed)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, tuple):
        return "[" + ", ".join(f'"{item}"' for item in value) + "]"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


@main.command("notes")
@click.argument(
    "meeting", required=False, type=click.Path(exists=True, path_type=Path), metavar="[PATH]"
)
@click.option(
    "--last",
    "last",
    is_flag=True,
    help="Use the newest meeting folder in the output home ([output] dir in "
    "settings.toml, else ~/Documents/Meetings) instead of naming a PATH.",
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
    export_dir: Path | None,
    no_export: bool,
) -> None:
    """Generate LLM meeting notes (summary, decisions, action items).

    PATH is a meeting folder (its transcript.json is used) or a transcript
    JSON file; --last picks the newest meeting folder in the output home
    instead. Notes are written as sibling .notes.md/.notes.json files; the
    meeting profile's glossary and attendees steer the prompt. Configure the
    backend in settings.toml under [notes].
    """
    import json as json_mod

    from stenograf.notes import NotesBackendError
    from stenograf.output import created_at_from_dir_name
    from stenograf.settings import SettingsError
    from stenograf.transcript import UnsupportedTranscriptVersion

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
            export_dir=export_dir,
            no_export=no_export,
        )
    except (NotesBackendError, SettingsError, ValueError, OSError) as exc:
        # The documented failure modes become clean CLI errors; anything else
        # is a bug and must propagate as a traceback, not masquerade as one.
        raise click.ClickException(str(exc)) from exc

    click.echo(f"wrote {', '.join(str(p) for p in written)}")


def _resolve_notes_target(meeting: Path | None, last: bool) -> Path:
    """The transcript JSON a ``steno notes`` invocation names.

    Exactly one of PATH and ``--last`` must be given. A directory PATH means
    its ``transcript.json``; ``--last`` scans the output home for the newest
    finished meeting folder (by name — the name encodes the start time)."""
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
    export_dir: Path | None = None,
    no_export: bool = False,
    notes_settings=None,
):
    """Generate notes and write ``<basename>.notes.md``/``.notes.json`` (plus the
    combined-note export when a target dir is configured). Returns
    ``(written_paths, notes)``; raises typed errors, writing nothing, on failure.

    ``notes_settings`` is the ``[notes]`` table a command already loaded at its
    start (so a ``--notes`` run uses the values in force when the meeting began);
    ``None`` loads it here (the standalone ``steno notes`` path)."""
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
    instructions = None
    if settings.instructions is not None:
        instructions = settings.instructions.read_text(encoding="utf-8")

    notes = generate_notes(
        transcript,
        backend,
        instructions=instructions,
        on_progress=lambda message: click.echo(f"notes: {message}"),
    )

    md_path = out_dir / f"{basename}.notes.md"
    json_path = out_dir / f"{basename}.notes.json"
    atomic_write_text(md_path, notes.to_markdown())
    atomic_write_text(json_path, notes.to_json())
    written = [md_path, json_path]

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
) -> None:
    """The opt-in ``--notes`` step after a transcript is safely written.

    Non-fatal by contract (PLAN.md §5 D6): the transcript already stands, so
    any notes failure warns and returns — rerun later with ``steno notes``."""
    try:
        written, _notes = _generate_and_write_notes(
            transcript, out_dir, basename, created_at=created_at, notes_settings=notes_settings
        )
    except Exception as exc:
        click.secho(f"notes failed: {exc}", fg="yellow")
        click.secho(f"  the transcript is safe — retry with `steno notes {out_dir}`", fg="yellow")
        return
    click.echo(f"notes: wrote {', '.join(str(p) for p in written)}")
