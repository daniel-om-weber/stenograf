"""``steno start`` — capture a meeting live, finalize on stop."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from stenograf.session import MeetingResult

from stenograf import loaders
from stenograf.cli.format import _MEETING_MAX_SPEAKERS, _report_speaker_counts
from stenograf.cli.run import (
    _apply_no_diarization,
    _echo_glossary,
    _finish_run,
    _load_reid,
    _notes_options,
    _prepare_output,
    _reid_format_options,
    _resolve_diarization,
    _resolve_run_config,
    _vocab_options,
)
from stenograf.config import Language, MeetingProfile
from stenograf.output import checkpoint_writer, cleanup_checkpoints, write_transcript
from stenograf.transcript import Transcript

# Sentinel for --record-audio given without a value (write next to the transcript).
_RECORD_DEFAULT = "\0default"

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


@click.command()
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
    "--diarization/--no-diarization",
    "diarization_flag",
    default=None,
    help="Run (or skip) speaker diarization; skipped, the diarizer model is "
    "never loaded and each captured channel is attributed to a single speaker "
    "(Local-1/Remote-1). --no-diarization conflicts with a --local/--remote "
    "count above 1 [default: [speakers] diarization in settings.toml, else on].",
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
@_reid_format_options
@_vocab_options
@click.option(
    "--full-finalize",
    is_flag=True,
    help="Re-transcribe everything at stop instead of reusing the live window "
    "pass's decodes. The live pass already decodes the exact windows the "
    "finalize pass would (so reuse is the default); this forces the "
    "from-scratch ASR pass for A/B comparison or paranoia.",
)
@_notes_options
def start(
    lang: str | None,
    local_speakers: int | None,
    remote_speakers: int | None,
    diarization_flag: bool | None,
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

    diarize = _resolve_diarization(
        diarization_flag, settings.speakers.diarization, local_speakers, remote_speakers
    )
    if not diarize and diarization_flag is None:  # settings turned it off — say so
        click.echo("diarization: off ([speakers] in settings.toml; --diarization to enable)")
    local_speakers, remote_speakers = _apply_no_diarization(
        not diarize, local_speakers, remote_speakers
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
    reid = _load_reid(diarizer, enabled=use_reid, threshold=reid_threshold, store=reid_store)
    _echo_glossary(glossary_terms, attendee_names)
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
        paths = write_transcript(transcript, out_dir, basename, write_formats)
        cleanup_checkpoints(out_dir, basename)
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
        result = _run_meeting(
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

    if result is None:
        # Defensive: a live view exited without producing a transcript. There is
        # nothing authoritative to write; leave any .partial checkpoint in place
        # for recovery rather than deleting it or crashing on None.
        raise click.ClickException(
            "meeting ended before a transcript was produced; any .partial checkpoint is kept"
        )

    # The canceller counts every 10 ms mic tick that ran without a usable system
    # reference — frames that never arrived, or a dead tap delivering bit-exact
    # zeros. A lost reference degrades to "no cancellation" by design — but
    # silently, so say how much of the meeting ran unprotected, and whether the
    # armed text backstop had to clean up after it.
    canceller = getattr(provider, "canceller", None)
    if canceller is not None and canceller.far_end_missing_ticks > 0:
        if result.dropped_echo_lines:
            backstop = (
                f"; the text backstop removed {result.dropped_echo_lines} mic "
                "line(s) that duplicated remote speech"
            )
        else:
            backstop = "; review Local lines in those spans for leaked remote speech"
        click.secho(
            f"echo cancellation ran without its reference for "
            f"{canceller.far_end_missing_ticks / 100:.1f}s — the system-audio tap "
            f"stalled or went silent{backstop}",
            fg="yellow",
        )

    transcript = result.transcript
    # Usually already persisted at the ``finalized`` event (the TUI path writes
    # while the app still shows the "done" screen); this is the no-op replay
    # then, and the write for the plain/batch paths — or the retry if the
    # event-time write failed, surfacing the error as a normal CLI error here.
    paths = persist(transcript)
    elapsed = time.monotonic() - started
    _report_speaker_counts(result.speaker_counts)
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
) -> MeetingResult | None:
    """Run the capture session through the right live view and return its result.

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
    from stenograf.session import CheckpointConfig

    checkpoint = CheckpointConfig(checkpoint_writer(out_dir, basename), flush_interval)
    if use_tui:
        from stenograf.ui.meeting import TextualLiveView

        view = TextualLiveView(
            profile, language=profile.language, stop=provider.stop, persist=persist
        )
        # The TUI speaks Transcript (that is what it renders); the run report
        # travels back to this caller beside it.
        results: list[MeetingResult] = []

        def meeting() -> Transcript:
            result = recorder.run(
                provider,
                live=True,
                view=view,
                on_frame=on_frame,
                checkpoint=checkpoint,
                max_seconds=max_seconds,
            )
            results.append(result)
            return result.transcript

        view.serve(meeting)
        return results[-1] if results else None
    if live:
        from stenograf.view import PlainLiveView

        with PlainLiveView() as view:
            return recorder.run(
                provider,
                live=True,
                view=view,
                on_frame=on_frame,
                checkpoint=checkpoint,
                max_seconds=max_seconds,
            )

    from stenograf.view import LiveView

    class _BatchEcho(LiveView):
        """Batch-mode sink: notices echo indented under the "capturing" line."""

        def status(self, message: str) -> None:
            click.echo(f"  {message}")

        def language(self, language: Language) -> None:
            click.echo(f"  detected language: {language.value}")

        def error(self, message: str) -> None:
            click.echo(f"  {message}")

    return recorder.run(
        provider,
        view=_BatchEcho(),
        on_frame=on_frame,
        checkpoint=CheckpointConfig(
            checkpoint_writer(out_dir, basename, announce=lambda m: click.echo(f"  {m}")),
            flush_interval,
        ),
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
