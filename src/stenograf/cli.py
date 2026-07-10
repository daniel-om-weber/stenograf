"""Command-line interface: ``stenograf`` / ``steno``."""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import click

from stenograf import __version__
from stenograf.config import Language, MeetingProfile, ResolvedParameters, resolve_value
from stenograf.doctor import run_checks
from stenograf.transcript import Transcript

# Sentinel for --record-audio given without a value (write next to the transcript).
_RECORD_DEFAULT = "\0default"

# The transcript formats stenograf can emit, and how each Transcript renders it.
# SRT/VTT re-flow the retained word timestamps into short subtitle cues.
_FORMATS: dict[str, str] = {
    "md": "to_markdown",
    "json": "to_json",
    "srt": "to_srt",
    "vtt": "to_vtt",
}
_DEFAULT_FORMATS = ("md", "json")

# Settable speaker-count ranges, kept in sync with the --local/--remote and
# --speakers IntRange bounds. The unconstrained diarizer can *detect* more (or, on
# silence, zero) speakers than the user can set, so the "lock the detected count"
# hint is clamped to these — never suggesting an out-of-range or nonsensical re-run.
_MEETING_MAX_SPEAKERS = 8
_FILE_MAX_SPEAKERS = 16


def _parse_formats(spec: str) -> list[str]:
    """Parse a ``--format`` value (comma-separated) into an ordered, de-duped list."""
    formats: list[str] = []
    for name in spec.split(","):
        name = name.strip().lower()
        if not name or name in formats:
            continue
        if name not in _FORMATS:
            raise click.BadParameter(
                f"unknown format {name!r}; choose from {', '.join(_FORMATS)}",
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
                help="Similarity 0–1 required to correct a term [default: 0.82].",
            ),
            click.option(
                "--profile-store",
                type=click.Path(dir_okay=False, path_type=Path),
                default=None,
                help="Use this re-ID profile store instead of the default location.",
            ),
        )
    ):
        func = option(func)
    return func


def _collect_terms(
    glossary: tuple[str, ...], glossary_file: Path | None, attendee: tuple[str, ...]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Gather glossary terms (inline + file) and attendee names from the options.

    Inline ``--glossary``/``--attendee`` values may each be comma-separated; the
    file is one term per line. Both are de-duplicated preserving first-seen order.
    """
    terms: list[str] = []
    for value in glossary:
        terms.extend(part.strip() for part in value.split(",") if part.strip())
    if glossary_file is not None:
        for raw in glossary_file.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip()
            if line:
                terms.append(line)
    names: list[str] = []
    for value in attendee:
        names.extend(part.strip() for part in value.split(",") if part.strip())
    return tuple(dict.fromkeys(terms)), tuple(dict.fromkeys(names))


@click.group()
@click.version_option(__version__, prog_name="stenograf")
def main() -> None:
    """Accuracy-first local meeting transcription. Audio never touches disk."""


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
    help="Write this meeting's transcript here instead of the managed archive dir "
    "(the meeting is still registered in the archive unless --no-archive).",
)
@click.option(
    "--title",
    default=None,
    metavar="TEXT",
    help="A human-readable title for this meeting (shown in `steno meetings`).",
)
@click.option(
    "--no-archive",
    "no_archive",
    is_flag=True,
    help="Don't file this meeting in the managed archive; write flat, timestamp-named "
    "transcript files to --out (or the current directory), as before Phase 4.",
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
    default=180.0,
    metavar="SECONDS",
    help="Flush a <transcript>.partial crash checkpoint every N seconds of capture "
    "(live: the already-committed captions, zero extra inference; batch: only the "
    "newest tail is finalized, off the capture thread); 0 disables it.",
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
    help="Cosine similarity required to match a saved profile [default: 0.5].",
)
@click.option(
    "--format",
    "formats",
    default=",".join(_DEFAULT_FORMATS),
    metavar="LIST",
    help="Comma-separated transcript formats to write: md, json, srt, vtt "
    "[default: md,json]. srt/vtt re-flow speaker turns into subtitle cues.",
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
@click.option("--print", "print_markdown", is_flag=True, help="Also print the transcript.")
def start(
    lang: str | None,
    local_speakers: int | None,
    remote_speakers: int | None,
    replay: str | None,
    out: Path | None,
    title: str | None,
    no_archive: bool,
    record_audio: str | None,
    flush_interval: float,
    max_seconds: float | None,
    live: bool,
    plain: bool,
    use_aec: bool,
    aec_dump: Path | None,
    use_reid: bool,
    reid_threshold: float | None,
    formats: str,
    glossary: tuple[str, ...],
    glossary_file: Path | None,
    attendee: tuple[str, ...],
    glossary_threshold: float | None,
    profile_store: Path | None,
    full_finalize: bool,
    print_markdown: bool,
) -> None:
    """Start transcribing a meeting (capture → finalize on stop)."""
    from stenograf.session import MeetingRecorder, plan_channels

    write_formats = _parse_formats(formats)
    glossary_terms, attendee_names = _collect_terms(glossary, glossary_file, attendee)

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
    provider = _make_provider(replay, plans, paced=live, aec=use_aec, aec_dump=aec_dump)
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

    # By default a meeting is filed in the managed archive: its own dir under the
    # data dir (or --out), holding transcript.{md,json,…} + optional audio.wav, plus
    # an index record. --no-archive restores the flat, timestamp-named output.
    created_at = datetime.now()
    archive, meeting_id, out_dir, basename, audio_default = _prepare_output(
        no_archive,
        out,
        created_at,
        legacy_dir=Path.cwd(),
        legacy_stem=f"meeting-{created_at:%Y%m%d-%H%M%S}",
    )

    started = time.monotonic()
    asr, vad, diarizer = _load_backends(need_diarizer=any(p.num_speakers != 1 for p in plans))
    reid = (
        _load_reid(enabled=use_reid, threshold=reid_threshold, store_path=profile_store)
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

    paths = _write_transcript(transcript, out_dir, basename, write_formats)
    _cleanup_checkpoints(out_dir, basename)  # the final transcript supersedes them
    elapsed = time.monotonic() - started
    _report_speaker_counts(recorder.speaker_counts)
    click.echo(f"wrote {', '.join(p.name for p in paths)} ({elapsed:.1f}s)")
    if archive is not None:
        archive.add(
            _meeting_record(
                meeting_id,
                created_at,
                transcript,
                write_formats,
                out_dir,
                audio_path=tee.path if tee is not None else None,
            )
        )
        click.echo(f"archived as {meeting_id} — see `steno meetings show {meeting_id}`")
    if print_markdown:
        click.echo()
        click.echo(transcript.to_markdown(), nl=False)


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
) -> Transcript:
    """Run the capture session through the right live view and return the transcript.

    Three shapes behind one call:

    - **TUI** (live, on a TTY, not ``--plain``): the Textual view runs the app on
      this thread while the meeting runs on a background thread; its quit binding
      crosses to ``provider.stop`` to end capture. Checkpoints are written silently
      (the TUI owns the screen).
    - **Plain live** (live, no TTY or ``--plain``): the meeting runs on this thread
      and streams committed captions to stdout; checkpoints written silently.
    - **Batch** (``--no-live``): no live pass; status and checkpoint notices echo
      as before.
    """
    if use_tui:
        from stenograf.tui import TextualLiveView

        view = TextualLiveView(profile, language=profile.language, stop=provider.stop)
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


def _checkpoint_writer(
    out_dir: Path, basename: str, announce: Callable[[str], None] | None = None
) -> Callable[[Transcript], None]:
    """Build the ``on_checkpoint`` sink that writes the ``.partial`` crash file.

    Live views keep the caption stream clean (``announce=None`` → write silently);
    the batch path narrates each write, as it always has. The final transcript
    supersedes these files, which ``_cleanup_checkpoints`` then removes.
    """

    def on_checkpoint(transcript: Transcript) -> None:
        # Checkpoints are crash recovery, always plain md+json — subtitles of a
        # partial transcript are pointless and the final write supersedes these.
        md = _write_transcript(transcript, out_dir, f"{basename}.partial")[0]
        if announce is not None:
            announce(f"checkpoint: {md.name} ({len(transcript.entries)} entries)")

    return on_checkpoint


def _make_tee(record_audio: str | None, default_path: Path, plans):
    """Create the audio tee if --record-audio was given, with a loud banner.

    ``default_path`` is where a bare ``--record-audio`` (no value) writes — the
    managed ``audio.wav`` when archiving, else ``<stem>.wav``; an explicit
    ``--record-audio PATH`` overrides it.
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


def _make_provider(
    replay: str | None,
    plans,
    *,
    paced: bool = False,
    aec: bool = True,
    aec_dump: Path | None = None,
):
    """Build the capture provider: file replay if given, else the native helper.

    When both channels are captured, the mic is echo-cancelled against the system
    channel — without it, remote participants coming out of the speakers land on
    the mic channel and get transcribed as the local speaker. ``aec_dump`` wraps
    even with ``--no-aec`` so the eval rig can record the uncancelled baseline.
    """
    from stenograf.capture.base import Channel

    provider = _base_provider(replay, plans, paced=paced)
    channels = {plan.channel for plan in plans}
    if (aec or aec_dump is not None) and {Channel.MIC, Channel.SYSTEM} <= channels:
        from stenograf.aec import EchoCancellingProvider

        return EchoCancellingProvider(provider, cancel=aec, dump_dir=aec_dump)
    return provider


def _base_provider(replay: str | None, plans, *, paced: bool = False):
    from stenograf.capture.base import Channel

    if replay is not None:
        from stenograf.capture.file import FileCaptureProvider

        paths = [p.strip() for p in replay.split(",") if p.strip()]
        channel_order = [Channel.MIC, Channel.SYSTEM]
        sources = dict(zip(channel_order, paths, strict=False))
        planned = {p.channel for p in plans}
        ignored = [ch.value for ch in sources if ch not in planned]
        if ignored:
            click.echo(f"note: ignoring replay for un-recorded channel(s): {', '.join(ignored)}")
        return FileCaptureProvider(
            {ch: p for ch, p in sources.items() if ch in planned}, paced=paced
        )

    if sys.platform != "darwin":
        raise click.ClickException(
            "live capture is macOS-only for now; on other platforms transcribe a "
            "recorded file with `steno transcribe`, or use `steno start --replay`."
        )
    from stenograf.capture.macos import HelperNotFoundError, MacOSCaptureProvider

    try:
        return MacOSCaptureProvider()
    except HelperNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc


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
    "1 skips diarization, omit to estimate.",
)
@click.option(
    "--out",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Write this transcript here instead of the managed archive dir (the "
    "transcription is still registered in the archive unless --no-archive).",
)
@click.option(
    "--title",
    default=None,
    metavar="TEXT",
    help="A human-readable title for this transcription (shown in `steno meetings`).",
)
@click.option(
    "--no-archive",
    "no_archive",
    is_flag=True,
    help="Don't file this transcription in the managed archive; write flat "
    "<name>.transcript.{md,json,…} files next to the input (or --out), as before.",
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
    help="Cosine similarity required to match a saved profile [default: 0.5].",
)
@click.option(
    "--format",
    "formats",
    default=",".join(_DEFAULT_FORMATS),
    metavar="LIST",
    help="Comma-separated transcript formats to write: md, json, srt, vtt "
    "[default: md,json]. srt/vtt re-flow speaker turns into subtitle cues.",
)
@_vocab_options
@click.option("--print", "print_markdown", is_flag=True, help="Also print the transcript.")
def transcribe(
    audio_file: Path,
    lang: str | None,
    speakers: int | None,
    out: Path | None,
    title: str | None,
    no_archive: bool,
    use_reid: bool,
    reid_threshold: float | None,
    formats: str,
    glossary: tuple[str, ...],
    glossary_file: Path | None,
    attendee: tuple[str, ...],
    glossary_threshold: float | None,
    profile_store: Path | None,
    print_markdown: bool,
) -> None:
    """Transcribe an audio/video file (batch finalize pass).

    Files the transcript in the managed archive by default (browse it with
    `steno meetings`), the same pipeline a live meeting runs on stop. Use
    --out to write elsewhere, or --no-archive to drop flat
    <name>.transcript.{md,json,…} files next to the input as before; --format
    also emits srt/vtt subtitles.
    """
    from stenograf.audio import SAMPLE_RATE, load_audio
    from stenograf.glossary import DEFAULT_THRESHOLD, apply_glossary
    from stenograf.pipeline import finalize_channel, relabel_speakers

    started = time.monotonic()
    write_formats = _parse_formats(formats)
    glossary_terms, attendee_names = _collect_terms(glossary, glossary_file, attendee)
    given_language = Language(lang) if lang else None
    language = given_language

    samples = load_audio(audio_file)
    duration = len(samples) / SAMPLE_RATE
    click.echo(f"audio: {audio_file.name} ({_fmt_duration(duration)})")

    asr, vad, diarizer = _load_backends(need_diarizer=speakers != 1)
    reid = (
        _load_reid(enabled=use_reid, threshold=reid_threshold, store_path=profile_store)
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

    entries = relabel_speakers(
        finalize_channel(
            samples,
            asr=asr,
            language=language,
            vad=vad,
            diarizer=diarizer,
            num_speakers=speakers,
            reid=reid,
            on_progress=progress,
        )
    )
    threshold = DEFAULT_THRESHOLD if glossary_threshold is None else glossary_threshold
    entries = apply_glossary(
        entries, glossary=glossary_terms, attendee_names=attendee_names, threshold=threshold
    )
    if language is None:
        from stenograf.lid import detect_language

        language = detect_language(" ".join(e.text for e in entries))
        if language is not None:
            click.echo(f"language: detected {language.value}")
    profile = MeetingProfile(
        language=given_language,
        glossary=glossary_terms,
        attendee_names=attendee_names,
        speaker_profile_store=profile_store,
        title=title,
    )
    # A file transcribe is one un-split stream (no local/remote model), so its
    # speaker provenance is recorded under a single "audio" channel (PLAN.md §5 3b).
    parameters = ResolvedParameters(
        language=resolve_value(given_language, language),
        speakers={"audio": resolve_value(speakers, len({e.speaker for e in entries}))},
    )
    transcript = Transcript(
        language=language, profile=profile, entries=entries, parameters=parameters
    )

    created_at = datetime.now()
    archive, meeting_id, out_dir, basename, _ = _prepare_output(
        no_archive, out, created_at, legacy_dir=audio_file.parent, legacy_stem=audio_file.stem
    )
    paths = _write_transcript(transcript, out_dir, basename, write_formats)
    if archive is not None:
        # The source file is already on disk, so reference it as this meeting's
        # audio — that enables archived playback / re-diarize (B4) at no extra cost
        # to the in-memory-only guarantee (which is about live capture).
        archive.add(
            _meeting_record(
                meeting_id, created_at, transcript, write_formats, out_dir, audio_path=audio_file
            )
        )
    elapsed = time.monotonic() - started
    speed = duration / elapsed if elapsed else 0.0
    if speakers is None:
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
    click.echo(f"wrote {', '.join(p.name for p in paths)} ({elapsed:.1f}s, {speed:.1f}x realtime)")
    if archive is not None:
        click.echo(f"archived as {meeting_id} — see `steno meetings show {meeting_id}`")
    if print_markdown:
        click.echo()
        click.echo(transcript.to_markdown(), nl=False)


def _load_backends(*, need_diarizer: bool):
    """Load the finalize backends (ASR, VAD, and optionally the diarizer).

    Shared by ``start`` and ``transcribe`` so both use the same committed
    defaults (parakeet-mlx, Silero VAD, sherpa-onnx diarization).
    """
    from stenograf import models
    from stenograf.asr import create_backend
    from stenograf.vad import SileroVAD

    asr = create_backend()  # the selection seam; a Linux backend registers alongside
    click.echo(f"asr: loading {getattr(asr, 'model_id', None) or asr.name}")
    asr.load()
    vad = SileroVAD(models.fetch(models.SILERO_VAD, _model_progress))
    diarizer = _load_diarizer(need=need_diarizer)
    return asr, vad, diarizer


def _load_diarizer(*, need: bool = True):
    """Build the diarizer (or ``None`` when a channel is single-speaker).

    When the stenodiar helper is present, unknown speaker counts go through
    speakrs' VBx estimation and explicit counts through sherpa; without it,
    sherpa handles both (its estimate mode over-splits badly — the helper is
    what makes "don't specify a count" usable).

    A seam of its own so ``steno profiles enroll`` computes its voiceprints with
    the exact same embedding path the finalize pass uses at match time (the two
    must agree for the cosine match to mean anything), and so tests can inject a
    fake without a real ONNX model.
    """
    if not need:
        return None
    from stenograf.diarization.sherpa import SherpaOnnxDiarizer
    from stenograf.diarization.speakrs import (
        DiarizerHelperNotFoundError,
        SpeakrsCliDiarizer,
        find_stenodiar,
    )

    sherpa = SherpaOnnxDiarizer(progress=_model_progress)
    try:
        find_stenodiar()
    except DiarizerHelperNotFoundError:
        return sherpa
    return SpeakrsCliDiarizer(sherpa)


def _load_reid(*, enabled: bool, threshold: float | None, store_path: Path | None = None):
    """Build the cross-meeting re-ID resolver from the saved profile store, or ``None``.

    Returns ``None`` when re-ID is turned off or the store holds no profiles for
    the active embedding model — so the finalize pass is byte-for-byte unchanged
    without enrolled profiles (match-only, zero behaviour change; PLAN.md Phase 3
    Task 1b/1c). ``threshold=None`` uses the store default (0.5). ``store_path``
    (``--profile-store`` / ``MeetingProfile.speaker_profile_store``) overrides the
    default store location.
    """
    if not enabled:
        return None
    from stenograf import models
    from stenograf.profiles import ProfileStore, SpeakerReID

    store = ProfileStore.load(store_path)
    model = models.SPEAKER_EMBEDDING.name
    if not store.for_model(model):
        return None
    return SpeakerReID(store, model, threshold=threshold)


def _prepare_output(
    no_archive: bool,
    out: Path | None,
    created_at: datetime,
    *,
    legacy_dir: Path,
    legacy_stem: str,
):
    """Resolve where a finalized transcript is written and whether it is archived.

    Returns ``(archive, meeting_id, out_dir, basename, audio_default)``.

    - **Archive-on (the default):** a managed per-meeting dir under the archive
      (``meetings/<id>/``) — or ``--out`` used as that meeting's dir — holding
      plainly named ``transcript.{fmt}`` + ``audio.wav`` files (the layout the
      B1 archive reads back), plus a live :class:`MeetingArchive` to register into.
    - **``--no-archive``:** the pre-Phase-4 flat layout — ``<stem>.transcript.{fmt}``
      into ``--out`` (or ``legacy_dir``), audio at ``<stem>.wav``, no archive.
    """
    if no_archive:
        out_dir = out or legacy_dir
        return None, None, out_dir, f"{legacy_stem}.transcript", out_dir / f"{legacy_stem}.wav"
    from stenograf.archive import AUDIO_NAME, TRANSCRIPT_STEM, MeetingArchive

    archive = MeetingArchive.load()
    meeting_id = archive.allocate_id(created_at)
    out_dir = out or archive.meeting_dir(meeting_id)
    return archive, meeting_id, out_dir, TRANSCRIPT_STEM, out_dir / AUDIO_NAME


def _meeting_record(
    meeting_id: str,
    created_at: datetime,
    transcript: Transcript,
    formats: tuple[str, ...] | list[str],
    out_dir: Path,
    *,
    audio_path: Path | None,
):
    """Build the archive index record for a just-written transcript.

    Denormalizes the same fields ``archive._record_from_dir`` recovers on
    reconcile (title, language, per-channel speaker counts, duration, formats),
    so a live-registered record and a re-adopted one describe the meeting alike.
    """
    from stenograf.archive import MeetingRecord

    speakers: dict[str, int | None] = {}
    if transcript.parameters is not None:
        speakers = {ch: rv.value for ch, rv in transcript.parameters.speakers.items()}  # type: ignore[misc]
    return MeetingRecord(
        id=meeting_id,
        title=transcript.profile.title,
        created_at=created_at.isoformat(timespec="seconds"),
        duration_s=max((e.end for e in transcript.entries), default=0.0),
        language=transcript.language,
        speakers=speakers,
        formats=tuple(formats),
        dir=out_dir,
        audio_path=audio_path,
    )


def _write_transcript(
    transcript: Transcript,
    out_dir: Path,
    basename: str,
    formats: tuple[str, ...] | list[str] = _DEFAULT_FORMATS,
) -> list[Path]:
    """Write the transcript in each requested format; returns the written paths.

    ``basename`` is the full file stem (extension excluded): ``transcript`` in the
    managed archive dir, or ``<name>.transcript`` for the flat ``--no-archive``
    layout. Markdown + JSON are the default (the only files stenograf emits unless
    the user asks for subtitles); SRT/VTT are opt-in via ``--format``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = out_dir / f"{basename}.{fmt}"
        _atomic_write_text(path, getattr(transcript, _FORMATS[fmt])())
        paths.append(path)
    return paths


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` via a temp file + ``os.replace`` (atomic on POSIX/Windows).

    A plain ``write_text`` truncates in place, so a crash mid-write leaves a
    corrupt file — and for the ``.partial`` crash-recovery checkpoint that also
    destroys the previous good copy, defeating the artifact meant to survive the
    crash. Writing a sibling temp then atomically renaming means a reader only ever
    sees the whole old file or the whole new one (PLAN.md §5 Phase 3→4 audit)."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _cleanup_checkpoints(out_dir: Path, basename: str) -> None:
    """Remove the crash-recovery checkpoints once the final transcript is written."""
    for fmt in ("md", "json"):
        (out_dir / f"{basename}.partial.{fmt}").unlink(missing_ok=True)


def _model_progress(name: str, done: int, total: int) -> None:
    if total and done == 0:
        click.echo(f"model: downloading {name} ({total >> 20} MB)")


def _fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


@main.command()
def doctor() -> None:
    """Check this machine's readiness (permissions, OS version, models)."""
    checks = run_checks()
    for check in checks:
        symbol = click.style("✓", fg="green") if check.ok else click.style("✗", fg="red")
        click.echo(f" {symbol} {check.name}: {check.detail}")
    if not all(check.ok for check in checks):
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
    if not models_only:
        _grant_capture_permissions()

    # Permissions first (they need the user at the keyboard), then the long
    # unattended part: everything a first meeting would otherwise stop to fetch.
    try:
        _prefetch_models()
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


def _prefetch_models() -> None:
    """Download the VAD/diarization assets and the ASR weights now, not mid-meeting."""
    from stenograf import models
    from stenograf.asr import backend_model_id, create_backend, get_spec
    from stenograf.doctor import _installed

    for asset in (models.SILERO_VAD, models.PYANNOTE_SEGMENTATION, models.SPEAKER_EMBEDDING):
        if models.cached_path(asset) is not None:
            click.echo(f"model: {asset.name} already cached")
        else:
            models.fetch(asset, _model_progress)

    # Gate on the backend's runtime deps the way doctor does: the backend
    # *module* imports fine everywhere (its heavy imports live inside load()),
    # so a try/except around create_backend() would not catch a missing MLX.
    spec = get_spec()
    if not all(_installed(module) for module in spec.requires):
        click.echo(f"ASR backend {spec.label} is not installed here; skipping its weights")
        return
    click.echo(f"model: fetching + loading ASR weights ({backend_model_id(spec)})")
    backend = create_backend()
    backend.load()  # pulls from HuggingFace on first run, then verifies it loads
    backend.unload()


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
    diarizer = _load_diarizer(need=True)
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


@main.group()
def meetings() -> None:
    """Browse the meeting archive.

    steno start and steno transcribe file each finalized transcript here by
    default — a managed library under the data dir — unless --no-archive.
    """


@meetings.command("list")
def meetings_list() -> None:
    """List archived meetings, most recent first."""
    from stenograf.archive import MeetingArchive, meetings_dir

    archive = MeetingArchive.load()
    if archive.root.exists():
        # Self-heal against the directory tree before listing (drop vanished
        # meetings, adopt any written while the index was unavailable). Skip the
        # save-on-read when nothing is there yet — an empty listing writes nothing.
        archive.reconcile()
    records = archive.records()
    if not records:
        click.echo(f"no meetings archived yet ({meetings_dir()})")
        click.echo("run `steno start` (or `steno transcribe FILE`) to record one.")
        return
    click.echo(f"meetings ({meetings_dir()}):")
    for record in sorted(records, key=lambda r: r.created_at, reverse=True):
        title = record.title or "(untitled)"
        lang = record.language.value if record.language else "?"
        when = record.created_at.replace("T", " ") if record.created_at else "unknown"
        audio = " ●rec" if record.has_audio() else ""
        click.echo(
            f"  {record.id}  {when}  [{lang}]  {_fmt_duration(record.duration_s)}  {title}{audio}"
        )


@meetings.command("show")
@click.argument("meeting_id")
def meetings_show(meeting_id: str) -> None:
    """Show the archive record for one meeting."""
    from stenograf.archive import MeetingArchive

    archive = MeetingArchive.load()
    record = archive.get(meeting_id)
    if record is None:
        raise click.ClickException(f"no meeting {meeting_id!r} in the archive")
    click.echo(f"{record.id}  {record.title or '(untitled)'}")
    click.echo(f"  created:  {record.created_at or 'unknown'}")
    click.echo(f"  language: {record.language.value if record.language else 'unknown'}")
    click.echo(f"  duration: {_fmt_duration(record.duration_s)}")
    if record.speakers:
        parts = ", ".join(
            f"{ch}={n if n is not None else '?'}" for ch, n in record.speakers.items()
        )
        click.echo(f"  speakers: {parts}")
    click.echo(f"  formats:  {', '.join(record.formats) or 'none'}")
    click.echo(f"  dir:      {record.dir}")
    click.echo(f"  audio:    {record.audio_path if record.has_audio() else 'none (in-memory)'}")


@meetings.command("rm")
@click.argument("meeting_id")
@click.option("--yes", is_flag=True, help="Skip the confirmation prompt.")
@click.option(
    "--keep-files",
    is_flag=True,
    help="Only unregister the meeting; leave its transcript files on disk.",
)
def meetings_rm(meeting_id: str, yes: bool, keep_files: bool) -> None:
    """Remove a meeting from the archive (and delete its managed files)."""
    import shutil

    from stenograf.archive import MeetingArchive

    archive = MeetingArchive.load()
    record = archive.get(meeting_id)
    if record is None:
        raise click.ClickException(f"no meeting {meeting_id!r} in the archive")
    # Only ever delete files stenograf manages — a dir that is the archive root's
    # own child. An explicit --out dir may hold unrelated files, so it is only
    # unregistered, never removed.
    managed = record.dir.parent == archive.root
    delete_files = managed and not keep_files
    if not yes:
        prompt = (
            f"remove meeting {meeting_id!r} and delete its files?"
            if delete_files
            else f"remove meeting {meeting_id!r} from the archive (files kept)?"
        )
        click.confirm(prompt, abort=True)
    archive.remove(meeting_id)
    if delete_files:
        shutil.rmtree(record.dir, ignore_errors=True)
        click.echo(f"removed {meeting_id} and its files")
    else:
        click.echo(f"unregistered {meeting_id} (files at {record.dir})")
