"""``steno start`` — capture a meeting live, finalize on stop.

A click-flag adapter over :class:`stenograf.flow.MeetingRun` — the one
meeting assembly sequence both front-ends share. This module only translates
flags into a :class:`~stenograf.flow.MeetingRequest` plus
:class:`~stenograf.flow.RunOptions`, prints the pre-run banner lines a
terminal wants, and picks the view (the live caption stream, or the indented
batch echo for ``--no-live``); everything between Start and the written
transcript happens in the library."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from stenograf.view import LiveView

from stenograf.cli.format import _MEETING_MAX_SPEAKERS, _report_speaker_counts
from stenograf.cli.run import (
    _apply_no_diarization,
    _echo_glossary,
    _library_errors,
    _notes_enabled,
    _notes_options,
    _reid_format_options,
    _resolve_diarization,
    _resolve_run_config,
    _vocab_options,
)
from stenograf.config import Language, MeetingProfile

# Sentinel for --record-audio given without a value (write next to the transcript).
_RECORD_DEFAULT = "\0default"


@click.command()
@click.option(
    "--preset",
    "preset",
    default=None,
    metavar="NAME",
    help="Meeting preset from settings.toml ([meetings.NAME]): title, language, "
    "vocabulary, notes setup and protocol template for this kind of meeting. "
    "`steno presets` lists them; typed flags still beat the preset's values.",
)
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
    "count above 1 [default: [speakers] diarization in settings.toml, else off].",
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
    "else Meetings/ in your documents folder). Refuses a directory that already "
    "holds a transcript unless --force.",
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
    "Off by default (unless [output] record_audio is set) — audio otherwise never "
    "touches disk. Give a PATH or omit it to write <transcript>.wav.",
)
@click.option(
    "--no-record-audio",
    "no_record_audio",
    is_flag=True,
    help="Do not save the audio for this run, even when [output] record_audio is "
    "set. The per-run opt-out of the standing default.",
)
@click.option(
    "--flush-interval",
    "--checkpoint-interval",
    "flush_interval",
    type=click.FloatRange(0),
    default=None,
    metavar="SECONDS",
    help="Flush a <transcript>.partial crash checkpoint every N seconds of capture "
    "(the already-committed live captions, zero extra inference — default 15); "
    "0 disables it. Batch (--no-live) runs never checkpoint.",
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
    hidden=True,
    # Accepted no-op: the line stream it once selected is now the only live
    # terminal mode, but external scripts still pass the flag, so deleting it
    # would break them for nothing.
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
@_library_errors
def start(
    preset: str | None,
    lang: str | None,
    local_speakers: int | None,
    remote_speakers: int | None,
    diarization_flag: bool | None,
    replay: str | None,
    out: Path | None,
    force: bool,
    title: str | None,
    record_audio: str | None,
    no_record_audio: bool,
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
    notes_flag: bool | None,
    notes_backend: str | None,
    notes_model: str | None,
    notes_instructions: Path | None,
    print_markdown: bool,
) -> None:
    """Start transcribing a meeting (capture → finalize on stop)."""
    from stenograf.capture.base import Channel
    from stenograf.flow import MeetingRequest, MeetingRun, RunOptions
    from stenograf.session import plan_channels

    if no_record_audio and record_audio is not None:  # before any model loads
        raise click.UsageError("--record-audio and --no-record-audio are mutually exclusive")

    cfg = _resolve_run_config(
        formats=formats,
        glossary=glossary,
        glossary_file=glossary_file,
        attendee=attendee,
        glossary_threshold=glossary_threshold,
        reid_threshold=reid_threshold,
        profile_store=profile_store,
        preset=preset,
    )
    if cfg.preset is not None:
        # Preset values are defaults a typed flag still beats.
        title = title or cfg.preset.title
        lang = lang or cfg.preset.language
        # Explicit, not just overlaid into settings: a CLI-level backend choice
        # (flag, then preset) beats STENOGRAF_NOTES_BACKEND, which would win
        # over the overlaid [notes] table inside create_backend.
        notes_backend = notes_backend or cfg.preset.notes.backend

    diarize = _resolve_diarization(
        diarization_flag, cfg.settings.speakers.diarization, local_speakers, remote_speakers
    )
    if not diarize and diarization_flag is None:  # off without an explicit flag — say so
        click.echo("diarization: off (--diarization or a speaker count enables it)")
    local_speakers, remote_speakers = _apply_no_diarization(
        not diarize, local_speakers, remote_speakers
    )
    try:
        profile = MeetingProfile(
            language=Language(lang) if lang else None,
            local_speakers=local_speakers,
            remote_speakers=remote_speakers,
            glossary=cfg.glossary_terms,
            attendee_names=cfg.attendee_names,
            speaker_profile_store=profile_store,
            title=title,
        )
    except ValueError as exc:  # e.g. --local 0 --remote 0 — report cleanly, not a traceback
        raise click.ClickException(str(exc)) from exc
    mode = profile.mode.value if profile.mode else "auto"
    click.echo(f"profile: language={profile.language or 'auto'} mode={mode}")

    # A bare --record-audio wins; absent one, [output] record_audio makes the
    # meeting folder's audio.wav the standing default. --no-record-audio is the
    # per-run opt-out of that default (the two are rejected together up top).
    if not no_record_audio and record_audio is None and cfg.settings.output.record_audio:
        record_audio = _RECORD_DEFAULT
    audio_path = None  # None → the meeting folder's audio.wav
    if record_audio is not None and record_audio != _RECORD_DEFAULT:
        audio_path = Path(record_audio)

    request = MeetingRequest(
        profile=profile,
        settings=cfg.settings,
        notes=_notes_enabled(notes_flag, cfg.settings),
        record_audio=record_audio is not None,
    )
    options = RunOptions(
        replay=replay,
        out=out,
        force=force,
        live=live,
        aec=use_aec,
        aec_dump=aec_dump,
        flush_interval=flush_interval,
        max_seconds=max_seconds,
        full_finalize=full_finalize,
        use_reid=use_reid,
        reid_threshold=cfg.reid_threshold,
        reid_store=cfg.reid_store,
        glossary_threshold=cfg.glossary_threshold,
        formats=tuple(cfg.write_formats),
        audio_path=audio_path,
        notes_backend=notes_backend,
        notes_model=notes_model,
        notes_instructions=notes_instructions,
        # The capture transports inherit stderr, so their chatter (and any
        # FATAL line) prints straight to the terminal — this command owns no
        # screen a raw write could corrupt.
        transport_stderr=True,
    )
    # Every meeting gets its own date-named folder in the visible output home
    # (or --out as the folder), holding transcript.{md,json,…} + optional
    # audio.wav — self-describing files, no index. An --out collision raises
    # here; the _library_errors boundary reports it cleanly.
    run = MeetingRun(request, options=options)

    plans = plan_channels(profile)
    channels = ", ".join(p.channel.value for p in plans)
    stop_hint = f"stops after {max_seconds:g}s" if max_seconds else "press Ctrl-C to stop"
    click.echo(f"capturing: {channels} ({stop_hint} and transcribe)")
    if len(plans) > 1:
        state = "on" if use_aec else "off"
        click.echo(f"echo cancellation: {state} (mic cancelled against system audio)")
    _echo_glossary(cfg.glossary_terms, cfg.attendee_names)
    if aec_dump is not None:
        # Same condition make_provider wraps the canceller on (dump wraps even
        # with --no-aec, recording the uncancelled baseline).
        if {Channel.MIC, Channel.SYSTEM} <= {p.channel for p in plans}:
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
    if request.record_audio:
        click.secho(
            f"● RECORDING AUDIO to {run.audio_path} — raw audio is being written to disk",
            fg="red",
            bold=True,
        )

    # Two view shapes behind the one run:
    #
    # - **Live** (the default): the meeting runs on this thread and streams
    #   committed captions to stdout; checkpoints written silently. Ctrl-C is
    #   the stop gesture (session.py catches it, stops the provider and joins
    #   capture; the finalize is shielded against a second Ctrl-C).
    # - **Batch** (``--no-live``): no live pass and no checkpoints; status
    #   notices echo indented, as they always have.
    #
    # A capture transport that dies with nothing recorded raises
    # CaptureHelperError (its FATAL detail already printed on inherited
    # stderr); the _library_errors boundary reports it cleanly.
    from stenograf.view import PlainLiveView

    view = PlainLiveView() if live else _batch_view()
    with view:
        transcript = run.run(view)
    if transcript is None:
        raise click.ClickException(
            "the meeting ended before a transcript was produced; "
            "any .partial checkpoint is kept"
        )
    if run.result is not None:
        _report_speaker_counts(run.result.speaker_counts)
    if print_markdown:
        click.echo()
        click.echo(transcript.to_markdown(), nl=False)


def _batch_view() -> LiveView:
    """Batch-mode sink: notices echo indented under the "capturing" line."""
    from stenograf.view import CallbackView

    def indent(message: str) -> None:
        if message:  # "" is a screen's clear-the-label event, not a line
            click.echo(f"  {message}")

    return CallbackView(on_status=indent)
