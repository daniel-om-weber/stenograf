"""``steno transcribe`` — batch finalize pass over an audio/video file."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import click

from stenograf import loaders
from stenograf.cli.format import (
    _FILE_MAX_SPEAKERS,
    _MEETING_MAX_SPEAKERS,
    _fmt_duration,
    _lock_hint,
    _report_speaker_counts,
)
from stenograf.cli.run import (
    _apply_no_diarization,
    _echo_glossary,
    _finish_run,
    _library_errors,
    _load_reid,
    _notes_options,
    _reid_format_options,
    _resolve_diarization,
    _resolve_run_config,
    _vocab_options,
)
from stenograf.config import Language, MeetingProfile
from stenograf.output import prepare_output, write_transcript


@click.command()
@click.argument("audio_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
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
    help="Audio language (transcript metadata; the default ASR model is "
    "multilingual). Omit to auto-detect from the transcript.",
)
@click.option(
    "--speakers",
    type=click.IntRange(1, _FILE_MAX_SPEAKERS),
    default=None,
    help="Known speaker count (the biggest diarization accuracy lever); "
    "a count above 1 turns diarization on, 1 skips it, omitted the count "
    "is estimated when diarization runs. Mixed single stream only — "
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
    "--diarization/--no-diarization",
    "diarization_flag",
    default=None,
    help="Run (or skip) speaker diarization; skipped, the diarizer model is "
    "never loaded and each voice channel (or the mixed stream) is attributed "
    "to a single speaker. --no-diarization conflicts with a speaker count "
    "above 1 [default: [speakers] diarization in settings.toml, else off].",
)
@click.option(
    "--out",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Use this directory as the transcription's folder instead of creating "
    "a date-named one under the output home ([output] dir in settings.toml, "
    "else Meetings/ in your documents folder). Refuses a directory that already "
    "holds a transcript unless --force.",
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
@_reid_format_options
@_vocab_options
@_notes_options
@_library_errors
def transcribe(
    audio_file: Path,
    preset: str | None,
    lang: str | None,
    speakers: int | None,
    channels_mode: str,
    local_speakers: int | None,
    remote_speakers: int | None,
    diarization_flag: bool | None,
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
    notes_flag: bool | None,
    notes_backend: str | None,
    notes_model: str | None,
    notes_instructions: Path | None,
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
    ([output] dir in settings.toml, else Meetings/ in your documents folder).
    Use --out to name the folder yourself; --format also emits srt/vtt subtitles.
    """
    from stenograf.audio import SAMPLE_RATE, load_audio
    from stenograf.pipeline import resolve_split_channels

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
    settings, write_formats = cfg.settings, cfg.write_formats
    glossary_terms, attendee_names = cfg.glossary_terms, cfg.attendee_names
    glossary_threshold, reid_threshold = cfg.glossary_threshold, cfg.reid_threshold
    reid_store = cfg.reid_store
    if cfg.preset is not None:
        # Preset values are defaults a typed flag still beats.
        title = title or cfg.preset.title
        lang = lang or cfg.preset.language
        notes_backend = notes_backend or cfg.preset.notes.backend  # flag > preset > env
    given_language = Language(lang) if lang else None
    language = given_language

    # Resolve (and overwrite-guard) the output folder before the transcription
    # work, so a refusal costs nothing — not minutes of ASR.
    created_at = datetime.now()
    out_dir, basename, _ = prepare_output(out, created_at, settings, force=force)

    try:
        split_pcms, correlation = resolve_split_channels(audio_file, channels_mode)
    except (RuntimeError, ValueError) as exc:
        # RuntimeError: unreadable input (ffmpeg could not decode it);
        # ValueError: --channels split on non-2-channel audio.
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
    if not _resolve_diarization(
        diarization_flag, settings.speakers.diarization, speakers, local_speakers, remote_speakers
    ):
        if (speakers or 0) > 1:
            raise click.UsageError("--no-diarization conflicts with a speaker count above 1")
        if diarization_flag is None:  # off without an explicit flag — say so, a flag user knows
            click.echo("diarization: off (--diarization or a speaker count enables it)")
        if split_pcms is None:
            speakers = 1
        else:
            local_speakers, remote_speakers = _apply_no_diarization(
                True, local_speakers, remote_speakers
            )

    meeting_result = None  # bound by the split-channel branch, which reports its counts
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
        _echo_glossary(glossary_terms, attendee_names)
        from stenograf.pipeline import transcribe_split_channels

        profile = MeetingProfile(
            language=given_language,
            local_speakers=local_speakers,
            remote_speakers=remote_speakers,
            glossary=glossary_terms,
            attendee_names=attendee_names,
            speaker_profile_store=profile_store,
            title=title,
        )
        meeting_result, elapsed = transcribe_split_channels(
            *split_pcms,
            profile=profile,
            view=_echo_view(),
            use_reid=use_reid,
            reid_threshold=reid_threshold,
            glossary_threshold=glossary_threshold,
            asr_backend=settings.asr.backend,
            asr_ep=settings.asr.ep,
            asr_boost=settings.asr.boost,
            profile_store=reid_store,
        )
        transcript = meeting_result.transcript
    else:
        from stenograf.pipeline import STAGE_ASR, STAGE_DIARIZATION, finalize_file

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
            asr_ep=settings.asr.ep,
            glossary=glossary_terms,
            attendee_names=attendee_names,
            boost=settings.asr.boost,
        )
        started = time.monotonic()  # post-load: the speed stat must not count a model download
        reid = _load_reid(diarizer, enabled=use_reid, threshold=reid_threshold, store=reid_store)
        _echo_glossary(glossary_terms, attendee_names)

        def progress(stage: str, done: int, total: int) -> None:
            if stage == STAGE_ASR and done == 0:
                click.echo(f"transcribing {total} windows")
            elif stage == STAGE_DIARIZATION:
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

    paths = write_transcript(transcript, out_dir, basename, write_formats)
    speed = duration / elapsed if elapsed else 0.0
    if meeting_result is not None:
        _report_speaker_counts(meeting_result.speaker_counts)
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
        notes_backend=notes_backend,
        notes_model=notes_model,
        notes_instructions=notes_instructions,
    )


def _echo_view():
    """The CLI's view for library calls: status to stdout, warnings to stderr."""
    from stenograf.view import CallbackView

    return CallbackView(
        on_status=click.echo,
        on_error=lambda message: click.echo(f"warning: {message}", err=True),
    )
