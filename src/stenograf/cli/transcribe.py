"""``steno transcribe`` — batch finalize pass over an audio/video file."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

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
    _load_reid,
    _notes_options,
    _prepare_output,
    _reid_format_options,
    _resolve_run_config,
    _vocab_options,
)
from stenograf.config import Language, MeetingProfile
from stenograf.output import write_transcript

if TYPE_CHECKING:
    import numpy as np


@click.command()
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
@_reid_format_options
@_vocab_options
@_notes_options
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
        profile = MeetingProfile(
            language=given_language,
            local_speakers=local_speakers,
            remote_speakers=remote_speakers,
            glossary=glossary_terms,
            attendee_names=attendee_names,
            speaker_profile_store=profile_store,
            title=title,
        )
        meeting_result, elapsed = _transcribe_split_channels(
            *split_pcms,
            profile=profile,
            use_reid=use_reid,
            reid_threshold=reid_threshold,
            glossary_threshold=glossary_threshold,
            asr_backend=settings.asr.backend,
            asr_provider=settings.asr.provider,
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
            asr_provider=settings.asr.provider,
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
    )


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
    view=None,
):
    """Transcribe two voice channels through the meeting finalize.

    This is the exact pipeline a live meeting runs on stop — per-channel ASR
    and diarization with the channel's speaker count, cross-channel echo-text
    dedup (armed conservatively: the recording's canceller state is unknown),
    glossary, one interleaved Local-N/Remote-N transcript — just fed from a
    file instead of a capture session. Returns ``(result, elapsed)``; the
    :class:`~stenograf.session.MeetingResult` carries the per-channel speaker
    counts for reporting, ``elapsed`` the processing seconds (clocked after
    model load, so a first-run weight download never masquerades as
    transcription speed).

    ``view`` receives the finalize's per-channel status lines; ``None`` (the
    CLI) echoes them to stdout. The launcher's TranscribeScreen passes its own
    — a click echo would print into a raw-mode Textual terminal.
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
    reid = _load_reid(
        diarizer,
        enabled=use_reid,
        threshold=reid_threshold,
        store=profile_store or profile.speaker_profile_store,
    )
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
    result = recorder.finalize(store, plans, view=view if view is not None else _StatusEcho())
    return result, time.monotonic() - started
