"""Command-line interface: ``stenograf`` / ``steno``."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click

from stenograf import __version__
from stenograf.config import Language, MeetingProfile
from stenograf.doctor import run_checks
from stenograf.transcript import Transcript

# Sentinel for --record-audio given without a value (write next to the transcript).
_RECORD_DEFAULT = "\0default"


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
    type=click.IntRange(0, 8),
    default=None,
    help="Number of speakers in the room; omit to auto-detect.",
)
@click.option(
    "--remote",
    "remote_speakers",
    type=click.IntRange(0, 8),
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
    help="Output directory for the transcript [default: current directory].",
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
    "--checkpoint-interval",
    type=click.FloatRange(0),
    default=180.0,
    metavar="SECONDS",
    help="Re-finalize and save a <transcript>.partial checkpoint every N seconds "
    "of capture (crash recovery); 0 disables it.",
)
@click.option(
    "--max-seconds",
    type=click.FloatRange(0, min_open=True),
    default=None,
    metavar="SECONDS",
    help="Stop capture automatically after this many seconds [default: until Ctrl-C].",
)
@click.option("--print", "print_markdown", is_flag=True, help="Also print the transcript.")
def start(
    lang: str | None,
    local_speakers: int | None,
    remote_speakers: int | None,
    replay: str | None,
    out: Path | None,
    record_audio: str | None,
    checkpoint_interval: float,
    max_seconds: float | None,
    print_markdown: bool,
) -> None:
    """Start transcribing a meeting (capture → finalize on stop)."""
    from stenograf.session import MeetingRecorder, plan_channels

    profile = MeetingProfile(
        language=Language(lang) if lang else None,
        local_speakers=local_speakers,
        remote_speakers=remote_speakers,
    )
    mode = profile.mode.value if profile.mode else "auto"
    click.echo(f"profile: language={profile.language or 'auto'} mode={mode}")

    plans = plan_channels(profile)
    provider = _make_provider(replay, plans)
    out_dir = out or Path.cwd()
    stem = f"meeting-{time.strftime('%Y%m%d-%H%M%S')}"

    started = time.monotonic()
    asr, vad, diarizer = _load_backends(
        need_diarizer=any(p.num_speakers != 1 for p in plans)
    )
    recorder = MeetingRecorder(
        profile, asr=asr, vad=vad, diarizer=diarizer, language=profile.language
    )

    tee = _make_tee(record_audio, out_dir, stem, plans)

    def on_checkpoint(transcript: Transcript) -> None:
        md, _ = _write_transcript(transcript, out_dir, f"{stem}.partial")
        click.echo(f"  checkpoint: {md} ({len(transcript.entries)} entries)")

    channels = ", ".join(p.channel.value for p in plans)
    stop_hint = f"stops after {max_seconds:g}s" if max_seconds else "press Ctrl-C to stop"
    click.echo(f"capturing: {channels} ({stop_hint} and transcribe)")
    try:
        transcript = recorder.run(
            provider,
            on_frame=tee.add if tee else None,
            on_status=lambda msg: click.echo(f"  {msg}"),
            on_checkpoint=on_checkpoint,
            checkpoint_interval=checkpoint_interval,
            max_seconds=max_seconds,
        )
    finally:
        if tee is not None:
            tee.close()
            click.echo(f"recorded audio: {tee.path}")

    md_path, _ = _write_transcript(transcript, out_dir, stem)
    _cleanup_checkpoints(out_dir, stem)  # the final transcript supersedes them
    elapsed = time.monotonic() - started
    found = len({e.speaker for e in transcript.entries})
    click.echo(f"speakers: {found} found")
    click.echo(f"wrote {md_path} and .json ({elapsed:.1f}s)")
    if print_markdown:
        click.echo()
        click.echo(transcript.to_markdown(), nl=False)


def _make_tee(record_audio: str | None, out_dir: Path, stem: str, plans):
    """Create the audio tee if --record-audio was given, with a loud banner."""
    if record_audio is None:
        return None
    from stenograf.recording import WavTee

    path = out_dir / f"{stem}.wav" if record_audio == _RECORD_DEFAULT else Path(record_audio)
    path.parent.mkdir(parents=True, exist_ok=True)
    tee = WavTee(path, {p.channel for p in plans})
    click.secho(
        f"● RECORDING AUDIO to {path} — raw audio is being written to disk",
        fg="red",
        bold=True,
    )
    return tee


def _make_provider(replay: str | None, plans):
    """Build the capture provider: file replay if given, else the native helper."""
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
        return FileCaptureProvider({ch: p for ch, p in sources.items() if ch in planned})

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
    type=click.IntRange(1, 16),
    default=None,
    help="Known speaker count (the biggest diarization accuracy lever); "
    "1 skips diarization, omit to estimate.",
)
@click.option(
    "--out",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Output directory [default: next to the input file].",
)
@click.option("--print", "print_markdown", is_flag=True, help="Also print the transcript.")
def transcribe(
    audio_file: Path,
    lang: str | None,
    speakers: int | None,
    out: Path | None,
    print_markdown: bool,
) -> None:
    """Transcribe an audio/video file (batch finalize pass).

    Writes <name>.transcript.md and <name>.transcript.json next to the
    input (or into --out). This is the same pipeline a live meeting runs
    on stop; use it for recorded meetings or re-transcription.
    """
    from stenograf.audio import SAMPLE_RATE, load_audio
    from stenograf.pipeline import finalize_channel, relabel_speakers

    started = time.monotonic()
    language = Language(lang) if lang else None

    samples = load_audio(audio_file)
    duration = len(samples) / SAMPLE_RATE
    click.echo(f"audio: {audio_file.name} ({_fmt_duration(duration)})")

    asr, vad, diarizer = _load_backends(need_diarizer=speakers != 1)

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
            on_progress=progress,
        )
    )
    if language is None:
        from stenograf.lid import detect_language

        language = detect_language(" ".join(e.text for e in entries))
        if language is not None:
            click.echo(f"language: detected {language.value}")
    transcript = Transcript(
        language=language, profile=MeetingProfile(language=language), entries=entries
    )

    md_path, _ = _write_transcript(transcript, out or audio_file.parent, audio_file.stem)
    elapsed = time.monotonic() - started
    speed = duration / elapsed if elapsed else 0.0
    found = len({e.speaker for e in entries})
    click.echo(f"speakers: {found} found" if speakers is None else f"speakers: {speakers} given")
    click.echo(f"wrote {md_path} and .json ({elapsed:.1f}s, {speed:.1f}x realtime)")
    if print_markdown:
        click.echo()
        click.echo(transcript.to_markdown(), nl=False)


def _load_backends(*, need_diarizer: bool):
    """Load the finalize backends (ASR, VAD, and optionally the diarizer).

    Shared by ``start`` and ``transcribe`` so both use the same committed
    defaults (parakeet-mlx, Silero VAD, sherpa-onnx diarization).
    """
    from stenograf import models
    from stenograf.asr.parakeet import ParakeetMLXBackend
    from stenograf.diarization.sherpa import SherpaOnnxDiarizer
    from stenograf.vad import SileroVAD

    asr = ParakeetMLXBackend()
    click.echo(f"asr: loading {asr.model_id}")
    asr.load()
    vad = SileroVAD(models.fetch(models.SILERO_VAD, _model_progress))
    diarizer = SherpaOnnxDiarizer(progress=_model_progress) if need_diarizer else None
    return asr, vad, diarizer


def _write_transcript(transcript: Transcript, out_dir: Path, stem: str) -> tuple[Path, Path]:
    """Write the Markdown + JSON transcript — the only files stenograf emits."""
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{stem}.transcript.md"
    json_path = out_dir / f"{stem}.transcript.json"
    md_path.write_text(transcript.to_markdown())
    json_path.write_text(transcript.to_json())
    return md_path, json_path


def _cleanup_checkpoints(out_dir: Path, stem: str) -> None:
    """Remove the crash-recovery checkpoints once the final transcript is written."""
    for suffix in (".partial.transcript.md", ".partial.transcript.json"):
        (out_dir / f"{stem}{suffix}").unlink(missing_ok=True)


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
