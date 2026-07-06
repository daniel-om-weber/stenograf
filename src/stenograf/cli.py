"""Command-line interface: ``stenograf`` / ``steno``."""

from __future__ import annotations

import time
from pathlib import Path

import click

from stenograf import __version__
from stenograf.config import Language, MeetingProfile
from stenograf.doctor import run_checks
from stenograf.transcript import Transcript


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
def start(lang: str | None, local_speakers: int | None, remote_speakers: int | None) -> None:
    """Start transcribing a meeting."""
    profile = MeetingProfile(
        language=Language(lang) if lang else None,
        local_speakers=local_speakers,
        remote_speakers=remote_speakers,
    )
    mode = profile.mode.value if profile.mode else "auto"
    click.echo(f"profile: language={profile.language or 'auto'} mode={mode}")
    raise click.ClickException("the capture/transcription pipeline is not implemented yet")


@main.command()
@click.argument("audio_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--lang",
    type=click.Choice([lang.value for lang in Language]),
    default=None,
    help="Audio language (transcript metadata; the default ASR model is multilingual).",
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
    from stenograf import models
    from stenograf.asr.parakeet import ParakeetMLXBackend
    from stenograf.audio import SAMPLE_RATE, load_audio
    from stenograf.diarization.sherpa import SherpaOnnxDiarizer
    from stenograf.pipeline import finalize_channel, relabel_speakers
    from stenograf.vad import SileroVAD

    started = time.monotonic()
    language = Language(lang) if lang else None

    samples = load_audio(audio_file)
    duration = len(samples) / SAMPLE_RATE
    click.echo(f"audio: {audio_file.name} ({_fmt_duration(duration)})")

    def model_progress(name: str, done: int, total: int) -> None:
        if total and done == 0:
            click.echo(f"model: downloading {name} ({total >> 20} MB)")

    asr = ParakeetMLXBackend()
    click.echo(f"asr: loading {asr.model_id}")
    asr.load()
    vad = SileroVAD(models.fetch(models.SILERO_VAD, model_progress))
    diarizer = None if speakers == 1 else SherpaOnnxDiarizer(progress=model_progress)

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
    transcript = Transcript(
        language=language, profile=MeetingProfile(language=language), entries=entries
    )

    out_dir = out or audio_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{audio_file.stem}.transcript.md"
    json_path = out_dir / f"{audio_file.stem}.transcript.json"
    md_path.write_text(transcript.to_markdown())
    json_path.write_text(transcript.to_json())

    elapsed = time.monotonic() - started
    speed = duration / elapsed if elapsed else 0.0
    found = len({e.speaker for e in entries})
    click.echo(f"speakers: {found} found" if speakers is None else f"speakers: {speakers} given")
    click.echo(f"wrote {md_path} and .json ({elapsed:.1f}s, {speed:.1f}x realtime)")
    if print_markdown:
        click.echo()
        click.echo(transcript.to_markdown(), nl=False)


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
