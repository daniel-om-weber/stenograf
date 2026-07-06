"""Command-line interface: ``stenograf`` / ``steno``."""

from __future__ import annotations

import click

from stenograf import __version__
from stenograf.config import Language, MeetingProfile
from stenograf.doctor import run_checks


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
def doctor() -> None:
    """Check this machine's readiness (permissions, OS version, models)."""
    checks = run_checks()
    for check in checks:
        symbol = click.style("✓", fg="green") if check.ok else click.style("✗", fg="red")
        click.echo(f" {symbol} {check.name}: {check.detail}")
    if not all(check.ok for check in checks):
        raise SystemExit(1)
