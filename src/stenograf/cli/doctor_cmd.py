"""``steno doctor`` and ``steno setup`` — machine readiness and first-run setup."""

from __future__ import annotations

import sys

import click

from stenograf import loaders
from stenograf.doctor import run_checks


@click.command()
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
        # Bare SystemExit, not ClickException: the table above already says
        # what failed, and ClickException would append a redundant "Error:" line.
        raise SystemExit(1)


@click.command()
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
