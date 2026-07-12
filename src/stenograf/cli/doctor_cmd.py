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
    """One-time setup: capture permissions, desktop launcher, model downloads.

    macOS: launches the capture helper so both permission prompts (mic +
    system audio) appear now instead of at the start of your first meeting —
    nothing is recorded; the grant is scoped to the app the helper was
    launched from, so re-run this from each terminal app (or IDE) you will
    run meetings from. Windows: no prompt exists, so the microphone privacy
    toggle is checked and named instead. Then installs a double-clickable
    launcher and downloads every model the first meeting would otherwise
    stop to fetch; the models are cached machine-wide.
    """
    if not models_only:
        if sys.platform == "darwin":
            _grant_capture_permissions()  # only macOS gates capture behind TCC prompts
        elif sys.platform == "win32":
            _check_windows_mic_access()  # no prompt exists — read the privacy toggle now

    # The launcher lands before the model download: the download can fail (and
    # models fetch on first use anyway), the shortcut shouldn't be lost to that.
    # --models-only skips it — headless machines and CI have no desktop.
    if not models_only:
        from stenograf.shortcut import install_shortcut

        if (shortcut := install_shortcut()) is not None:
            click.echo(click.style("✓", fg="green") + f" launcher installed: {shortcut}")
            if sys.platform.startswith("linux"):  # menu entry; macOS/Windows land on the Desktop
                click.echo('  Look for "Stenograf" in your application menu.')
            else:
                click.echo("  Double-click it to start stenograf — no terminal needed.")

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


def _check_windows_mic_access() -> None:
    """Fail setup loud when the Windows mic privacy toggle denies capture.

    Windows never prompts desktop apps for the microphone (no TCC
    equivalent) — a denied toggle just makes the stream deliver zeros — so
    setup reads the consent store up front, mirroring the macOS grant step's
    fail-before-models behavior, and tells the user no prompt is coming.
    """
    from stenograf.capture.windows import mic_access_blocked

    if (blocked := mic_access_blocked()) is not None:
        raise click.ClickException(f"{blocked}, then re-run `steno setup`")
    click.echo(
        click.style("✓", fg="green") + " microphone access is allowed in Windows privacy settings."
    )
    click.echo(
        "  Windows shows no permission prompt — if the mic ever records only silence, "
        "check Settings > Privacy & security > Microphone."
    )


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
