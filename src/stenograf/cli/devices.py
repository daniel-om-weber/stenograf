"""``steno devices`` — the microphones this machine can record from.

The only place a user learns the id that ``[capture] mic_device`` and
``steno start --mic-device`` take: the ids are what survive a reboot and a
re-plug, and on Windows they are endpoint GUIDs nobody would type from memory.
Names are accepted by both too, and printed here beside their ids so either can
be copied.
"""

from __future__ import annotations

import click

from stenograf.cli.run import _library_errors


@click.command()
@_library_errors
def devices() -> None:
    """List the microphones available for recording."""
    from stenograf.capture.helper import list_input_devices, match_input_device
    from stenograf.flow import resolve_mic_device, standing_settings
    from stenograf.settings import settings_path

    listed = list_input_devices()
    if not listed:
        click.echo("No microphone is connected.")
        return

    # Resolved the way a run resolves it, so this can never describe a
    # selection differently from the meeting it is about to start.
    configured = resolve_mic_device(None, standing_settings())
    width = max(len(device.id) for device in listed)
    for device in listed:
        marks = " (system default)" if device.is_default else ""
        click.echo(f"{device.id.ljust(width)}  {device.name}{marks}")

    click.echo()
    if configured:
        # Whether the configured device is actually among them is the question
        # this command exists to answer, so it is answered rather than implied.
        found, problem = match_input_device(listed, configured)
        state = f"→ {found.name}" if found is not None else f"— {problem}, meetings will stop"
        click.echo(f"[capture] mic_device = {configured!r} {state}")
    else:
        click.echo(
            "Recording follows the system default. To fix one device, put its id in "
            f"[capture] mic_device in {settings_path()} (`steno settings edit`), "
            "or pass --mic-device to `steno start`."
        )
