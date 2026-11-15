"""Human-facing rendering helpers shared across the CLI commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from stenograf.capture.base import Channel
    from stenograf.session import SpeakerCount


def _describe_channel(channel: Channel) -> tuple[str, str]:
    """The human name and CLI flag for a channel's speaker count."""
    from stenograf.capture.base import Channel

    return ("local", "--local") if channel is Channel.MIC else ("remote", "--remote")


def _report_speaker_counts(counts: list[SpeakerCount]) -> None:
    """Print per-channel speaker counts, flagging estimated ones as editable.

    Explicit counts are echoed as given; an auto-detected count shows what the
    finalize found and the exact flag to lock or correct it by re-running over
    the retained/recorded audio (just re-run finalize)."""
    if not counts:
        click.echo("speakers: none found")
        return
    parts, corrections = [], []
    for count in counts:
        name, flag = _describe_channel(count.channel)
        if count.requested is None:
            parts.append(f"{count.detected} {name} (detected)")
            # A silent channel (0 found) has nothing to lock; `--local 0` would
            # mean "no microphone", not "this count".
            if count.detected >= 1:
                corrections.append(f"{flag} {count.detected}")
        else:
            parts.append(f"{count.requested} {name} (given)")
    click.echo("speakers: " + ", ".join(parts))
    if corrections:
        click.echo(f"  estimated — re-run with {' '.join(corrections)} to lock or correct")


def _fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
