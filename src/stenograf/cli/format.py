"""Human-facing rendering helpers shared across the CLI commands."""

from __future__ import annotations

import click

# Settable speaker-count ranges, kept in sync with the --local/--remote and
# --speakers IntRange bounds. The unconstrained diarizer can *detect* more (or, on
# silence, zero) speakers than the user can set, so the "lock the detected count"
# hint is clamped to these — never suggesting an out-of-range or nonsensical re-run.
_MEETING_MAX_SPEAKERS = 8
_FILE_MAX_SPEAKERS = 16


def _describe_channel(channel) -> tuple[str, str]:
    """The human name and CLI flag for a channel's speaker count."""
    from stenograf.capture.base import Channel

    return ("local", "--local") if channel is Channel.MIC else ("remote", "--remote")


def _report_speaker_counts(counts) -> None:
    """Print per-channel speaker counts, flagging estimated ones as editable.

    Explicit counts are echoed as given; an auto-detected count shows what the
    finalize found and the exact flag to lock or correct it by re-running over
    the retained/recorded audio (PLAN.md §5 Stage 3a — a wrong estimate is never
    fatal, just re-run finalize)."""
    if not counts:
        click.echo("speakers: none found")
        return
    parts, corrections = [], []
    capped = False
    for count in counts:
        name, flag = _describe_channel(count.channel)
        if count.requested is None:
            parts.append(f"{count.detected} {name} (detected)")
            hint = _lock_hint(count.detected, _MEETING_MAX_SPEAKERS)
            if hint is not None:  # None → nothing to lock (a silent channel, 0 found)
                value, was_capped = hint
                corrections.append(f"{flag} {value}")
                capped = capped or was_capped
        else:
            parts.append(f"{count.requested} {name} (given)")
    click.echo("speakers: " + ", ".join(parts))
    if corrections:
        note = f" (estimate exceeded the {_MEETING_MAX_SPEAKERS}-speaker max)" if capped else ""
        click.echo(f"  estimated — re-run with {' '.join(corrections)} to lock or correct{note}")


def _lock_hint(detected: int, max_settable: int) -> tuple[int, bool] | None:
    """The value to suggest for locking an estimated count, clamped to the settable
    range, or ``None`` when there is nothing sensible to lock.

    Returns ``(value, capped)``: ``value`` is ``detected`` clamped into
    ``[1, max_settable]`` and ``capped`` flags that the raw estimate exceeded that
    range (an over-cluster artifact of unconstrained clustering — the displayed
    count stays the raw estimate; only the suggested lock value is capped).
    ``None`` when no speaker was found (``detected < 1``), so a silent channel never
    produces a nonsensical ``--local 0`` hint (PLAN.md §5 Phase 3→4 audit)."""
    if detected < 1:
        return None
    if detected > max_settable:
        return max_settable, True
    return detected, False


def _fmt_setting(value) -> str:
    """One effective value, TOML-flavored (bools lowercase, arrays bracketed)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, tuple):
        return "[" + ", ".join(f'"{item}"' for item in value) + "]"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
