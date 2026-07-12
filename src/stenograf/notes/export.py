"""Combined-note export: one self-contained markdown file per meeting.

Written for an Obsidian vault but deliberately Obsidian-free: a vault is a
folder, a note is markdown, and the ``> [!quote]-`` collapsible-transcript
callout degrades to an ordinary blockquote elsewhere. Point ``[notes.export]
dir`` at any directory and every summarized meeting lands there as
``YYYY-MM-DD – Title.md`` (PLAN.md §5 D7).
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from stenograf.notes.model import ActionItem, MeetingNotes
from stenograf.output import atomic_write_text
from stenograf.transcript import Transcript, format_timestamp

_SLUG_MAX_CHARS = 80
_STRIP = re.compile(r"[][#^|]")  # markdown/Obsidian link syntax — drop silently
_REPLACE = re.compile(r'[/\\:*?"<>]')  # path separators & Windows-invalid — space out


def export_note(
    transcript: Transcript,
    notes: MeetingNotes,
    directory: Path,
    *,
    created_at: datetime,
) -> Path:
    """Write the combined note into ``directory`` and return its path.

    Never overwrites: a filename collision gets an `` (2)`` suffix, so two
    meetings titled alike on the same day both keep their notes."""
    directory = directory.expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    base = f"{created_at:%Y-%m-%d} – {_slug(notes.title)}"
    path = directory / f"{base}.md"
    n = 2
    while path.exists():
        path = directory / f"{base} ({n}).md"
        n += 1
    atomic_write_text(path, render_note(transcript, notes, created_at=created_at))
    return path


def render_note(transcript: Transcript, notes: MeetingNotes, *, created_at: datetime) -> str:
    lines = [
        "---",
        f'title: "{notes.title.replace(chr(34), chr(39))}"',
        f"date: {created_at:%Y-%m-%d}",
        f"created: {created_at:%Y-%m-%dT%H:%M}",
        "source: stenograf",
        "type: meeting",
    ]
    if transcript.language is not None:
        lines.append(f"language: {transcript.language.value}")
    lines += ["tags: [meeting]", "---", "", f"# {notes.title}", "", notes.summary.rstrip()]
    if notes.decisions:
        lines += ["", "## Decisions", ""]
        lines += [f"- {d}" for d in notes.decisions]
    if notes.action_items:
        lines += ["", "## Action items", ""]
        lines += _action_items_by_owner(notes.action_items)
    if notes.open_questions:
        lines += ["", "## Open questions", ""]
        lines += [f"- {q}" for q in notes.open_questions]
    lines += ["", "> [!quote]- Transcript"]
    lines += _quoted_transcript(transcript)
    return "\n".join(lines) + "\n"


def _action_items_by_owner(items: tuple[ActionItem, ...]) -> list[str]:
    """Group action items under their owner so each participant sees theirs at
    a glance; unassigned items close the section."""
    by_owner: dict[str | None, list[ActionItem]] = {}
    for item in items:
        by_owner.setdefault(item.owner, []).append(item)
    lines: list[str] = []
    unassigned = by_owner.pop(None, [])
    for owner, owned in by_owner.items():
        lines.append(f"**{owner}**")
        lines += [_item_line(i) for i in owned]
        lines.append("")
    if unassigned:
        if by_owner:
            lines.append("**Unassigned**")
        lines += [_item_line(i) for i in unassigned]
        lines.append("")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _item_line(item: ActionItem) -> str:
    line = f"- [ ] {item.task}"
    if item.due:
        line += f" (due {item.due})"
    return line


def _quoted_transcript(transcript: Transcript) -> list[str]:
    lines = []
    for entry in transcript.entries:
        marker = " *(overlap)*" if entry.provisional else ""
        stamp = format_timestamp(entry.start)
        lines.append(f"> **{entry.speaker}** [{stamp}]{marker}: {entry.text}")
        lines.append(">")
    if lines:
        lines.pop()  # no trailing empty quote line
    return lines


def _slug(title: str) -> str:
    slug = _STRIP.sub("", title)
    slug = _REPLACE.sub(" ", slug)
    slug = " ".join(slug.split())
    return slug[:_SLUG_MAX_CHARS].rstrip() or "Meeting"

