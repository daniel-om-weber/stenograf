"""Combined-note export: one self-contained markdown file per meeting.

Written for an Obsidian vault but deliberately Obsidian-free: a vault is a
folder, a note is markdown, and the ``> [!quote]-`` collapsible-transcript
callout degrades to an ordinary blockquote elsewhere. Point ``[notes.export]
dir`` at any directory and every summarized meeting lands there as
``YYYY-MM-DD – Title.md``.

The note's body is the model's markdown verbatim — same body as the sibling
``.notes.md``. Provenance goes into the frontmatter (the vault-idiomatic
place, out of the reading view) rather than a footer.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from stenograf.notes.model import MeetingNotes
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
    if notes.provenance is not None:
        lines.append(f"notes_backend: {notes.provenance.backend}")
        if notes.provenance.model:
            lines.append(f'notes_model: "{notes.provenance.model}"')
    if transcript.language is not None:
        lines.append(f"language: {transcript.language.value}")
    lines += ["tags: [meeting]", "---", "", f"# {notes.title}", "", notes.body.rstrip()]
    lines += ["", "> [!quote]- Transcript"]
    lines += _quoted_transcript(transcript)
    return "\n".join(lines) + "\n"


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
