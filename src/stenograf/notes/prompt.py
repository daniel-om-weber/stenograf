"""Prompt construction for notes generation — shared by every backend.

The anti-hallucination rules live here, once: notes must cite what was said
(speaker + timestamp), never invent attendees, decisions, or due dates. A
user's own style guide (``[notes] instructions`` in settings.toml) is
*appended* to — never replaces — the built-in system prompt, so those rules
survive customization (PLAN.md §5 D4).
"""

from __future__ import annotations

import json

from stenograf.config import Language
from stenograf.transcript import Transcript, TranscriptEntry, format_timestamp

NOTES_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": "Short, specific meeting title derived from the content.",
        },
        "summary": {
            "type": "string",
            "description": "A few paragraphs covering what was discussed and concluded.",
        },
        "decisions": {"type": "array", "items": {"type": "string"}},
        "action_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "owner": {"type": ["string", "null"]},
                    "due": {"type": ["string", "null"]},
                    "timestamp": {"type": ["number", "null"]},
                },
                "required": ["task"],
            },
        },
        "highlights": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker": {"type": "string"},
                    "highlight": {"type": "string"},
                },
                "required": ["speaker", "highlight"],
            },
        },
        "open_questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "summary", "decisions", "action_items", "open_questions"],
}

def schema_instruction(schema: dict) -> str:
    """The instruction a backend *without* decode-time grammar appends.

    Ollama enforces the schema server-side (``format=``) and must NOT use
    this; mlx and command inline it and rely on the tolerant JSON extraction
    in :mod:`.generate`. One wording, here, so the two grammarless backends
    can never drift apart."""
    return (
        "Respond with exactly one JSON object matching this JSON Schema — "
        "no other text before or after it:\n" + json.dumps(schema, ensure_ascii=False)
    )


_LANGUAGE_NAMES = {Language.GERMAN: "German", Language.ENGLISH: "English"}


def _language_name(language: Language | None) -> str:
    """The name the system prompt writes notes in; undetected stays generic."""
    if language is None:
        return "the language of the transcript"
    return _LANGUAGE_NAMES.get(language, "the language of the transcript")


_SYSTEM = """\
You turn a meeting transcript into precise written notes.

Rules — follow them exactly:
- Report only what the transcript supports. Never invent decisions, action \
items, owners, or due dates; when the transcript doesn't say, use null.
- Attribute claims to the speaker labels as they appear in the transcript.
- For each action item, set "timestamp" to the [h:mm:ss] second where it was \
raised (as a number of seconds), or null if unclear.
- Speaker labels like "Local-1"/"Remote-2" are automatic; if attendee names \
are given below, map labels to names only when the transcript itself makes \
the mapping obvious.
- The title must be short and specific to this meeting's content.
- Write the notes in {language}."""


def build_messages(
    transcript: Transcript,
    *,
    instructions: str | None = None,
    entries: list[TranscriptEntry] | None = None,
    partial: bool = False,
) -> list[dict[str, str]]:
    """Chat messages asking for notes over ``entries`` (default: the whole
    transcript). ``partial=True`` marks a map-reduce chunk, whose notes are
    later merged — the model is told not to pad missing context."""
    language = _language_name(transcript.language)
    system = _SYSTEM.format(language=language)
    context = _context_lines(transcript)
    if context:
        system += "\n\nMeeting context:\n" + "\n".join(context)
    if partial:
        system += (
            "\n\nThis is one portion of a longer meeting. Take notes on this "
            "portion only; do not guess at what came before or after."
        )
    if instructions:
        system += "\n\nAdditional instructions from the user:\n" + instructions.strip()
    body = _render_entries(entries if entries is not None else transcript.entries)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"The meeting transcript:\n\n{body}"},
    ]


def build_reduce_messages(
    transcript: Transcript,
    partials: list[str],
    *,
    instructions: str | None = None,
) -> list[dict[str, str]]:
    """The reduce step: merge per-chunk notes JSON into one set of notes."""
    language = _language_name(transcript.language)
    system = _SYSTEM.format(language=language)
    context = _context_lines(transcript)
    if context:
        system += "\n\nMeeting context:\n" + "\n".join(context)
    if instructions:
        system += "\n\nAdditional instructions from the user:\n" + instructions.strip()
    joined = "\n\n".join(
        f"Portion {i} notes:\n{partial}" for i, partial in enumerate(partials, start=1)
    )
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                "The meeting was summarized in consecutive portions. Merge these "
                "portion notes into ONE set of notes for the whole meeting — "
                "deduplicate, keep every distinct decision and action item, and "
                "write one coherent summary:\n\n" + joined
            ),
        },
    ]


def chunk_entries(entries: list[TranscriptEntry], *, max_chars: int) -> list[list[TranscriptEntry]]:
    """Split whole speaker turns into chunks of at most ``max_chars`` rendered
    characters. No entry is ever dropped; a single over-budget turn becomes its
    own chunk rather than being truncated."""
    chunks: list[list[TranscriptEntry]] = []
    current: list[TranscriptEntry] = []
    size = 0
    for entry in entries:
        cost = len(_render_entry(entry)) + 2
        if current and size + cost > max_chars:
            chunks.append(current)
            current, size = [], 0
        current.append(entry)
        size += cost
    if current:
        chunks.append(current)
    return chunks


def _context_lines(transcript: Transcript) -> list[str]:
    profile = transcript.profile
    lines = []
    if profile.title:
        lines.append(f"- Title: {profile.title}")
    if profile.attendee_names:
        lines.append(f"- Attendees: {', '.join(profile.attendee_names)}")
    if profile.glossary:
        lines.append(f"- Domain terms: {', '.join(profile.glossary)}")
    return lines


def _render_entries(entries: list[TranscriptEntry]) -> str:
    return "\n\n".join(_render_entry(e) for e in entries) + "\n"


def _render_entry(entry: TranscriptEntry) -> str:
    return f"{entry.speaker} [{format_timestamp(entry.start)}]: {entry.text}"

