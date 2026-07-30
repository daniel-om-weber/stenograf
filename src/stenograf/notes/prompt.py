"""Prompt construction for notes generation — shared by every backend.

The anti-hallucination rules live here, once: notes must cite what was said
(speaker + timestamp), never invent attendees, decisions, or due dates. A
user's own style guide (``[notes] instructions`` in settings.toml) is
*appended* to — never replaces — the built-in system prompt, so those rules
survive customization.
"""

from __future__ import annotations

from stenograf.config import Language
from stenograf.transcript import Transcript, TranscriptEntry, format_timestamp


def template_instruction(template: str) -> str:
    """The format instruction that rides at the very END of the prompt.

    Position is deliberate and load-bearing: the old schema instruction was
    appended to the *last* message by both grammarless backends (recency —
    the format spec must not sit in front of up to 400k chars of transcript),
    and the template inherits that seat. One wording, here, so backends can
    never drift apart. Headings are kept verbatim — the template's language
    is its author's choice; the prose language is the system prompt's."""
    return (
        "Fill in this markdown template for the meeting above. Keep its "
        "headings exactly as written (do not translate them); replace the "
        "angle-bracket placeholders with content, or leave a section empty "
        "when the meeting had none. Respond with the completed markdown "
        "only — no code fence, nothing before or after it:\n\n" + template
    )


_PARTIAL_INSTRUCTION = (
    'Return the notes for this portion as plain markdown bullets (lines starting with "- "), '
    "covering what was discussed, decided, assigned, and left open — "
    "no headings, no title, no other text."
)
"""The map step's format: neutral bullets. Shape and context belong to the
reduce call alone — a portion must not fill the template, or N portions
produce N conflicting layouts the reduce step has to un-merge."""


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
items, owners, or due dates; leave out what the transcript does not say.
- Attribute claims to the speaker labels as they appear in the transcript.
- For each action item, keep the [h:mm:ss] timestamp where it was raised, \
when the transcript makes that clear.
- Speaker labels like "Local-1"/"Remote-2" are automatic; if attendee names \
are given below, map labels to names only when the transcript itself makes \
the mapping obvious.
- The title must be short and specific to this meeting's content.
- Write the notes in {language}. Template headings are not prose — they stay \
exactly as the template gives them."""


def _system_prompt(
    transcript: Transcript, *, instructions: str | None, partial: bool = False
) -> str:
    """The system message every notes request shares — assembled once, here,
    so the anti-hallucination rules can never differ between the map and
    reduce steps of a long meeting."""
    system = _SYSTEM.format(language=_language_name(transcript.language))
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
    return system


def system_overhead(transcript: Transcript, *, instructions: str | None) -> int:
    """Rendered size of this meeting's system prompt, in chars.

    The *map* variant, which is the longest one a run can send — a chunk budget
    derived from it can never exceed what any actual call has left. Exists so
    :mod:`.generate` can charge the system prompt against
    ``max_input_chars`` instead of letting instructions and meeting context
    ride for free. The template is charged separately there — it rides in the
    user message, not here."""
    return len(_system_prompt(transcript, instructions=instructions, partial=True))


def build_messages(
    transcript: Transcript,
    *,
    instructions: str | None = None,
    entries: list[TranscriptEntry] | None = None,
    partial: bool = False,
    template: str | None = None,
) -> list[dict[str, str]]:
    """Chat messages asking for notes over ``entries`` (default: the whole
    transcript). ``partial=True`` marks a map-reduce chunk — neutral bullets,
    no template (content in map, shape in reduce); a full request carries
    ``template`` as the last thing in the prompt."""
    system = _system_prompt(transcript, instructions=instructions, partial=partial)
    body = _render_entries(entries if entries is not None else transcript.entries)
    user = f"The meeting transcript:\n\n{body}"
    if partial:
        user += "\n" + _PARTIAL_INSTRUCTION
    elif template is not None:
        user += "\n" + template_instruction(template)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_reduce_messages(
    transcript: Transcript,
    partials: list[str],
    *,
    instructions: str | None = None,
    template: str | None = None,
) -> list[dict[str, str]]:
    """The reduce step: merge per-portion bullet notes into the one note."""
    system = _system_prompt(transcript, instructions=instructions)
    joined = "\n\n".join(
        f"Portion {i} notes:\n{partial}" for i, partial in enumerate(partials, start=1)
    )
    user = (
        "The meeting was summarized in consecutive portions. Merge these "
        "portion notes into ONE set of notes for the whole meeting — "
        "deduplicate, keep every distinct decision and action item, and "
        "write one coherent summary:\n\n" + joined
    )
    if template is not None:
        user += "\n\n" + template_instruction(template)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
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
