"""Backend-agnostic notes generation: prompt → complete → parse → validate.

Single-shot for a normal meeting; whole-turn map-reduce for one too long for a
single completion window. Nothing is ever written on failure — a typed error
propagates and the transcript stands untouched (PLAN.md §5 D4).
"""

from __future__ import annotations

import json

from stenograf.notes.backend import (
    NotesBackend,
    NotesBackendUnavailableError,
    NotesGenerationError,
)
from stenograf.notes.model import ActionItem, MeetingNotes, NotesProvenance, SpeakerHighlight
from stenograf.notes.prompt import (
    NOTES_SCHEMA,
    build_messages,
    build_reduce_messages,
    chunk_entries,
)
from stenograf.transcript import Transcript

MAX_SINGLE_SHOT_CHARS = 48_000
"""Rendered-transcript budget for one completion (~12k tokens of speech —
roughly a one-hour meeting). Sized for the small *local* models the Ollama
backend defaults to; hosted models would take far more, but a smaller
single-shot ceiling only costs long meetings one extra merge pass."""


def generate_notes(
    transcript: Transcript, backend: NotesBackend, *, instructions: str | None = None
) -> MeetingNotes:
    """Produce :class:`MeetingNotes` for ``transcript`` via ``backend``.

    Raises :class:`NotesBackendUnavailableError` before any model work if the
    backend can't run, and :class:`NotesGenerationError` when the model's
    output can't be parsed into valid notes. A title the user set on the
    meeting always wins over the derived one."""
    if not transcript.entries:
        raise NotesGenerationError("the transcript has no entries — nothing to summarize")
    if not backend.is_available():
        raise NotesBackendUnavailableError(
            f"notes backend {backend.name!r} is not available — "
            "see `steno doctor` for what it needs"
        )
    return _generate(transcript, backend, instructions=instructions)


def _generate(
    transcript: Transcript, backend: NotesBackend, *, instructions: str | None
) -> MeetingNotes:
    chunks = chunk_entries(transcript.entries, max_chars=MAX_SINGLE_SHOT_CHARS)
    if len(chunks) == 1:
        messages = build_messages(transcript, instructions=instructions)
        obj = _parse_notes_object(backend.complete(messages, NOTES_SCHEMA))
        strategy = "single-shot"
    else:
        partials = []
        for chunk in chunks:
            messages = build_messages(
                transcript, instructions=instructions, entries=chunk, partial=True
            )
            partial = _parse_notes_object(backend.complete(messages, NOTES_SCHEMA))
            partials.append(json.dumps(partial, ensure_ascii=False))
        reduce_messages = build_reduce_messages(transcript, partials, instructions=instructions)
        obj = _parse_notes_object(backend.complete(reduce_messages, NOTES_SCHEMA))
        strategy = f"map-reduce ({len(chunks)} portions)"
    notes = _notes_from_object(obj)
    return MeetingNotes(
        title=transcript.profile.title or notes.title,
        summary=notes.summary,
        decisions=notes.decisions,
        action_items=notes.action_items,
        highlights=notes.highlights,
        open_questions=notes.open_questions,
        provenance=NotesProvenance(
            backend=backend.name,
            model=backend.model,
            strategy=strategy,
            language=transcript.language.value if transcript.language else None,
        ),
    )


def _parse_notes_object(raw: str) -> dict:
    """Extract the first top-level JSON object from ``raw``.

    Ollama's ``format=schema`` returns bare JSON, but a generic command backend
    may wrap it in prose or a ``` fence — scan for each ``{`` and try a real
    decode from there, so anything containing one valid object parses."""
    decoder = json.JSONDecoder()
    index = raw.find("{")
    while index != -1:
        try:
            obj, _ = decoder.raw_decode(raw, index)
        except json.JSONDecodeError:
            index = raw.find("{", index + 1)
            continue
        if isinstance(obj, dict):
            return obj
        index = raw.find("{", index + 1)
    raise NotesGenerationError(
        f"the notes backend returned no JSON object; response started: {raw.strip()[:200]!r}"
    )


def _notes_from_object(obj: dict) -> MeetingNotes:
    """Validate the parsed object against what :class:`MeetingNotes` needs.

    Deliberately lenient about extras and null-vs-absent, strict about the
    fields the note is built from — a malformed response must fail loudly here,
    not render a broken note."""
    title = obj.get("title")
    summary = obj.get("summary")
    if not isinstance(title, str) or not title.strip():
        raise NotesGenerationError("notes response is missing a usable 'title'")
    if not isinstance(summary, str) or not summary.strip():
        raise NotesGenerationError("notes response is missing a usable 'summary'")
    return MeetingNotes(
        title=title.strip(),
        summary=summary.strip(),
        decisions=_str_tuple(obj, "decisions"),
        action_items=tuple(_action_item(a) for a in _list_of_dicts(obj, "action_items")),
        highlights=tuple(
            SpeakerHighlight(speaker=str(h.get("speaker", "")), highlight=str(h["highlight"]))
            for h in _list_of_dicts(obj, "highlights")
            if h.get("highlight")
        ),
        open_questions=_str_tuple(obj, "open_questions"),
    )


def _action_item(a: dict) -> ActionItem:
    task = a.get("task")
    if not isinstance(task, str) or not task.strip():
        raise NotesGenerationError(f"action item without a 'task': {a!r:.100}")
    timestamp = a.get("timestamp")
    return ActionItem(
        task=task.strip(),
        owner=_opt_str(a.get("owner")),
        due=_opt_str(a.get("due")),
        timestamp=float(timestamp) if isinstance(timestamp, int | float) else None,
    )


def _opt_str(value: object) -> str | None:
    return value.strip() or None if isinstance(value, str) else None


def _str_tuple(obj: dict, key: str) -> tuple[str, ...]:
    value = obj.get(key, ())
    if not isinstance(value, list):
        raise NotesGenerationError(f"notes response field {key!r} is not a list")
    return tuple(str(item) for item in value if str(item).strip())


def _list_of_dicts(obj: dict, key: str) -> list[dict]:
    value = obj.get(key, ())
    if not isinstance(value, list) or not all(isinstance(i, dict) for i in value):
        raise NotesGenerationError(f"notes response field {key!r} is not a list of objects")
    return value
