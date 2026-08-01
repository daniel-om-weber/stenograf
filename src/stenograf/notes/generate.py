"""Backend-agnostic notes generation: prompt → complete → unwrap → validate.

Single-shot for a normal meeting; whole-turn map-reduce for one too long for a
single completion window. Nothing is ever written on failure — a typed error
propagates and the transcript stands untouched.

The template (:mod:`.template`) is the output schema. The gates in front of it
replace the four jobs the old JSON schema did, each named:

- *sanitizing* — :func:`~.markdown.unwrap_markdown` strips reasoning blocks,
  preamble chatter, and a fence wrapper;
- *refusal detection* — a response matching none of the template's headings,
  or one that returns the template unchanged, hard-fails;
- *truncation detection* — the backends check their own completion signal
  (mlx ``finish_reason``, Ollama ``done_reason``) and raise; the command
  backend has no such signal, so heading validation is its only net;
- *structure* — missing headings and empty sections become warnings, returned
  on the note's provenance so the evidence survives the screen.
"""

from __future__ import annotations

from collections.abc import Callable

from stenograf.notes.backend import (
    NotesBackend,
    NotesBackendUnavailableError,
    NotesGenerationError,
)
from stenograf.notes.markdown import strip_reasoning, unwrap_markdown
from stenograf.notes.model import MeetingNotes, NotesProvenance
from stenograf.notes.prompt import (
    build_messages,
    build_reduce_messages,
    chunk_entries,
    system_overhead,
    template_instruction,
)
from stenograf.notes.template import DEFAULT_TEMPLATE, content_lines, h1, headings, sections
from stenograf.transcript import Transcript

_MIN_CHUNK_BUDGET = 4_000
"""Smallest transcript-chunk budget worth running. Below this the system
prompt has eaten the window and chunking degenerates toward one model call per
speaker turn — a runaway, not a meeting summary. Fail loudly instead and name
what to cut."""

_FALLBACK_TITLE = "Meeting"
"""Last-resort title when the profile has none, the model returned no H1, and
the caller supplied no fallback — the same word the export's empty-slug
fallback uses."""


def generate_notes(
    transcript: Transcript,
    backend: NotesBackend,
    *,
    instructions: str | None = None,
    on_progress: Callable[[str], None] | None = None,
    template: str = DEFAULT_TEMPLATE,
    fallback_title: str | None = None,
) -> MeetingNotes:
    """Produce :class:`MeetingNotes` for ``transcript`` via ``backend``.

    Raises :class:`NotesBackendUnavailableError` before any model work if the
    backend can't run, and :class:`NotesGenerationError` when the model's
    output fails the gates above. A title the user set on the meeting always
    wins over the derived one; ``fallback_title`` (the CLI passes the meeting
    date) covers a response with no H1. ``on_progress`` receives one
    human-readable line before each model call — a long meeting through a slow
    model must not look like a hang."""
    if not transcript.entries:
        raise NotesGenerationError("the transcript has no entries — nothing to summarize")
    if not backend.is_available():
        raise NotesBackendUnavailableError(
            f"notes backend {backend.name!r} is not available — "
            "see `steno doctor` for what it needs"
        )
    return _generate(
        transcript,
        backend,
        instructions=instructions,
        on_progress=on_progress,
        template=template,
        fallback_title=fallback_title,
    )


def _generate(
    transcript: Transcript,
    backend: NotesBackend,
    *,
    instructions: str | None,
    on_progress: Callable[[str], None] | None,
    template: str,
    fallback_title: str | None,
) -> MeetingNotes:
    progress = on_progress or (lambda _message: None)
    # The system prompt (rules, meeting context, [notes] instructions) and the
    # template ride on every call, so they are charged against the backend's
    # input budget — only the remainder is available for transcript entries.
    overhead = system_overhead(transcript, instructions=instructions) + len(
        template_instruction(template)
    )
    budget = backend.max_input_chars - overhead
    if budget < _MIN_CHUNK_BUDGET:
        raise NotesGenerationError(
            f"the notes prompt (rules, meeting context, [notes] instructions, template) "
            f"is {overhead} chars, leaving {budget} of the backend's "
            f"{backend.max_input_chars}-char input budget for the transcript — "
            "shorten the [notes] instructions file or raise [notes] max_input_chars"
        )
    chunks = chunk_entries(transcript.entries, max_chars=budget)
    if len(chunks) == 1:
        progress(f"summarizing with {backend.name} ({backend.model}), single pass")
        messages = build_messages(transcript, instructions=instructions, template=template)
        raw = backend.complete(messages)
        strategy = "single-shot"
    else:
        partials = []
        for i, chunk in enumerate(chunks, start=1):
            progress(f"summarizing portion {i}/{len(chunks)} with {backend.name}")
            messages = build_messages(
                transcript, instructions=instructions, entries=chunk, partial=True
            )
            partials.append(_checked_partial(backend.complete(messages), portion=i))
        progress(f"merging {len(chunks)} portion notes")
        reduce_messages = build_reduce_messages(
            transcript, partials, instructions=instructions, template=template
        )
        # The reduce call is where every portion converges; over the budget it
        # would degrade silently (server-side truncation, output cut-off), so
        # check it the same way the chunks were budgeted.
        reduce_size = sum(len(m["content"]) for m in reduce_messages)
        if reduce_size > backend.max_input_chars:
            raise NotesGenerationError(
                f"the merge step's prompt is {reduce_size} chars, over the backend's "
                f"{backend.max_input_chars}-char input budget — raise [notes] "
                "max_input_chars, or shorten the instructions file"
            )
        raw = backend.complete(reduce_messages)
        strategy = f"map-reduce ({len(chunks)} portions)"

    model_title, body = unwrap_markdown(raw)
    warnings = _validate(body, template, raw=raw)
    if model_title is not None and model_title == h1(template):
        model_title = None  # the template's own placeholder H1, echoed back
    title = transcript.profile.title or model_title
    if title is None:
        title = fallback_title or _FALLBACK_TITLE
        warnings.append(f'the response had no title; "{title}" stands in')
    return MeetingNotes(
        title=title.strip(),
        body=body,
        provenance=NotesProvenance(
            backend=backend.name,
            model=backend.model,
            strategy=strategy,
            language=transcript.language.value if transcript.language else None,
            warnings=tuple(warnings),
        ),
    )


def _checked_partial(raw: str, *, portion: int) -> str:
    """One map portion's bullet notes, gated.

    The gate is honest about its strength: the old JSON parse caught any
    refusal (no ``{``); requiring at least one bullet line catches empty and
    prose-shaped refusals ("I can't summarize this") but not one that emits
    bullets. Accepted — the reduce output still faces the full validation."""
    text = strip_reasoning(raw).strip()
    if not any(line.lstrip().startswith(("- ", "* ")) for line in text.splitlines()):
        raise NotesGenerationError(
            f"portion {portion} produced no bullet notes; response started: {text[:200]!r}"
        )
    return text


def _validate(body: str, template: str, *, raw: str) -> list[str]:
    """The shape gate: hard failures raise, soft ones return as warnings.

    Matching is verbatim against the headings of the template actually used —
    which is what keeps a German or user-authored template exactly as valid
    as the built-in one."""
    if not body.strip():
        raise NotesGenerationError(
            f"the notes backend returned no usable markdown; "
            f"response started: {raw.strip()[:200]!r}"
        )
    wanted = headings(template)
    if not wanted:
        # A template with no ## headings is legal but degrades the gate to
        # "some content exists" — stated as weaker in the plan, warned here.
        if not content_lines(body, template):
            raise NotesGenerationError("the model returned the template unchanged — no content")
        return ["the template has no ## headings, so structure was not validated"]
    present = sections(body)
    matched = [heading for heading in wanted if heading in present]
    if not matched:
        raise NotesGenerationError(
            f"the response matched none of the template's headings — "
            f"not a note; response started: {raw.strip()[:200]!r}"
        )
    if not content_lines(body, template):
        raise NotesGenerationError("the model returned the template unchanged — an empty note")
    warnings: list[str] = []
    template_lines = {line.strip() for line in template.splitlines() if line.strip()}
    for heading in wanted:
        if heading not in present:
            warnings.append(f'section "## {heading}" is missing')
        elif not any(
            line.strip() and line.strip() not in template_lines for line in present[heading]
        ):
            warnings.append(f'section "## {heading}" is empty')
    return warnings
