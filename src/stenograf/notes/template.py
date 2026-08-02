"""The notes template — the markdown skeleton the model fills in.

The template *is* the output schema: its headings are what validation checks
the response against (:mod:`.generate`), so this module is the one owner of
both the built-in layout and heading extraction. A template is plain literal
markdown — its headings are matched verbatim, which keeps a translated or
user-authored template language-safe by construction; the *prose* language is
a separate instruction (:mod:`.prompt`).
"""

from __future__ import annotations

import re
from collections.abc import Iterator

DEFAULT_TEMPLATE = """\
# <Short, specific meeting title derived from the content>

<A few paragraphs covering what was discussed and concluded.>

## Decisions

- <one decision per bullet>

## Action items

- [ ] <task> — **<owner>** (due <date as spoken>) [<h:mm:ss where it was raised>]

## Highlights

- **<speaker>**: <the statement worth keeping>

## Open questions

- <one open question per bullet>
"""
"""The built-in layout: today's fixed sections, as markdown. Placeholders are
angle-bracketed so a response that merely echoes the template is detectable
(:func:`content_lines`)."""

_HEADING = re.compile(r"^(#{2,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_H1 = re.compile(r"^#\s+(.+?)\s*#*\s*$")


def is_fence(line: str) -> bool:
    """Whether ``line`` is a code-fence marker (``` or ~~~)."""
    return _FENCE.match(line) is not None


def fence_walk(text: str) -> Iterator[tuple[str, bool, bool]]:
    """``(line, is_marker, in_fence)`` per line — the ONE fence-state walk.

    Every markdown reader here must agree on what is inside a fence, or a
    ``#`` line in a code block reads as a heading to one of them and not
    another. A marker line reports the state *after* its toggle (an opener is
    ``in_fence=True``); content lines report the state they sit in.
    """
    in_fence = False
    for line in text.splitlines():
        marker = is_fence(line)
        if marker:
            in_fence = not in_fence
        yield line, marker, in_fence


def headings(template: str) -> list[str]:
    """The ``##``-and-deeper heading texts of ``template``, in order.

    Fence-aware: ``#`` lines inside a code fence are content, not phantom
    headings. The H1 is the title slot, not a section — it is never part of
    the validation set. A template with zero ``##`` headings is legal but
    degrades validation to its weaker non-empty form (see :mod:`.generate`).
    """
    found: list[str] = []
    for line, marker, in_fence in fence_walk(template):
        if marker or in_fence:
            continue
        match = _HEADING.match(line)
        if match:
            found.append(match.group(2))
    return found


def h1(text: str) -> str | None:
    """The first H1's text in ``text`` (fence-aware), or ``None``."""
    for line, marker, in_fence in fence_walk(text):
        if marker or in_fence:
            continue
        match = _H1.match(line)
        if match:
            return match.group(1)
    return None


def sections(body: str) -> dict[str, list[str]]:
    """Map of ``##``-heading text → the raw lines under it, fence-aware.

    Lines before the first heading (the summary area) are nobody's section
    and are not in the map. A heading line inside a code fence is content of
    the section it sits in, not a new section."""
    result: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line, marker, in_fence in fence_walk(body):
        match = None if marker or in_fence else _HEADING.match(line)
        if match:
            current = result.setdefault(match.group(2), [])
            continue
        if current is not None:
            current.append(line)
    return result


def content_lines(body: str, template: str) -> list[str]:
    """The lines of ``body`` that carry actual content.

    Headings don't count, blank lines don't count, and lines that appear
    verbatim in ``template`` don't count either — so a model that returns the
    template unchanged (headings plus angle-bracket placeholders) has zero
    content, which validation treats as a failed generation rather than a
    complete-looking empty note."""
    template_lines = {line.strip() for line in template.splitlines() if line.strip()}
    kept: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or _HEADING.match(stripped) or _H1.match(stripped):
            continue
        if stripped in template_lines:
            continue
        kept.append(stripped)
    return kept


__all__ = [
    "DEFAULT_TEMPLATE",
    "content_lines",
    "fence_walk",
    "h1",
    "headings",
    "is_fence",
    "sections",
]
