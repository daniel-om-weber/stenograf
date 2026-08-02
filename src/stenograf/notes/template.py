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


def headings(template: str) -> list[str]:
    """The ``##``-and-deeper heading texts of ``template``, in order.

    Fence-aware: ``#`` lines inside a code fence are content, not phantom
    headings. The H1 is the title slot, not a section — it is never part of
    the validation set. A template with zero ``##`` headings is legal but
    degrades validation to its weaker non-empty form (see :mod:`.generate`).
    """
    found: list[str] = []
    in_fence = False
    for line in template.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _HEADING.match(line)
        if match:
            found.append(match.group(2))
    return found


def h1(text: str) -> str | None:
    """The first H1's text in ``text`` (fence-aware), or ``None``."""
    in_fence = False
    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
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
    in_fence = False
    for line in body.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            if current is not None:
                current.append(line)
            continue
        match = None if in_fence else _HEADING.match(line)
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


__all__ = ["DEFAULT_TEMPLATE", "content_lines", "h1", "headings", "sections"]
