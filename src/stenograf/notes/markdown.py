"""Markdown response hygiene: reasoning blocks off, chat wrapping off.

The markdown counterpart of the old ``{``-scan JSON extraction: a model may
wrap the note in a code fence, put chatter before it ("Here are your
notes!"), after it ("Anything else?"), or think out loud first — none of that
may reach the written note. Shared by every backend (mlx used to strip its
own ``<think>`` block; the command backend can leak one from any model), so
it lives here, once.
"""

from __future__ import annotations

import re

from stenograf.notes.template import h1 as _outside_h1

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_OPEN_THINK = re.compile(r"<think>.*\Z", re.DOTALL)
_FENCE_LINE = re.compile(r"^\s*(```|~~~)")
_BARE_FENCE = re.compile(r"^\s*(```|~~~)\s*$")
_H1_LINE = re.compile(r"^#\s+(.+?)\s*#*\s*$")


def strip_reasoning(text: str) -> str:
    """Drop ``<think>…</think>`` blocks, wherever they sit.

    Deliberately unanchored (mlx's old strip was ``\\A``-anchored): a think
    block that doesn't start the output would otherwise survive, and a ``# ``
    line inside it would read as the note's title. An *unclosed* ``<think>``
    means the output was cut mid-reasoning — everything from it on is
    reasoning, not note."""
    text = _THINK_BLOCK.sub("", text)
    return _OPEN_THINK.sub("", text)


def unwrap_markdown(raw: str) -> tuple[str | None, str]:
    """``(title, body)`` of the note inside a model's raw response.

    - reasoning is stripped first (see :func:`strip_reasoning`);
    - with an H1 outside any code fence, everything before it is preamble and
      dropped;
    - with no such H1, a fenced block *containing* one is the note delivered
      inside a fence — its content replaces the whole response, which also
      discards suffix chatter after the fence (the counterpart of the JSON
      path's ``raw_decode`` stopping at the closing brace);
    - a trailing unbalanced fence line (the leftover of a half-stripped
      wrapper) is dropped;
    - the first H1 line becomes the title and leaves the body, so the
      renderer emits exactly one H1 from the *resolved* title.

    Unfenced chatter *after* the last section is accepted as body — there is
    no delimiter to stop at, and cutting real prose would be worse. ``title``
    is ``None`` when the response has no H1 (validation warns; the caller's
    fallback takes over)."""
    text = strip_reasoning(raw)
    if _outside_h1(text) is not None:
        text = _from_first_h1(text)
    else:
        fenced = _fenced_block_with_h1(text)
        if fenced is not None:
            text = fenced
    text = _strip_trailing_orphan_fence(text)
    return _split_title(text)


def _from_first_h1(text: str) -> str:
    lines = text.splitlines()
    in_fence = False
    for i, line in enumerate(lines):
        if _FENCE_LINE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence and _H1_LINE.match(line):
            return "\n".join(lines[i:])
    return text


def _fenced_block_with_h1(text: str) -> str | None:
    """The content of the first fenced block that holds an H1, if any."""
    lines = text.splitlines()
    block: list[str] | None = None
    for line in lines:
        if _FENCE_LINE.match(line):
            if block is None:
                block = []
                continue
            if any(_H1_LINE.match(inner) for inner in block):
                return "\n".join(block)
            block = None
            continue
        if block is not None:
            block.append(line)
    # An unclosed fence that holds the H1: everything after the opener is the
    # note (the closing fence was cut off with the suffix).
    if block is not None and any(_H1_LINE.match(inner) for inner in block):
        return "\n".join(block)
    return None


def _strip_trailing_orphan_fence(text: str) -> str:
    lines = text.rstrip().splitlines()
    if lines and _BARE_FENCE.match(lines[-1]):
        markers = sum(1 for line in lines if _FENCE_LINE.match(line))
        if markers % 2 == 1:
            lines.pop()
    return "\n".join(lines).rstrip()


def _split_title(text: str) -> tuple[str | None, str]:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        match = _H1_LINE.match(stripped)
        if match:
            return match.group(1), "\n".join(lines[i + 1 :]).strip("\n")
        break
    return None, text.strip("\n")


__all__ = ["strip_reasoning", "unwrap_markdown"]
