"""Gathering a run's vocabulary: the standing baseline plus per-run terms.

Glossary terms and attendee names drive both vocabulary layers — they boost
the decoder as it transcribes (:mod:`stenograf.asr.biasing`) and then snap the
near-misses it still got wrong (:mod:`stenograf.glossary`). This module only
*collects* them: one precedence rule shared by every entry point, so a CLI
flag, the ``[vocab]`` settings table and a meeting preset can never disagree
about what a run's terms are.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from stenograf.settings import SettingsError

if TYPE_CHECKING:
    from pathlib import Path

    from stenograf.settings import VocabSettings


def collect_terms(
    glossary: tuple[str, ...],
    glossary_file: Path | None,
    attendee: tuple[str, ...],
    *,
    vocab: VocabSettings | None = None,
    extra_vocab: VocabSettings | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Gather glossary terms (inline + file) and attendee names for one run.

    ``vocab`` (the ``[vocab]`` settings table) is the standing baseline: its
    glossary file and attendees come first and per-run ``--glossary``/
    ``--glossary-file``/``--attendee`` values *merge* on top — configuring a
    vocabulary must never make the flags stop working, or vice versa.
    ``extra_vocab`` (a meeting preset's ``[meetings.*.vocab]``) merges the same
    way: a preset adds vocabulary for its kind of meeting, it never removes the
    baseline. Inline values may each be comma-separated; a file is one term per
    line. Both lists are de-duplicated preserving first-seen order.
    """
    terms: list[str] = []
    names: list[str] = []
    if vocab is not None:
        if vocab.glossary_file is not None:
            terms.extend(read_glossary_lines(vocab.glossary_file, source="[vocab] glossary_file"))
        names.extend(vocab.attendees)
    if extra_vocab is not None:
        if extra_vocab.glossary_file is not None:
            terms.extend(
                read_glossary_lines(
                    extra_vocab.glossary_file, source="[meetings.*.vocab] glossary_file"
                )
            )
        names.extend(extra_vocab.attendees)
    for value in glossary:
        terms.extend(part.strip() for part in value.split(",") if part.strip())
    if glossary_file is not None:
        terms.extend(read_glossary_lines(glossary_file))
    for value in attendee:
        names.extend(part.strip() for part in value.split(",") if part.strip())
    return tuple(dict.fromkeys(terms)), tuple(dict.fromkeys(names))


def read_glossary_lines(path: Path, *, source: str | None = None) -> list[str]:
    """Terms from a glossary file (# comments and blank lines ignored).

    ``source`` names the setting that configured the path — the CLI flag
    validates existence itself (``exists=True``), but a stale path in
    settings.toml must say where it came from, not just fail to open. Raises
    :class:`~stenograf.settings.SettingsError` with that user-facing message."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        where = f" ({source} in settings.toml)" if source else ""
        raise SettingsError(f"cannot read glossary file {path}{where}: {exc}") from exc
    terms = []
    for raw_line in raw.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            terms.append(line)
    return terms


def attendee_name_forms(names: Iterable[str]) -> list[str]:
    """Each attendee's full name plus each of its parts.

    A name is usually misheard one part at a time, so both vocabulary layers
    register the parts individually — and the full name too: the biasing
    tree's phrase reward grows with depth, and the glossary's whole-name match
    wins where both parts drifted. The one expansion rule, shared so the
    decoder and the post-correction can never disagree about a name's forms.
    """
    forms: list[str] = []
    for name in names:
        parts = name.split()
        if not parts:
            continue
        forms.append(name)
        if len(parts) > 1:
            forms.extend(parts)
    return forms
