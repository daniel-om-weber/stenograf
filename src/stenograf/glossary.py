"""Deterministic glossary + attendee-name post-correction (PLAN.md §5 Task 2b).

The *second* of stenograf's two glossary layers, and the one that answers to no
acoustics. ``stenograf.asr.biasing`` steers the decoder toward the same terms while
it transcribes, which is strictly the better place to act — it can still hear the
audio. This layer runs after the finalize pass, over finished text: it fuzzy-matches
each user-supplied term against the transcribed words and snaps close
misrecognitions to the canonical spelling. Deterministic (stdlib ``difflib``, no
ML), model-agnostic, and testable.

It exists because biasing cannot reach every error. Boosting only re-ranks a token
the decoder was already about to emit, so a term the model did not *nearly* hear —
one glued into a neighbour, or lost to a word boundary that was never there — is
past its reach. Text matching has no such constraint, and pays for it in the
opposite currency: with no acoustics to answer to, it will rewrite a word that was
correct if the spelling is close enough. Hence the deliberately high threshold, and
hence ``eval/bias.py --post``, which prices both layers on the same benchmark.

Matching is diacritic- and case-insensitive (so German umlaut/ß spellings match
their ASCII-ish transcriptions) and operates on whole word tokens: an ``n``-word
window is replaced by the term's ``n`` canonical tokens, preserving each word's
timing and attached punctuation — so the retained word timestamps (Task 0e) and
the SRT/VTT cues derived from them stay in sync with the corrected text. Known
limit: a term and its transcription must share a *token count* (no split/merge
across word boundaries), so attendee names are also registered token-by-token and
first/last names correct individually.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, replace
from difflib import SequenceMatcher

from stenograf.asr.base import Word
from stenograf.transcript import TranscriptEntry

DEFAULT_THRESHOLD = 0.95
"""Minimum normalized similarity (0–1) for a window to be snapped to a term.

High by design: over-correction (rewriting a correct common word into a glossary
term) is worse than a missed correction. Tune per run via the CLI if needed.

**0.95, not the 0.82 this shipped with, and the gap between them is not a matter of
taste — 0.82 fails the same acceptance bar that set `[asr] boost`** (measured
2026-07-13, ``eval/bias.py --post``, 500-utt subsample, 100-term lists). At 0.82 the
layer buys rare words by rewriting words that were never wrong: U-WER +6.5 % German
and **+86 % English**, with 84–86 false insertions against decode-time biasing's 3.
That is worse than the ``boost = 2.0`` config we rejected for exactly this failure.
The reason is structural — fuzzy matching answers to no acoustics, so nothing stops
it from snapping a correct common word onto a glossary term that merely looks like
it, and B-WER alone cannot see the difference.

At 0.95 the layer is a free win instead: it still *adds* to biasing (B-WER −30.3 %
vs −27.0 % German, −36.3 % vs −34.9 % English) while U-WER stays flat or improves,
false insertions land at biasing's own level (4 / 3), and on real meeting audio with
a realistic 30-term glossary it inserts **nothing** biasing did not already insert.
Below ~0.92 the damage climbs fast; do not lower this without re-running that tier."""

_MIN_TERM_CHARS = 4
"""Terms shorter than this (normalized) are too collision-prone to correct."""

_PUNCT = re.compile(r"^(\W*)(.*?)(\W*)$", re.DOTALL)
"""Split a token into (leading punctuation, core, trailing punctuation). Under
Unicode ``\\W``, accented letters/digits are word chars, so only real punctuation
is peeled off."""


@dataclass(frozen=True)
class _Term:
    tokens: tuple[str, ...]  # canonical display spelling, per token
    norm: str  # normalized full form, for scoring

    @property
    def size(self) -> int:
        return len(self.tokens)


def build_terms(glossary: Iterable[str] = (), attendee_names: Iterable[str] = ()) -> list[_Term]:
    """Compile canonical terms from a glossary and attendee names.

    Attendee names are registered both whole (``"Daniel Weber"``) and per token
    (``"Daniel"``, ``"Weber"``), since a name is usually mis-transcribed one part
    at a time. Terms are de-duplicated by their match key (first spelling wins),
    and terms too short to correct safely are dropped.
    """
    terms: dict[str, _Term] = {}

    def add(phrase: str) -> None:
        tokens = tuple(tok for tok in phrase.split() if tok)
        if not tokens:
            return
        norm = _norm(phrase)
        if len(norm) < _MIN_TERM_CHARS:
            return
        terms.setdefault(norm, _Term(tokens, norm))

    for phrase in glossary:
        add(phrase)
    for name in attendee_names:
        add(name)
        for token in name.split():
            add(token)
    return list(terms.values())


def apply_glossary(
    entries: list[TranscriptEntry],
    *,
    glossary: Iterable[str] = (),
    attendee_names: Iterable[str] = (),
    threshold: float = DEFAULT_THRESHOLD,
) -> list[TranscriptEntry]:
    """Correct near-misses of the glossary/attendee terms across all entries.

    Returns the entries unchanged (same list) when there is nothing to correct,
    so this is a cheap no-op in the common case of no glossary.
    """
    terms = build_terms(glossary, attendee_names)
    if not terms:
        return entries
    return [_correct_entry(entry, terms, threshold) for entry in entries]


def _correct_entry(entry: TranscriptEntry, terms: list[_Term], threshold: float) -> TranscriptEntry:
    # Correct the word list and the flat text independently with the same terms,
    # rather than rebuilding one from the other: for a well-formed entry both hold
    # the same tokens so they get identical corrections and stay in sync, but this
    # never truncates `text` if a backend's words don't fully cover it (a wordless
    # backend has `words=()` — only its text is corrected).
    new_words = _correct_words(entry.words, terms, threshold) if entry.words else None
    new_text = _correct_text(entry.text, terms, threshold)
    if new_words is None and new_text is None:
        return entry
    return replace(
        entry,
        words=new_words if new_words is not None else entry.words,
        text=new_text if new_text is not None else entry.text,
    )


def _correct_words(
    words: tuple[Word, ...], terms: list[_Term], threshold: float
) -> tuple[Word, ...] | None:
    """Snap matching word runs to their canonical spelling, keeping timing +
    surrounding punctuation. ``None`` when nothing matched."""
    parts = [_split_punct(w.text) for w in words]
    plan = _plan_corrections([core for _, core, _ in parts], terms, threshold)
    if not plan:
        return None
    out = []
    for i, word in enumerate(words):
        if i in plan:
            lead, _, trail = parts[i]
            out.append(replace(word, text=f"{lead}{plan[i]}{trail}"))
        else:
            out.append(word)
    return tuple(out)


def _correct_text(text: str, terms: list[_Term], threshold: float) -> str | None:
    """Correction over a plain whitespace-tokenized string. ``None`` when unchanged."""
    tokens = text.split()
    parts = [_split_punct(tok) for tok in tokens]
    plan = _plan_corrections([core for _, core, _ in parts], terms, threshold)
    if not plan:
        return None
    out = [
        f"{parts[i][0]}{plan[i]}{parts[i][2]}" if i in plan else tok for i, tok in enumerate(tokens)
    ]
    return " ".join(out)


def _plan_corrections(cores: list[str], terms: list[_Term], threshold: float) -> dict[int, str]:
    """Greedy, non-overlapping left-to-right match → {word index: canonical token}.

    At each position the longest term size that yields a qualifying match wins
    (more specific), and within a size the highest-scoring term; the window is
    then consumed so corrections never overlap.
    """
    norms = [_norm(core) for core in cores]
    by_size: dict[int, list[_Term]] = {}
    for term in terms:
        by_size.setdefault(term.size, []).append(term)
    sizes = sorted(by_size, reverse=True)

    plan: dict[int, str] = {}
    i, n = 0, len(cores)
    while i < n:
        advance = 1
        for size in sizes:
            if i + size > n:
                continue
            window = "".join(norms[i : i + size])
            if not window:
                continue
            best = _best_term(window, by_size[size], threshold)
            if best is not None:
                for j, token in enumerate(best.tokens):
                    plan[i + j] = token
                advance = size
                break
        i += advance
    return plan


def _best_term(window: str, terms: list[_Term], threshold: float) -> _Term | None:
    """The highest-scoring term at/above ``threshold``; first term wins ties."""
    best: _Term | None = None
    best_score = threshold
    for term in terms:
        score = SequenceMatcher(None, window, term.norm).ratio()
        if score >= threshold and (best is None or score > best_score):
            best, best_score = term, score
    return best


def _norm(text: str) -> str:
    """Casefold, strip accents/umlauts, drop non-alphanumerics — the match key."""
    folded = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in folded if c.isalnum())


def _split_punct(token: str) -> tuple[str, str, str]:
    match = _PUNCT.match(token)
    assert match is not None  # the pattern matches any string
    return match.group(1), match.group(2), match.group(3)


__all__ = ["DEFAULT_THRESHOLD", "apply_glossary", "build_terms"]
