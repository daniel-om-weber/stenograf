"""SentencePiece token → word merging, shared by the Parakeet backends.

Both parakeet-mlx and onnx-asr decode Parakeet's SentencePiece vocabulary,
so both emit pieces with the word-boundary marker rendered as a leading space
and per-token start/end times; only the container object differs. Backends
adapt their tokens to :class:`Token` (or anything with ``text``/``start``/
``end``) and share one merge.
"""

from __future__ import annotations

from dataclasses import dataclass

from stenograf.asr.base import Word


@dataclass(frozen=True)
class Token:
    text: str
    start: float
    end: float


def merge_tokens(tokens) -> list[Word]:
    """Merge subword tokens into words.

    Tokens are SentencePiece pieces with the word-boundary marker rendered as
    a leading space; a token without one continues the previous word. Numbers
    arrive as a *bare* space token followed by digit pieces (" und", " ", "1",
    "5", ".", "7", "."), so a whitespace-only token carries no text but must
    still open the boundary — dropping it silently glued "und 15.7." into
    "und15.7.".
    """
    words: list[Word] = []
    boundary = False  # a pending word break left by a bare space token
    for token in tokens:
        text = token.text.strip()
        if not text:
            boundary = boundary or bool(token.text)
            continue
        if token.text.startswith(" ") or boundary or not words:
            words.append(Word(text=text, start=token.start, end=token.end))
        else:
            prev = words[-1]
            words[-1] = Word(text=prev.text + text, start=prev.start, end=token.end)
        boundary = False
    return words
