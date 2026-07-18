"""SentencePiece tokens, shared by the Parakeet backends.

Both parakeet-mlx and onnx-asr decode Parakeet's SentencePiece vocabulary,
so both emit pieces with the word-boundary marker rendered as a leading space
and per-token start/end times; only the container object differs. Backends
adapt their tokens to :class:`Token` (or anything with ``text``/``start``/
``end``) and share one merge — and, for contextual biasing, one *encoder*
(:func:`load_encoder`), because a boosting tree only matches if it is built
from the very token ids the decoder emits.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

from stenograf.asr.base import Word

TOKENIZER_REPO = "mlx-community/parakeet-tdt-0.6b-v3"
TOKENIZER_FILE = "tokenizer.model"
"""Where Parakeet's SentencePiece model comes from.

The two backends load the *same* Parakeet-v3 weights from different repos, and
their vocabularies are byte-identical index for index (8192 pieces; onnx-asr
appends ``<blk>`` as 8192, which is exactly parakeet-mlx's ``blank == len(vocab)``
convention) — so one encoder is correct for both. Only the MLX repo publishes the
SentencePiece model itself; onnx-asr's ships a bare ``vocab.txt`` with no merge
information, from which the model's own segmentation cannot be reconstructed.
Hence both backends read the tokenizer from here. It is a ~300 kB file and needs
no MLX, so this costs the ONNX platforms nothing but the download."""


@dataclass(frozen=True)
class Token:
    text: str
    start: float
    end: float
    confidence: float | None = None
    """The model's certainty when it emitted this token, read from the *unbiased*
    distribution (see ``parakeet._biased_decode_greedy``) — so a boosted token reports
    what the model actually believed, not what we told it to prefer. Optional because
    a backend may not expose it; ``merge_tokens`` then leaves the word's confidence
    ``None`` rather than inventing one."""


@lru_cache(maxsize=1)
def load_encoder() -> Callable[[str], list[list[int]]]:
    """Parakeet's own text → token-ids encoder, returning *every* tokenization of
    a term the decoder might emit.

    Used to compile glossary terms for the boosting tree, which matches on token
    ids: an approximate segmentation would build arcs the decoder never walks, so
    this has to be the model's real SentencePiece encoder, not a longest-match
    over the vocabulary.

    Two tokenizations, because German puts terms inside words. SentencePiece marks
    a word start with ``▁``, so "Dashboard" standing alone is ``▁Das h bo ard`` —
    but in "Grafana-Dashboard" the decoder emits ``- D as h bo ard``, with no
    marker at all. A tree built only from the word-initial form can never be
    *entered* on the compound, which is precisely where German technical
    vocabulary lives (and precisely where the model gets it wrong: measured
    2026-07-13, the decoder wrote "Grafana-Dashboot" and word-start-only biasing
    could not touch it). So each term is compiled both ways.
    """
    import sentencepiece
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(TOKENIZER_REPO, TOKENIZER_FILE)
    # sentencepiece is SWIG-generated: the model_file kwarg and the snake_case
    # encode alias are injected at runtime, so static analysis can't see them.
    sp = sentencepiece.SentencePieceProcessor(model_file=path)  # pyright: ignore[reportCallIssue]

    def encode(text: str) -> list[list[int]]:
        forms = [sp.encode(text)]  # pyright: ignore[reportAttributeAccessIssue]
        tail = _compound_tail(sp, text)
        if tail is not None and tail != forms[0]:
            forms.append(tail)
        return forms

    return encode


def _compound_tail(sp, text: str) -> list[int] | None:
    """``text`` tokenized as it appears *inside* a word, with no ``▁`` marker.

    SentencePiece has no "encode as a continuation" mode, so we ask it for a word
    that has ``text`` as its tail — ``X-Dashboard`` — and keep everything after the
    separator. ``None`` when the tokenizer folds the separator into the following
    piece and the trick does not hold; the round-trip check is what proves it did.
    """
    pieces = sp.encode(f"X-{text}", out_type=str)
    if "-" not in pieces:
        return None
    tail = pieces[pieces.index("-") + 1 :]
    if "".join(tail) != text:
        return None
    return [sp.piece_to_id(piece) for piece in tail]


def _weakest(left: float | None, right: float | None) -> float | None:
    """A word is only as certain as its least certain piece.

    ``min``, not a mean: a word the model was sure about *except* for one token is
    exactly the word a glossary correction should be allowed to touch, and averaging
    would bury that token under its confident neighbours. ``None`` (a backend that
    reports no confidence) is not a low score — it is no score, and must not drag a
    word's confidence down.
    """
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


def merge_tokens(tokens) -> list[Word]:
    """Merge subword tokens into words, carrying each word's confidence with it.

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
        confidence = getattr(token, "confidence", None)
        if not text:
            boundary = boundary or bool(token.text)
            continue
        if token.text.startswith(" ") or boundary or not words:
            words.append(Word(text=text, start=token.start, end=token.end, confidence=confidence))
        else:
            prev = words[-1]
            words[-1] = Word(
                text=prev.text + text,
                start=prev.start,
                end=token.end,
                confidence=_weakest(prev.confidence, confidence),
            )
        boundary = False
    return words
