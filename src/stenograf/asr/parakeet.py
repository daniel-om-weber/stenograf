"""Parakeet-TDT-0.6B-v3 via parakeet-mlx — the committed default backend
(PLAN.md Phase 0 verdict: tied Whisper large-v3 on real meetings at ~10×
the speed; native word timestamps; no hallucination on silence).

Audio is passed as in-memory arrays (never temp files): parakeet-mlx's own
``transcribe(path)`` is just load → logmel → generate, so we call the last
two steps directly.
"""

from __future__ import annotations

import numpy as np

from stenograf.asr.base import ASRBackend, Segment, Word
from stenograf.audio import to_float32
from stenograf.config import Language

MODEL_ID = "mlx-community/parakeet-tdt-0.6b-v3"


class ParakeetMLXBackend(ASRBackend):
    name = "parakeet"

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.model_id = model_id
        self._model = None

    def load(self) -> None:
        from parakeet_mlx import from_pretrained

        self._model = from_pretrained(self.model_id)

    def transcribe(self, samples: np.ndarray, language: Language) -> list[Segment]:
        # Parakeet v3 is multilingual with no language switch; ``language``
        # is intentionally unused.
        if self._model is None:
            self.load()
        import mlx.core as mx
        from parakeet_mlx.audio import get_logmel

        audio = mx.array(to_float32(samples))
        mel = get_logmel(audio, self._model.preprocessor_config)
        (result,) = self._model.generate(mel)

        segments = []
        for sentence in result.sentences:
            words = _merge_tokens(sentence.tokens)
            if not words:
                continue
            segments.append(
                Segment(
                    text=sentence.text.strip(),
                    start=sentence.start,
                    end=sentence.end,
                    words=tuple(words),
                )
            )
        return segments

    def unload(self) -> None:
        self._model = None
        import mlx.core as mx

        mx.clear_cache()


def _merge_tokens(tokens) -> list[Word]:
    """Merge subword tokens into words.

    parakeet-mlx tokens are SentencePiece pieces with the word-boundary
    marker rendered as a leading space; a token without one continues the
    previous word.
    """
    words: list[Word] = []
    for token in tokens:
        starts_word = token.text.startswith(" ") or not words
        text = token.text.strip()
        if not text:
            continue
        if starts_word:
            words.append(Word(text=text, start=token.start, end=token.end))
        else:
            prev = words[-1]
            words[-1] = Word(text=prev.text + text, start=prev.start, end=token.end)
    return words
