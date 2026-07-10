"""Parakeet-TDT-0.6B-v3 via parakeet-mlx — the committed default backend
(PLAN.md Phase 0 verdict: tied Whisper large-v3 on real meetings at ~10×
the speed; native word timestamps; no hallucination on silence).

Audio is passed as in-memory arrays (never temp files): parakeet-mlx's own
``transcribe(path)`` is just load → logmel → generate, so we call the last
two steps directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from stenograf.asr.base import ASRBackend, Segment, Word
from stenograf.audio import to_float32
from stenograf.config import Language

MODEL_ID = "mlx-community/parakeet-tdt-0.6b-v3"

CACHE_LIMIT_BYTES = 2 << 30
"""Cap on MLX's Metal buffer cache while this backend is loaded.

MLX never returns a freed buffer to the OS on its own; every distinct window
length leaves its own activation-sized buffers behind, and a finalize pass
over a long meeting decodes hundreds of variable-length windows back to back
— measured 12.8 GB of cache after 35 windows (~15 min of one channel), ~60 GB
over a full two-channel meeting, all invisible in the process RSS (Metal
buffers count against unified memory, not RSS). One window's working set is
~1-2 GB, so 2 GB keeps same-shape reuse while excess buffers are freed."""


def _cached_snapshot(model_id: str) -> str | None:
    """Directory of the fully downloaded HF snapshot, or ``None`` to go online.

    ``hf_hub_download`` phones home to huggingface.co on *every* call to
    revision-check ``main``, even with a complete cache — that's the Hub's
    unauthenticated-request warning on each run and up to a 10 s stall per
    file (``DEFAULT_ETAG_TIMEOUT``) on a network that hangs instead of
    refusing. Resolving with ``local_files_only=True`` never touches the
    network and raises on files that were never (or only partially)
    downloaded, so a hit means the load is safe fully offline and a miss —
    first run, interrupted download — falls back to the online path.

    The two filenames mirror what ``parakeet_mlx.from_pretrained`` reads; if
    a parakeet_mlx upgrade ever needs more files, this misses and the online
    path stays correct.
    """
    from huggingface_hub import hf_hub_download

    try:
        config = hf_hub_download(model_id, "config.json", local_files_only=True)
        hf_hub_download(model_id, "model.safetensors", local_files_only=True)
    except Exception:
        return None
    return str(Path(config).parent)


class ParakeetMLXBackend(ASRBackend):
    name = "parakeet"

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.model_id = model_id
        self._model = None

    def load(self) -> None:
        import mlx.core as mx
        from parakeet_mlx import from_pretrained

        # Load from the local cache when it's complete (no network, no Hub
        # warning); from_pretrained treats a directory path like a repo id.
        self._model = from_pretrained(_cached_snapshot(self.model_id) or self.model_id)
        # Bound the Metal buffer cache (see CACHE_LIMIT_BYTES): without a limit
        # a long batch pass accumulates tens of GB of dead buffers and swaps
        # the machine. Process-global, which is fine — this is the process's
        # one Metal workload.
        mx.set_cache_limit(CACHE_LIMIT_BYTES)
        # Materialize the weights on the load thread. MLX is lazy and its GPU
        # streams are thread-local: left lazy, the freshly loaded weights carry a
        # pending computation bound to *this* thread's Stream(gpu, 0), and the
        # first decode on another thread — the live pass's LiveWorker — then dies
        # with "There is no Stream(gpu, 0) in current thread". Forcing them
        # concrete here makes the one loaded backend safe to call from the worker
        # thread and the finalize thread alike (Phase 2, Task 3).
        mx.eval(self._model.parameters())

    def transcribe(self, samples: np.ndarray, language: Language | None) -> list[Segment]:
        # Parakeet v3 is multilingual with no language switch; ``language``
        # is intentionally unused (may be None until LID runs over the text).
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
    previous word. Numbers arrive as a *bare* space token followed by digit
    pieces (" und", " ", "1", "5", ".", "7", "."), so a whitespace-only token
    carries no text but must still open the boundary — dropping it silently
    glued "und 15.7." into "und15.7.".
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
