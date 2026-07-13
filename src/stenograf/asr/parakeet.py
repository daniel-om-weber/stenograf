"""Parakeet-TDT-0.6B-v3 via parakeet-mlx — the committed default backend
(PLAN.md Phase 0 verdict: tied Whisper large-v3 on real meetings at ~10×
the speed; native word timestamps; no hallucination on silence).

Audio is passed as in-memory arrays (never temp files): parakeet-mlx's own
``transcribe(path)`` is just load → logmel → generate, so we call the last
two steps directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from stenograf.asr.base import ASRBackend, Segment
from stenograf.asr.biasing import DEFAULT_ALPHA, BoostingTree
from stenograf.asr.biasing import build as build_tree
from stenograf.asr.tokens import load_encoder, merge_tokens
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


def _biased_decode_greedy(model, tree: BoostingTree, alpha: float, mx):
    """parakeet-mlx's greedy TDT loop, with phrase boosting.

    A line-for-line mirror of ``parakeet_mlx.parakeet.ParakeetTDT.decode_greedy``
    (which offers no hook, hence the copy — ``tests/test_biasing_loops.py`` diffs
    the two on every run). Everything is upstream's except the two-stage token
    selection and the tree's state update.
    """
    from parakeet_mlx import tokenizer
    from parakeet_mlx.alignment import AlignedToken

    blank = len(model.vocabulary)
    # advance() is memoized in numpy; this caches the Metal-side copy so the loop
    # does not re-upload the same boost vector on every emitted token.
    vectors: dict[int, tuple] = {}

    def boost_at(state: int):
        cached = vectors.get(state)
        if cached is None:
            scores, next_states = tree.advance(state)
            cached = (mx.array(scores), next_states)
            vectors[state] = cached
        return cached

    def decode_greedy(features, lengths=None, last_token=None, hidden_state=None, *, config):
        B, S, *_ = features.shape
        if hidden_state is None:
            hidden_state = list([None] * B)
        if lengths is None:
            lengths = mx.array([S] * B)
        if last_token is None:
            last_token = list([None] * B)

        results = []
        for batch in range(B):
            hypothesis = []
            feature = features[batch : batch + 1]
            length = int(lengths[batch])
            # One tree state per utterance, reset per decoded window. A phrase
            # straddling a window boundary therefore loses its boost — the live
            # pass decodes in windows and cannot see across them. Benign: the
            # finalize pass re-decodes and is where the transcript comes from.
            boost, next_states = boost_at(0)

            step = 0
            new_symbols = 0

            while step < length:
                decoder_out, (hidden, cell) = model.decoder(
                    mx.array([[last_token[batch]]]) if last_token[batch] is not None else None,
                    hidden_state[batch],
                )
                decoder_out = decoder_out.astype(feature.dtype)
                decoder_hidden = (hidden.astype(feature.dtype), cell.astype(feature.dtype))

                joint_out = model.joint(feature[:, step : step + 1], decoder_out)
                token_logits = joint_out[0, 0, :, : blank + 1]

                # Stage 1: token or blank, decided *unbiased* (see the ONNX loop's
                # note — folding blank into the biased argmax corrupts alignment).
                pred_token = int(mx.argmax(token_logits))
                if pred_token != blank:
                    pred_token = int(mx.argmax(token_logits[..., :blank] + alpha * boost))

                # Confidence stays the model's own: it is read from the unbiased
                # distribution, so a boosted word does not report false certainty.
                token_probs = mx.softmax(token_logits, axis=-1)
                entropy = -mx.sum(token_probs * mx.log(token_probs + 1e-10), axis=-1)
                max_entropy = mx.log(mx.array(blank + 1, dtype=token_probs.dtype))
                confidence = float(1.0 - (entropy / max_entropy))

                decision = int(mx.argmax(joint_out[0, 0, :, blank + 1 :]))

                if pred_token != blank:
                    hypothesis.append(
                        AlignedToken(
                            int(pred_token),
                            start=step * model.time_ratio,
                            duration=model.durations[decision] * model.time_ratio,
                            confidence=confidence,
                            text=tokenizer.decode([pred_token], model.vocabulary),
                        )
                    )
                    last_token[batch] = pred_token
                    hidden_state[batch] = decoder_hidden
                    boost, next_states = boost_at(int(next_states[pred_token]))

                step += model.durations[int(decision)]

                new_symbols += 1
                if model.durations[int(decision)] != 0:
                    new_symbols = 0
                elif model.max_symbols is not None and model.max_symbols <= new_symbols:
                    step += 1
                    new_symbols = 0

            results.append(hypothesis)

        return results, hidden_state

    return decode_greedy


class ParakeetMLXBackend(ASRBackend):
    name = "parakeet"
    model_id: str  # always set; narrows the base's optional declaration

    def __init__(
        self,
        model_id: str = MODEL_ID,
        *,
        glossary: Sequence[str] = (),
        boost: float = DEFAULT_ALPHA,
    ) -> None:
        self.model_id = model_id
        self._model = None
        self._glossary = tuple(glossary)
        self._boost = boost
        # MLX must not import at module top (Apple-Silicon-only); load() binds
        # these once instead of re-importing on every transcribe call.
        self._mx = None
        self._get_logmel = None

    def load(self) -> None:
        import mlx.core as mx
        from parakeet_mlx import from_pretrained
        from parakeet_mlx.audio import get_logmel

        self._mx = mx
        self._get_logmel = get_logmel

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

        # Swap in the boosting decode loop only when there is something to boost.
        # With no glossary — or with `[asr] boost = 0`, which disables biasing
        # outright — the stock loop stays untouched and the run pays nothing: not a
        # tree lookup per token, not even the tokenizer download.
        if self._glossary and self._boost:
            tree = build_tree(
                self._glossary, load_encoder(), vocab_size=len(self._model.vocabulary)
            )
            if tree is not None:
                self._model.decode_greedy = _biased_decode_greedy(
                    self._model, tree, self._boost, mx
                )

    def transcribe(self, samples: np.ndarray, language: Language | None) -> list[Segment]:
        # Parakeet v3 is multilingual with no language switch; ``language``
        # is intentionally unused (may be None until LID runs over the text).
        if self._model is None:
            self.load()
        model, mx, get_logmel = self._model, self._mx, self._get_logmel
        assert model is not None and mx is not None and get_logmel is not None  # set by load()

        audio = mx.array(to_float32(samples))
        mel = get_logmel(audio, model.preprocessor_config)
        (result,) = model.generate(mel)

        segments = []
        for sentence in result.sentences:
            words = merge_tokens(sentence.tokens)
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
        if self._mx is not None:  # never loaded → nothing cached
            self._mx.clear_cache()
