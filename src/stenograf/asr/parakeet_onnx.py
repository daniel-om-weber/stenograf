"""Parakeet-TDT-0.6B-v3 via onnx-asr — the cross-platform CPU backend
(Phase 5). The *same* model the MLX backend runs, fp32 ONNX export, so
transcripts stay comparable across platforms.

Decision A (PLAN.md Phase 5) resolved to onnx-asr on *accuracy*, not
timestamps: the pinned sherpa-onnx<1.13 does decode Parakeet-v3 with real
per-token timestamps, but its only published v3 export is int8, and int8
costs real accuracy — measured 2026-07-11 on the eval WAVs, cross-WER
against MLX grew from 2.0–6.8 % (fp32) to 5.8–20.8 % (int8), German worst.
fp32 on CPU was no slower (~36–44× realtime, all cores). onnx-asr is the
plan's designated fallback (small MIT dep, isolated ONNX Runtime, leaves the
diarization sherpa pin untouched); ``quantization="int8"`` remains available
for RAM-constrained boxes.

onnx-asr returns per-token *start* timestamps but no durations, so token
ends are approximated as the next token's start, capped at TDT's own
duration ceiling (4 × 80 ms frames). Word ends only set entry-gap splits and
midpoints for speaker attribution — an 0.32 s cap cannot move a word across
a diarization turn boundary farther than the model's real durations could.
"""

from __future__ import annotations

import numpy as np

from stenograf.asr.base import ASRBackend, Segment, Word
from stenograf.asr.tokens import Token, merge_tokens
from stenograf.audio import SAMPLE_RATE, to_float32
from stenograf.config import Language

MODEL_ID = "nemo-parakeet-tdt-0.6b-v3"

_MAX_TOKEN_SECONDS = 0.32
"""Cap on an approximated token duration: TDT emits at most 4 × 80 ms frames
per token, so a real token never lasts longer — without the cap the last
token before a silence would stretch to the next utterance."""

_SENTENCE_END = (".", "!", "?", "…")


class ParakeetOnnxBackend(ASRBackend):
    name = "parakeet-onnx"

    def __init__(
        self,
        model_id: str = MODEL_ID,
        *,
        quantization: str | None = None,
        provider: str | None = None,
    ) -> None:
        self.model_id = model_id
        self._quantization = quantization
        self._model = None
        self.provider = provider or "cpu"
        """Requested provider name (``cpu``/``dml``/``cuda``/``auto``); the CLI
        sets it from ``[asr] provider`` / ``STENOGRAF_ASR_PROVIDER``."""
        self.active_provider: str | None = None
        """The provider actually running after :meth:`load` (post-fallback)."""
        self.provider_fallback: str | None = None
        """Why an accelerated provider was abandoned for CPU, or ``None``."""

    def load(self) -> None:
        from stenograf.asr.providers import ort_providers, resolve, unavailable_reason

        # Never let onnx-asr default to "all available providers": ORT's CoreML
        # provider fails on this model (verified 2026-07-11), and acceleration
        # must be an explicit request so CPU stays the zero-surprise default.
        requested = resolve(self.provider)
        if requested != "cpu":
            # Pre-check the build: ORT does not raise on an unlisted provider,
            # it warns and silently runs on what remains — the canary would
            # pass on CPU and the run would claim acceleration it isn't getting.
            reason = unavailable_reason(requested)
            if reason is not None:
                self.provider_fallback = f"{requested}: {reason}"
            else:
                try:
                    model = self._load_with(ort_providers(requested))
                    # Canary: session creation succeeding does not mean
                    # inference works (CoreML initialized, then died decoding) —
                    # commit only after the provider survives a second of silence.
                    model.recognize(
                        np.zeros(SAMPLE_RATE, dtype=np.float32), sample_rate=SAMPLE_RATE
                    )
                    self._model = model
                    self.active_provider = requested
                    return
                except Exception as exc:  # noqa: BLE001 — any init/run failure means CPU
                    text = str(exc).strip()
                    first_line = text.splitlines()[0] if text else repr(exc)
                    self.provider_fallback = f"{requested}: {first_line}"
        self._model = self._load_with(ort_providers("cpu"))
        self.active_provider = "cpu"

    def _load_with(self, providers: list[str]):
        import onnx_asr

        return onnx_asr.load_model(
            self.model_id,
            quantization=self._quantization,
            providers=providers,
        ).with_timestamps()

    def transcribe(self, samples: np.ndarray, language: Language | None) -> list[Segment]:
        # Parakeet v3 is multilingual with no language switch; ``language``
        # is intentionally unused (may be None until LID runs over the text).
        if self._model is None:
            self.load()
        result = self._model.recognize(to_float32(samples), sample_rate=SAMPLE_RATE)
        return _split_sentences(merge_tokens(_approximate_ends(result.tokens, result.timestamps)))

    def unload(self) -> None:
        self._model = None


def _approximate_ends(texts: list[str], starts: list[float]) -> list[Token]:
    """Tokens with end ≈ min(next start, start + the TDT duration ceiling)."""
    tokens = []
    for i, (text, start) in enumerate(zip(texts, starts, strict=True)):
        next_start = starts[i + 1] if i + 1 < len(starts) else float("inf")
        end = min(next_start, start + _MAX_TOKEN_SECONDS)
        tokens.append(Token(text=text, start=start, end=end))
    return tokens


def _split_sentences(words: list[Word]) -> list[Segment]:
    """Group words into segments at sentence-final punctuation.

    onnx-asr returns one flat utterance (parakeet-mlx returns sentences), so
    this restores comparable entry granularity. Deliberately naive (an
    abbreviation or a German ordinal like "3." ends a segment early): segment
    boundaries only set entry granularity — the diarized path flattens
    segments back to words, so an extra split never changes attribution.
    """
    segments: list[Segment] = []
    run: list[Word] = []
    for word in words:
        run.append(word)
        if word.text.endswith(_SENTENCE_END):
            segments.append(_segment(run))
            run = []
    if run:
        segments.append(_segment(run))
    return segments


def _segment(run: list[Word]) -> Segment:
    return Segment(
        text=" ".join(w.text for w in run),
        start=run[0].start,
        end=run[-1].end,
        words=tuple(run),
    )
