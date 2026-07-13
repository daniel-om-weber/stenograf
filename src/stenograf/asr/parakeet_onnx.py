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

from collections.abc import Iterable, Iterator, Sequence

import numpy as np

from stenograf.asr.base import ASRBackend, Segment, Word
from stenograf.asr.biasing import DEFAULT_ALPHA, BoostingTree
from stenograf.asr.biasing import build as build_tree
from stenograf.asr.tokens import Token, load_encoder, merge_tokens
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
    # Always set; narrow the base's optional declarations (provider "cpu",
    # never None — this backend is provider-configurable).
    model_id: str
    provider: str

    def __init__(
        self,
        model_id: str = MODEL_ID,
        *,
        quantization: str | None = None,
        provider: str | None = None,
        glossary: Sequence[str] = (),
        boost: float = DEFAULT_ALPHA,
    ) -> None:
        self.model_id = model_id
        self._quantization = quantization
        self._model = None
        self._glossary = tuple(glossary)
        self._boost = boost
        # Diagnostics surface declared on ASRBackend; "cpu" (never None)
        # marks this backend as provider-configurable.
        self.provider = provider or "cpu"
        self.active_provider = None
        self.provider_fallback = None

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

        model = onnx_asr.load_model(
            self.model_id,
            quantization=self._quantization,
            providers=providers,
        ).with_timestamps()
        self._bias(model)
        return model

    def _bias(self, model) -> None:
        """Swap in the boosting decode loop when there is a glossary to boost.

        ``with_timestamps()`` returns an adapter holding the model as ``.asr``;
        that is the object owning the decode loop. With no glossary — or with
        ``[asr] boost = 0``, which disables biasing outright — the stock loop stays
        untouched, so a meeting with no vocabulary pays nothing at all: not a tree
        lookup per token, not even the tokenizer download.
        """
        if not self._glossary or not self._boost:
            return
        asr = model.asr
        tree = build_tree(self._glossary, load_encoder(), vocab_size=asr._blank_idx)
        if tree is None:
            return
        asr._decoding = _biased_decoding(asr, tree, self._boost)

    def transcribe(self, samples: np.ndarray, language: Language | None) -> list[Segment]:
        # Parakeet v3 is multilingual with no language switch; ``language``
        # is intentionally unused (may be None until LID runs over the text).
        if self._model is None:
            self.load()
        model = self._model
        assert model is not None  # load() sets it or raises
        result = model.recognize(to_float32(samples), sample_rate=SAMPLE_RATE)
        # with_timestamps() at load time guarantees both are present.
        assert result.tokens is not None and result.timestamps is not None
        return _split_sentences(merge_tokens(_approximate_ends(result.tokens, result.timestamps)))

    def unload(self) -> None:
        self._model = None


def _biased_decoding(asr, tree: BoostingTree, alpha: float):
    """onnx-asr's greedy transducer loop, with phrase boosting.

    A line-for-line mirror of ``onnx_asr.asr._AsrWithTransducerDecoding._decoding``
    (which offers no hook, hence the copy — ``tests/test_biasing_loops.py`` diffs
    the two on every run, so an upstream change to the loop fails loudly instead of
    silently un-biasing us). Everything here is upstream's except the two-stage
    token selection and the tree's state update.
    """
    from onnx_asr.utils import log_softmax

    blank = asr._blank_idx
    # The boost vector spans the non-blank vocabulary, so blank has to be the last
    # index for `logits[:blank]` to mean "every label". It is, for Parakeet-v3.
    assert blank == asr._vocab_size - 1

    def decoding(
        encoder_out: np.ndarray, encoder_out_lens: np.ndarray, /, **kwargs: object | None
    ) -> Iterator[tuple[Iterable[int], Iterable[int], Iterable[float] | None]]:
        need_logprobs = kwargs.get("need_logprobs")
        if asr.use_low_precision:
            encoder_out_lens = np.minimum(encoder_out_lens, encoder_out.shape[1])

        for encodings, encodings_len in zip(encoder_out, encoder_out_lens, strict=True):
            prev_state = asr._create_state()
            tokens: list[int] = []
            timestamps: list[int] = []
            logprobs: list[float] = []
            # One tree state per utterance. It moves only on an emitted token, so
            # a partial match survives blanks and TDT frame skips untouched.
            boost, next_states = tree.advance(0)

            t = 0
            emitted_tokens = 0
            while t < encodings_len:
                logits, step, state = asr._decode(tokens, prev_state, encodings[t])
                assert logits.shape[-1] <= asr._vocab_size

                # Stage 1: is this frame a token at all? Decided on the *unbiased*
                # logits, blank included. Stage 2 then re-ranks within the labels.
                # Boosting must not be able to answer this question: the refund for
                # abandoning a phrase is the only large negative score in play, and
                # blank is the one token it cannot touch — fold them together and a
                # failing match hands its frame to blank, dropping the token and
                # shifting every timestamp after it.
                token = int(logits.argmax())
                if token != blank:
                    token = int((logits[:blank] + alpha * boost).argmax())

                if token != blank:
                    prev_state = state
                    tokens.append(token)
                    timestamps.append(t)
                    emitted_tokens += 1
                    if need_logprobs:
                        # Unbiased: this is the model's confidence, not ours.
                        logprobs.append(log_softmax(logits)[token])
                    boost, next_states = tree.advance(int(next_states[token]))

                if step > 0:
                    t += step
                    emitted_tokens = 0
                elif token == blank or emitted_tokens == asr._max_tokens_per_step:
                    t += 1
                    emitted_tokens = 0

            yield tokens, timestamps, logprobs if need_logprobs else None

    return decoding


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
