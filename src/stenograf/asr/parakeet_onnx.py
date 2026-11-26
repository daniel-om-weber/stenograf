"""Parakeet Ultra via onnx-asr — the cross-platform CPU backend.
The *same* weights the MLX backend runs, fp32 ONNX export, so
transcripts stay comparable across platforms.

onnx-asr won over sherpa-onnx on *accuracy*, not
timestamps: sherpa-onnx does decode Parakeet with real
per-token timestamps, but its only published export was int8, and int8
costs real accuracy — measured 2026-07-11 on the eval WAVs with
Parakeet-v3, cross-WER against MLX grew from 2.0–6.8 % (fp32) to
5.8–20.8 % (int8), German worst. fp32 on CPU was no slower (~36–44×
realtime, all cores). onnx-asr is a small MIT dep with an isolated ONNX
Runtime that leaves the sherpa pin — the VAD — untouched.

onnx-asr returns per-token *start* timestamps but no durations, so token
ends are approximated as the next token's start, capped at TDT's own
duration ceiling (4 × 80 ms frames). Word ends only set entry-gap splits and
midpoints for speaker attribution — an 0.32 s cap cannot move a word across
a diarization turn boundary farther than the model's real durations could.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from stenograf.asr.base import ASRBackend, BackendUnavailableError, Segment, Word
from stenograf.asr.biasing import DEFAULT_ALPHA, BoostingTree
from stenograf.asr.biasing import build as build_tree
from stenograf.asr.tokens import Token, load_encoder, merge_tokens
from stenograf.audio import SAMPLE_RATE, to_float32
from stenograf.config import Language
from stenograf.cpu import ThreadPlan, confined, live_plan, session_options

MODEL_ID = "VoiceVibe/parakeet-ultra-nounk-onnx"
_REVISION = "ed66bdeef2ec0086cbf96fa8b046c1a2ea59998d"
"""A third party's export of Moondream's Parakeet Ultra in onnx-asr's layout,
pinned because nothing but the revision ties it to what was checked: the graph
is byte-identical to ``istupakov/parakeet-tdt-0.6b-v3-onnx``'s and every weight
is ``moondream/parakeet-ultra``'s (the 48 conv+norm tensors folded, equal to
2e-7), except the ``<unk>`` output bias, lowered by 10 000 exactly as in the
MLX conversion (``parakeet.REVISION``). Re-run ``eval/asr_weights.py`` before
moving the pin. Only fp32 is published."""

_ONNX_ASR_TYPE = "nemo-parakeet-tdt-0.6b-v3"
"""What onnx-asr calls the architecture; Ultra is v3 with other weights."""


def model_dir(quantization: str | None = None) -> Path:
    """The model's files as plain files in stenograf's own cache, downloaded once.

    Never huggingface_hub's cache: from hub 1.32 (2026-09) it keeps large files
    in a cache-wide store sharded by hash, so ``encoder-model.onnx`` and its
    ``encoder-model.onnx.data`` resolve into different directories, and every
    onnxruntime before 1.32 refuses external data outside the graph's
    directory — DirectML, the Windows flavor, ends at 1.24.4 and will never get
    the fix (measured 2026-11-15, ``eval/README.md``). A directory of real files
    loads on every version. onnx-asr treats an existing directory as complete,
    so completeness is ours: the marker is written only after every file is in,
    and an interrupted download resumes on the next call.
    """
    from stenograf.paths import cache_dir

    variant = quantization or "fp32"
    target = cache_dir() / f"parakeet-ultra-onnx-{_REVISION[:12]}"
    marker = target / f".complete-{variant}"
    if marker.exists():
        return target
    from huggingface_hub import snapshot_download

    suffix = f".{quantization}" if quantization else ""
    snapshot_download(
        MODEL_ID,
        revision=_REVISION,
        local_dir=target,
        allow_patterns=[
            "config.json",
            "vocab.txt",
            f"encoder-model{suffix}.onnx",
            f"encoder-model{suffix}.onnx.data",
            f"decoder_joint-model{suffix}.onnx",
        ],
    )
    marker.touch()
    return target

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
    ep: str

    def __init__(
        self,
        model_id: str = MODEL_ID,
        *,
        quantization: str | None = None,
        ep: str | None = None,
        glossary: Sequence[str] = (),
        boost: float = DEFAULT_ALPHA,
    ) -> None:
        self.model_id = model_id
        self._quantization = quantization
        self._model = None
        self._plan: ThreadPlan | None = None
        self._glossary = tuple(glossary)
        self._boost = boost
        # "cpu" (never None) marks this backend as provider-configurable.
        self.ep = ep or "cpu"

    def load(self) -> None:
        from stenograf.asr.ep import ort_providers, resolve, unavailable_reason

        # Never let onnx-asr default to "all available providers": ORT's CoreML
        # provider fails on this model (verified 2026-07-11), and acceleration
        # must be an explicit request so CPU stays the zero-surprise default.
        requested = resolve(self.ep)
        if requested != "cpu":
            # An explicit provider that can't deliver is a loud error, never a
            # silent CPU run at a fraction of the requested speed — the user
            # asked for acceleration, so its absence is theirs to resolve.
            # Pre-check the build: ORT does not raise on an unlisted provider,
            # it warns and silently runs on what remains — the canary would
            # pass on CPU and the run would claim acceleration it isn't getting.
            reason = unavailable_reason(requested)
            if reason is not None:
                raise BackendUnavailableError(
                    f"ASR provider {requested!r} cannot run here ({reason}) — "
                    'set [asr] provider = "cpu" in settings.toml, or unset '
                    "STENOGRAF_ASR_PROVIDER"
                )
            try:
                model = self._load_with(ort_providers(requested))
                # Canary: session creation succeeding does not mean inference
                # works (CoreML initialized, then died decoding) — commit only
                # after the provider survives a second of silence.
                model.recognize(np.zeros(SAMPLE_RATE, dtype=np.float32), sample_rate=SAMPLE_RATE)
            except Exception as exc:
                text = str(exc).strip()
                first_line = text.splitlines()[0] if text else repr(exc)
                raise BackendUnavailableError(
                    f"ASR provider {requested!r} failed to initialize or decode "
                    f'({first_line}) — set [asr] provider = "cpu" in settings.toml, '
                    "or unset STENOGRAF_ASR_PROVIDER"
                ) from exc
            self._model = model
            return
        # CPU decodes, the live captions above all, run on the efficiency cores
        # at normal priority (stenograf.cpu): on a Ryzen AI 9 HX 370 a third of
        # ONNX Runtime's default energy at the same speed, word for word the
        # same (2026-11-17, eval/README.md "Background threads, by platform").
        # Accelerated providers keep ONNX Runtime's defaults: unmeasured.
        self._plan = live_plan()
        self._model = self._load_with(ort_providers("cpu"), session_options(self._plan))

    def _load_with(self, providers: list[str], options: Any = None):
        import onnx_asr

        pinned = self.model_id == MODEL_ID
        model = onnx_asr.load_model(
            _ONNX_ASR_TYPE if pinned else self.model_id,
            model_dir(self._quantization) if pinned else None,
            quantization=self._quantization,
            providers=providers,
            sess_options=options,
        ).with_timestamps()
        self._bias(model)
        return model

    def _bias(self, model: Any) -> None:
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
        # Parakeet is multilingual with no language switch; ``language``
        # is intentionally unused (may be None until LID runs over the text).
        if self._model is None:
            self.load()
        model = self._model
        assert model is not None  # load() sets it or raises
        with confined(self._plan):
            result = model.recognize(to_float32(samples), sample_rate=SAMPLE_RATE)
        # with_timestamps() at load time guarantees both are present.
        assert result.tokens is not None and result.timestamps is not None
        return _split_sentences(merge_tokens(_approximate_ends(result.tokens, result.timestamps)))

    def unload(self) -> None:
        self._model = None


def _biased_decoding(asr: Any, tree: BoostingTree, alpha: float):
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
    # index for `logits[:blank]` to mean "every label". It is, for Parakeet.
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
