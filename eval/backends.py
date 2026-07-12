"""Eval-only ASR backend wrappers.

Deliberately minimal: each wraps one model+runtime candidate from PLAN.md §1 and
returns plain dicts. The shipped package gets proper ``stenograf.asr`` backends
later, informed by what wins here.

Backends run one-per-process (see transcribe.py) so peak-memory numbers are
attributable to a single model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

Result = dict[str, Any]
# {"text": str, "segments": [{"text", "start", "end", "words": [...]}],
#  "detected_language": str | None}


class Backend:
    name: str
    model_id: str

    def load(self) -> None:
        raise NotImplementedError

    def transcribe(self, wav: Path, language: str | None) -> Result:
        raise NotImplementedError


class WhisperMLX(Backend):
    """Whisper large-v3 via mlx-whisper — the mature fallback."""

    name = "whisper"
    model_id = "mlx-community/whisper-large-v3-mlx"

    def load(self) -> None:
        # mlx-whisper has no explicit load API — weights download and load inside
        # transcribe(), so warm up on a second of silence to keep timings honest.
        import mlx_whisper
        import numpy as np

        mlx_whisper.transcribe(
            np.zeros(16000, dtype=np.float32), path_or_hf_repo=self.model_id, language="en"
        )

    def transcribe(self, wav: Path, language: str | None) -> Result:
        import mlx_whisper

        raw = mlx_whisper.transcribe(
            str(wav),
            path_or_hf_repo=self.model_id,
            language=language,
            word_timestamps=True,
            # Phase 0 finding: with condition_on_previous_text=True, decoder
            # loops snowballed across windows (up to 220 repeated words on
            # overlap/silence regions). False stops the propagation; the plan's
            # cross-window-consistency preference loses to that in practice.
            condition_on_previous_text=False,
            hallucination_silence_threshold=2.0,
        )
        segments = [
            {
                "text": seg["text"].strip(),
                "start": seg["start"],
                "end": seg["end"],
                "words": [
                    {"text": w["word"].strip(), "start": w["start"], "end": w["end"]}
                    for w in seg.get("words", [])
                ],
            }
            for seg in raw["segments"]
        ]
        return {
            "text": raw["text"].strip(),
            "segments": segments,
            "detected_language": raw.get("language"),
        }


class ParakeetMLX(Backend):
    """Parakeet-TDT-0.6B-v3 via parakeet-mlx — the live-pass model, baseline here."""

    name = "parakeet"
    model_id = "mlx-community/parakeet-tdt-0.6b-v3"

    def load(self) -> None:
        from parakeet_mlx import from_pretrained

        self._model = from_pretrained(self.model_id)

    def transcribe(self, wav: Path, language: str | None) -> Result:
        # Parakeet v3 is multilingual without a language switch.
        result = self._model.transcribe(str(wav))
        segments = [
            {
                "text": sentence.text.strip(),
                "start": sentence.start,
                "end": sentence.end,
                "words": [
                    {"text": token.text.strip(), "start": token.start, "end": token.end}
                    for token in getattr(sentence, "tokens", [])
                ],
            }
            for sentence in result.sentences
        ]
        return {"text": result.text.strip(), "segments": segments, "detected_language": None}


class VoxtralMLX(Backend):
    """Voxtral Small 24B (4-bit) via mlx-voxtral — max-accuracy challenger.

    Text only: Voxtral emits no timestamps, so it competes on WER alone.
    ~14 GB download on first use; ~13 GB wired while running — close
    memory-hungry apps first on a 48 GB machine.

    Audio goes in as ~30 s windows cut at silence, mirroring the production
    finalize pass (VAD → batch ASR). Long windows are unsafe here: greedy
    decoding without repetition penalty can fall into multi-hundred-word
    loops (observed on a 5-minute window), and a penalty would corrupt
    verbatim transcription of naturally repetitive speech. Short windows
    both prevent that and bound the damage to one chunk.
    """

    name = "voxtral"
    model_id = "VincentGOURBIN/voxtral-small-4bit-mixed"

    def load(self) -> None:
        import mlx.core as mx
        from mlx_voxtral import VoxtralForConditionalGeneration, VoxtralProcessor

        # Keep MLX's buffer cache from stacking gigabytes on top of the weights.
        mx.set_cache_limit(1 << 30)
        self._model = VoxtralForConditionalGeneration.from_pretrained(self.model_id)
        self._processor = VoxtralProcessor.from_pretrained(self.model_id)

    def transcribe(self, wav: Path, language: str | None) -> Result:
        import tempfile

        from common import split_at_silences, to_wav16k

        if language is None:
            raise ValueError("voxtral needs an explicit language; set it in manifest.json")
        texts = []
        with tempfile.TemporaryDirectory() as tmp:
            for i, (start, end) in enumerate(split_at_silences(wav)):
                # The input is already the eval wire format, so this is a pure cut.
                chunk = Path(tmp) / f"chunk{i}.wav"
                to_wav16k(wav, chunk, start=start, end=end)
                texts.append(self._transcribe_chunk(chunk, language, end - start))
        return {"text": " ".join(t for t in texts if t), "segments": [], "detected_language": None}

    def _transcribe_chunk(self, chunk: Path, language: str, duration_s: float) -> str:
        # (sic) apply_transcrition_request is the library's actual method name.
        inputs = self._processor.apply_transcrition_request(language=language, audio=str(chunk))
        outputs = self._model.generate(
            input_ids=inputs.input_ids,
            input_features=inputs.input_features,
            max_new_tokens=max(256, int(duration_s * 16)),
            temperature=0.0,
            repetition_penalty=1.0,
        )
        prompt_len = inputs.input_ids.shape[1]
        return self._processor.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()


class CanaryNeMo(Backend):
    """Canary-1B-v2 via NeMo on PyTorch MPS — accuracy-ceiling reference ONLY.

    Research verdict (July 2026): no MLX/CoreML runtime emits Canary word
    timestamps (PyPI canary-mlx is an abandoned template; mlx-audio returns
    hardcoded 0.0 timestamps; onnx-asr supports timestamps only for
    TDT/CTC/RNNT). NeMo with MPS fallback is the one real path — slow, heavy,
    never shippable. Needs its own environment:

        uv sync --group eval-canary
    """

    name = "canary"
    model_id = "nvidia/canary-1b-v2"

    def load(self) -> None:
        import os

        # Must be set before torch initializes; several Canary ops lack MPS kernels.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        try:
            from nemo.collections.asr.models import ASRModel
        except ImportError as e:
            raise RuntimeError(
                "NeMo not installed — run `uv sync --group eval-canary` "
                "(conflicts with the mlx eval group; re-sync `--group eval` afterwards)"
            ) from e
        self._model = ASRModel.from_pretrained(self.model_id)

    def transcribe(self, wav: Path, language: str | None) -> Result:
        if language is None:
            raise ValueError("canary needs an explicit language; set it in manifest.json")
        (output,) = self._model.transcribe(
            [str(wav)], timestamps=True, source_lang=language, target_lang=language
        )
        words = [
            {"text": w["word"], "start": w["start"], "end": w["end"]}
            for w in output.timestamp.get("word", [])
        ]
        segments = [
            {
                "text": s["segment"],
                "start": s["start"],
                "end": s["end"],
                "words": [w for w in words if s["start"] <= w["start"] < s["end"]],
            }
            for s in output.timestamp.get("segment", [])
        ]
        return {"text": output.text.strip(), "segments": segments, "detected_language": None}


BACKENDS: dict[str, type[Backend]] = {
    backend.name: backend for backend in (WhisperMLX, ParakeetMLX, VoxtralMLX, CanaryNeMo)
}
