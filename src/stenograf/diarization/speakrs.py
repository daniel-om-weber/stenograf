"""Speaker diarization via the ``stenodiar`` helper (speakrs).

speakrs reimplements the pyannote community-1 pipeline (segmentation →
embeddings → PLDA → VBx clustering) in Rust; VBx is what makes its *automatic*
speaker-count estimation trustworthy, where sherpa's threshold clustering finds
13–25 "speakers" in a 3-person meeting (measured 2026-07-10 on the eval
segments). It exposes no way to force a known count, so this backend covers
exactly the estimate case and delegates known counts to the sherpa backend —
whose accuracy with an explicit count was never the problem.

Audio is piped to the helper as raw PCM (meeting audio never touches disk) and
turns come back as JSON. Re-ID voiceprints keep coming from sherpa's
``SpeakerEmbeddingExtractor`` no matter which backend diarized: saved speaker
profiles are cosine-matched against ``models.SPEAKER_EMBEDDING`` vectors, so a
second embedding model would silently break every enrolled profile.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from importlib import resources
from pathlib import Path

import numpy as np

from stenograf.diarization.base import DiarizationResult, Diarizer, SpeakerTurn
from stenograf.diarization.sherpa import SherpaOnnxDiarizer, cluster_embeddings

HELPER_NAME = "stenodiar"
_ENV_OVERRIDE = "STENOGRAF_DIAR_HELPER"

_TIMEOUT_S = 1800
"""Hard cap on one helper run. Warm runs take under a second per meeting-hour;
the first run ever also downloads the models and compiles them for CoreML
(minutes, then cached per machine), so the cap is generous, not tight."""

DEFAULT_MODE = "coreml" if sys.platform == "darwin" else "cpu"
"""stenodiar execution mode: CoreML on macOS, ONNX Runtime CPU elsewhere
(GPU execution providers are a later opt-in, mirroring the ASR backend)."""


class DiarizerHelperNotFoundError(RuntimeError):
    """The stenodiar diarization helper binary could not be located."""


def find_stenodiar() -> Path:
    """Locate the ``stenodiar`` binary: env override, packaged bin, then dev build."""
    override = os.environ.get(_ENV_OVERRIDE)
    if override:
        return Path(override)

    packaged = resources.files("stenograf") / "bin" / HELPER_NAME
    if packaged.is_file():
        path = Path(str(packaged))
        if not os.access(path, os.X_OK):
            path.chmod(path.stat().st_mode | 0o755)
        return path

    dev = Path(__file__).resolve().parents[3] / "native" / "stenodiar" / HELPER_NAME
    if dev.is_file():
        return dev

    raise DiarizerHelperNotFoundError(
        f"diarization helper '{HELPER_NAME}' not found. Build it with "
        f"native/stenodiar/build.sh, or set {_ENV_OVERRIDE} to its path."
    )


class SpeakrsCliDiarizer(Diarizer):
    """Estimate-mode diarization through stenodiar, known counts through sherpa.

    ``command`` overrides the helper argv prefix (tests point it at a fake);
    production locates the real binary via :func:`find_stenodiar`.
    """

    def __init__(
        self,
        sherpa: SherpaOnnxDiarizer,
        *,
        command: list[str] | None = None,
        mode: str = DEFAULT_MODE,
    ) -> None:
        self._sherpa = sherpa
        self._command = command
        self._mode = mode

    def diarize(self, samples: np.ndarray, num_speakers: int | None = None) -> list[SpeakerTurn]:
        if num_speakers is not None:
            return self._sherpa.diarize(samples, num_speakers)
        return self._run_helper(samples)

    def diarize_with_embeddings(
        self, samples: np.ndarray, num_speakers: int | None = None
    ) -> DiarizationResult:
        turns = self.diarize(samples, num_speakers)
        return DiarizationResult(
            turns=turns, embeddings=cluster_embeddings(turns, samples, self._sherpa.embed)
        )

    def _run_helper(self, samples: np.ndarray) -> list[SpeakerTurn]:
        command = self._command or [str(find_stenodiar())]
        pcm = _to_int16(samples).tobytes()
        try:
            proc = subprocess.run(
                [*command, "--mode", self._mode, "--stdin"],
                input=pcm,
                capture_output=True,
                timeout=_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"{HELPER_NAME} timed out after {_TIMEOUT_S}s") from None
        if proc.returncode != 0:
            detail = proc.stderr.decode(errors="replace").strip().splitlines()
            raise RuntimeError(
                f"{HELPER_NAME} failed (exit {proc.returncode}): "
                f"{detail[-1] if detail else 'no error output'}"
            )
        try:
            payload = json.loads(proc.stdout)
            turns = [
                SpeakerTurn(
                    speaker=_normalize_label(t["speaker"]),
                    start=float(t["start"]),
                    end=float(t["end"]),
                )
                for t in payload["turns"]
            ]
        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(f"{HELPER_NAME} returned unparseable output: {exc}") from exc
        return sorted(turns, key=lambda t: (t.start, t.end))


def _normalize_label(label: str) -> str:
    """speakrs' ``SPEAKER_07`` → the ``S7`` convention every other backend emits."""
    prefix, _, index = label.rpartition("_")
    if prefix == "SPEAKER" and index.isdigit():
        return f"S{int(index)}"
    return label


def _to_int16(samples: np.ndarray) -> np.ndarray:
    if samples.dtype == np.int16:
        return samples
    return (np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0) * 32767.0).astype(np.int16)
