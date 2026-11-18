"""NVIDIA TitaNet-Large through sherpa-onnx's ``SpeakerEmbeddingExtractor``.

The graph is NeMo's own ONNX export of the network (its metadata says
``framework = nemo``; inputs ``audio_signal``/``length``, outputs
``logits``/``embs`` — the encoder and decoder only). The log-mel front end is
NOT NeMo's code: sherpa-onnx recomputes it in C++ from the metadata (hann
25/10 ms, 80 bins, per-feature normalisation). Parity of that front end
against NeMo's ``AudioToMelSpectrogramPreprocessor`` was not measured —
installing NeMo to check it costs more than this candidate is worth unless it
places near the top.

Weights: ``eval/audio/models/titanet/nemo_en_titanet_large.onnx`` from the
sherpa-onnx ``speaker-recongition-models`` release (sha256 d51abcf3…). Runs in
the project environment, which already ships sherpa-onnx::

    uv run --group eval eval/embedders/titanet.py --manifest M.json --out titanet-l.npz

``--device`` is accepted for interface uniformity; this path is CPU only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import sherpa_onnx
from embedio import SAMPLE_RATE, parser, run

MODEL = Path(__file__).resolve().parents[1] / "audio/models/titanet/nemo_en_titanet_large.onnx"


def main() -> None:
    p = parser(__doc__.splitlines()[0])
    p.add_argument("--model", type=Path, default=MODEL)
    args = p.parse_args()
    # One thread by default, like the shipped embedder, so speeds compare.
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(args.model), num_threads=args.threads or 1
        )
    )

    def embed(audio: np.ndarray) -> np.ndarray | None:
        stream = extractor.create_stream()
        stream.accept_waveform(SAMPLE_RATE, np.ascontiguousarray(audio, dtype=np.float32))
        stream.input_finished()
        if not extractor.is_ready(stream):
            return None
        return np.asarray(extractor.compute(stream), dtype=np.float32)

    run("titanet-l", embed, args)


if __name__ == "__main__":
    main()
