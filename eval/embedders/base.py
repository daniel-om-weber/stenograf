"""The shipped voice embedder (ERes2Net-base via sherpa-onnx), through the
production class itself — the baseline every candidate is scored against.

Runs in the project environment, not a PEP 723 one, so it embeds with exactly
the sherpa-onnx and model file a user's install uses::

    uv run --group eval eval/embedders/base.py --manifest M.json --out base.npz

``--device`` is accepted for interface uniformity; this path is CPU only.
"""

from __future__ import annotations

from embedio import parser, run

from stenograf.diarization.embedding import VoiceEmbedder


def main() -> None:
    args = parser(__doc__.splitlines()[0]).parse_args()
    embedder = VoiceEmbedder()
    run("base", embedder.embed, args)


if __name__ == "__main__":
    main()
