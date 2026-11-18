"""The shipped voice embedder, through the production class itself.

Runs in the project environment, not a PEP 723 one, so it embeds with exactly
the onnxruntime and model file a user's install uses — which makes its output
against a reference runner's the parity check of the export::

    uv run --group eval eval/embedders/shipped.py --manifest M.json --out shipped.npz

``--device`` is accepted for interface uniformity; this path is CPU only.
"""

from __future__ import annotations

from embedio import parser, run

from stenograf.diarization.embedding import VoiceEmbedder


def main() -> None:
    args = parser(__doc__.splitlines()[0]).parse_args()
    embedder = VoiceEmbedder()
    run("shipped", embedder.embed, args)


if __name__ == "__main__":
    main()
