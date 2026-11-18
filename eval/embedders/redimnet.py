# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "numpy>=1.26",
#     "torch==2.8.0",
#     "torchaudio==2.8.0",
#     "scipy>=1.11",
# ]
# ///
"""ReDimNet2 (PalabraAI) in the repo's own PyTorch code, front end in-graph.

The model is built exactly as ``redimnet2.load_custom`` (the torch.hub entry)
builds it — ``ReDimNet2Wrap(**ckpt["model_config"])`` and a strict
``load_state_dict`` — and called on a (1, samples) float waveform as the README
does. Two differences from that path: the checkpoint is read from disk rather
than ``torch.hub.load_state_dict_from_url``, and with ``weights_only=True``
(the file holds only tensors and plain config values, so nothing is lost).

``--checkpoint`` picks the model string:

==========================  ==================================  =========================
model string                release asset (v1.0.0)              training data
==========================  ==================================  =========================
redimnet2-b6-vb2-lm         b6-vb2+vox2_v0-lm.pt                VoxBlink2 + VoxCeleb2
redimnet2-b6-vb2-ptn        b6-vb2+vox2_v0-ptn.pt               same, before LM fine-tune
redimnet2-b6-vb2cnc-lm      b6-vb2+vox2+cnc2_v0-lm.pt           + CN-Celeb2
redimnet2-b6-vox2-lm        b6-vox2-lm.pt                       VoxCeleb2 only
redimnet2-b3-vb2cnc-lm      b3-vb2+vox2+cnc2_v0-lm.pt           VoxBlink2+VoxCeleb2+CN-Celeb2
redimnet2-b{3,2,0}-vox2-lm  b{3,2,0}-vox2-lm.pt                 VoxCeleb2 only
==========================  ==================================  =========================

The ``ptn`` checkpoint is in the set because it was trained on 2 s crops
before the large-margin fine-tune moved to longer ones, so it may hold up
better on short meeting turns than its Vox1-O row suggests.

Sources (gitignored, ``eval/audio/models/``): code ``src/redimnet2`` =
github.com/PalabraAI/redimnet2 @ c5bbe0b7; weights ``redimnet2-weights/``.

Run::

    uv run eval/embedders/redimnet.py --checkpoint redimnet2-b6-vb2-lm \\
        --manifest M.json --out redimnet2-b6-vb2-lm.npz [--device mps]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from embedio import done, parser, run

MODELS = Path(__file__).resolve().parents[1] / "audio/models"
REPO = MODELS / "src/redimnet2"
WEIGHTS = MODELS / "redimnet2-weights"
CHECKPOINTS = {
    "redimnet2-b6-vb2-lm": "b6-vb2+vox2_v0-lm.pt",
    "redimnet2-b6-vb2-ptn": "b6-vb2+vox2_v0-ptn.pt",
    "redimnet2-b6-vb2cnc-lm": "b6-vb2+vox2+cnc2_v0-lm.pt",
    "redimnet2-b6-vox2-lm": "b6-vox2-lm.pt",
    "redimnet2-b3-vb2cnc-lm": "b3-vb2+vox2+cnc2_v0-lm.pt",
    "redimnet2-b3-vox2-lm": "b3-vox2-lm.pt",
    "redimnet2-b2-vox2-lm": "b2-vox2-lm.pt",
    "redimnet2-b0-vox2-lm": "b0-vox2-lm.pt",
}


def load(name: str, device: str):
    sys.path.insert(0, str(REPO))
    from redimnet2.redimnet2 import ReDimNet2Wrap

    ckpt = torch.load(WEIGHTS / CHECKPOINTS[name], map_location="cpu", weights_only=True)
    model = ReDimNet2Wrap(**ckpt["model_config"])
    res = model.load_state_dict(ckpt["state_dict"])
    if res.missing_keys or res.unexpected_keys:
        raise RuntimeError(f"{name}: missing {res.missing_keys}, unexpected {res.unexpected_keys}")
    return model.to(device).eval()


def main() -> None:
    p = parser(__doc__.splitlines()[0])
    p.add_argument("--checkpoint", choices=sorted(CHECKPOINTS), required=True)
    args = p.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if done(args, args.checkpoint):
        return
    model = load(args.checkpoint, args.device)

    @torch.inference_mode()
    def embed(audio: np.ndarray) -> np.ndarray:
        return model(torch.from_numpy(audio).unsqueeze(0).to(args.device)).squeeze(0).cpu().numpy()

    run(args.checkpoint, embed, args)


if __name__ == "__main__":
    main()
