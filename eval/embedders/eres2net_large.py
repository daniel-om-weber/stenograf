# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "numpy>=1.26",
#     "torch==2.8.0",
#     "torchaudio==2.8.0",
#     "scipy>=1.11",
#     "pyyaml>=6",
# ]
# ///
"""ERes2Net-large (3D-Speaker, VoxCeleb) in 3D-Speaker's own PyTorch code.

The path is ``speakerlab/bin/infer_sv.py``'s: Kaldi fbank 80 with per-utterance
mean normalisation (``speakerlab.process.processor.FBank(80, mean_nor=True)``,
dither 0) into ``speakerlab.models.eres2net.ERes2Net``. infer_sv.py's model
table lacks this checkpoint; its constructor arguments come from the ModelScope
repo's ``configuration.json`` (``embed_dim`` 192, ``channels`` 64, which is
the 3D-Speaker recipe's "change m_channels from 32 to 64 for eres2net-large").
The ModelScope card reports Vox1-O EER 0.57 % for this file — the plain
checkpoint, not the 0.52 % large-margin one in the 3D-Speaker README, which was
never published.

Sources (gitignored, ``eval/audio/models/``):

- code: ``src/3D-Speaker`` = github.com/modelscope/3D-Speaker @ 065629c3
- weights: ``eres2net-large/pretrained_eres2net.ckpt`` = ModelScope
  iic/speech_eres2net_large_sv_en_voxceleb_16k @ a7e82865 (git-lfs),
  sha256 c5e4d642…

Run::

    uv run eval/embedders/eres2net_large.py --manifest M.json --out eres2net-large.npz \\
        [--device mps]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from embedio import done, parser, run

MODELS = Path(__file__).resolve().parents[1] / "audio/models"
REPO = MODELS / "src/3D-Speaker"
CHECKPOINT = MODELS / "eres2net-large/pretrained_eres2net.ckpt"


def load(device: str):
    sys.path.insert(0, str(REPO))
    from speakerlab.models.eres2net.ERes2Net import ERes2Net
    from speakerlab.process.processor import FBank

    model = ERes2Net(feat_dim=80, embedding_size=192, m_channels=64)
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True))
    model.to(device).eval()
    return model, FBank(80, sample_rate=16000, mean_nor=True)


def main() -> None:
    args = parser(__doc__.splitlines()[0]).parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if done(args, "eres2net-large"):
        return
    model, fbank = load(args.device)

    @torch.inference_mode()
    def embed(audio: np.ndarray) -> np.ndarray:
        feat = fbank(torch.from_numpy(audio).unsqueeze(0)).unsqueeze(0).to(args.device)
        return model(feat).squeeze(0).cpu().numpy()

    run("eres2net-large", embed, args)


if __name__ == "__main__":
    main()
