# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "numpy>=1.26",
#     "torch==2.8.0",
#     "torchaudio==2.8.0",
#     "scipy>=1.11",
# ]
# ///
"""ReDimNet2+ (lab260): ReDimNet2-B6 retrained for channel and codec shift.

Run in the authors' vendored model code, built as their inference script
builds it (the B6 preset, ``return_2d_output=True`` — their model card's
``False`` does not fit the weights) and fed the model's own front end on a
(1, samples) waveform. The same weights in PalabraAI's code give cosine
0.9989–1.0000 to this path (six spans, 2026-11-18), so the two are not
interchangeable bit for bit and the authors' path is the reference.

Sources (gitignored, ``eval/audio/models/``): code
``src/redimnet2-plus/asv/models/redimnet2_vendor`` =
github.com/lab260ru/redimnet2-plus @ 2a8d6db; weights
``redimnet2-weights/redimnet2-plus-lmft.bin`` = ``pytorch_model_fsdp.bin``
there and at huggingface.co/lab260/redimnet2-plus, read with
``weights_only=True``. Only the vendored model directory is imported; the
package's own ``__init__`` files are not run.

Run::

    uv run eval/embedders/redimnet_plus.py --manifest M.json --out redimnet2-plus-lmft.npz
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch
from embedio import done, parser, run

MODELS = Path(__file__).resolve().parents[1] / "audio/models"
REPO = MODELS / "src/redimnet2-plus"
WEIGHTS = MODELS / "redimnet2-weights/redimnet2-plus-lmft.bin"
NAME = "redimnet2-plus-lmft"
B6 = {
    "causal": "none",
    "feat_type": "tf",
    "C": 64,
    "out_channels": 224,
    "return_2d_output": True,
    "stages_setup": [
        ((1, 1), 3, 3, [(3, 3)], 64),
        ((2, 1), 4, 2, [(3, 3)], 64),
        ((1, 2), 5, 2, [(3, 3)], 48),
        ((2, 1), 5, 1, [(3, 3)], 48),
        ((1, 2), 4, 0.75, [(3, 3)], 32),
        ((2, 1), 3, 0.5, [(3, 3)], 24),
    ],
}


def load(device: str):
    for name in ("asv", "asv.models"):
        package = types.ModuleType(name)
        package.__path__ = [str(REPO / name.replace(".", "/"))]
        sys.modules[name] = package
    from asv.models.redimnet2_vendor.redimnet2 import (
        ReDimNet2Wrap,  # type: ignore[import-not-found]
    )

    raw = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    prefix = "encoder.model."
    state = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in raw.items()}
    model = ReDimNet2Wrap(**B6)
    model.load_state_dict({k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)})
    return model.to(device).eval()


def main() -> None:
    args = parser(__doc__.splitlines()[0]).parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if done(args, NAME):
        return
    model = load(args.device)

    @torch.inference_mode()
    def embed(audio: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(audio).unsqueeze(0).to(args.device)
        return model(model.spec(x)).squeeze(0).cpu().numpy()

    run(NAME, embed, args)


if __name__ == "__main__":
    main()
