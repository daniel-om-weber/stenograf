# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "numpy>=1.26",
#     "torch==2.8.0",
#     "torchaudio==2.8.0",
#     "scipy>=1.11",
#     "soundfile>=0.12",
#     "hyperpyyaml>=1.2",
#     "pyyaml>=6",
#     "calflops==0.3.2",
#     "peft==0.17.1",
#     # Installed only because peft declares it; the repo's vendored copy
#     # shadows it on sys.path. Pinned so the resolver cannot pull 5.x, whose
#     # hub/tokenizers floors the vendored copy's import-time check rejects.
#     "transformers==4.57.1",
#     "tokenizers>=0.22,<0.23",
#     "huggingface-hub>=0.34,<1.0",
#     "safetensors>=0.4.3",
# ]
# ///
"""w2v-BERT 2.0 speaker verification (Li, Cheng & Li 2025) in its own code.

The path is ``recipes/DeepASV/utils/get_embd_w2v.py``'s: the modules are built
from the checkpoint's own ``train.yaml`` through ``read_hyperyaml`` (it
instantiates only repo classes and torch/os callables — read before use), the
checkpoint is copied in with that script's shape-matched partial load, and the
speaker model is called on the 1-D float waveform, cut at ``max_valid_dur``
(60 s) exactly as that script cuts it. The repo vendors a modified
transformers 4.57-dev; its only changes to w2v-BERT add structured-pruning
masks, all disabled by ``config_prune_tea.json`` (``prune: false``), so the
network is upstream w2v-BERT 2.0 plus the repo's adapters and pooling. The
encoder config named "prune" also means facebook/w2v-bert-2.0's own weights are
never read — every encoder tensor comes from the checkpoint.

Two deliberate differences from that script: it runs on CPU/MPS in float32
where the authors scored on CUDA under bfloat16 autocast (``use_amp: true``),
and the checkpoint loads with ``weights_only=True`` (it holds only tensors).

``--checkpoint`` picks the model string:

- ``w2v-bert-sv``: ``model_lmft_0.14.pth`` + ``train.yaml`` (VoxCeleb2 +
  VoxBlink2, large-margin fine-tuned, Vox1-O EER 0.14 %), 580 M + 6 M params.
- ``w2v-bert-sv-tidyvoice``: ``TidyVoice2026_GRL/ckpt_0044.pth`` + its
  ``train.yaml`` (``Audio2Vec_based_Adapter_Language``) — a language-invariant
  variant. NOT runnable: that class is absent from the public repo, so the
  checkpoint cannot be built from the code that trained it; the weights were
  not downloaded.

Sources (gitignored, ``eval/audio/models/``): code ``src/w2v-bert-sv`` =
github.com/ZXHY-82/w2v-BERT-2.0_SV @ 98ff0e29; weights ``w2v-bert-sv/`` =
huggingface.co/zl389/w2v-bert-2.0_SV @ 4e7b92e8 (sha256 454398fe…).

Run on MPS for corpus-sized manifests (speeds in ``eval/embedders/README.md``;
MPS agrees with CPU to cosine ≥ 0.999999999, 2026-11-17)::

    uv run eval/embedders/w2v_bert.py --manifest M.json --out w2v-bert-sv.npz --device mps
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from embedio import SAMPLE_RATE, done, parser, run

MODELS = Path(__file__).resolve().parents[1] / "audio/models"
REPO = MODELS / "src/w2v-bert-sv"
WEIGHTS = MODELS / "w2v-bert-sv"
CHECKPOINTS = {
    "w2v-bert-sv": ("model_lmft_0.14.pth", "train.yaml"),
}


def load(name: str, device: str):
    """Modules and hparams, built and loaded the way get_embd_w2v.py does."""
    ckpt_file, yaml_file = CHECKPOINTS[name]
    # get_embd_w2v.py's own sys.path, made absolute; spk_model.py appends
    # cwd-relative paths at import, which resolve inside the repo only from
    # the directory the authors ran it in.
    for p in (
        REPO / "deeplab/pretrained/audio2vector/module/transformers/src",
        REPO,
        REPO / "recipes/DeepASV",
    ):
        sys.path.insert(0, str(p))
    cwd = os.getcwd()
    os.chdir(REPO / "recipes/DeepASV/utils")
    try:
        from deeplab.utils.fileio import read_hyperyaml

        hparams = read_hyperyaml(path=str(WEIGHTS / yaml_file))
    finally:
        os.chdir(cwd)

    ckpt = torch.load(WEIGHTS / ckpt_file, map_location="cpu", weights_only=True)
    modules = hparams["modules"]
    report = {}
    for key, module in modules.items():
        if key == "classifier":
            continue
        state = module.state_dict()
        loaded = ckpt["modules"][key]
        unmatched = [k for k in state if not (k in loaded and state[k].shape == loaded[k].shape)]
        for k in state:
            if k not in unmatched:
                state[k] = loaded[k]
        module.load_state_dict(state)
        module.eval().to(device)
        report[key] = (len(state), unmatched, [k for k in loaded if k not in state])
    return modules["spk_model"], hparams, report


def main() -> None:
    p = parser(__doc__.splitlines()[0])
    p.add_argument("--checkpoint", choices=sorted(CHECKPOINTS), default="w2v-bert-sv")
    args = p.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if done(args, args.checkpoint):
        return
    model, hparams, report = load(args.checkpoint, args.device)
    for key, (n, unmatched, unused) in report.items():
        print(
            f"[{args.checkpoint}] {key}: {n - len(unmatched)}/{n} tensors loaded, "
            f"unmatched {unmatched[:5]}, unused checkpoint keys {unused[:5]}",
            file=sys.stderr,
        )
    max_len = int(hparams["max_valid_dur"] * hparams["sample_rate"])
    assert hparams["sample_rate"] == SAMPLE_RATE

    @torch.inference_mode()
    def embed(audio: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(audio[:max_len]).float()
        return model(x).float().squeeze(0).cpu().numpy()

    run(args.checkpoint, embed, args)


if __name__ == "__main__":
    main()
