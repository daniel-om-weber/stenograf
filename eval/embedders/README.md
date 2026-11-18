# Speaker-embedding runners

One runner per candidate model. Each reads a manifest, embeds every span in
the model's own reference implementation (3D-Speaker / ReDimNet2 / w2v-BERT
PyTorch code, sherpa-onnx for the ONNX models) and writes one `.npz`, so a
scorer compares models on sample-identical inputs. The shared contract —
manifest format, slicing, L2 normalisation, NaN rows for refused spans,
caching, timing — is `embedio.py`; `check.py` asserts an output honours it.

```sh
# project environment (sherpa-onnx is already a dependency)
uv run --group eval eval/embedders/base.py     --manifest M.json --out out/base.npz
uv run --group eval eval/embedders/titanet.py  --manifest M.json --out out/titanet-l.npz
# self-contained PEP 723 environments (torch 2.8)
uv run eval/embedders/eres2net_large.py --manifest M.json --out out/eres2net-large.npz
uv run eval/embedders/redimnet.py --checkpoint redimnet2-b6-vb2-lm --manifest M.json --out out/redimnet2-b6-vb2-lm.npz
#   --checkpoint also: redimnet2-b6-vb2-ptn, redimnet2-b6-vb2cnc-lm, redimnet2-b6-vox2-lm
uv run eval/embedders/w2v_bert.py --manifest M.json --out out/w2v-bert-sv.npz --device mps
python3 eval/embedders/check.py --manifest M.json out/*.npz
```

Every runner takes `--manifest`, `--out`, `--device cpu|mps` (the two
sherpa-onnx runners are CPU only) and `--threads`; an existing `--out` with
the same model, item ids and span counts is reused. Each module docstring names
its exact code commit, weight file and hash, and every way it departs from the
upstream inference script.

Code and weights live under `eval/audio/models/` (gitignored): the cloned
repos in `src/<repo>`, weights in one directory per model. Nothing from those
repos is installed or executed beyond the model and feature modules each
runner imports; checkpoints load with `weights_only=True`.

## Reference parity and speed (M4 Max, 2026-11-17)

Parity: each PyTorch runner against its repo's own inference path (the
upstream loader and file reader on three 10-s AMI clips written to WAV) — min
cosine 1.0000000 for all six PyTorch models; CPU vs MPS ≥ 0.999999999. The two
sherpa-onnx runners *are* the reference path (`base` calls the shipped
`VoiceEmbedder`). w2v-BERT's authors scored under CUDA bfloat16 autocast; the
runner computes in float32, which differs from CPU bf16 autocast by cosine
0.99994.

Compute per 10 s of speech (eight 10-s spans, one run each, warm):

| model | CPU 1 thread | CPU 12 threads | MPS |
|---|---|---|---|
| base | 0.15 s | — | — |
| titanet-l | 0.07 s | — | — |
| eres2net-large | 0.24 s | 0.11 s | 0.05 s |
| redimnet2-b6 (any checkpoint) | 0.51 s | 0.31 s | 0.10 s |
| w2v-bert-sv | 0.76 s | 1.31 s | 0.12 s |
