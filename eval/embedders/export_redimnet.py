# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "numpy>=1.26",
#     "torch==2.8.0",
#     "torchaudio==2.8.0",
#     "scipy>=1.11",
#     "onnx>=1.16",
#     "onnxruntime>=1.20",
# ]
# ///
"""Export a ReDimNet2 checkpoint to one ONNX file, front end included.

The reference front end (``layers/features_tf.py``) is itself built from
convolutions, so the whole of ``ReDimNet2Wrap.forward`` exports: the graph
takes a ``(1, samples)`` float32 16 kHz waveform in [-1, 1) of any length and
returns the ``(1, 192)`` unnormalised embedding. No mel code exists outside
the graph, so none can drift from the reference. The publishers' own ONNX
export is fixed at 6 s of input; meeting spans are 0.5 to 10 s.

After writing, the file is run on onnxruntime against the PyTorch model on
noise of several lengths; the script fails unless every cosine is at least
0.9999. It prints the file's size and sha256 — what ``stenograf.assets``
pins.

    uv run eval/embedders/export_redimnet.py --checkpoint redimnet2-b6-vb2-lm \\
        --out eval/out/reid_models/redimnet2-b6-vb2-lm.onnx
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from redimnet import CHECKPOINTS, load

LENGTHS_S = (0.5, 0.73, 1.0, 2.0, 3.17, 6.0, 10.0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", choices=sorted(CHECKPOINTS), required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    model = load(args.checkpoint, "cpu")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (torch.zeros(1, 32000),),
        str(args.out),
        input_names=["waveform"],
        output_names=["embedding"],
        dynamic_axes={"waveform": {1: "samples"}},
        opset_version=17,
        dynamo=False,
    )

    import onnxruntime

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    session = onnxruntime.InferenceSession(
        str(args.out), options, providers=["CPUExecutionProvider"]
    )
    rng = np.random.default_rng(0)
    worst = 1.0
    for seconds in LENGTHS_S:
        audio = rng.normal(0, 0.05, (1, round(seconds * 16000))).astype(np.float32)
        with torch.inference_mode():
            want = model(torch.from_numpy(audio)).numpy()[0]
        got = session.run(None, {"waveform": audio})[0][0]
        cosine = float(want @ got / np.linalg.norm(want) / np.linalg.norm(got))
        worst = min(worst, cosine)
        print(f"{seconds:5.2f} s: cosine {cosine:.7f}")
    if worst < 0.9999:
        raise SystemExit(f"export does not match the reference (worst cosine {worst})")
    data = args.out.read_bytes()
    print(f"{args.out}: {len(data)} bytes, sha256 {hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
