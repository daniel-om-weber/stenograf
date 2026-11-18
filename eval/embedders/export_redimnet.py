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

Two things are done to the traced graph before it is checked. The stage
aggregation — stack up to seven feature maps, multiply by constant weights,
sum — is rewritten as one multiply per map and a sum, which never builds the
stacked tensor: 11 % less CPU and 7–23 % less peak memory with the same
vectors (2026-11-18, 80 AMI spans, onnxruntime 1.27). And the file's metadata
names its origin and licence, since the bytes travel without this repository.

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
SOURCE = "https://github.com/PalabraAI/redimnet2"
LICENCE = "CC BY-NC-SA 4.0"


def unstack_aggregations(model) -> int:
    """Rewrite every ``ReduceSum(Concat(Unsqueeze(x_i)) * W)`` as
    ``Sum(x_i * W[:, i])``; returns how many were rewritten."""
    from onnx import helper, numpy_helper

    graph = model.graph
    made_by = {out: node for node in graph.node for out in node.output}
    constants = {i.name: i for i in graph.initializer}

    def constant(name: str) -> np.ndarray:
        if name in constants:
            return numpy_helper.to_array(constants[name])
        return numpy_helper.to_array(made_by[name].attribute[0].t)

    replaced: dict[str, list] = {}
    dropped: set[str] = set()
    for node in graph.node:
        if node.op_type != "ReduceSum":
            continue
        product = made_by.get(node.input[0])
        if product is None or product.op_type != "Mul":
            continue
        weight = next((i for i in product.input if i in constants), None)
        stack = next(
            (made_by[i] for i in product.input if i in made_by and made_by[i].op_type == "Concat"),
            None,
        )
        if weight is None or stack is None:
            continue
        keepdims = next((a.i for a in node.attribute if a.name == "keepdims"), 1)
        if list(np.atleast_1d(constant(node.input[1]))) != [1] or keepdims != 0:
            raise SystemExit(f"{node.name}: not a sum over the stacked axis")
        weights = numpy_helper.to_array(constants[weight])
        if weights.shape[1] != len(stack.input):
            raise SystemExit(f"{node.name}: {weights.shape} weights for {len(stack.input)} maps")
        terms = []
        for k, stacked in enumerate(stack.input):
            lifted = made_by[stacked]
            if lifted.op_type != "Unsqueeze" or list(np.atleast_1d(constant(lifted.input[1]))) != [
                1
            ]:
                raise SystemExit(f"{node.name}: input {k} is not a map lifted onto axis 1")
            name = f"{node.name}_w{k}"
            graph.initializer.append(
                numpy_helper.from_array(np.ascontiguousarray(weights[:, k]), name)
            )
            terms.append(
                helper.make_node(
                    "Mul", [lifted.input[0], name], [f"{node.name}_t{k}"], name=f"{node.name}_t{k}"
                )
            )
            dropped.add(lifted.name)
        total = helper.make_node(
            "Sum", [t.output[0] for t in terms], [node.output[0]], name=f"{node.name}_sum"
        )
        replaced[node.name] = [*terms, total]
        dropped |= {node.name, product.name, stack.name}
    nodes = []
    for node in graph.node:
        if node.name in replaced:
            nodes += replaced[node.name]
        elif node.name not in dropped:
            nodes.append(node)
    del graph.node[:]
    graph.node.extend(nodes)
    # Constants that fed only the removed nodes: left in, onnxruntime warns
    # about each one on every load.
    outputs = {o.name for o in graph.output}
    while True:
        used = {i for node in graph.node for i in node.input} | outputs
        dangling = [node for node in graph.node if not used.intersection(node.output)]
        if not dangling:
            break
        for node in dangling:
            graph.node.remove(node)
    for stale in [i for i in graph.initializer if i.name not in used]:
        graph.initializer.remove(stale)
    return len(replaced)


def finish(path: Path, checkpoint: str) -> None:
    """Unstack the aggregations and stamp origin and licence into the file."""
    import onnx

    model = onnx.load(str(path))
    count = unstack_aggregations(model)
    if not count:
        raise SystemExit("no stage aggregation found to rewrite")
    for key, value in {
        "source": SOURCE,
        "checkpoint": CHECKPOINTS[checkpoint],
        "license": LICENCE,
        "converted_by": "stenograf eval/embedders/export_redimnet.py",
    }.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    print(f"rewrote {count} stage aggregations; {len(model.graph.node)} nodes")


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
    finish(args.out, args.checkpoint)

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
