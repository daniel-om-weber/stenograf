"""Are the pinned Parakeet Ultra files Moondream's weights, and nothing else?

Moondream publishes Parakeet Ultra in the transformers layout for its own
runtime. stenograf's two backends load third-party conversions of it
(``asr/parakeet.py``, ``asr/parakeet_onnx.py``), and a conversion is a file
somebody else wrote: this compares every tensor of both against the original,
so the pins rest on arithmetic and not on a model card. Run it before moving
either pin.

- **MLX**: the original, renamed and transposed into parakeet-mlx's layout
  (:func:`mlx_layout`), must equal the pinned file tensor for tensor.
- **ONNX**: every weight in the two graphs must be one of the original's,
  as it is or transposed. Three kinds are stored transformed and are rebuilt
  here instead: each conformer block's depthwise convolution with its batch
  norm folded in (equal to rounding, not bit for bit — the fold is float
  arithmetic), the LSTM's gates in ONNX's order, and its two bias vectors
  concatenated.

One difference is expected in both and reported, not failed: the ``<unk>``
output bias, lowered by 10 000 (``parakeet.REVISION``).

Usage (the ``onnx`` package only reads the graph; stenograf does not need it):
    uv run --group eval --with onnx eval/asr_weights.py
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ORIGINAL = "moondream/parakeet-ultra"
ORIGINAL_REVISION = "73175eb7aeb0d82f1e2a6b53b3aabc10a90bcd0b"
UNK = 0
FOLD_TOLERANCE = 1e-5
"""Relative. The folded tensors of the v3 export, rebuilt the same way from
v3's own weights, land within 2.4e-7 (2026-11-26)."""

Tensors = dict[str, np.ndarray]

_ATTENTION = {
    "q_proj.weight": "linear_q.weight",
    "k_proj.weight": "linear_k.weight",
    "v_proj.weight": "linear_v.weight",
    "o_proj.weight": "linear_out.weight",
    "relative_k_proj.weight": "linear_pos.weight",
    "bias_u": "pos_bias_u",
    "bias_v": "pos_bias_v",
}


def mlx_layout(original: Tensors) -> Tensors:
    """The transformers checkpoint under parakeet-mlx's names and axis orders."""
    out: Tensors = {}
    for key, tensor in original.items():
        if key.startswith("vad_head.") or key.endswith("num_batches_tracked"):
            continue  # Moondream's own segmenter and a training counter
        if m := re.fullmatch(r"encoder\.subsampling\.layers\.(\d+)\.(weight|bias)", key):
            name = f"encoder.pre_encode.conv.{m[1]}.{m[2]}"
            out[name] = tensor.transpose(0, 2, 3, 1) if m[2] == "weight" else tensor
        elif m := re.fullmatch(r"encoder\.subsampling\.linear\.(weight|bias)", key):
            out[f"encoder.pre_encode.out.{m[1]}"] = tensor
        elif m := re.fullmatch(r"encoder\.layers\.(\d+)\.(.+)", key):
            prefix, rest = f"encoder.layers.{m[1]}.", m[2]
            if rest.removeprefix("self_attn.") in _ATTENTION and rest.startswith("self_attn."):
                out[prefix + "self_attn." + _ATTENTION[rest.removeprefix("self_attn.")]] = tensor
            elif rest.startswith("conv.norm."):
                out[prefix + "conv.batch_norm." + rest.removeprefix("conv.norm.")] = tensor
            elif re.fullmatch(r"conv\.\w+_conv\d?\.weight", rest):
                out[prefix + rest] = tensor.transpose(0, 2, 1)
            else:
                out[prefix + rest] = tensor
        elif key == "decoder.embedding.weight":
            out["decoder.prediction.embed.weight"] = tensor
        elif m := re.fullmatch(r"decoder\.lstm\.weight_(ih|hh)_l(\d+)", key):
            which = "Wx" if m[1] == "ih" else "Wh"
            out[f"decoder.prediction.dec_rnn.lstm.{m[2]}.{which}"] = tensor
        elif m := re.fullmatch(r"decoder\.lstm\.bias_ih_l(\d+)", key):
            # MLX's LSTM has one bias where torch has two that are only ever added.
            hidden = original[f"decoder.lstm.bias_hh_l{m[1]}"]
            out[f"decoder.prediction.dec_rnn.lstm.{m[1]}.bias"] = tensor + hidden
        elif re.fullmatch(r"decoder\.lstm\.bias_hh_l\d+", key):
            continue
        elif m := re.fullmatch(r"decoder\.decoder_projector\.(weight|bias)", key):
            out[f"joint.pred.{m[1]}"] = tensor
        elif m := re.fullmatch(r"encoder_projector\.(weight|bias)", key):
            out[f"joint.enc.{m[1]}"] = tensor
        elif m := re.fullmatch(r"joint\.head\.(weight|bias)", key):
            out[f"joint.joint_net.2.{m[1]}"] = tensor
        else:
            raise KeyError(f"no place in the MLX layout for {key}")
    return out


class Report:
    def __init__(self, title: str) -> None:
        self.title = title
        self.counts: Counter[str] = Counter()
        self.failures: list[str] = []
        self.unk: list[float] = []

    def compare(self, name: str, got: np.ndarray, want: np.ndarray, kind: str) -> None:
        if got.shape != want.shape:
            self.failures.append(f"{name}: shape {got.shape}, expected {want.shape}")
        elif np.array_equal(got, want):
            self.counts[f"{kind}, bit-identical"] += 1
        elif kind == "folded conv+norm" and (
            np.max(np.abs(got - want)) <= FOLD_TOLERANCE * np.max(np.abs(want))
        ):
            self.counts[f"{kind}, equal to rounding"] += 1
        else:
            differing = np.argwhere(got != want)
            if name.endswith("bias") and differing.tolist() == [[UNK]]:
                self.counts[f"{kind}, bit-identical"] += 1
                self.unk.append(float(got[UNK] - want[UNK]))
            else:
                self.failures.append(f"{name}: {len(differing)} of {got.size} values differ")

    def print(self) -> bool:
        print(self.title)
        for kind, count in sorted(self.counts.items()):
            print(f"  {count:4d} {kind}")
        print(f"  <unk> output bias moved by {self.unk or 'nothing'}")
        for failure in self.failures[:20]:
            print(f"  FAIL {failure}")
        return not self.failures


def check_mlx(original: Tensors, path: Path) -> bool:
    from safetensors.numpy import load_file

    report = Report(f"MLX: {path}")
    want, got = mlx_layout(original), load_file(str(path))
    for name in sorted(set(want) ^ set(got)):
        report.failures.append(f"{name}: in only one of the two")
    for name in sorted(set(want) & set(got)):
        report.compare(name, got[name].astype(np.float32), want[name], "tensor")
    return report.print()


def _initializers(path: Path) -> Tensors:
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(path), load_external_data=False)
    data: np.memmap | None = None
    out: Tensors = {}
    for init in model.graph.initializer:
        if init.data_location != onnx.TensorProto.EXTERNAL:
            out[init.name] = numpy_helper.to_array(init)
            continue
        info = {entry.key: entry.value for entry in init.external_data}
        if data is None:
            data = np.memmap(path.parent / info["location"], mode="r")
        start = int(info.get("offset", 0))
        dtype = onnx.helper.tensor_dtype_to_np_dtype(init.data_type)
        raw = np.frombuffer(data[start : start + int(info["length"])], dtype=dtype)
        out[init.name] = raw.reshape(tuple(init.dims))
    return out


def _onnx_gates(tensor: np.ndarray) -> np.ndarray:
    """torch's LSTM gate order (i, f, g, o) → ONNX's (i, o, f, c)."""
    i, f, g, o = np.split(tensor, 4)
    return np.concatenate([i, o, f, g])


def check_onnx(original: Tensors, directory: Path) -> bool:
    report = Report(f"ONNX: {directory}")
    rebuilt: dict[str, tuple[str, np.ndarray]] = {}  # consumed original name → (kind, tensor)
    folded: list[np.ndarray] = []
    for layer in range(24):
        prefix = f"encoder.layers.{layer}.conv."
        norm = {
            n: original[f"{prefix}norm.{n}"]
            for n in ("weight", "bias", "running_mean", "running_var")
        }
        scale = (norm["weight"] / np.sqrt(norm["running_var"] + np.float32(1e-5))).astype(
            np.float32
        )
        folded.append(original[prefix + "depthwise_conv.weight"] * scale[:, None, None])
        folded.append((norm["bias"] - norm["running_mean"] * scale).astype(np.float32))
        rebuilt.update({f"{prefix}norm.{n}": ("", norm[n]) for n in norm})
        rebuilt[prefix + "depthwise_conv.weight"] = ("", folded[-2])
    lstm: dict[tuple[int, ...], list[np.ndarray]] = {}
    for layer in range(2):
        for which in ("ih", "hh"):
            weight = _onnx_gates(original[f"decoder.lstm.weight_{which}_l{layer}"])[None]
            lstm.setdefault(weight.shape, []).append(weight)
        biases = [_onnx_gates(original[f"decoder.lstm.bias_{w}_l{layer}"]) for w in ("ih", "hh")]
        lstm.setdefault((1, biases[0].size * 2), []).append(np.concatenate(biases)[None])
    plain = {
        name: tensor
        for name, tensor in original.items()
        if name not in rebuilt and ".lstm." not in name and tensor.size >= 16
    }
    by_range: dict[tuple[int, float, float], list[str]] = {}
    for name, tensor in plain.items():
        by_range.setdefault((tensor.size, float(tensor.min()), float(tensor.max())), []).append(
            name
        )
    used: Counter[str] = Counter()

    for graph in ("encoder-model.onnx", "decoder_joint-model.onnx"):
        for name, got in _initializers(directory / graph).items():
            if got.dtype != np.float32 or got.size < 16:
                continue  # shape constants
            if got.shape in lstm and any(np.array_equal(got, want) for want in lstm[got.shape]):
                report.counts["LSTM gates reordered, bit-identical"] += 1
                continue
            key = (got.size, float(got.min()), float(got.max()))
            source = next(
                (
                    n
                    for n in by_range.get(key, [])
                    if any(np.array_equal(got, v) for v in _views(plain[n], got.shape))
                ),
                None,
            )
            if source is not None:
                used[source] += 1
                report.counts["weight, bit-identical"] += 1
            elif got.shape == original["joint.head.bias"].shape:
                used["joint.head.bias"] += 1
                report.compare(name, got, original["joint.head.bias"], "weight")
            elif folded and got.shape == folded[0].shape:
                report.compare(name, got, folded.pop(0), "folded conv+norm")
            else:
                report.failures.append(f"{name} {got.shape}: none of the original's tensors")
    for name in plain:
        if used[name] != 1:
            report.failures.append(f"{name}: used {used[name]} times by the graphs")
    if folded:
        report.failures.append(f"{len(folded)} folded conv+norm tensors never appeared")
    return report.print()


def _views(tensor: np.ndarray, shape: tuple[int, ...]) -> list[np.ndarray]:
    views = [tensor, tensor.T] if tensor.ndim == 2 else [tensor]
    return [view for view in views if view.shape == shape]


def main() -> int:
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file

    from stenograf.asr import parakeet, parakeet_onnx

    path = hf_hub_download(ORIGINAL, "model.safetensors", revision=ORIGINAL_REVISION)
    original = {
        name: tensor.astype(np.float32)
        for name, tensor in load_file(path).items()
        if tensor.dtype.kind == "f" and not name.startswith("vad_head.")
    }
    print(f"original: {ORIGINAL} @ {ORIGINAL_REVISION[:12]}, {len(original)} tensors")
    ok = check_mlx(original, Path(parakeet.model_dir()) / "model.safetensors")
    ok = check_onnx(original, parakeet_onnx.model_dir()) and ok
    print("the pinned files are the original's weights" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
