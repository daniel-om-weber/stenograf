"""Manifest in, ``.npz`` out: the contract every embedder runner shares.

Each runner supplies one function — a float32 16 kHz mono slice in [-1, 1) to
an unnormalised embedding, or None when the model cannot embed it — and this
module does the rest identically for all of them, so a scorer comparing models
compares the models and nothing else: the same slices (sample-exact, from the
same integer bounds), the same normalisation, the same NaN rows for spans a
model refuses.

Output ``.npz`` keys: ``emb`` float32 (total_spans × dim, rows L2-normalised,
NaN rows for refused spans), ``offsets`` int64 (n_items + 1), ``item_ids``,
``model``, ``seconds_cpu`` (wall-clock inside the embed calls only — on MPS it
is still wall-clock, the name is the interface's) and ``audio_seconds``.

Numpy and the standard library only: runners live in separate PEP 723
environments that share nothing else.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000

SHORT_SPAN_S = 2.0
"""A model that raises on a span shorter than this has met its minimum input
length — that span becomes a NaN row. A raise on a longer span is a bug and
propagates."""

EmbedFn = Callable[[np.ndarray], "np.ndarray | None"]


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    p.add_argument(
        "--threads",
        type=int,
        default=None,
        help="CPU threads for the forward pass (default: the runtime's own)",
    )
    return p


def pcm16(path: str | Path) -> np.ndarray:
    """A memory-mapped int16 view of a 16 kHz mono 16-bit WAV's samples.

    Mapped, not read: manifests point many items at the same hour-long
    channel, and only the spans are ever touched."""
    path = Path(path)
    with path.open("rb") as f:
        riff, _, wave_id = struct.unpack("<4sI4s", f.read(12))
        if riff != b"RIFF" or wave_id != b"WAVE":
            raise ValueError(f"{path}: not a RIFF/WAVE file")
        fmt = None
        while True:
            header = f.read(8)
            if len(header) < 8:
                raise ValueError(f"{path}: no data chunk")
            chunk_id, size = struct.unpack("<4sI", header)
            if chunk_id == b"fmt ":
                fmt = struct.unpack("<HHIIHH", f.read(16))
                f.seek(size - 16 + (size & 1), 1)
            elif chunk_id == b"data":
                offset = f.tell()
                break
            else:
                f.seek(size + (size & 1), 1)
    if fmt is None or fmt[0] != 1 or fmt[1] != 1 or fmt[2] != SAMPLE_RATE or fmt[5] != 16:
        raise ValueError(f"{path}: need 16 kHz mono 16-bit PCM, got fmt {fmt}")
    n = min(size, path.stat().st_size - offset) // 2
    return np.memmap(path, dtype="<i2", mode="r", offset=offset, shape=(n,))


def span_slice(samples: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    """float32 in [-1, 1) — torchaudio.load's and soundfile's scaling, so a
    reference implementation's own loader would hand its model the same
    numbers."""
    a = max(0, round(start_s * SAMPLE_RATE))
    b = min(len(samples), round(end_s * SAMPLE_RATE))
    return np.asarray(samples[a:b], dtype=np.float32) / 32768.0


def load_manifest(path: Path) -> list[dict]:
    items = json.loads(path.read_text())["items"]
    ids = [it["id"] for it in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate item ids")
    return items


def cached(out: Path, items: list[dict], model: str) -> bool:
    if not out.exists():
        return False
    with np.load(out, allow_pickle=False) as z:
        counts = np.diff(z["offsets"]).tolist()
        return (
            str(z["model"]) == model
            and list(z["item_ids"]) == [it["id"] for it in items]
            and counts == [len(it["spans"]) for it in items]
        )


def done(args: argparse.Namespace, model: str) -> bool:
    """True (and says so) when ``args.out`` is already this run's output —
    for runners to check before paying for a multi-GB model load."""
    if cached(args.out, load_manifest(args.manifest), model):
        print(f"[{model}] cached: {args.out}", file=sys.stderr)
        return True
    return False


def run(model: str, embed: EmbedFn, args: argparse.Namespace) -> Path:
    """Embed every span of ``args.manifest`` with ``embed`` into ``args.out``
    (skipped when ``args.out`` already holds this model's embeddings of the
    same items)."""
    items = load_manifest(args.manifest)
    if cached(args.out, items, model):
        print(f"[{model}] cached: {args.out}", file=sys.stderr)
        return args.out

    # Lazy loads, first-call graph builds and allocator growth land here,
    # outside the timed loop, so seconds_cpu measures steady-state inference.
    embed(np.random.default_rng(0).normal(0, 0.01, 3 * SAMPLE_RATE).astype(np.float32))

    rows: list[np.ndarray | None] = []
    offsets = [0]
    seconds = 0.0
    audio_seconds = 0.0
    refused = 0
    total = sum(len(it["spans"]) for it in items)
    t_report = time.monotonic()
    for item in items:
        samples = pcm16(item["wav"])
        for start_s, end_s in item["spans"]:
            audio = span_slice(samples, start_s, end_s)
            audio_seconds += len(audio) / SAMPLE_RATE
            t0 = time.perf_counter()
            try:
                vec = embed(audio) if len(audio) else None
            except Exception:
                if len(audio) >= SHORT_SPAN_S * SAMPLE_RATE:
                    raise
                vec = None
            seconds += time.perf_counter() - t0
            if vec is not None:
                vec = np.asarray(vec, dtype=np.float64).reshape(-1)
                norm = np.linalg.norm(vec)
                vec = (vec / norm).astype(np.float32) if norm > 0 and np.isfinite(norm) else None
            refused += vec is None
            rows.append(vec)
            if time.monotonic() - t_report > 30:
                t_report = time.monotonic()
                print(
                    f"[{model}] {len(rows)}/{total} spans, {seconds:.0f} s compute", file=sys.stderr
                )
        offsets.append(len(rows))

    dims = {len(r) for r in rows if r is not None}
    if len(dims) != 1:
        raise RuntimeError(f"[{model}] embedding dims {dims}: none embedded, or inconsistent")
    (dim,) = dims
    emb = np.full((len(rows), dim), np.nan, dtype=np.float32)
    for i, r in enumerate(rows):
        if r is not None:
            emb[i] = r

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp.npz")
    np.savez(
        tmp,
        emb=emb,
        offsets=np.asarray(offsets, dtype=np.int64),
        item_ids=np.asarray([it["id"] for it in items], dtype=str),
        model=np.asarray(model),
        seconds_cpu=np.asarray(seconds, dtype=np.float64),
        audio_seconds=np.asarray(audio_seconds, dtype=np.float64),
    )
    tmp.replace(args.out)
    rtf = seconds / audio_seconds if audio_seconds else float("nan")
    print(
        f"[{model}] {len(rows)} spans, dim {dim}, {refused} refused (NaN), "
        f"{seconds:.1f} s compute for {audio_seconds:.1f} s audio "
        f"({10 * rtf:.3f} s per 10 s) -> {args.out}",
        file=sys.stderr,
    )
    return args.out
