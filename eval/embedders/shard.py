"""Split a manifest into shards and merge the shards' outputs back.

A runner is one process at one thread per forward pass; on a many-core CPU the
throughput lever is processes, not threads (the same finding as the shipped
embedder's pool, `stenograf.diarization.embedding._pool_workers`). MPS is no
shortcut here: every span is a new input length, and the per-shape cost made
it ~10× slower than its fixed-shape benchmark (2026-11-18, eres2net-large,
0.53 vs 0.05 s per 10 s of audio).

    python3 eval/embedders/shard.py split M.json N        # → M.shard{i}.json
    python3 eval/embedders/shard.py merge M.json OUT.npz OUT.shard*.npz
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def split(manifest: Path, n: int) -> None:
    items = json.loads(manifest.read_text())["items"]
    weight = np.array([sum(e - s for s, e in i["spans"]) + 0.05 * len(i["spans"]) for i in items])
    edges = np.searchsorted(np.cumsum(weight), np.linspace(0, weight.sum(), n + 1)[1:-1])
    for k, part in enumerate(np.split(np.arange(len(items)), edges)):
        path = manifest.with_name(f"{manifest.stem}.shard{k}.json")
        path.write_text(json.dumps({"items": [items[i] for i in part]}))
        print(path, len(part))


def merge(manifest: Path, out: Path, shards: list[Path]) -> None:
    order = [i["id"] for i in json.loads(manifest.read_text())["items"]]
    rows: dict[str, np.ndarray] = {}
    model, seconds, audio = None, 0.0, 0.0
    for path in shards:
        d = np.load(path, allow_pickle=False)
        if model is not None and str(d["model"]) != model:
            raise SystemExit(f"{path}: model {d['model']} != {model}")
        model = str(d["model"])
        seconds += float(d["seconds_cpu"])
        audio += float(d["audio_seconds"])
        # An .npz member is decompressed again on every access.
        emb, offsets = d["emb"], d["offsets"]
        for k, iid in enumerate(d["item_ids"]):
            rows[str(iid)] = emb[offsets[k] : offsets[k + 1]]
    missing = [i for i in order if i not in rows]
    if missing:
        raise SystemExit(f"{len(missing)} items missing, first {missing[0]}")
    parts = [rows[i] for i in order]
    bounds = np.concatenate([[0], np.cumsum([len(p) for p in parts])]).astype(np.int64)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(
        tmp,
        emb=np.concatenate(parts).astype(np.float32),
        offsets=bounds,
        item_ids=np.array(order),
        model=np.array(model),
        seconds_cpu=np.array(seconds),
        audio_seconds=np.array(audio),
    )
    tmp.replace(out)
    print(f"{out}: {len(order)} items from {len(shards)} shards")


if __name__ == "__main__":
    if sys.argv[1] == "split":
        split(Path(sys.argv[2]), int(sys.argv[3]))
    else:
        merge(Path(sys.argv[2]), Path(sys.argv[3]), [Path(p) for p in sys.argv[4:]])
