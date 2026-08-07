"""Persist each loop channel's segmentation + pair embeddings — compute once.

Everything upstream of clustering is invariant across partitioner arms
(segmentation, per-(chunk, speaker) embeddings) and everything downstream of
the turns is either cheap or reusable (the decoded words don't depend on the
diarization at all). Persisting the freeze makes a full clustering arm cost
seconds (``loop_arm.py``) instead of a ~40-minute matrix re-run; the
linkage-sweep round that recomputed and discarded exactly these arrays is the
measured argument (2026-08-07).

Writes ``out/diar/freeze/<id>.npz`` per multi-speaker channel: ``labels``
(chunks × FRAMES × 3 bool), ``pairs`` (m × 2 int64), ``vectors`` (m × d
float32), ``n`` (samples). Existing files are kept — delete the directory to
re-freeze (e.g. after a segmentation-config change; a stride arm needs its own
freeze directory).

Run::

    uv run --group eval eval/loop_freeze.py
"""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np
from common import OUT_DIR, read_pcm16

from stenograf.audio import to_float32
from stenograf.diarization.loop import OwnDiarizer

FREEZE_DIR = OUT_DIR / "diar" / "freeze"


def freeze_sha(channel_id: str) -> str:
    return hashlib.sha256((FREEZE_DIR / f"{channel_id}.npz").read_bytes()).hexdigest()[:16]


def load_emb_cache(partitioner: str, channel_id: str) -> dict[str, np.ndarray] | None:
    """Cached raw-k+1 cluster embeddings, or None if absent or stale.

    Stamped with the freeze's sha: a cache written against different
    segmentation/embedding artifacts must recompute, never be silently reused
    (2026-08-07 review). Values carry ~1e-8 ONNX extraction jitter relative to
    an in-process compute — semantically nil, but the reason cache-path
    emb.json bytes differ from inline-path ones."""
    path = FREEZE_DIR / f"emb-{partitioner}" / f"{channel_id}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if data.get("freeze") != freeze_sha(channel_id):
        return None
    return {k: np.array(v, dtype=np.float32) for k, v in data["embeddings"].items()}


def save_emb_cache(partitioner: str, channel_id: str, embeddings: dict[str, np.ndarray]) -> None:
    path = FREEZE_DIR / f"emb-{partitioner}" / f"{channel_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "freeze": freeze_sha(channel_id),
                "embeddings": {k: [float(x) for x in v] for k, v in embeddings.items()},
            }
        )
    )


def main() -> int:
    import ami

    channels = [c for c in ami.load_channels(include_duo=True) if c.num_speakers > 1]
    if not channels:
        raise SystemExit("no corpus channels — run `eval/ami.py fetch` first")
    FREEZE_DIR.mkdir(parents=True, exist_ok=True)

    diarizer = OwnDiarizer()
    for channel in channels:
        out = FREEZE_DIR / f"{channel.id}.npz"
        if out.exists():
            print(f"[{channel.id}] exists, kept")
            continue
        started = time.monotonic()
        audio = to_float32(read_pcm16(channel.wav_path))
        labels = diarizer._chunk_labels(audio)
        pairs, vectors = diarizer._pair_embeddings(audio, labels)
        np.savez_compressed(
            out,
            labels=np.stack(labels),
            pairs=np.array(pairs, dtype=np.int64),
            vectors=np.stack(vectors).astype(np.float32),
            n=np.int64(len(audio)),
        )
        print(
            f"[{channel.id}] {len(labels)} chunks, {len(pairs)} embeddings, "
            f"{time.monotonic() - started:.0f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
