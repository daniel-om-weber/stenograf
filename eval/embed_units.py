"""Clustering-unit kill-tests: fewer/longer embedding units, before any matrix.

`PLAN-DIARIZATION-SPEED.md` step 5 buys a full harness matrix (plus
`threshold_pick.py` and `auto_update.py`) only for an arm that survives a
one-channel kill-test. Both arms here consume `loop_freeze.py`'s artifacts —
segmentation labels and per-(chunk, speaker) embeddings are the frozen
invariants; what changes is which vectors go INTO ward and how the rest get
their cluster:

- **L2** (``--arm l2 --nth N``): cluster only every Nth chunk's pair vectors
  (frozen, so values are bit-identical to production's); propagate cluster
  labels to the skipped pairs by global-frame overlap, cosine-to-centroid
  where a pair overlaps nothing. The label grid, vote grid and ``starts``
  are byte-identical to production by construction — the mechanism the
  declined stride bullet names is not engaged. Embed-stage cost at N: ~1/N.

- **L3** (``--arm l3``): embed each maximal global sole-speaker interval
  once (speakers-per-frame == 1 runs on the RF grid, ≥0.5 s), ward those
  interval vectors, then assign every frozen pair to the interval-cluster
  it overlaps most (cosine-to-centroid fallback). Embed-stage cost: one
  pass over exclusive speech (~10× less audio than today's overlapping
  windows).

Reported per channel: DER vs the corpus reference, the shipped ward arm's
DER (`ami-loop-ward-sv08`), cluster count after the production fold, and the
minimum cosine between each folded cluster's naming embedding and its
turn-overlap-matched baseline cluster — the number the 0.56 naming
calibration cares about.

Run::

    uv run --group eval eval/embed_units.py --arm l2 --nth 3
    uv run --group eval eval/embed_units.py --arm l3 --segments ES2003c.loop
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
from common import OUT_DIR, read_pcm16
from der import score_der
from loop_freeze import freeze_dir
from rttm import Turn, parse_rttm

from stenograf.audio import to_float32
from stenograf.diarization.loop import FRAMES, RF_SHIFT, OwnDiarizer, _cluster, _runs
from stenograf.diarization.sherpa import cluster_embeddings
from stenograf.pipeline import fold_excess_clusters

BASELINE = "ami-loop-ward-sv08"
MIN_INTERVAL_S = 0.5


def _pair_frames(labels: list[np.ndarray], pair: tuple[int, int], shift: int) -> np.ndarray:
    """A pair's clean active frames on the global RF grid (production's
    overlap-zeroing, then offset by the chunk's start frame)."""
    chunk_index, speaker = pair
    chunk = labels[chunk_index]
    clean = chunk.copy()
    clean[np.sum(chunk, axis=1) >= 2] = False
    start = int(chunk_index * shift / RF_SHIFT + 0.5)
    return np.flatnonzero(clean[:, speaker]) + start


def _speakers_per_frame(labels: list[np.ndarray], shift: int) -> np.ndarray:
    num_frames = (160_000 + (len(labels) - 1) * shift) // RF_SHIFT + 1
    count = np.zeros(num_frames)
    weight = np.zeros(num_frames)
    for i, chunk in enumerate(labels):
        start = int(i * shift / RF_SHIFT + 0.5)
        count[start : start + FRAMES] += np.sum(chunk, axis=1)
        weight[start : start + FRAMES] += 1.0
    return (count / (weight + 1e-12) + 0.5).astype(np.int64)


def _propagate(
    all_pairs: list[tuple[int, int]],
    labels: list[np.ndarray],
    shift: int,
    seed_pairs: dict[tuple[int, int], int],
    vectors: dict[tuple[int, int], np.ndarray],
    num_clusters: int,
) -> np.ndarray:
    """Cluster id per pair: seeds keep theirs, the rest vote by global-frame
    overlap with seed frames, cosine-to-centroid when nothing overlaps."""
    frames_of = {p: _pair_frames(labels, p, shift) for p in all_pairs}
    cluster_frames: list[set[int]] = [set() for _ in range(num_clusters)]
    for pair, cluster in seed_pairs.items():
        cluster_frames[cluster].update(frames_of[pair].tolist())
    centroids = np.zeros((num_clusters, next(iter(vectors.values())).shape[0]), dtype=np.float64)
    counts = np.zeros(num_clusters)
    for pair, cluster in seed_pairs.items():
        centroids[cluster] += vectors[pair]
        counts[cluster] += 1
    centroids /= np.maximum(counts, 1)[:, None]

    assigned = np.zeros(len(all_pairs), dtype=np.int64)
    for i, pair in enumerate(all_pairs):
        if pair in seed_pairs:
            assigned[i] = seed_pairs[pair]
            continue
        frames = set(frames_of[pair].tolist())
        overlaps = [len(cluster_frames[c] & frames) for c in range(num_clusters)]
        if max(overlaps) > 0:
            assigned[i] = int(np.argmax(overlaps))
        else:
            # gated-finite inputs; macOS Accelerate raises spurious FP flags
            # on clean GEMMs (see loop.py::_cluster)
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                sims = centroids @ vectors[pair]
            assigned[i] = int(np.argmax(sims))
    return assigned


def run_channel(channel, arm: str, nth: int, diarizer: OwnDiarizer) -> dict:
    root = freeze_dir(1.0)
    data = np.load(root / f"{channel.id}.npz")
    labels = list(data["labels"])
    pairs = [(int(a), int(b)) for a, b in data["pairs"]]
    vectors = {p: v.astype(np.float32) for p, v in zip(pairs, data["vectors"], strict=True)}
    n = int(data["n"])
    shift = 16_000
    k = channel.num_speakers
    audio = to_float32(read_pcm16(channel.wav_path))

    embedded_seconds = 0.0
    if arm == "l2":
        seeds = [p for p in pairs if p[0] % nth == 0]
        seed_matrix = np.stack([vectors[p] for p in seeds])
        clusters = _cluster(seed_matrix, 0.5, k + 1, "ward")
        seed_map = {p: int(c) for p, c in zip(seeds, clusters, strict=True)}
        num_clusters = int(clusters.max()) + 1
        # cost model: the skipped pairs' audio is what L2 saves — report it
        seed_frames = sum(len(_pair_frames(labels, p, shift)) for p in seeds)
        embedded_seconds = seed_frames * RF_SHIFT / 16_000
    else:  # l3
        spf = _speakers_per_frame(labels, shift)
        intervals = [
            (a, b)
            for a, b in _runs(spf == 1)
            if (b - a) * RF_SHIFT / 16_000 >= MIN_INTERVAL_S
        ]
        spans = [(a, b, a * RF_SHIFT, min(b * RF_SHIFT, n)) for a, b in intervals]
        spans = [(a, b, lo, hi) for a, b, lo, hi in spans if hi > lo]
        pooled = diarizer._executor().map(diarizer.embed, [audio[lo:hi] for _, _, lo, hi in spans])
        unit_vectors = []
        kept = []
        for (a, b, lo, hi), vec in zip(spans, pooled, strict=True):
            if vec is not None and np.all(np.isfinite(vec)) and np.any(vec):
                unit_vectors.append(vec)
                kept.append((a, b))
                embedded_seconds += (hi - lo) / 16_000
        clusters = _cluster(np.stack(unit_vectors), 0.5, k + 1, "ward")
        num_clusters = int(clusters.max()) + 1
        interval_frames: list[tuple[set[int], int]] = [
            (set(range(a, b)), int(c)) for (a, b), c in zip(kept, clusters, strict=True)
        ]
        seed_map = {}
        # seed each pair directly from its best-overlapping interval
        for p in pairs:
            frames = set(_pair_frames(labels, p, shift).tolist())
            best, best_overlap = -1, 0
            for iframes, cluster in interval_frames:
                overlap = len(iframes & frames)
                if overlap > best_overlap:
                    best, best_overlap = cluster, overlap
            if best >= 0:
                seed_map[p] = best

    assigned = _propagate(pairs, labels, shift, seed_map, vectors, num_clusters)
    turns = diarizer._assemble(n, labels, pairs, assigned)
    embeddings = cluster_embeddings(turns, audio, diarizer.embed, pool=diarizer._executor())
    folded_turns, folded_emb = fold_excess_clusters(turns, embeddings, k)

    ref = parse_rttm(
        OUT_DIR.parent / "refs" / "ami" / f"{channel.id}.rttm"
    )
    hyp = [Turn(t.speaker, t.start, t.end) for t in folded_turns]
    der = score_der(ref, hyp).der

    base_dir = OUT_DIR / "diar" / BASELINE
    base_turns = parse_rttm(base_dir / f"{channel.id}.rttm")
    base_der = score_der(ref, base_turns).der
    base_emb = {
        key: np.asarray(v, dtype=np.float32)
        for key, v in json.loads((base_dir / f"{channel.id}.emb.json").read_text()).items()
    }

    # map folded clusters -> baseline clusters by turn overlap, compare naming cosines
    def spans(turn_list):
        by = {}
        for t in turn_list:
            by.setdefault(t.speaker, []).append((t.start, t.end))
        return by

    hyp_spans, base_spans = spans(hyp), spans(base_turns)

    def overlap(a, b):
        return sum(
            max(0.0, min(e1, e2) - max(s1, s2)) for s1, e1 in a for s2, e2 in b
        )

    cosines = []
    for label, vec in folded_emb.items():
        if label not in hyp_spans:
            continue
        best = max(base_spans, key=lambda bl: overlap(hyp_spans[label], base_spans[bl]))
        if best in base_emb:
            cosines.append(float(vec @ base_emb[best]))
    return {
        "id": channel.id,
        "der": der,
        "base_der": base_der,
        "clusters": len({t.speaker for t in folded_turns}),
        "k": k,
        "min_naming_cos": min(cosines) if cosines else float("nan"),
        "embedded_seconds": embedded_seconds,
    }


def main() -> int:
    import ami

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("l2", "l3"))
    parser.add_argument("--nth", type=int, default=3)
    parser.add_argument("--segments", default="ES2003c.loop")
    args = parser.parse_args()

    wanted = set(args.segments.split(","))
    channels = [c for c in ami.load_channels() if c.id in wanted]
    if not channels:
        raise SystemExit("no matching corpus channels")
    diarizer = OwnDiarizer()
    for channel in channels:
        started = time.monotonic()
        row = run_channel(channel, args.arm, args.nth, diarizer)
        print(
            f"[{row['id']}] {args.arm}: DER {row['der']:.3f} (baseline {row['base_der']:.3f}), "
            f"clusters {row['clusters']}/k={row['k']}, min naming cos {row['min_naming_cos']:.4f}, "
            f"embedded {row['embedded_seconds']:.0f} s, {time.monotonic() - started:.0f} s"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
