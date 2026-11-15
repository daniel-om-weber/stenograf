"""Does ``pipeline.collapse_single_voice`` earn its place behind Nemotron 3?

The rule folds an estimated-count channel to one speaker when every pair of its
cluster voices is at least ``COLLAPSE_SIMILARITY`` alike. It was measured on
the owned estimators, which split every monologue 2–3 ways; Nemotron with no
count over-counts far less. Kept only if it fixes Nemotron's single-speaker
over-counts without merging a real pair — so it runs on Nemotron's uncapped
turns (the no-count path) over three sets, from probabilities already on disk:

- ``heldout`` — AMI's held-out meetings, the library port's probabilities
  (``heldout.py run --arm library``): 12 single-speaker mic channels, 24 with
  two or three people;
- ``calls`` — the user's 17 one-to-one calls (``owncalls.py``), two speakers;
- ``pairs`` — 22 spliced dialogues of two remote voices, the hardest pairs.

Only the voice embeddings are computed (cached per channel under
``<set>/collapse_embeddings/``, so a rerun only scores)::

    uv run --group eval eval/nemotron_collapse.py [--sets heldout calls pairs]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import ami
import heldout
import numpy as np
from common import read_pcm16
from rttm import parse_rttm

from stenograf.diarization.base import SpeakerTurn
from stenograf.diarization.embedding import VoiceEmbedder
from stenograf.diarization.nemotron import turns_from_probs
from stenograf.pipeline import collapse_single_voice

OWN = Path(__file__).resolve().parent / "out" / "owncalls"
SETS = {
    # name: (root holding channels.json, probabilities, audio, reference directory)
    "heldout": (heldout.OUT_DIR, "probs_nemotron_library", ami.CHANNELS_DIR, ami.AMI_REFS_DIR),
    "calls": (OWN, "probs_nemotron_fp32_webgpu", OWN / "audio", OWN),
    "pairs": (OWN / "pairs", "probs_nemotron_fp16_cpu", OWN / "pairs" / "audio", OWN / "pairs"),
}


def _dicts(turns: list[SpeakerTurn]) -> list[dict]:
    return [{"speaker": t.speaker, "start": t.start, "end": t.end} for t in turns]


def _embeddings(cache: Path, turns, pcm, embedder: VoiceEmbedder) -> dict[str, np.ndarray]:
    if cache.exists():
        with np.load(cache) as stored:
            return {k: stored[k] for k in stored.files}
    vectors = embedder.cluster_embeddings(turns, pcm)
    np.savez(cache, **vectors)
    return vectors


def run_set(name: str, embedder: VoiceEmbedder) -> list[dict]:
    root, probs_name, audio_dir, refs = SETS[name]
    cache_dir = root / "collapse_embeddings"
    cache_dir.mkdir(exist_ok=True)
    rows = []
    for channel in json.loads((root / "channels.json").read_text()):
        cid, k = channel["id"], channel["num_speakers"]
        probs_path = root / probs_name / f"{cid}.npy"
        if not probs_path.exists():
            continue
        turns = turns_from_probs(np.load(probs_path).astype(np.float32))
        started = time.monotonic()
        pcm = read_pcm16(audio_dir / f"{cid}.wav")
        vectors = _embeddings(cache_dir / f"{cid}.npz", turns, pcm, embedder)
        secs = time.monotonic() - started
        collapsed, _ = collapse_single_voice(turns, vectors)
        labels = sorted({t.speaker for t in turns})
        pairs = [float(vectors[a] @ vectors[b]) for a, b in combinations(labels, 2)
                 if a in vectors and b in vectors]
        ref = parse_rttm(refs / f"{cid}.rttm")
        before, after = _dicts(turns), _dicts(collapsed)
        row = {
            "set": name,
            "id": cid,
            "k": k,
            "found": len(labels),
            "after": len({t.speaker for t in collapsed}),
            "min_cos": min(pairs) if pairs else None,
            "max_cos": max(pairs) if pairs else None,
            "embedded": f"{len(vectors)}/{len(labels)}",
            "attr_before": heldout.attribution(ref, before),
            "attr_after": heldout.attribution(ref, after),
        }
        if name == "heldout":
            row["explained"] = heldout.explained_extras(cid, ref, before)
        rows.append(row)
        print(
            f"{name:8s} {cid:28s} k={k} found={row['found']} → {row['after']}  "
            f"min/max cos={_fmt(row['min_cos'])}/{_fmt(row['max_cos'])}  "
            f"attr {100 * row['attr_before']:5.1f} → {100 * row['attr_after']:5.1f}"
            + (f"  expl={row['explained']}" if "explained" in row else "")
            + f"  ({secs:.0f} s)",
            flush=True,
        )
    return rows


def _fmt(x: float | None) -> str:
    return "  —  " if x is None else f"{x:5.3f}"


def summarize(rows: list[dict]) -> None:
    print()
    for name in SETS:
        rs = [r for r in rows if r["set"] == name]
        if not rs:
            continue
        for label, group in (("one speaker", [r for r in rs if r["k"] == 1]),
                             ("several", [r for r in rs if r["k"] > 1])):
            if not group:
                continue
            over = [r for r in group if r["found"] > r["k"]]
            fired = [r for r in group if r["after"] < r["found"]]
            fixed = [r for r in fired if r["after"] == r["k"]]
            wrong = [r for r in fired if r["after"] < r["k"]]
            attr_b = 100 * np.mean([r["attr_before"] for r in group])
            attr_a = 100 * np.mean([r["attr_after"] for r in group])
            multi = [r["min_cos"] for r in group if r["min_cos"] is not None]
            print(
                f"{name:8s} {label:12s} n={len(group):2d}  over-counted {len(over):2d}  "
                f"collapse fired {len(fired):2d} (to the true count {len(fixed)}, "
                f"below it {len(wrong)})  attr {attr_b:5.1f} → {attr_a:5.1f} %  "
                f"highest min pair cos {max(multi) if multi else float('nan'):.3f}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sets", nargs="+", choices=list(SETS), default=list(SETS))
    args = parser.parse_args()
    embedder = VoiceEmbedder()
    rows = [r for name in args.sets for r in run_set(name, embedder)]
    summarize(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
