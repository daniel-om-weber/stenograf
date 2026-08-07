"""Partitioner × fold-rule matrix on the frozen loop — the fold is not neutral.

The 2026-08-07 adversarial review of the ward-vs-complete verdict found the
shipped fold (spare by duration, ``pipeline.fold_excess_clusters``) is
co-tuned to complete-linkage clusters: under ward the k+1 spare can be half a
dominant talker, the duration rule folds the wrong cluster, and an
oracle-best single merge showed ~2.6 pts of ward's partition quality being
discarded (mean 12.4% vs 15.0%). The 2026-08-06 decline of the max-pair rule
was measured against complete's clusters only and does not transfer. So the
partitioner and the fold rule are one decision, measured here as a matrix.

Rules (each applied iteratively while clusters exceed the known count, with
production's merge bookkeeping — keep the longer side's label, embeddings
merge to the duration-weighted mean):

- ``duration`` — the pre-gate production rule: spare by duration, always.
- ``maxpair``  — the pre-c78b856 rule: merge the globally most-similar pair.
- ``samevoice`` — merge the most-similar pair only if its cosine clears the
  gate (default ``pipeline.FOLD_PAIR_SIMILARITY``; the shipped rule), else
  fall back to the duration spare. The first candidate gated at
  ``COLLAPSE_SIMILARITY`` (0.6) admitted 4 cross-speaker merges the gate
  audit caught (``fold_gate_audit.py``); 0.8 sits in the measured empty band.
- ``oracle``   — best single merge by DER. A bound, not shippable.

Per-partitioner raw-k+1 cluster embeddings are cached under
``out/diar/freeze/emb-<partitioner>/`` (the one nontrivial cost, ~10 min per
partitioner first run; every fold rule afterwards scores in seconds).

Run::

    uv run --group eval eval/fold_sweep.py
"""

from __future__ import annotations

import time
from dataclasses import replace
from itertools import combinations

import numpy as np
from common import OUT_DIR, read_pcm16
from der import score_der
from loop_freeze import FREEZE_DIR, load_emb_cache, save_emb_cache
from rttm import Turn, parse_rttm

from stenograf.diarization.base import SpeakerTurn
from stenograf.diarization.loop import OwnDiarizer, _cluster
from stenograf.diarization.sherpa import cluster_embeddings
from stenograf.pipeline import FOLD_PAIR_SIMILARITY
from stenograf.voiceprints import l2_normalize

PARTITIONERS = ("complete", "ward", "nmesc")
RULES = ("duration", "maxpair", "samevoice", "oracle")


def _merge(
    turns: list[SpeakerTurn],
    embeddings: dict[str, np.ndarray],
    keep: str,
    fold: str,
    durations: dict[str, float],
) -> tuple[list[SpeakerTurn], dict[str, np.ndarray]]:
    turns = [replace(t, speaker=keep) if t.speaker == fold else t for t in turns]
    merged = embeddings[keep] * durations[keep] + embeddings[fold] * durations[fold]
    embeddings = dict(embeddings)
    embeddings[keep] = l2_normalize(merged)
    del embeddings[fold]
    return turns, embeddings


def _durations(turns: list[SpeakerTurn]) -> dict[str, float]:
    durations: dict[str, float] = {}
    for t in turns:
        durations[t.speaker] = durations.get(t.speaker, 0.0) + (t.end - t.start)
    return durations


def _der(ref: list[Turn], turns: list[SpeakerTurn]) -> float:
    return score_der(ref, [Turn(t.speaker, t.start, t.end) for t in turns]).der


def fold_with_rule(
    rule: str,
    turns: list[SpeakerTurn],
    embeddings: dict[str, np.ndarray],
    num_speakers: int,
    ref: list[Turn],
    gate: float = FOLD_PAIR_SIMILARITY,
) -> tuple[list[SpeakerTurn], dict[str, np.ndarray]]:
    """Folded turns AND embeddings — merged with production's bookkeeping
    (duration-weighted mean), so downstream naming material is what a shipped
    version of the rule would produce. ``ref`` is only read by ``oracle``;
    ``gate`` is samevoice's pair threshold (the gate audit measured
    cross-speaker fold pairs ≤0.697 and needed merges ≥0.846 — the shipped
    default sits in that band). ``duration`` is implemented here rather than
    calling production, which now IS gated samevoice."""
    if rule == "duration":
        rule, gate = "samevoice", float("inf")  # a gate that never fires
    turns = list(turns)
    embeddings = dict(embeddings)
    while len({t.speaker for t in turns}) > num_speakers:
        durations = _durations(turns)
        embedded = [c for c in durations if c in embeddings]
        if len(embedded) < 2:
            return turns, embeddings
        if rule == "oracle":
            best = None
            for a, b in combinations(embedded, 2):
                keep, fold = (a, b) if durations[a] >= durations[b] else (b, a)
                candidate, _ = _merge(turns, embeddings, keep, fold, durations)
                der = _der(ref, candidate)
                if best is None or der < best[0]:
                    best = (der, keep, fold)
            assert best is not None
            _, keep, fold = best
        else:
            sim, a, b = max(
                (float(embeddings[x] @ embeddings[y]), x, y)
                for x, y in combinations(embedded, 2)
            )
            if rule == "samevoice" and sim < gate:
                fold_c = min(embedded, key=lambda c: durations[c])
                a = max(
                    (c for c in embedded if c != fold_c),
                    key=lambda c: float(embeddings[c] @ embeddings[fold_c]),
                )
                b = fold_c
            keep, fold = (a, b) if durations[a] >= durations[b] else (b, a)
        turns, embeddings = _merge(turns, embeddings, keep, fold, durations)
    return turns, embeddings


def main() -> int:
    import ami

    channels = [c for c in ami.load_channels() if c.num_speakers > 1]
    diarizer = OwnDiarizer()
    results: dict[str, dict[str, dict[str, float]]] = {p: {} for p in PARTITIONERS}

    for partitioner in PARTITIONERS:
        for channel in channels:
            started = time.monotonic()
            data = np.load(FREEZE_DIR / f"{channel.id}.npz")
            pairs = [(int(a), int(b)) for a, b in data["pairs"]]
            clusters = _cluster(
                data["vectors"], 0.5, channel.num_speakers + 1, partitioner
            )
            turns = diarizer._assemble(int(data["n"]), list(data["labels"]), pairs, clusters)

            embeddings = load_emb_cache(partitioner, channel.id)
            if embeddings is None:
                pcm = read_pcm16(channel.wav_path)
                embeddings = cluster_embeddings(turns, pcm, diarizer.embed)
                save_emb_cache(partitioner, channel.id, embeddings)

            ref = parse_rttm(channel.ref_path)
            row = {}
            for rule in RULES:
                folded, _ = fold_with_rule(rule, turns, embeddings, channel.num_speakers, ref)
                row[rule] = _der(ref, folded)
            results[partitioner][channel.id] = row
            print(
                f"[{partitioner}/{channel.id}] "
                + " ".join(f"{r}={row[r]:.1%}" for r in RULES)
                + f" ({time.monotonic() - started:.0f}s)"
            )

    lines = ["## Partitioner × fold-rule DER matrix (fold_sweep.py)", ""]
    lines.append("| partitioner | " + " | ".join(RULES) + " |")
    lines.append("|---|" + "---|" * len(RULES))
    for partitioner in PARTITIONERS:
        rows = results[partitioner]
        means = {r: float(np.mean([rows[c][r] for c in rows])) for r in RULES}
        lines.append(
            f"| {partitioner} | " + " | ".join(f"{means[r]:.1%}" for r in RULES) + " |"
        )
    for partitioner in PARTITIONERS:
        lines += ["", f"### {partitioner}, per channel", ""]
        lines.append("| channel | " + " | ".join(RULES) + " |")
        lines.append("|---|" + "---|" * len(RULES))
        for cid, row in results[partitioner].items():
            lines.append(
                f"| {cid} | " + " | ".join(f"{row[r]:.1%}" for r in RULES) + " |"
            )
    text = "\n".join(lines)
    print()
    print(text)
    (OUT_DIR / "diar-fold-sweep.md").write_text(text + "\n")
    print(f"wrote {OUT_DIR / 'diar-fold-sweep.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
