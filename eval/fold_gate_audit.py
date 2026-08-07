"""Price the samevoice gate as a gate, and close the k=2 evidence hole.

The 2026-08-07 re-review's two blockers against shipping samevoice into
``fold_excess_clusters``: (a) the gate's admissions were never classified —
its n=16 audit found 3 of ward's 14 firings merged CROSS-speaker (two of them
stranger-into-enrolled, the profile-poisoning path); (b) k=2 rooms — the one
place a similarity-gated fold can delete a participant — had zero harness
evidence. This measures both on the frozen channels, duos included:

For every fold step (k+1 → k) under complete and ward, log the most-similar
pair's cosine and whether the pair shares a dominant reference speaker; then
which merge each rule performs (duration spare vs gated pair) and whether THAT
merge is same-speaker. The published table is the same/cross similarity
distribution — whether a band separates them is the whole question — plus the
duo matrix: per k=2 channel, DER under both rules and whether the two real
speakers' clusters ever cleared the gate.

Run (after ``loop_freeze.py`` over the duo channels)::

    uv run --group eval eval/fold_gate_audit.py
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from common import OUT_DIR, read_pcm16
from der import score_der
from fold_sweep import _durations, _merge
from loop_freeze import FREEZE_DIR, load_emb_cache, save_emb_cache
from rttm import Turn, parse_rttm

from stenograf.diarization.loop import OwnDiarizer, _cluster
from stenograf.diarization.sherpa import cluster_embeddings
from stenograf.pipeline import FOLD_PAIR_SIMILARITY

PARTITIONERS = ("complete", "ward")


def _dominant(turns, ref):
    import ami

    return ami.dominant_speaker([Turn(t.speaker, t.start, t.end) for t in turns], ref)


def main() -> int:
    import ami

    channels = [c for c in ami.load_channels(include_duo=True) if c.num_speakers > 1]
    diarizer = OwnDiarizer()
    firings: list[str] = []
    sims: dict[str, list[float]] = {"same": [], "cross": []}
    duo_rows: list[str] = []

    for channel in channels:
        data = np.load(FREEZE_DIR / f"{channel.id}.npz")
        pairs = [(int(a), int(b)) for a, b in data["pairs"]]
        ref = parse_rttm(channel.ref_path)
        duo = channel.id.endswith(".duo")
        duo_cells: dict[str, str] = {}
        for partitioner in PARTITIONERS:
            clusters = _cluster(
                data["vectors"], 0.5, channel.num_speakers + 1, partitioner
            )
            turns = diarizer._assemble(int(data["n"]), list(data["labels"]), pairs, clusters)

            embeddings = load_emb_cache(partitioner, channel.id)
            if embeddings is None:
                embeddings = cluster_embeddings(
                    turns, read_pcm16(channel.wav_path), diarizer.embed
                )
                save_emb_cache(partitioner, channel.id, embeddings)

            by_rule = {}
            for rule in ("duration", "samevoice"):
                r_turns, r_emb = list(turns), dict(embeddings)
                while len({t.speaker for t in r_turns}) > channel.num_speakers:
                    durations = _durations(r_turns)
                    embedded = [c for c in durations if c in r_emb]
                    if len(embedded) < 2:
                        break
                    sim, a, b = max(
                        (float(r_emb[x] @ r_emb[y]), x, y)
                        for x, y in combinations(embedded, 2)
                    )
                    who = {
                        c: _dominant([t for t in r_turns if t.speaker == c], ref)
                        for c in embedded
                    }
                    pair_kind = "same" if who[a] == who[b] and who[a] else "cross"
                    fired = sim >= FOLD_PAIR_SIMILARITY
                    if rule == "samevoice" and fired:
                        keep, fold = (a, b) if durations[a] >= durations[b] else (b, a)
                    else:
                        fold = min(embedded, key=lambda c: durations[c])
                        keep = max(
                            (c for c in embedded if c != fold),
                            key=lambda c: float(r_emb[c] @ r_emb[fold]),
                        )
                        if durations[keep] < durations[fold]:
                            keep, fold = fold, keep
                    merge_kind = "same" if who[keep] == who[fold] and who[keep] else "cross"
                    if rule == "samevoice":
                        sims[pair_kind].append(sim)
                        firings.append(
                            f"| {channel.id} | {partitioner} | {sim:.3f} | {pair_kind} "
                            f"| {'pair' if fired else 'duration'} | {merge_kind} "
                            f"| {min(durations[keep], durations[fold]) / 60:.1f}m |"
                        )
                    r_turns, r_emb = _merge(r_turns, r_emb, keep, fold, durations)
                by_rule[rule] = score_der(
                    ref, [Turn(t.speaker, t.start, t.end) for t in r_turns]
                ).der
            if duo:
                duo_cells[partitioner] = (
                    f"{by_rule['duration']:.1%} / {by_rule['samevoice']:.1%}"
                )
        if duo:
            duo_rows.append(
                f"| {channel.id} | " + " | ".join(duo_cells[p] for p in PARTITIONERS) + " |"
            )

    lines = [
        "## samevoice gate audit: every fold step, same vs cross speaker "
        "(fold_gate_audit.py)",
        "",
        "Max-pair cosine at each fold step (samevoice's decision input):",
        "",
        f"- same-speaker pairs: n={len(sims['same'])}, "
        + (
            f"min {min(sims['same']):.3f}, max {max(sims['same']):.3f}"
            if sims["same"]
            else "none"
        ),
        f"- cross-speaker pairs: n={len(sims['cross'])}, "
        + (
            f"min {min(sims['cross']):.3f}, max {max(sims['cross']):.3f}"
            if sims["cross"]
            else "none"
        ),
        f"- gate = FOLD_PAIR_SIMILARITY = {FOLD_PAIR_SIMILARITY}",
        "",
        "| channel | partitioner | max-pair sim | pair | branch | merged | smaller side |",
        "|---|---|---|---|---|---|---|",
        *firings,
        "",
        "### k=2 duo channels: DER duration / samevoice",
        "",
        "| channel | " + " | ".join(PARTITIONERS) + " |",
        "|---|" + "---|" * len(PARTITIONERS),
        *duo_rows,
    ]
    text = "\n".join(lines)
    print(text)
    (OUT_DIR / "diar-fold-gate-audit.md").write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
