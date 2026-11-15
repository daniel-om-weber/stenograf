"""Re-ID on AMI's held-out meetings: does 0.56 still name the right people when
the turns come from a different diarizer?

The threshold was picked on the corpus harness's owned-loop clusters
(`threshold_pick.py`), and a diarizer change moves what a "cluster" is — its
purity, its length, how much crosstalk it carries. This scores the naming stage
on the held-out channels `heldout.py fetch` built, for every arm whose turns are
in ``eval/out/heldout/``, with the same open-set convention as the harness
(`reid_score.py`): galleries enroll from session **a**, the alphabetically-last
participant per meeting stays unenrolled (a stranger by construction), and every
cluster of sessions b–d is a trial against its own meeting's gallery.

Two enrollment sources, as `rename_once.py` defines them:

- ``headset`` — reference spans sliced from each participant's raw session-a
  headset. Identical for every arm, so differences between arms come from the
  trial clusters alone.
- ``clusters`` — what ``steno profiles assign`` would store: the arm's own
  session-a clusters (mic + loop channels) under their reference-majority name,
  multi-entry profiles scored by mean cosine. A participant the arm fused away
  in session a has nothing to assign and is a stranger in later trials.

Every arm's cluster embedding is recomputed from its final turns with
production's ``cluster_embeddings`` (overlap-cleaned, ERes2Net through
sherpa-onnx). Production's known-count arm instead carries the fold's
duration-weighted mean of its k+1 cluster embeddings; recomputing keeps all arms
on one rule. Trials are also re-embedded from their first 3 s and 2 s of clean
speech, the short-cluster regime where scores collapse.

    uv run --group eval eval/heldout_reid.py          # writes out/heldout/reid.md
"""

from __future__ import annotations

import json
import sys

import ami
import heldout
import numpy as np
from common import read_pcm16
from naming_gate import clean_spans, embed_spans, truncate_spans
from reid_score import Trial, dir_at_far, eer_point, operating_point
from rttm import Turn, parse_rttm

from stenograf.diarization.base import SpeakerTurn

OUT = heldout.OUT_DIR
ENROLL = "a"
BUDGETS: tuple[float | None, ...] = (None, 3.0, 2.0)
SHIPPED = 0.56
SWEEP = (0.45, 0.50, 0.53, 0.56, 0.60, 0.65, 0.70)


def arm_turns() -> dict[str, dict[str, list[SpeakerTurn]]]:
    """arm → channel id → final turns, for every arm with results on disk."""
    channels = json.loads((OUT / "channels.json").read_text())
    arms: dict[str, dict[str, list[SpeakerTurn]]] = {}
    for name, path in (
        ("owned loop, true k", OUT / "turns_known.json"),
        ("stenodiar estimate", OUT / "turns_estimate.json"),
        ("Nemotron, no count", OUT / "turns_nemotron_fp32_cpu.json"),
    ):
        if path.exists():
            data = json.loads(path.read_text())
            arms[name] = {
                cid: [SpeakerTurn(t["speaker"], t["start"], t["end"]) for t in v["turns"]]
                for cid, v in data.items()
            }
    if "owned loop, true k" in arms:
        # heldout.py records a count-1 channel as one channel-long turn, and the
        # embedder's memory is linear in slice length (~14 MB per second, 26 GB
        # at the kill, 2026-11-15). Production embeds a solo channel over its
        # transcript entries (session.py, solo-channel bypass); the reference's
        # speech segments stand in for them.
        for c in channels:
            if c["num_speakers"] == 1 and c["id"] in arms["owned loop, true k"]:
                ref = parse_rttm(ami.AMI_REFS_DIR / f"{c['id']}.rttm")
                arms["owned loop, true k"][c["id"]] = [
                    SpeakerTurn("S0", t.start, t.end) for t in ref
                ]
    probs = OUT / "probs_nemotron_fp32_cpu"
    if probs.exists():
        arms["Nemotron, count applied after"] = {
            c["id"]: [
                SpeakerTurn(t["speaker"], t["start"], t["end"])
                for t in heldout.count_capped(
                    np.load(probs / f"{c['id']}.npy").astype(np.float32), c["num_speakers"]
                )
            ]
            for c in channels
            if (probs / f"{c['id']}.npy").exists()
        }
    return arms


def participants(meeting: str) -> dict[str, tuple[str, str]]:
    """name → (agent, headset) for one meeting."""
    annotations = ami.RAW_DIR / "annotations"
    meetings = ami.parse_meetings_xml(annotations / "corpusResources" / "meetings.xml")
    return {name: (agent, headset) for agent, (headset, name) in meetings[meeting].items()}


def headset_gallery(meeting: str, embed, pool) -> dict[str, list[np.ndarray]]:
    from stenograf.diarization.sherpa import cluster_embeddings

    gallery = {}
    who = participants(meeting)
    for name in sorted(who)[:-1]:
        agent, headset = who[name]
        words = ami.RAW_DIR / "annotations" / "words" / f"{meeting}.{agent}.words.xml"
        turns = [SpeakerTurn(name, s, e) for s, e in ami.merge_spans(ami.parse_words_xml(words))]
        wav = ami.RAW_DIR / meeting / f"{meeting}.Headset-{headset}.wav"
        gallery[name] = [cluster_embeddings(turns, read_pcm16(wav), embed, pool=pool)[name]]
    return gallery


def channel_vectors(
    cid: str, turns: list[SpeakerTurn], embed, pool
) -> dict[str, tuple[str | None, dict[float | None, np.ndarray | None]]]:
    """cluster → (true speaker, budget → embedding)."""
    from stenograf.diarization.sherpa import cluster_embeddings

    pcm = read_pcm16(ami.CHANNELS_DIR / f"{cid}.wav")
    ref = parse_rttm(ami.AMI_REFS_DIR / f"{cid}.rttm")
    full = cluster_embeddings(turns, pcm, embed, pool=pool)
    by_cluster: dict[str, list[SpeakerTurn]] = {}
    for t in turns:
        by_cluster.setdefault(t.speaker, []).append(t)
    spans = clean_spans(by_cluster)
    out = {}
    for label, cluster in by_cluster.items():
        true = ami.dominant_speaker([Turn(t.speaker, t.start, t.end) for t in cluster], ref)
        vectors: dict[float | None, np.ndarray | None] = {None: full.get(label)}
        for budget in BUDGETS[1:]:
            vectors[budget] = embed_spans(pcm, truncate_spans(spans[label], budget), embed)
        out[label] = (true, vectors)
    return out


def score(vector: np.ndarray, gallery: dict[str, list[np.ndarray]]) -> dict[str, float]:
    return {n: float(np.mean([vector @ e for e in entries])) for n, entries in gallery.items()}


def table(rows: list[tuple[str, list[Trial]]]) -> list[str]:
    lines = [
        f"| arm | trials (known/unknown) | DIR / FAR / FRR @ {SHIPPED} | DIR@FAR0 (thr) "
        "| DIR@FAR≤5% (thr) | EER point |",
        "|---|---|---|---|---|---|",
    ]
    for label, trials in rows:
        known = sum(t.known for t in trials)
        p = operating_point(trials, SHIPPED)
        strict, loose, eer = dir_at_far(trials, 0.0), dir_at_far(trials, 0.05), eer_point(trials)
        fmt = lambda q: f"{q.dir_:.1%} ({q.threshold:.3f})" if q else "—"  # noqa: E731
        lines.append(
            f"| {label} | {known}/{len(trials) - known} "
            f"| {p.dir_:.1%} / {p.far:.1%} / {p.frr:.1%} | {fmt(strict)} | {fmt(loose)} "
            f"| {eer.dir_:.1%} / {eer.far:.1%} @ {eer.threshold:.3f} |"
            if eer
            else f"| {label} | {known}/{len(trials) - known} | — | — | — | — |"
        )
    return lines


def main() -> int:
    from stenograf.diarization.loop import OwnDiarizer

    arms = arm_turns()
    if not arms:
        print("no held-out turns — run heldout.py / nemotron_arm.py first", file=sys.stderr)
        return 1
    channels = json.loads((OUT / "channels.json").read_text())
    diarizer = OwnDiarizer()
    embed, pool = diarizer.embed, diarizer._executor()
    meetings = sorted({c["id"].split(".")[0] for c in channels})
    groups = sorted({c["group"] for c in channels})

    headset = {g: headset_gallery(g + ENROLL, embed, pool) for g in groups}
    print(f"headset galleries: { {g: sorted(v) for g, v in headset.items()} }", flush=True)

    trials: dict[tuple[str, str, float | None], list[Trial]] = {}
    for arm, by_channel in arms.items():
        clusters: dict[str, dict[str, list[np.ndarray]]] = {g: {} for g in groups}
        later = []
        for c in channels:
            cid, group = c["id"], c["group"]
            if cid not in by_channel:
                continue
            vectors = channel_vectors(cid, by_channel[cid], embed, pool)
            if c["session"] == ENROLL:
                if cid.endswith(".duo"):
                    continue
                enrollable = set(sorted(participants(cid.split(".")[0]))[:-1])
                for true, by_budget in vectors.values():
                    if true in enrollable and by_budget[None] is not None:
                        clusters[group].setdefault(true, []).append(by_budget[None])
            else:
                later += [(cid, group, label, v) for label, v in vectors.items()]
        print(
            f"[{arm}] cluster galleries: { {g: sorted(v) for g, v in clusters.items()} }",
            flush=True,
        )
        for source, galleries in (("headset", headset), ("clusters", clusters)):
            for budget in BUDGETS:
                trials[(arm, source, budget)] = [
                    Trial(f"{cid}/{label}", true, score(vec[budget], galleries[group]))
                    for cid, group, label, (true, vec) in later
                    if vec[budget] is not None and galleries[group]
                ]

    lines = ["## Re-ID on the held-out meetings (heldout_reid.py)", ""]
    for source in ("headset", "clusters"):
        for budget in BUDGETS:
            label = "full duration" if budget is None else f"{budget:g} s clean"
            lines += [f"### {source} enrollment, {label}", ""]
            lines += table([(arm, trials[(arm, source, budget)]) for arm in arms])
            lines.append("")
    lines += ["### Threshold sweep, cluster enrollment, full duration (DIR / FAR / FRR)", ""]
    lines += [
        "| arm | " + " | ".join(f"{t:.2f}" for t in SWEEP) + " |",
        "|---|" + "---|" * len(SWEEP),
    ]
    for arm in arms:
        ts = trials[(arm, "clusters", None)]
        cells = [operating_point(ts, t) for t in SWEEP]
        lines.append(
            f"| {arm} | "
            + " | ".join(f"{p.dir_:.0%}/{p.far:.0%}/{p.frr:.0%}" for p in cells)
            + " |"
        )
    lines += ["", "### Strangers named at the shipped threshold, cluster enrollment, full", ""]
    for arm in arms:
        hits = [t for t in trials[(arm, "clusters", None)] if not t.known and t.top[1] >= SHIPPED]
        lines.append(
            f"- {arm}: "
            + (
                ", ".join(
                    f"{t.name}→{t.top[0]} {t.top[1]:.3f} (true {t.true_speaker})" for t in hits
                )
                or "none"
            )
        )
    text = "\n".join(lines)
    print(text)
    (OUT / "reid.md").write_text(text + "\n")
    json.dump(
        {
            f"{a}|{s}|{b}": [[t.name, t.true_speaker, t.scores] for t in ts]
            for (a, s, b), ts in trials.items()
        },
        (OUT / "reid_trials.json").open("w"),
    )
    print(f"meetings {meetings}; wrote {OUT / 'reid.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
