"""The re-ID operating point, picked from measured curves at last.

`PLAN-DIARIZATION.md` step 2.4: the shipped ``DEFAULT_THRESHOLD`` (0.5) was a
starting guess — verification defaults cluster at raw cosine 0.25–0.40,
sherpa's identification default is 0.6, and no toolkit publishes threshold and
error rate together. The grown same-group harness measures it: at 0.5,
hard-stranger FAR is 11.8 % while DIR is *flat* at 88.7 % all the way from
0.322 to 0.605 — the FAR0 point. Raising the threshold to the strict point
costs one known trial of FRR and buys every stranger back.

One curve is not a decision. The pick must hold on the enrollment source the
product actually uses (rename-once *cluster* enrollment, not clean headsets),
in the short-cluster regime where scores collapse (2–3 s truncation), and for
the solo-channel 1:1 flow whose margins are wide (0.935+ known). This script
reports all of it against candidate thresholds, same-group trials only
(`ami.build_trials` convention):

- ``headset`` arm — the matrix trials verbatim (``out/reid/trials.json``).
- ``clusters`` arm — session-a cluster enrollment (what ``steno profiles
  assign`` stores), restricted to the convention's enrollable names, scored
  over the same matrix clusters at full duration and 3 s / 2 s truncation.
- Stranger listings above 0.5 per arm, so the pick's FAR is auditable trial
  by trial (the known IS1009 impure-profile pathology — FIO084 pulled to
  0.860 by a fused enrollment cluster — is steps-4/5 territory and no
  threshold's business; it is listed, not hidden).

Requires the cached matrix (``eval/ami.py run``)::

    uv run --group eval eval/threshold_pick.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
from common import OUT_DIR, read_pcm16
from reid_score import Trial, dir_at_far, load_trials, operating_point

HYP_DIR = OUT_DIR / "diar" / "ami"
CANDIDATES = (0.50, 0.53, 0.55, 0.56, 0.57, 0.58, 0.60, 0.62, 0.65, 0.70)
BUDGETS = (None, 3.0, 2.0)


def arm_report(label: str, trials: list[Trial]) -> list[str]:
    known = sum(1 for t in trials if t.known)
    lines = [
        f"### {label} ({len(trials)} trials: {known} known, {len(trials) - known} unknown)",
        "",
        "| threshold | DIR | FAR | FRR |",
        "|---|---|---|---|",
    ]
    strict = dir_at_far(trials, 0.0)
    for threshold in sorted({*CANDIDATES, round(strict.threshold, 3)}):
        p = operating_point(trials, threshold)
        tag = " (FAR0-strict)" if abs(threshold - strict.threshold) < 5e-4 else ""
        lines.append(f"| {threshold:.3f}{tag} | {p.dir_:.1%} | {p.far:.1%} | {p.frr:.1%} |")
    return lines


def strangers_above(trials: list[Trial], floor: float) -> list[str]:
    hits = sorted(
        (t for t in trials if not t.known and t.top[1] >= floor),
        key=lambda t: -t.top[1],
    )
    return [
        f"- {t.name} → {t.top[0]} @ {t.top[1]:.3f} (true {t.true_speaker})" for t in hits
    ] or ["- none"]


def main() -> int:
    import ami
    from naming_gate import Cluster, embed_spans, load_clusters, truncate_spans
    from rename_once import cluster_galleries

    from stenograf.diarization.loop import OwnDiarizer

    trials_path = OUT_DIR / "reid" / "trials.json"
    if not trials_path.exists():
        print("no matrix trials — run `eval/ami.py run` first", file=sys.stderr)
        return 1

    headset = load_trials(trials_path)
    channels = ami.load_channels()
    trial_channels = [c for c in channels if c.session != ami.ENROLL_SESSION]
    enroll_channels = [c for c in channels if c.session == ami.ENROLL_SESSION]

    enrolled = cluster_galleries(enroll_channels)
    galleries = {
        group: {
            n: found[n]
            for n in ami.participants(group, ami.ENROLL_SESSION)[:-1]
            if n in found
        }
        for group in [*ami.AMI_GROUPS, ami.ICSI_GROUP]
        if (found := enrolled.get(group)) is not None
    }

    embed = OwnDiarizer().embed
    by_channel: dict[str, list[Cluster]] = {}
    for cluster in load_clusters(HYP_DIR, trial_channels):
        by_channel.setdefault(cluster.channel_id, []).append(cluster)
    group_of = {c.id: c.group for c in trial_channels}

    cluster_trials: dict[float | None, list[Trial]] = {b: [] for b in BUDGETS}
    for channel_id, clusters in sorted(by_channel.items()):
        gallery = galleries.get(group_of[channel_id]) or {}
        if not gallery:
            continue
        embeddings = {
            k: np.asarray(v, dtype=np.float32)
            for k, v in json.loads((HYP_DIR / f"{channel_id}.emb.json").read_text()).items()
        }
        pcm = read_pcm16(ami.CHANNELS_DIR / f"{channel_id}.wav")  # one at a time: ~30 wavs
        for cluster in clusters:
            for budget in BUDGETS:
                vector = (
                    embeddings.get(cluster.label)
                    if budget is None
                    else embed_spans(pcm, truncate_spans(cluster.spans, budget), embed)
                )
                if vector is None:
                    continue
                scores = {
                    n: float(np.mean([vector @ e for e in entries]))
                    for n, entries in gallery.items()
                }
                cluster_trials[budget].append(Trial(cluster.name, cluster.true_speaker, scores))

    solo = [t for t in headset if ".mic/" in t.name and t.known]
    lines = [
        "## The re-ID operating point (threshold_pick.py)",
        "",
        *arm_report("Headset enrollment, full duration (matrix trials)", headset),
        "",
        "Strangers ≥ 0.5:",
        *strangers_above(headset, 0.5),
    ]
    for budget in BUDGETS:
        label = "full duration" if budget is None else f"{budget:g} s clean"
        lines += [
            "",
            *arm_report(f"Cluster enrollment, {label}", cluster_trials[budget]),
            "",
            "Strangers ≥ 0.5:",
            *strangers_above(cluster_trials[budget], 0.5),
        ]
    lines += [
        "",
        "### Solo (mic) channel margins, headset arm",
        "",
        f"{len(solo)} known solo trials; correct-top scores "
        f"{min(t.top[1] for t in solo):.3f}–{max(t.top[1] for t in solo):.3f} "
        f"(wrong-top: {sum(1 for t in solo if t.top[0] != t.true_speaker)}).",
    ]

    text = "\n".join(lines)
    print(text)
    out = OUT_DIR / "diar-threshold-pick.md"
    out.write_text(text + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
