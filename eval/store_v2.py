"""Multi-meeting profiles: score averaging vs embedding averaging, measured.

`PLAN-DIARIZATION.md` step 2.1 replaces the profile store's single running
mean with a per-meeting embedding set matched by mean cosine — on the
literature's verdict that score averaging beats embedding averaging
(2.05 % vs 2.85 % EER on identical data; the i-vector-era "average the
embeddings" rule is retracted for modern embeddings). This measures that
choice on the step-0 harness before it ships.

Setup: enroll each group's participants from sessions **a and b** (each
session's reference spans on the participant's own raw close-talk channel —
two independent meetings per speaker; Bmr included since 2026-08-07, names
enrollable in both sessions only); trials are the cached known-count matrix
clusters of sessions **c and d**, scored against every group's gallery. Arms
differ only in how the two enrollments make one profile score:

- ``single a`` / ``single b`` — one-meeting profiles (today's store after one
  enrollment; the two arms bound enrollment-session luck).
- ``emb avg``   — v1 ``reinforce``: one running mean, ``l2(a + b)``.
- ``score avg`` — v2: mean cosine against the stored set.
- ``score max`` — max cosine (control; the plan picks mean).

Full-duration trial clusters are scored from the cached matrix embeddings;
because the arms may only separate where matching is hard, the same trials are
also re-embedded at 2 s and 3 s of clean audio (``naming_gate.py``'s
truncation) — the short-cluster regime where scores measurably collapse.

Requires the cached matrix (``eval/ami.py run``)::

    uv run --group eval eval/store_v2.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
from common import OUT_DIR
from reid_score import Trial, dir_at_far, eer_point
from rttm import parse_rttm

HYP_DIR = OUT_DIR / "diar" / "ami"
ENROLL_SESSIONS = ("a", "b")
TRIAL_SESSIONS = ("c", "d")


def arm_scores(
    vector: np.ndarray, enrollments: dict[str, list[np.ndarray]]
) -> dict[str, dict[str, float]]:
    """Every arm's per-profile score for one trial vector."""
    from stenograf.audio import l2_normalize

    scores: dict[str, dict[str, float]] = {arm: {} for arm in ARMS}
    for name, (a, b) in enrollments.items():
        cos_a, cos_b = float(vector @ a), float(vector @ b)
        scores["single a"][name] = cos_a
        scores["single b"][name] = cos_b
        scores["emb avg"][name] = float(vector @ l2_normalize(a + b))
        scores["score avg"][name] = (cos_a + cos_b) / 2
        scores["score max"][name] = max(cos_a, cos_b)
    return scores


ARMS = ("single a", "single b", "emb avg", "score avg", "score max")


def score_arms(
    vectors: dict[str, tuple[np.ndarray, str | None]],
    enrollments: dict[str, dict[str, list[np.ndarray]]],
) -> dict[str, list[Trial]]:
    """Score every (trial vector × group gallery) under every arm."""
    trials: dict[str, list[Trial]] = {arm: [] for arm in ARMS}
    for trial_name, (vector, true) in vectors.items():
        for group, profiles in enrollments.items():
            by_arm = arm_scores(vector, profiles)
            for arm in ARMS:
                trials[arm].append(Trial(f"{group}:{trial_name}", true, by_arm[arm]))
    return trials


def arm_table(trials: dict[str, list[Trial]]) -> list[str]:
    lines = [
        "| arm | DIR@FAR0 (thr) | DIR@FAR≤5% (thr) | EER point (DIR/FAR/FRR) |",
        "|---|---|---|---|",
    ]
    for arm in ARMS:
        strict, loose = dir_at_far(trials[arm], 0.0), dir_at_far(trials[arm], 0.05)
        eer = eer_point(trials[arm])
        lines.append(
            f"| {arm} | {strict.dir_:.1%} ({strict.threshold:.3f}) "
            f"| {loose.dir_:.1%} ({loose.threshold:.3f}) "
            f"| {eer.dir_:.1%} / {eer.far:.1%} / {eer.frr:.1%} @ {eer.threshold:.3f} |"
        )
    return lines


def main() -> int:
    import ami
    from common import read_pcm16
    from naming_gate import embed_spans, load_clusters, truncate_spans

    from stenograf.diarization.sherpa import SherpaOnnxDiarizer

    if not HYP_DIR.exists():
        print("cached matrix outputs missing — run `eval/ami.py run` first", file=sys.stderr)
        return 1

    embed = SherpaOnnxDiarizer().embed
    galleries = [ami.build_galleries(embed, session) for session in ENROLL_SESSIONS]
    # group → name → (embedding_a, embedding_b). Restricted to names enrollable
    # in BOTH sessions: guaranteed for AMI's fixed foursomes, data-dependent for
    # Bmr (attendance churns; today a∩b covers all five session-a enrollables).
    enrollments = {
        group: {
            name: [g[group][name] for g in galleries]
            for name in galleries[0][group]
            if all(name in g[group] for g in galleries)
        }
        for group in galleries[0]
    }

    channels = [c for c in ami.load_channels() if c.session in TRIAL_SESSIONS]
    full: dict[str, tuple[np.ndarray, str | None]] = {}
    for channel in channels:
        embeddings = {
            k: np.asarray(v, dtype=np.float32)
            for k, v in json.loads((HYP_DIR / f"{channel.id}.emb.json").read_text()).items()
        }
        hyp = parse_rttm(HYP_DIR / f"{channel.id}.rttm")
        ref = parse_rttm(channel.ref_path)
        for cluster, vector in embeddings.items():
            cluster_turns = [t for t in hyp if t.speaker == cluster]
            true = ami.dominant_speaker(cluster_turns, ref)
            full[f"{channel.id}/{cluster}"] = (vector, true)

    trials = score_arms(full, enrollments)
    known = sum(1 for t in trials[ARMS[0]] if t.known)
    total = len(trials[ARMS[0]])
    lines = [
        "## Multi-meeting profile scoring (store_v2.py)",
        "",
        f"Enrolled sessions {'+'.join(ENROLL_SESSIONS)}; {total} trials "
        f"({known} known, {total - known} unknown) from sessions c–d + ICSI.",
        "",
        "### Full-duration trial clusters",
        "",
        *arm_table(trials),
    ]

    clusters = load_clusters(HYP_DIR, channels)
    pcms = {
        cid: read_pcm16(ami.CHANNELS_DIR / f"{cid}.wav")
        for cid in sorted({c.channel_id for c in clusters})
    }
    for budget in (3.0, 2.0):
        truncated: dict[str, tuple[np.ndarray, str | None]] = {}
        for cluster in clusters:
            spans = truncate_spans(cluster.spans, budget)
            vector = embed_spans(pcms[cluster.channel_id], spans, embed)
            if vector is not None:
                truncated[cluster.name] = (vector, cluster.true_speaker)
        lines += [
            "",
            f"### Trial clusters truncated to {budget:g} s clean audio",
            "",
            *arm_table(score_arms(truncated, enrollments)),
        ]

    text = "\n".join(lines)
    print(text)
    out = OUT_DIR / "diar-store-v2.md"
    out.write_text(text + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
