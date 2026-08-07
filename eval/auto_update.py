"""Gated automatic profile updates: the poisoning risk, measured.

`PLAN-DIARIZATION.md` step 2.3 proposes that an auto-matched cluster may add
its embedding to the matched profile without a user correction, behind a
high-confidence gate. The research record's warning is specific: *ungated*
iterative updates measured below the no-update baseline (memory poisoning —
one wrong absorption pulls the profile toward another voice, which attracts
more wrong matches). This measures the whole policy space on our stack before
anything ships.

Simulation, per AMI group and per policy independently: the store starts as
rename-once enrollment from session **a** (each session-a cluster's cached
embedding under its reference-majority name — exactly what `steno profiles
assign` stores; the alphabetically-last participant stays unenrolled). Then
sessions **b** and **c** replay the deployed loop: every cluster of the
meeting is scored against the store as it stands (mean cosine, the v2 store),
clusters at or above the shipped threshold are "named", and the policy decides
which named clusters append their embedding — all of a meeting's resolutions
before any of its updates, as production resolves a meeting. Policies:

- ``none``    — the no-update baseline (today's shipped behavior).
- ``ungated`` — every named cluster appends (the research record's loser).
- ``margin m``  — append only at score ≥ threshold + m.
- ``margin + clean`` — margin plus a minimum clean-speech duration
  (`naming_gate.py` measured wrong namings are *long* — 9.5 s / 75.1 s — so
  this is expected to gate little; measured rather than assumed).
- ``oracle``  — append only correctly-named clusters: the upper bound any
  gate could reach.

Evaluation is held out: session **d** clusters (never update material) scored
against each policy's final store, full-duration plus 3 s / 2 s truncation —
the short-cluster regime where `store_v2.py` measured multi-meeting profiles
as insurance, i.e. where auto-updates would earn their keep. The known
poisoning seed is in the update material by construction:
IS1009's quiet FIO084 has no session-a cluster (fused away → unenrolled) and
their later clusters pull toward the impure FIO087 profile at up to 0.860
(`rename_once.py`), scores no margin gate survives.

Requires the cached matrix (``eval/ami.py run``)::

    uv run --group eval eval/auto_update.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

import numpy as np
from common import OUT_DIR, read_pcm16
from reid_score import Trial, dir_at_far, eer_point, operating_point
from rttm import parse_rttm

HYP_DIR = OUT_DIR / "diar" / "ami"
ENROLL_SESSION = "a"
UPDATE_SESSIONS = ("b", "c")
EVAL_SESSIONS = ("d",)


@dataclass(frozen=True)
class Policy:
    """One update rule: when may a named cluster append its embedding?"""

    label: str
    margin: float | None = None  # None: never append (baseline)
    min_clean_s: float = 0.0
    oracle: bool = False

    def applies(self, score: float, threshold: float, clean_s: float, correct: bool) -> bool:
        if self.margin is None:
            return False
        return (
            score >= threshold + self.margin
            and clean_s >= self.min_clean_s
            and (correct or not self.oracle)
        )


POLICIES = (
    Policy("none"),
    Policy("ungated", margin=0.0),
    Policy("margin .10", margin=0.10),
    Policy("margin .20", margin=0.20),
    Policy("margin .10 + clean ≥ 5 s", margin=0.10, min_clean_s=5.0),
    Policy("oracle (correct only)", margin=0.0, oracle=True),
)

Gallery = dict[str, list[np.ndarray]]  # name → stored embeddings (mean cosine)


def mean_cosine(vector: np.ndarray, entries: list[np.ndarray]) -> float:
    return float(np.mean([vector @ e for e in entries]))


def load_channel_embeddings(channel_id: str) -> dict[str, np.ndarray]:
    return {
        k: np.asarray(v, dtype=np.float32)
        for k, v in json.loads((HYP_DIR / f"{channel_id}.emb.json").read_text()).items()
    }


def simulate(
    update_meetings: list[tuple[str, list]],
    initial: dict[str, Gallery],
    threshold: float,
) -> tuple[dict[str, dict[str, Gallery]], list[str], list[str], list[str]]:
    """Replay sessions b–c under every policy; return final galleries + logs.

    ``update_meetings``: (group, clusters) in meeting order; clusters
    carry spans/true_speaker (`naming_gate.Cluster`). Returns
    ``galleries[policy][group]``, the per-policy summary lines, the wrong-
    update listing, and the discriminator audit (every ungated-arm candidate's
    score, top-minus-second gap, and clean duration — right vs wrong in one
    table, so "would any cutoff separate them?" is read off, not argued)."""
    galleries = {
        p.label: {g: {n: list(v) for n, v in gal.items()} for g, gal in initial.items()}
        for p in POLICIES
    }
    counts = {p.label: {"ok": 0, "bad": 0, "skip_ok": 0, "skip_bad": 0} for p in POLICIES}
    wrong: list[str] = []
    audit = [
        "| candidate | named as | score | top−2nd | clean s | correct? |",
        "|---|---|---|---|---|---|",
    ]
    for group, clusters in update_meetings:
        vectors = {}
        for cluster in clusters:
            vectors.setdefault(cluster.channel_id, load_channel_embeddings(cluster.channel_id))
        for policy in POLICIES:
            gallery = galleries[policy.label][group]
            pending: list[tuple[str, np.ndarray]] = []
            for cluster in clusters:
                vector = vectors[cluster.channel_id].get(cluster.label)
                if vector is None or not gallery:
                    continue
                ranked = sorted(
                    ((mean_cosine(vector, e), n) for n, e in gallery.items()), reverse=True
                )
                score, name = ranked[0]
                if score < threshold:
                    continue  # unnamed: no update path exists
                correct = name == cluster.true_speaker
                if policy.label == "ungated":
                    gap = score - ranked[1][0] if len(ranked) > 1 else float("nan")
                    audit.append(
                        f"| {cluster.name} | {name} | {score:.3f} | {gap:.3f} "
                        f"| {cluster.clean_s:.1f} | {'yes' if correct else 'NO'} |"
                    )
                if policy.applies(score, threshold, cluster.clean_s, correct):
                    pending.append((name, vector))
                    counts[policy.label]["ok" if correct else "bad"] += 1
                    if not correct:
                        wrong.append(
                            f"- {policy.label}: {cluster.name} → {name} @ {score:.3f} "
                            f"(true {cluster.true_speaker}, clean {cluster.clean_s:.1f} s)"
                        )
                else:
                    counts[policy.label]["skip_ok" if correct else "skip_bad"] += 1
            for name, vector in pending:  # after the meeting resolves, as production would
                gallery[name].append(vector)
    summary = [
        "| policy | appended ok | appended wrong | skipped ok | skipped wrong |",
        "|---|---|---|---|---|",
        *(
            f"| {p.label} | {c['ok']} | {c['bad']} | {c['skip_ok']} | {c['skip_bad']} |"
            for p in POLICIES
            if (c := counts[p.label]) is not None
        ),
    ]
    return galleries, summary, wrong, audit


def score_trials(
    vectors: dict[str, tuple[np.ndarray, str | None]],
    galleries: dict[str, dict[str, Gallery]],
) -> dict[str, list[Trial]]:
    trials: dict[str, list[Trial]] = {p.label: [] for p in POLICIES}
    for policy in POLICIES:
        for trial_name, (vector, true) in vectors.items():
            for group, gallery in galleries[policy.label].items():
                scores = {n: mean_cosine(vector, e) for n, e in gallery.items()}
                trials[policy.label].append(Trial(f"{group}:{trial_name}", true, scores))
    return trials


def arm_table(trials: dict[str, list[Trial]], threshold: float) -> list[str]:
    lines = [
        f"| policy | DIR@FAR0 (thr) | DIR@FAR≤5% (thr) | DIR/FAR/FRR @{threshold:g} "
        "| EER point |",
        "|---|---|---|---|---|",
    ]
    for policy in POLICIES:
        arm = trials[policy.label]
        strict, loose = dir_at_far(arm, 0.0), dir_at_far(arm, 0.05)
        shipped = operating_point(arm, threshold)
        eer = eer_point(arm)
        lines.append(
            f"| {policy.label} | {strict.dir_:.1%} ({strict.threshold:.3f}) "
            f"| {loose.dir_:.1%} ({loose.threshold:.3f}) "
            f"| {shipped.dir_:.1%} / {shipped.far:.1%} / {shipped.frr:.1%} "
            f"| {eer.dir_:.1%} / {eer.far:.1%} / {eer.frr:.1%} @ {eer.threshold:.3f} |"
        )
    return lines


def stranger_accepts(trials: dict[str, list[Trial]], threshold: float) -> list[str]:
    """Strangers named at the shipped threshold, per policy — poisoning made
    visible as the specific false accepts it causes."""
    lines = []
    for policy in POLICIES:
        hits = [
            f"{t.name} → {t.top[0]} @ {t.top[1]:.3f}"
            for t in trials[policy.label]
            if not t.known and t.top[1] >= threshold
        ]
        lines.append(f"- {policy.label}: " + ("; ".join(hits) if hits else "none"))
    return lines


def main() -> int:
    import ami
    from naming_gate import embed_spans, load_clusters, truncate_spans
    from rename_once import cluster_galleries

    from stenograf.diarization.loop import OwnDiarizer
    from stenograf.voiceprints import DEFAULT_THRESHOLD

    if not HYP_DIR.exists():
        print("cached matrix outputs missing — run `eval/ami.py run` first", file=sys.stderr)
        return 1

    channels = ami.load_channels()
    enroll_channels = [c for c in channels if c.session == ENROLL_SESSION]

    # Rename-once enrollment, restricted to the trial convention's enrollable
    # names (alphabetically-last participant stays a stranger).
    enrolled = cluster_galleries(enroll_channels)
    groups = [*ami.AMI_GROUPS, ami.ICSI_GROUP]
    initial: dict[str, Gallery] = {}
    for group in groups:
        names = ami.participants(group, ENROLL_SESSION)
        found = enrolled.get(group, {})
        initial[group] = {n: found[n] for n in names[:-1] if n in found}

    update_meetings = [
        (
            group,
            load_clusters(
                HYP_DIR,
                [c for c in channels if c.group == group and c.session == session],
            ),
        )
        for session in UPDATE_SESSIONS
        for group in groups
    ]
    galleries, summary, wrong, audit = simulate(update_meetings, initial, DEFAULT_THRESHOLD)

    eval_channels = [c for c in channels if c.session in EVAL_SESSIONS]
    full: dict[str, tuple[np.ndarray, str | None]] = {}
    for channel in eval_channels:
        embeddings = load_channel_embeddings(channel.id)
        hyp = parse_rttm(HYP_DIR / f"{channel.id}.rttm")
        ref = parse_rttm(channel.ref_path)
        for cluster, vector in embeddings.items():
            turns = [t for t in hyp if t.speaker == cluster]
            full[f"{channel.id}/{cluster}"] = (vector, ami.dominant_speaker(turns, ref))

    trials = score_trials(full, galleries)
    known = sum(1 for t in trials[POLICIES[0].label] if t.known)
    total = len(trials[POLICIES[0].label])
    lines = [
        "## Gated automatic profile updates (auto_update.py)",
        "",
        f"Enrollment: session-{ENROLL_SESSION} clusters; updates replay sessions "
        f"{'+'.join(UPDATE_SESSIONS)} at threshold {DEFAULT_THRESHOLD:g}; held-out "
        f"evaluation: session d ({total} trials, {known} known).",
        "",
        "### Updates applied",
        "",
        *summary,
        "",
        "Wrong updates absorbed:",
        "",
        *(wrong or ["- none"]),
        "",
        "### Discriminator audit (every named cluster in the ungated arm)",
        "",
        *audit,
        "",
        "### Held-out naming, full-duration trial clusters",
        "",
        *arm_table(trials, DEFAULT_THRESHOLD),
        "",
        f"Strangers named at {DEFAULT_THRESHOLD:g}:",
        "",
        *stranger_accepts(trials, DEFAULT_THRESHOLD),
    ]

    embed = OwnDiarizer().embed
    clusters = load_clusters(HYP_DIR, eval_channels)
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
        short = score_trials(truncated, galleries)
        lines += [
            "",
            f"### Held-out naming, trials truncated to {budget:g} s clean audio",
            "",
            *arm_table(short, DEFAULT_THRESHOLD),
            "",
            f"Strangers named at {DEFAULT_THRESHOLD:g}:",
            "",
            *stranger_accepts(short, DEFAULT_THRESHOLD),
        ]

    text = "\n".join(lines)
    print(text)
    out = OUT_DIR / "diar-auto-update.md"
    out.write_text(text + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
