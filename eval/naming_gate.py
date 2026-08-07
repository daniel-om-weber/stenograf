"""Should a cluster with little clean speech be barred from profile naming?

`PLAN-DIARIZATION.md` step 1.5 proposes: a cluster with under ~3 s of clean
(non-overlap) speech keeps its local S-label instead of being matched against
profiles — the literature's short-turn cliff (EER at 2 s is 2.4× the
full-duration figure, and ERes2Net-base specifically collapses on short
turns). Three measurements decide it, all on the step-0 corpus harness:

A. **The cliff, on our stack.** Every known-count trial cluster's clean audio
   is truncated to D ∈ {1, 2, 3, 5, 8, full} seconds, embedded exactly the way
   production embeds a cluster, and scored against the session-a galleries.
   If naming reliability really collapses under some D, this shows where.

B. **The shipped path.** Clean durations of the known-count matrix clusters,
   and the re-ID operating points with under-bar trials forced to reject.
   Known-count folding leaves few small clusters, so this is expected to be a
   near-no-op — that is the non-regression evidence, not the motivation.

C. **Estimate mode, where short clusters actually live.** The cached
   estimate-mode clusters (``out/diar/ami-est``, sessions b–d + ICSI) are
   embedded fresh, run through ``collapse_single_voice`` (the shipped order),
   and matched against all enrolled profiles at the shipped threshold. Per
   candidate cutoff: how many wrong namings the gate prevents vs how many
   correct namings (merge-at-naming recoveries included) it loses.

Requires the cached matrix outputs (``eval/ami.py run``) and the est-mode
cache (``eval/split_recovery.py``)::

    uv run --group eval eval/naming_gate.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
from common import OUT_DIR, read_pcm16
from reid_score import Trial, dir_at_far, operating_point
from rttm import parse_rttm

CUTOFFS = (1.0, 2.0, 3.0, 5.0)
"""Candidate minimum-clean-duration gates (s)."""

DURATIONS = (1.0, 2.0, 3.0, 5.0, 8.0, None)
"""Section A truncation targets (s); None = the cluster's full clean audio."""

EST_DIR = OUT_DIR / "diar" / "ami-est"
HYP_DIR = OUT_DIR / "diar" / "ami"


def clean_spans(turns_by_cluster: dict[str, list]) -> dict[str, list[tuple[float, float]]]:
    """Per-cluster speech spans with every overlapped stretch removed.

    Mirrors what production embeds (``cluster_embeddings``): a cluster whose
    speech is entirely overlapped keeps its raw spans as the fallback."""
    from stenograf.diarization.sherpa import _overlap_spans, _subtract_spans

    overlap = _overlap_spans(turns_by_cluster)
    spans = {}
    for cluster, turns in turns_by_cluster.items():
        raw = [(t.start, t.end) for t in turns]
        spans[cluster] = _subtract_spans(raw, overlap) or raw
    return spans


def truncate_spans(
    spans: list[tuple[float, float]], budget: float | None
) -> list[tuple[float, float]]:
    """The first spans in time order up to ``budget`` seconds total (the
    cluster as if the speaker had only spoken this much)."""
    if budget is None:
        return spans
    kept: list[tuple[float, float]] = []
    remaining = budget
    for start, end in sorted(spans):
        if remaining <= 0:
            break
        kept.append((start, min(end, start + remaining)))
        remaining -= kept[-1][1] - kept[-1][0]
    return kept


def embed_spans(pcm: np.ndarray, spans: list[tuple[float, float]], embed) -> np.ndarray | None:
    """Duration-weighted mean embedding of ``spans`` — the tail of
    ``cluster_embeddings`` for one cluster (long-span preference included)."""
    from stenograf.audio import SAMPLE_RATE, l2_normalize
    from stenograf.diarization.sherpa import MIN_EMBED_SECONDS

    long = [s for s in spans if s[1] - s[0] >= MIN_EMBED_SECONDS]
    vectors, weights = [], []
    for start, end in long or spans:
        vector = embed(pcm[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)])
        if vector is not None:
            vectors.append(vector)
            weights.append(end - start)
    if not vectors:
        return None
    return l2_normalize(np.average(vectors, axis=0, weights=weights))


@dataclass(frozen=True)
class Cluster:
    """One hypothesis cluster with everything the sections score."""

    channel_id: str
    group: str | None
    label: str
    true_speaker: str | None
    spans: list[tuple[float, float]]  # clean (non-overlap) spans

    @property
    def clean_s(self) -> float:
        return sum(e - s for s, e in self.spans)

    @property
    def name(self) -> str:
        return f"{self.channel_id}/{self.label}"


def load_clusters(hyp_dir, channels) -> list[Cluster]:
    """Hypothesis clusters of every non-enrollment channel in ``hyp_dir``."""
    import ami

    clusters: list[Cluster] = []
    for channel in channels:
        rttm_path = hyp_dir / f"{channel.id}.rttm"
        if not rttm_path.exists():
            print(f"  no {rttm_path.name} — cache incomplete", file=sys.stderr)
            continue
        hyp = parse_rttm(rttm_path)
        ref = parse_rttm(channel.ref_path)
        by_cluster: dict[str, list] = {}
        for t in hyp:
            by_cluster.setdefault(t.speaker, []).append(t)
        spans = clean_spans(by_cluster)
        for label, turns in by_cluster.items():
            clusters.append(
                Cluster(
                    channel_id=channel.id,
                    group=channel.group,
                    label=label,
                    true_speaker=ami.dominant_speaker(turns, ref),
                    spans=spans[label],
                )
            )
    return clusters


def trial_rows(points: dict[str, list[Trial]]) -> list[str]:
    """One table row per arm: DIR@FAR0, the shipped-threshold point, score stats."""
    rows = [
        "| arm | known | unknown | DIR@FAR0 (thr) | DIR/FAR/FRR @0.5 "
        "| known top-correct μ/min | stranger top μ/max |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm, trials in points.items():
        known = [t for t in trials if t.known]
        unknown = [t for t in trials if not t.known]
        correct = [t.top[1] for t in known if t.top[0] == t.true_speaker]
        stranger = [t.top[1] for t in unknown]
        strict = dir_at_far(trials, 0.0)
        shipped = operating_point(trials, 0.5)
        rows.append(
            f"| {arm} | {len(known)} | {len(unknown)} "
            f"| {strict.dir_:.1%} ({strict.threshold:.3f}) "
            f"| {shipped.dir_:.1%} / {shipped.far:.1%} / {shipped.frr:.1%} "
            f"| {np.mean(correct):.3f} / {min(correct):.3f} "
            f"| {np.mean(stranger):.3f} / {max(stranger):.3f} |"
            if known and correct and stranger and strict
            else f"| {arm} | {len(known)} | {len(unknown)} | — | — | — | — |"
        )
    return rows


def section_a(clusters: list[Cluster], galleries, embed) -> list[str]:
    """Truncation sweep: naming reliability as a function of clean duration."""
    import ami

    channels = {c.channel_id for c in clusters}
    pcms = {cid: read_pcm16(ami.CHANNELS_DIR / f"{cid}.wav") for cid in sorted(channels)}
    arms: dict[str, list[Trial]] = {}
    for budget in DURATIONS:
        arm = f"{budget:g} s" if budget is not None else "full"
        trials: list[Trial] = []
        for cluster in clusters:
            vector = embed_spans(
                pcms[cluster.channel_id], truncate_spans(cluster.spans, budget), embed
            )
            if vector is None:
                continue
            for group, gallery in galleries.items():
                scores = {name: float(vector @ emb) for name, emb in gallery.items()}
                trials.append(Trial(f"{group}:{cluster.name}", cluster.true_speaker, scores))
        arms[arm] = trials
    return ["### A. Naming vs clean duration (truncated matrix clusters)", "", *trial_rows(arms)]


def section_b(clusters: list[Cluster], trials: list[Trial]) -> list[str]:
    """The gate on the shipped known-count path: who is under the bar, and the
    operating points with those trials forced to reject."""
    durations = {c.name: c.clean_s for c in clusters}
    smallest = sorted(clusters, key=lambda c: c.clean_s)[:5]
    lines = [
        "### B. The shipped known-count path",
        "",
        "Smallest clean durations in the matrix clusters: "
        + ", ".join(f"{c.name} {c.clean_s:.1f}s" for c in smallest),
        "",
        "| gate | gated known | gated unknown | DIR@FAR0 (thr) | DIR@FAR≤5% (thr) |",
        "|---|---|---|---|---|",
    ]
    for cutoff in (0.0, *CUTOFFS):
        # A gated trial always rejects (top score −1) but must keep its
        # known/unknown status, which Trial derives from score membership.
        gated = [
            Trial(t.name, t.true_speaker, {t.true_speaker if t.known else "gated": -1.0})
            if durations[t.name.split(":", 1)[1]] < cutoff
            else t
            for t in trials
        ]
        n_known = sum(1 for g, t in zip(gated, trials, strict=True) if g is not t and t.known)
        n_unknown = sum(
            1 for g, t in zip(gated, trials, strict=True) if g is not t and not t.known
        )
        strict, loose = dir_at_far(gated, 0.0), dir_at_far(gated, 0.05)
        lines.append(
            f"| {cutoff:g} s | {n_known} | {n_unknown} "
            f"| {strict.dir_:.1%} ({strict.threshold:.3f}) "
            f"| {loose.dir_:.1%} ({loose.threshold:.3f}) |"
        )
    return lines


def section_c(channels, galleries, embed) -> list[str]:
    """Estimate mode through the shipped order (collapse, then naming at the
    shipped threshold): what each cutoff would have gated, and at what cost."""
    import ami

    from stenograf.diarization.base import SpeakerTurn
    from stenograf.diarization.sherpa import cluster_embeddings
    from stenograf.pipeline import collapse_single_voice
    from stenograf.voiceprints import DEFAULT_THRESHOLD

    profiles = {name: emb for gallery in galleries.values() for name, emb in gallery.items()}
    outcomes: list[tuple[Cluster, str, str, float]] = []  # cluster, outcome, top name, score
    for channel in channels:
        rttm_path = EST_DIR / f"{channel.id}.rttm"
        if not rttm_path.exists():
            print(f"  no {rttm_path.name} — run eval/split_recovery.py first", file=sys.stderr)
            continue
        pcm = read_pcm16(ami.CHANNELS_DIR / f"{channel.id}.wav")
        turns = [
            SpeakerTurn(t.speaker, t.start, t.end) for t in parse_rttm(rttm_path)
        ]
        embeddings = cluster_embeddings(turns, pcm, embed)
        turns, embeddings = collapse_single_voice(turns, embeddings)
        ref = parse_rttm(channel.ref_path)
        by_cluster: dict[str, list] = {}
        for t in turns:
            by_cluster.setdefault(t.speaker, []).append(t)
        spans = clean_spans(by_cluster)
        for label, cluster_turns in by_cluster.items():
            if label not in embeddings:
                continue
            cluster = Cluster(
                channel_id=channel.id,
                group=channel.group,
                label=label,
                true_speaker=ami.dominant_speaker(cluster_turns, ref),
                spans=spans[label],
            )
            scores = {name: float(embeddings[label] @ emb) for name, emb in profiles.items()}
            top_name, top_score = max(scores.items(), key=lambda kv: kv[1])
            if top_score < DEFAULT_THRESHOLD:
                outcome = "unnamed"
            elif top_name == cluster.true_speaker:
                outcome = "correct"
            else:
                outcome = "wrong"
            outcomes.append((cluster, outcome, top_name, top_score))

    named = [o for o in outcomes if o[1] != "unnamed"]
    lines = [
        "### C. Estimate mode (collapsed clusters, shipped threshold "
        f"{DEFAULT_THRESHOLD})",
        "",
        f"{len(outcomes)} clusters reach naming; "
        f"{sum(1 for o in named if o[1] == 'correct')} named correct, "
        f"{sum(1 for o in named if o[1] == 'wrong')} named wrong.",
        "",
        "| gate | gated clusters | wrong namings prevented | correct namings lost |",
        "|---|---|---|---|",
    ]
    for cutoff in CUTOFFS:
        gated = [o for o in outcomes if o[0].clean_s < cutoff]
        lines.append(
            f"| {cutoff:g} s | {len(gated)} "
            f"| {sum(1 for o in gated if o[1] == 'wrong')} "
            f"| {sum(1 for o in gated if o[1] == 'correct')} |"
        )
    lines += ["", "Named clusters, smallest first (clean s → outcome):", ""]
    for cluster, outcome, top_name, score in sorted(named, key=lambda o: o[0].clean_s)[:15]:
        lines.append(
            f"- {cluster.name} {cluster.clean_s:.1f}s → {outcome} "
            f"({top_name} @ {score:.3f}, true {cluster.true_speaker})"
        )
    return lines


def main() -> int:
    import ami

    from stenograf.diarization.loop import OwnDiarizer

    trials_path = OUT_DIR / "reid" / "trials.json"
    if not trials_path.exists() or not HYP_DIR.exists():
        print("cached matrix outputs missing — run `eval/ami.py run` first", file=sys.stderr)
        return 1
    channels = [c for c in ami.load_channels() if c.session != ami.ENROLL_SESSION]

    embed = OwnDiarizer().embed
    galleries = ami.build_galleries(embed)
    clusters = load_clusters(HYP_DIR, channels)

    from reid_score import load_trials

    report = [
        "## Minimum clean-duration gate for naming (naming_gate.py)",
        "",
        *section_a(clusters, galleries, embed),
        "",
        *section_b(clusters, load_trials(trials_path)),
        "",
        *section_c(channels, galleries, embed),
        "",
    ]
    text = "\n".join(report)
    print(text)
    out = OUT_DIR / "diar-naming-gate.md"
    out.write_text(text + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
