"""Rename-once enrollment: the meeting's own cluster embedding, measured.

`PLAN-DIARIZATION.md` step 2.2 turns a user's "this speaker is Alice"
correction into an enrollment: the corrected cluster's embedding — computed
from the meeting's own audio by the pipeline that just ran — is stored on
Alice's profile. The research record says enrollment from the meeting's own
channel *beats* a clean sample (channel match +18 % rel; conversational vs
read speech 5×); this measures the claim on our stack before the flow ships,
because our "meeting embedding" is not a clean channel slice — it is a
diarized cluster's overlap-cleaned mean, far-field and possibly impure on the
loop channel.

Arms differ only in what session *a* enrolls; trials are identical (every
cached matrix cluster of sessions b–d + ICSI, scored against both group
galleries exactly like ``ami.py trials``):

- ``headset`` — reference spans sliced from the raw headset (the trial
  builder's gallery; reproduces the matrix operating points by construction).
- ``clusters`` — what ``steno profiles assign`` would store: session-a matrix
  cluster embeddings, mapped to speakers by reference majority — the mic
  wearer's from the mic channel, everyone else's from the loop. A speaker
  with several clusters gets a multi-entry profile; score = mean cosine (the
  v2 store).
- ``both`` — headset + clusters in one profile (enroll-then-assign, the store
  after one clean enrollment and one correction).
- ``headset ∩`` — the headset arm restricted to the names that *have* a
  session-a cluster. The control that separates enrollment **source** from
  enrollment **coverage**: a speaker the enrollment meeting's diarization
  fused into a groupmate's cluster has nothing to assign, and their later
  clusters become stranger trials against a gallery missing them — a penalty
  any enrollment source pays. Compare ``clusters`` against this arm, not
  against full-coverage ``headset``.

The report also prints the cross-speaker leakage of every cluster profile
(mean cosine against each *other* enrolled name's clean headset): an impure
enrolled cluster — the diarizer fused someone else's speech into it — shows up
here as a high off-diagonal, and is the one failure mode cluster enrollment
adds over a clean sample.

Because full-duration trials saturate (``store_v2.py``), the same trials are
also re-embedded at 3 s and 2 s of clean audio — the regime where enrollment
quality separates.

Requires the cached matrix (``eval/ami.py run``)::

    uv run --group eval eval/rename_once.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
from common import OUT_DIR, read_pcm16
from reid_score import Trial, dir_at_far, eer_point
from rttm import parse_rttm

HYP_DIR = OUT_DIR / "diar" / "ami"
ENROLL_SESSION = "a"
ARMS = ("headset", "clusters", "both", "headset ∩")

Galleries = dict[str, dict[str, dict[str, list[np.ndarray]]]]
"""arm → group → name → enrollment vectors (scored by mean cosine)."""


def cluster_galleries(channels) -> dict[str, dict[str, list[np.ndarray]]]:
    """Per-group galleries from the enrollment session's cached matrix clusters.

    Every session-a cluster is mapped to its reference-majority speaker — the
    correction a user would make — and contributes its embedding to that
    speaker's profile."""
    import ami

    galleries: dict[str, dict[str, list[np.ndarray]]] = {}
    for channel in channels:
        embeddings = {
            k: np.asarray(v, dtype=np.float32)
            for k, v in json.loads((HYP_DIR / f"{channel.id}.emb.json").read_text()).items()
        }
        hyp = parse_rttm(HYP_DIR / f"{channel.id}.rttm")
        ref = parse_rttm(channel.ref_path)
        gallery = galleries.setdefault(channel.group, {})
        for cluster, vector in embeddings.items():
            turns = [t for t in hyp if t.speaker == cluster]
            true = ami.dominant_speaker(turns, ref)
            if true is not None:
                gallery.setdefault(true, []).append(vector)
    return galleries


def score_trials(
    vectors: dict[str, tuple[np.ndarray, str | None]], galleries: Galleries
) -> dict[str, list[Trial]]:
    """Every (trial vector × group gallery) under every arm; mean-cosine scoring."""
    trials: dict[str, list[Trial]] = {arm: [] for arm in ARMS}
    for arm, groups in galleries.items():
        for trial_name, (vector, true) in vectors.items():
            for group, profiles in groups.items():
                scores = {
                    name: float(np.mean([vector @ e for e in entries]))
                    for name, entries in profiles.items()
                }
                trials[arm].append(Trial(f"{group}:{trial_name}", true, scores))
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
    from naming_gate import embed_spans, load_clusters, truncate_spans

    from stenograf.diarization.loop import OwnDiarizer

    if not HYP_DIR.exists():
        print("cached matrix outputs missing — run `eval/ami.py run` first", file=sys.stderr)
        return 1

    embed = OwnDiarizer().embed
    headset = ami.build_galleries(embed, ENROLL_SESSION)
    enroll_channels = [c for c in ami.load_channels() if c.session == ENROLL_SESSION]
    clusters_a = cluster_galleries(enroll_channels)

    # The stranger convention carries over: only headset-enrolled names get a
    # cluster profile, so the alphabetically-last participant stays a stranger
    # in every arm.
    galleries: Galleries = {arm: {} for arm in ARMS}
    coverage: list[str] = []
    for group, gallery in headset.items():
        for arm in ARMS:
            galleries[arm][group] = {}
        for name, vector in gallery.items():
            found = clusters_a.get(group, {}).get(name, [])
            galleries["headset"][group][name] = [vector]
            galleries["both"][group][name] = [vector, *found]
            note = f"{len(found)} session-a cluster(s)"
            if found:
                galleries["clusters"][group][name] = found
                galleries["headset ∩"][group][name] = [vector]
                leaks = (
                    f"{other} {np.mean([v @ hv for v in found]):.3f}"
                    for other, hv in sorted(gallery.items())
                    if other != name
                )
                note += "; cluster profile vs groupmate headsets: " + ", ".join(leaks)
            else:
                note += " — unassignable (fused into a groupmate's cluster?)"
            coverage.append(f"- {group}:{name} — {note}")

    trial_channels = [c for c in ami.load_channels() if c.session != ENROLL_SESSION]
    full: dict[str, tuple[np.ndarray, str | None]] = {}
    for channel in trial_channels:
        embeddings = {
            k: np.asarray(v, dtype=np.float32)
            for k, v in json.loads((HYP_DIR / f"{channel.id}.emb.json").read_text()).items()
        }
        hyp = parse_rttm(HYP_DIR / f"{channel.id}.rttm")
        ref = parse_rttm(channel.ref_path)
        for cluster, vector in embeddings.items():
            turns = [t for t in hyp if t.speaker == cluster]
            full[f"{channel.id}/{cluster}"] = (vector, ami.dominant_speaker(turns, ref))

    trials = score_trials(full, galleries)
    known = sum(1 for t in trials[ARMS[0]] if t.known)
    total = len(trials[ARMS[0]])
    lines = [
        "## Rename-once enrollment: meeting clusters vs clean headset (rename_once.py)",
        "",
        f"Session-{ENROLL_SESSION} enrollment; {total} trials ({known} known, "
        f"{total - known} unknown) from sessions b–d + ICSI.",
        "",
        *coverage,
        "",
        "### Full-duration trial clusters",
        "",
        *arm_table(trials),
    ]

    clusters = load_clusters(HYP_DIR, trial_channels)
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
            *arm_table(score_trials(truncated, galleries)),
        ]

    text = "\n".join(lines)
    print(text)
    out = OUT_DIR / "diar-rename-once.md"
    out.write_text(text + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
