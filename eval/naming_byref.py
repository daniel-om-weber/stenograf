"""Naming scored per REFERENCE speaker — the partitioner-independent metric.

`naming_trials`-style DIR/FAR is per hypothesis cluster, so the trial set is a
function of the partitioner (2026-08-07 review): fragmenting a known speaker
mints extra known trials, absorbing a stranger deletes their trial while doing
something strictly worse to the user. Here the denominators are reference
speech time, identical across arms: for every reference speaker of each
non-enroll loop channel, how much of their speech ends up under their own
name, under someone else's (the unrecoverable failure — wrong names poison
profiles and transcripts alike), or unnamed (recoverable: S-labels are
correctable after the meeting). Clusters are named production-style: best
gallery cosine at ``DEFAULT_THRESHOLD``, else unnamed. Mic channels are
excluded — their files are byte-identical across arms and only dilute.

Both weightings are published (2026-08-07 review): time-weighted flatters
every arm by letting talkative speakers dominate; the macro average (equal
weight per enrolled speaker) exposes the residual failure rate, and the
count of speakers >50% misnamed is the product-legible number. The three
time columns are not a strict partition: overlapping hypothesis turns can
count one reference span under two named clusters at once (rows sum to
~100.2%, not 100%), and ``unnamed`` clamps at zero — the ranking is
unaffected but the columns are rates, not shares.

Run (after the arm matrices)::

    uv run --group eval eval/naming_byref.py
"""

from __future__ import annotations

import json

import numpy as np
from common import OUT_DIR
from rttm import parse_rttm

ARMS = {
    "complete": "ami-loop",
    "ward": "ami-loop-ward",
    "ward-sv": "ami-loop-ward-sv",
    "ward-sv08": "ami-loop-ward-sv08",
    "ward-s3": "ami-loop-ward-s3",
    "ward-s2": "ami-loop-ward-s2",
    "nmesc": "ami-loop-nmesc",
}


def _overlap(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    return sum(
        max(0.0, min(e1, e2) - max(s1, s2)) for s1, e1 in a for s2, e2 in b
    )


def main() -> int:
    import ami

    from stenograf.diarization.sherpa import SherpaOnnxDiarizer
    from stenograf.voiceprints import DEFAULT_THRESHOLD

    embed = SherpaOnnxDiarizer().embed
    galleries = ami.build_galleries(embed)
    channels = [
        c
        for c in ami.load_channels()
        if c.session != ami.ENROLL_SESSION and c.num_speakers > 1 and c.group
    ]

    lines = [
        "## Naming per reference speaker (naming_byref.py)",
        "",
        "| arm | enrolled: own name (time / macro) | wrong name (time / macro) "
        "| unnamed | stranger: wrong name | speakers >50% misnamed |",
        "|---|---|---|---|---|---|",
    ]
    details: dict[str, list[str]] = {}
    for arm, name in ARMS.items():
        out_dir = OUT_DIR / "diar" / name
        own = wrong = unnamed = 0.0
        stranger_total = stranger_wrong = 0.0
        enrolled_total = 0.0
        macro_own: list[float] = []
        macro_wrong: list[float] = []
        rows: list[str] = []
        for channel in channels:
            gallery = galleries.get(channel.group or "") or {}
            if not gallery:
                continue
            embeddings = {
                k: np.asarray(v, dtype=np.float32)
                for k, v in json.loads((out_dir / f"{channel.id}.emb.json").read_text()).items()
            }
            hyp: dict[str, list[tuple[float, float]]] = {}
            for t in parse_rttm(out_dir / f"{channel.id}.rttm"):
                hyp.setdefault(t.speaker, []).append((t.start, t.end))
            named: dict[str, str] = {}
            for label in hyp:
                vector = embeddings.get(label)
                if vector is None:
                    continue
                scores = {n: float(vector @ e) for n, e in gallery.items()}
                best = max(scores, key=lambda n: scores[n])
                if scores[best] >= DEFAULT_THRESHOLD:
                    named[label] = best

            ref: dict[str, list[tuple[float, float]]] = {}
            for t in parse_rttm(channel.ref_path):
                ref.setdefault(t.speaker, []).append((t.start, t.end))
            for speaker, spans in sorted(ref.items()):
                total = sum(e - s for s, e in spans)
                right = sum(
                    _overlap(hyp[label], spans)
                    for label, n in named.items()
                    if n == speaker
                )
                misnamed = sum(
                    _overlap(hyp[label], spans)
                    for label, n in named.items()
                    if n != speaker
                )
                if speaker in gallery:
                    enrolled_total += total
                    own += right
                    wrong += misnamed
                    unnamed += max(0.0, total - right - misnamed)
                    macro_own.append(right / total)
                    macro_wrong.append(misnamed / total)
                    if misnamed > 0.05 * total:
                        rows.append(
                            f"| {channel.id} | {speaker} | {misnamed / total:.0%} misnamed |"
                        )
                else:
                    stranger_total += total
                    stranger_wrong += misnamed
                    if misnamed > 0.05 * total:
                        rows.append(
                            f"| {channel.id} | {speaker} (stranger) "
                            f"| {misnamed / total:.0%} misnamed |"
                        )
        catastrophic = sum(1 for w in macro_wrong if w > 0.5)
        lines.append(
            f"| {arm} | {own / enrolled_total:.1%} / {float(np.mean(macro_own)):.1%} "
            f"| {wrong / enrolled_total:.1%} / {float(np.mean(macro_wrong)):.1%} "
            f"| {unnamed / enrolled_total:.1%} | {stranger_wrong / stranger_total:.1%} "
            f"| {catastrophic} |"
        )
        details[arm] = rows

    for arm, rows in details.items():
        if rows:
            lines += ["", f"### {arm}: speakers >5% misnamed", ""]
            lines += ["| channel | speaker | misnamed |", "|---|---|---|"]
            lines += rows
    text = "\n".join(lines)
    print(text)
    (OUT_DIR / "diar-naming-byref.md").write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
