"""Clustering A/B on the owned loop: every partitioner arm, full gate.

`PLAN-DIARIZATION.md` step 4 Phase B: the known-count partitioner is the one
piece production actually exercises (estimate mode belongs to stenodiar), and
the harness's residual error is concentrated in cluster *confusion* (IS1009,
TS3010a/c/d — impure clusters that also poison naming and enrollment). Each
arm is a full 40-channel own-loop matrix (`diarize.py --own-loop --cluster
<m>`); this scores them: loop DER decomposed (the confusion component is the
target), word attribution, mic attribution, and same-group naming trials
under the shipped galleries (identical across arms — only the partitions
differ, so naming deltas are purely clustering quality).

Gate: a candidate must beat `complete` (the parity-verified reference
behavior) by a visible margin to ship; the confusion channels are where to
look. Run after the arm matrices exist::

    uv run --group eval eval/cluster_ab.py
"""

from __future__ import annotations

import sys

import numpy as np
from common import OUT_DIR
from der import _load_words, score_attribution, score_der
from embedder_ab import naming_trials
from reid_score import dir_at_far, operating_point
from rttm import parse_rttm

ARMS = {
    "complete": "ami-loop",
    "average": "ami-loop-average",
    "nmesc": "ami-loop-nmesc",
    "ward": "ami-loop-ward",
    "ward-sv": "ami-loop-ward-sv",
    "ward-sv08": "ami-loop-ward-sv08",
    "ward-s3": "ami-loop-ward-s3",
    "ward-s2": "ami-loop-ward-s2",
}

FOCUS = (
    "IS1009a.loop",
    "IS1009b.loop",
    "IS1009d.loop",
    "TS3010a.loop",
    "TS3010c.loop",
    "TS3010d.loop",
    "ES2003d.loop",
    "Bmr030.loop",
    "Bmr024.loop",
    "Bmr025.loop",
    "ES2007d.loop",
)
"""The confusion-heavy channels the program's residual error lives on, PLUS
every channel where a candidate arm regresses hard (Bmr024/Bmr025/ES2007d are
ward's losses) — a focus table that only shows a candidate's wins misleads
(2026-08-07 review)."""


def main() -> int:
    import ami

    from stenograf.diarization.sherpa import SherpaOnnxDiarizer
    from stenograf.voiceprints import DEFAULT_THRESHOLD

    channels = ami.load_channels()
    missing = [
        (arm, c.id)
        for arm, name in ARMS.items()
        for c in channels
        if not (OUT_DIR / "diar" / name / f"{c.id}.rttm").exists()
    ]
    if missing:
        print(f"incomplete arms (first: {missing[0]}) — run the arm matrices", file=sys.stderr)
        return 1

    embed = SherpaOnnxDiarizer().embed
    summary = [
        "| arm | loop DER | miss | FA | confusion | loop attribution | mic attribution "
        f"| DIR@FAR0 full (thr) | DIR/FAR/FRR @{DEFAULT_THRESHOLD:g} |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    focus_rows = [
        "| channel | " + " | ".join(f"{arm} DER (conf)" for arm in ARMS) + " | "
        + " | ".join(f"{arm} attr" for arm in ARMS) + " |",
        "|---|" + "---|" * (2 * len(ARMS)),
    ]
    focus_data: dict[str, list[str]] = {c: [] for c in FOCUS}

    for arm, name in ARMS.items():
        out_dir = OUT_DIR / "diar" / name
        ders, misses, fas, confs, attrs, mic_attrs = [], [], [], [], [], []
        per_channel: dict[str, tuple[float, float, float]] = {}
        for channel in channels:
            ref = parse_rttm(channel.ref_path)
            words = _load_words(out_dir / f"{channel.id}.words.json")
            attribution = score_attribution(words, ref).accuracy
            if channel.num_speakers > 1:
                score = score_der(ref, parse_rttm(out_dir / f"{channel.id}.rttm"))
                ders.append(score.der)
                misses.append(score.missed)
                fas.append(score.false_alarm)
                confs.append(score.confusion)
                attrs.append(attribution)
                per_channel[channel.id] = (score.der, score.confusion, attribution)
            else:
                mic_attrs.append(attribution)
        trials = naming_trials(out_dir, embed)
        strict = dir_at_far(trials[None], 0.0)
        shipped = operating_point(trials[None], DEFAULT_THRESHOLD)
        summary.append(
            f"| {arm} | {np.mean(ders):.1%} | {np.mean(misses):.1%} | {np.mean(fas):.1%} "
            f"| {np.mean(confs):.1%} | {np.mean(attrs):.1%} | {np.mean(mic_attrs):.1%} "
            f"| {strict.dir_:.1%} ({strict.threshold:.3f}) "
            f"| {shipped.dir_:.1%} / {shipped.far:.1%} / {shipped.frr:.1%} |"
        )
        for cid in FOCUS:
            der, conf, attr = per_channel[cid]
            focus_data[cid].append((f"{der:.1%} ({conf:.1%})", f"{attr:.1%}"))

    for cid in FOCUS:
        ders = " | ".join(d for d, _ in focus_data[cid])
        attrs = " | ".join(a for _, a in focus_data[cid])
        focus_rows.append(f"| {cid} | {ders} | {attrs} |")

    lines = [
        "## Clustering A/B on the owned loop (cluster_ab.py)",
        "",
        *summary,
        "",
        "### Confusion-heavy channels",
        "",
        *focus_rows,
    ]
    text = "\n".join(lines)
    print(text)
    out = OUT_DIR / "diar-cluster-ab.md"
    out.write_text(text + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
