"""Would all-pairs-similar collapse wrongly fire on a TRUE 2-speaker channel?

`split_recovery.py` measured a clean separation for the collapse rule — split
solo channels have min pairwise cluster similarity 0.74–0.98, the corpus loops
0.01–0.18 — but every corpus loop has 3–7 speakers, and cross-speaker cluster
sims reach 0.95 there, so a genuinely-two-speaker channel (the common remote
channel in production) might have ALL clusters mutually over the bar and
collapse into one "speaker". This probe synthesizes what the corpus lacks:
every pair of loop participants per AMI meeting, masked with the meeting's own
crosstalk gate and mixed exactly like the loop channels (8 meetings × 3 pairs
= 24 channels), then estimate-mode diarization + the collapse test on each.

The verdict is the false-collapse count. DER (est turns vs collapsed turns,
against the pair's reference spans) prices what a false collapse costs.

    uv run --group eval eval/collapse_probe.py
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from common import OUT_DIR, read_pcm16
from rttm import Turn
from split_recovery import _pairwise

from stenograf.voiceprints import DEFAULT_THRESHOLD


def main() -> int:
    import ami
    from der import score_der
    from diarize import _build_diarizer

    probe_dir = ami.CHANNELS_DIR / "probe2"
    probe_dir.mkdir(parents=True, exist_ok=True)
    annotations = ami.RAW_DIR / "annotations"
    meetings_map = ami.parse_meetings_xml(annotations / "corpusResources" / "meetings.xml")
    estimator = _build_diarizer(sherpa_only=False)

    rows = []
    for group, sessions in ami.AMI_GROUPS.items():
        for session in sessions:
            meeting = group + session
            speakers: dict[str, tuple[np.ndarray, list[tuple[float, float]]]] = {}
            for agent, (headset, name) in meetings_map[meeting].items():
                wav = ami._ami_headset(meeting, headset)
                spans = ami.parse_words_xml(annotations / "words" / f"{meeting}.{agent}.words.xml")
                speakers[name] = (read_pcm16(wav), ami.merge_spans(spans))
            names = sorted(speakers)
            masks = dict(
                zip(names, ami.crosstalk_masks([speakers[n][0] for n in names]), strict=True)
            )
            others = names[1:]  # the loop participants (mic person is names[0])

            for a, b in combinations(others, 2):
                channel_id = f"{meeting}.{a}-{b}"
                wav_path = probe_dir / f"{channel_id}.wav"
                if not wav_path.exists():
                    ami._write_wav(
                        wav_path,
                        ami.mix_pcm(
                            ami.apply_mask(speakers[n][0], masks[n]) for n in (a, b)
                        ),
                    )
                ref = [Turn(n, s, e) for n in (a, b) for s, e in speakers[n][1]]

                result = estimator.diarize_with_embeddings(read_pcm16(wav_path), None)
                turns = [Turn(t.speaker, t.start, t.end) for t in result.turns]
                sims = _pairwise(result.embeddings)
                collapses = bool(sims) and min(sims) >= DEFAULT_THRESHOLD
                der_est = score_der(ref, turns).der
                der_collapsed = score_der(
                    ref, [Turn("S0", t.start, t.end) for t in turns]
                ).der
                rows.append(
                    (channel_id, len(result.embeddings), sims, collapses, der_est, der_collapsed)
                )
                print(
                    f"[{channel_id}] est k={len(result.embeddings)} "
                    f"minsim={min(sims, default=1.0):.2f} "
                    f"{'COLLAPSES' if collapses else 'kept'} "
                    f"DER {der_est:.1%} → collapsed {der_collapsed:.1%}",
                    flush=True,
                )

    collapsed = [r for r in rows if r[3]]
    lines = [
        "# Collapse probe — true 2-speaker channels vs the all-pairs rule",
        "",
        f"**{len(collapsed)} of {len(rows)} channels would falsely collapse** at",
        f"threshold {DEFAULT_THRESHOLD}.",
        "",
        "| Channel | est k | min sim | max sim | collapses | DER est | DER collapsed |",
        "|---|---|---|---|---|---|---|",
    ]
    for channel_id, est_k, sims, collapses, der_est, der_collapsed in rows:
        lines.append(
            f"| {channel_id} | {est_k} | {min(sims, default=1.0):.2f} "
            f"| {max(sims, default=1.0):.2f} | {'YES' if collapses else '—'} "
            f"| {der_est:.1%} | {der_collapsed:.1%} |"
        )
    report = OUT_DIR / "diar-collapse-probe.md"
    report.write_text("\n".join(lines) + "\n")
    print(f"\n{len(collapsed)}/{len(rows)} false collapses → {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
