"""Score backend hypotheses against hand-corrected references.

Usage:
    uv run --group eval eval/score.py

Reads eval/refs/<segment>.txt (verbatim reference, plain text) and every
eval/out/<backend>/<segment>.json, prints a WER/CER/speed table, and writes
eval/out/report.md.

Normalization before scoring: lowercase, punctuation stripped (umlauts and ß
kept), whitespace collapsed. References should be verbatim — write numbers the
way they were spoken ("dreiundzwanzig", not "23"), because no inverse-text
normalization is applied.
"""

from __future__ import annotations

import json
import re
import sys

import jiwer
from backends import BACKENDS
from common import OUT_DIR, load_manifest


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)  # \w is unicode-aware: äöüß survive
    text = re.sub(r"_", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def main() -> int:
    segments = [s for s in load_manifest() if s.ref_path.exists()]
    if not segments:
        print("no references in eval/refs/ yet — nothing to score", file=sys.stderr)
        return 1

    rows = []
    for segment in segments:
        reference = normalize(segment.ref_path.read_text())
        for backend in sorted(BACKENDS):
            hyp_path = segment.hyp_path(backend)
            if not hyp_path.exists():
                continue
            record = json.loads(hyp_path.read_text())
            hypothesis = normalize(record["text"])
            rows.append(
                {
                    "segment": segment.id,
                    "language": record["language"] or segment.language or "?",
                    "backend": backend,
                    "wer": jiwer.wer(reference, hypothesis),
                    "cer": jiwer.cer(reference, hypothesis),
                    "speed_x_rt": record["speed_x_rt"],
                    "peak_rss_mb": record["peak_rss_mb"],
                }
            )

    if not rows:
        print("references exist but no hypotheses — run transcribe.py first", file=sys.stderr)
        return 1

    lines = [
        "| Segment | Lang | Backend | WER | CER | Speed | Peak RSS |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['segment']} | {r['language']} | {r['backend']} "
            f"| {r['wer']:.1%} | {r['cer']:.1%} "
            f"| {r['speed_x_rt']}x RT | {r['peak_rss_mb']} MB |"
        )

    # Per-backend mean WER across scored segments — the headline number.
    lines += ["", "| Backend | Mean WER | Segments |", "|---|---|---|"]
    for backend in sorted({r["backend"] for r in rows}):
        scored = [r["wer"] for r in rows if r["backend"] == backend]
        lines.append(f"| {backend} | {sum(scored) / len(scored):.1%} | {len(scored)} |")

    report = "\n".join(lines) + "\n"
    print(report)
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "report.md").write_text(report)
    print(f"wrote {OUT_DIR / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
