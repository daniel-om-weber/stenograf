"""Does a boundary margin on diarization turns improve word attribution?

TST-Bench measured +0.26 DIR from padding turn onsets/offsets ~0.1 s before
word intersection — on systems that drop or misattribute words falling outside
every turn. stenograf's ``merge_words_turns`` never drops a word: an uncovered
midpoint snaps to the *nearest* turn, which already splits every inter-turn
gap at its center, and symmetric padding clamped at neighbors cannot move a
center split. If that argument is right, the sweep is flat; if it is wrong
(overlapping turns, words spanning three turns, and clamp asymmetries are the
candidate leaks), the sweep says by how much.

Turns come from the matrix's ``<id>.rttm`` hypotheses and word times from its
``<id>.words.json``, so nothing re-runs diarization or ASR — only the padding
and the word→turn merge differ between arms, on the multi-speaker (loop)
channels where attribution is non-trivial. Run ``eval/ami.py run`` first.

    uv run --group eval eval/margin_sweep.py [--segments ES2003a.loop,...]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from bisect import bisect_left, bisect_right
from pathlib import Path

from common import OUT_DIR
from der import Word, score_attribution, score_der
from rttm import Turn, parse_rttm

from stenograf.asr.base import Word as PipelineWord
from stenograf.pipeline import merge_words_turns

MARGINS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25)


def pad_turns(turns: list[Turn], margin: float) -> list[Turn]:
    """Pad every onset/offset by ``margin``, clamped at neighboring turns.

    A turn's onset moves earlier by ``margin`` but never past the latest
    neighboring offset at or before it (and never below 0); its offset moves
    later but never past the earliest neighboring onset at or after it. Turns
    that already overlap are left overlapping — the clamp only stops padding
    from swallowing a neighbor, it never shrinks a turn.
    """
    if margin == 0.0:
        return list(turns)
    starts = sorted(t.start for t in turns)
    ends = sorted(t.end for t in turns)
    padded = []
    for t in turns:
        i = bisect_right(ends, t.start)
        prev_end = ends[i - 1] if i else 0.0
        j = bisect_left(starts, t.end)
        next_start = starts[j] if j < len(starts) else math.inf
        padded.append(
            Turn(
                speaker=t.speaker,
                start=max(t.start - margin, min(prev_end, t.start), 0.0),
                end=min(t.end + margin, max(next_start, t.end)),
            )
        )
    return padded


def _load_word_times(path: Path) -> tuple[list[PipelineWord], list[str]]:
    """Word times plus the matrix's own per-word labels (the parity check).

    Sorted by start (stably, matching the merge's own ordering) so the labels
    ``_labels`` returns line up index-for-index with the words passed in."""
    record = json.loads(path.read_text())
    rows = sorted(record["words"], key=lambda w: w["start"])
    words = [PipelineWord(text=w["text"], start=w["start"], end=w["end"]) for w in rows]
    return words, [w["speaker"] for w in rows]


def _labels(words: list[PipelineWord], turns: list[Turn]) -> list[str]:
    """Per-word speaker labels from the real pipeline merge, in word-time order."""
    entries = merge_words_turns(list(words), list(turns))
    return [entry.speaker for entry in entries for _ in entry.words]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", help="comma-separated channel ids (default: every loop one)")
    args = parser.parse_args()

    import ami

    wanted = set(args.segments.split(",")) if args.segments else None
    channels = [
        c
        for c in ami.load_channels()
        if c.num_speakers > 1 and (wanted is None or c.id in wanted)
    ]
    if not channels:
        raise SystemExit("no multi-speaker corpus channels — run `eval/ami.py fetch` first")

    hyp_dir = OUT_DIR / "diar" / "ami"
    attr_rows, der_rows = [], []
    for channel in channels:
        words_path = hyp_dir / f"{channel.id}.words.json"
        turns_path = hyp_dir / f"{channel.id}.rttm"
        if not (words_path.exists() and turns_path.exists()):
            print(f"  no matrix output for {channel.id} — run eval/ami.py run", file=sys.stderr)
            continue
        ref = parse_rttm(channel.ref_path)
        turns = parse_rttm(turns_path)
        words, matrix_labels = _load_word_times(words_path)

        baseline = _labels(words, turns)
        if baseline != matrix_labels:
            diffs = sum(a != b for a, b in zip(baseline, matrix_labels, strict=True))
            print(
                f"  {channel.id}: offline merge differs from the matrix on {diffs} words "
                "— stale matrix output?",
                file=sys.stderr,
            )

        accs, ders, changed = [], [], []
        for margin in MARGINS:
            padded = pad_turns(turns, margin)
            labels = _labels(words, padded)
            scored = score_attribution(
                [Word(w.text, w.start, w.end, s) for w, s in zip(words, labels, strict=True)],
                ref,
            )
            accs.append(scored.accuracy)
            ders.append(score_der(ref, padded).der)
            changed.append(sum(a != b for a, b in zip(labels, baseline, strict=True)))
        attr_rows.append((channel.id, accs, changed))
        der_rows.append((channel.id, ders))
        print(
            f"[{channel.id}] "
            + "  ".join(
                f"{m:.2f}s: {a:.1%} ({c} moved)"
                for m, a, c in zip(MARGINS, accs, changed, strict=True)
            ),
            flush=True,
        )

    header = " | ".join(f"{m:.2f}s" for m in MARGINS)
    lines = [
        "# Boundary-margin sweep — pad turns before word intersection",
        "",
        "Word-attribution accuracy per margin (padded turns, real merge); *moved* =",
        "words whose speaker changed vs margin 0. DER is the padded turns scored as",
        "the hypothesis (0.25 s collar), informational only — the pipeline's turn",
        "output would not carry the pad.",
        "",
        f"| Channel | {header} | moved (max) |",
        "|---" * (len(MARGINS) + 2) + "|",
    ]
    for channel_id, accs, changed in attr_rows:
        cells = " | ".join(f"{a:.2%}" for a in accs)
        lines.append(f"| {channel_id} | {cells} | {max(changed)} |")
    if attr_rows:
        means = [sum(r[1][k] for r in attr_rows) / len(attr_rows) for k in range(len(MARGINS))]
        lines.append(
            "| **mean** | " + " | ".join(f"{a:.2%}" for a in means)
            + f" | {sum(max(r[2]) for r in attr_rows)} |"
        )
    lines += ["", f"| Channel (DER) | {header} |", "|---" * (len(MARGINS) + 1) + "|"]
    for channel_id, ders in der_rows:
        lines.append(f"| {channel_id} | " + " | ".join(f"{d:.1%}" for d in ders) + " |")

    report = OUT_DIR / "diar-margin-sweep.md"
    report.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
