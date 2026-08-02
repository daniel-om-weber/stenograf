"""Does the model's own confidence separate a real fix from a false insertion?

Measures — never gates: decodes the German tier exactly as shipped (boost = 1.0),
applies post-correction at the old 0.82 threshold, and asks what the model believed
about each word it rewrote. Rewrites are classified per utterance against the
normalized reference (original in reference, replacement not → false insertion;
the converse → true fix; anything else excluded and counted) — a bag-of-words
diagnostic, unbiased between the two classes, deliberately cheaper than alignment.

**Verdict: the gate is dead; do not build it.** The measured distributions, why
they cannot carry a threshold, and the follow-up question that *would* be
discriminative are in **eval/README.md** ("The confidence gate we did NOT build").
This script is kept as the thing that would price that follow-up.

Usage:
    uv run --group eval eval/bias_confidence.py            # 500-utt German subsample
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import bias_score  # noqa: E402
from bias import SEED, SUBSAMPLE, Biaser, Config, load_audio, ref_path, subsample  # noqa: E402

from stenograf.asr import create_backend  # noqa: E402
from stenograf.glossary import _norm, apply_glossary  # noqa: E402
from stenograf.transcript import TranscriptEntry  # noqa: E402

OLD_THRESHOLD = 0.82
"""The threshold that caused the damage. The gate has to earn its keep *here* — at 0.95
there is barely any damage left to prevent, and barely any recall left to win back."""


def classify(original: str, replacement: str, ref_words: set[str]) -> str | None:
    was_right = _norm(original) in ref_words
    now_right = _norm(replacement) in ref_words
    if was_right and not now_right:
        return "false insertion"
    if now_right and not was_right:
        return "true fix"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default="parakeet")
    parser.add_argument("--limit", type=int, default=SUBSAMPLE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--threshold", type=float, default=OLD_THRESHOLD)
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    refs_all = bias_score.read_refs(ref_path("german", args.n))
    uttids = subsample(refs_all, args.limit, args.seed, "german")
    audio = load_audio("german", set(uttids))

    backend = create_backend(args.backend)
    backend.load()
    # Decode exactly as we ship: the decoder biased at boost = 1.0. The gate must be
    # judged on the transcript the user actually gets, not on an unbiased one.
    biaser = Biaser(backend, Config(alpha=1.0))

    buckets: dict[str, list[float]] = {"false insertion": [], "true fix": []}
    ambiguous = no_confidence = 0
    examples: dict[str, list[str]] = {"false insertion": [], "true fix": []}

    for index, uttid in enumerate(sorted(uttids), start=1):
        ref = refs_all[uttid]
        ref_words = {_norm(w) for w in ref.text.split()}
        biaser.arm(list(ref.biasing_words))
        segments = backend.transcribe(audio[uttid], None)

        for segment in segments:
            if not segment.words:
                continue
            entry = TranscriptEntry(
                speaker="",
                text=segment.text,
                start=segment.start,
                end=segment.end,
                words=segment.words,
            )
            out = apply_glossary(
                [entry], glossary=list(ref.biasing_words), threshold=args.threshold
            )[0]
            for before, after in zip(entry.words, out.words, strict=True):
                if before.text == after.text:
                    continue
                verdict = classify(before.text, after.text, ref_words)
                if verdict is None:
                    ambiguous += 1
                    continue
                if before.confidence is None:
                    no_confidence += 1
                    continue
                buckets[verdict].append(before.confidence)
                if len(examples[verdict]) < 6:
                    examples[verdict].append(
                        f"{before.text} → {after.text} (conf {before.confidence:.3f})"
                    )
        if index % 100 == 0:
            print(f"  {index}/{len(uttids)}", flush=True)

    biaser._restore()
    backend.unload()

    print(f"\npost-correction at threshold {args.threshold:g}, boost=1.0, {len(uttids)} utts")
    print(f"ambiguous rewrites excluded: {ambiguous}")
    if no_confidence:
        print(f"WARNING: {no_confidence} rewrites had no confidence — backend reports none")

    if not buckets["false insertion"] or not buckets["true fix"]:
        print("\nNot enough of one class to compare — the diagnostic cannot answer.")
        return 1

    print()
    for verdict, values in buckets.items():
        values.sort()
        print(
            f"{verdict:<16} n={len(values):<5} "
            f"median={statistics.median(values):.3f}  mean={statistics.fmean(values):.3f}  "
            f"p10={values[len(values) // 10]:.3f}  p90={values[len(values) * 9 // 10]:.3f}"
        )
    for verdict, shown in examples.items():
        print(f"\n{verdict} examples:")
        for line in shown:
            print(f"  {line}")

    # The decision the gate would make: "refuse to correct a word the model was at least
    # this sure about". A useful gate blocks most false insertions while sparing most
    # true fixes; if no threshold does both, the confidence signal cannot carry a gate.
    print("\ngate: refuse to rewrite a word whose confidence is >= c")
    print(f"{'c':>6} {'FI blocked':>12} {'fixes lost':>12}")
    fi, tf = buckets["false insertion"], buckets["true fix"]
    for c in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
        blocked = sum(1 for v in fi if v >= c) / len(fi)
        lost = sum(1 for v in tf if v >= c) / len(tf)
        print(f"{c:>6.2f} {blocked:>11.0%} {lost:>12.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
