"""Does the model's own confidence separate a real fix from a false insertion?

The whole argument for a confidence gate on ``stenograf.glossary`` rests on one
empirical claim, and this script is the cheapest possible test of it. The claim:

    When post-correction rewrites a word it should NOT have (``pound`` → ``compound``),
    the model was *confident* about that word — it heard it perfectly well. When it
    rewrites a word it SHOULD (a genuine misrecognition), the model was *unsure*.

If that holds, a gate on ``Word.confidence`` lets us drop the similarity threshold back
toward 0.82 — recovering the recall we surrendered when the over-correction disaster
forced it to 0.95 — while refusing the rewrites that caused the disaster. If it does
not hold (RNN-T greedy confidences are notoriously overconfident, so it may not), the
design is dead and we have spent one decode pass learning that instead of a refactor.

**This measures, it does not gate.** Nothing here changes shipped behaviour: it decodes
the German tier exactly as stenograf ships it (boost = 1.0), applies post-correction at
the *old* 0.82 threshold, and asks what the model believed about each word it rewrote.

**How a rewrite is classified.** Per utterance, against the normalized reference:

    original word in the reference, replacement not  → FALSE INSERTION (we broke it)
    replacement in the reference, original not       → TRUE FIX      (we mended it)
    anything else                                    → ambiguous, excluded

That is a bag-of-words test, not an alignment: a word repeated in an utterance can be
misjudged. It is the right trade for a diagnostic — it is unbiased between the two
classes (nothing about the rule favours one), and the alternative costs an alignment
pass to sharpen a number that only has to be *directionally* true to justify building
the real thing. Every excluded case is counted and printed, so the discard rate is
visible rather than assumed.

**VERDICT: the gate is dead. Do not build it.** (Measured 2026-07-13, German, 500 utts,
boost 1.0, threshold 0.82.) The claim is *directionally* true and *practically* useless:

    false insertions  n=53   median confidence 0.999   p10 0.938
    true fixes        n=165  median confidence 0.951   p10 0.858

The words we wrongly overwrite really are the ones the model was surer about — and the
distributions overlap so heavily that no threshold separates them. The best operating
point (c = 0.95) blocks 45 of 53 false insertions and destroys **84 of 165 true fixes**;
every looser point is worse (c = 0.90 → 98 % blocked, 81 % of fixes lost). ``der`` →
``deri`` (a false insertion) scored 1.000; ``finde`` → ``find`` (a real fix) scored
0.997. They are not distinguishable, because the entropy-normalized confidence of a
greedy RNN-T saturates at ~1.0 for nearly everything.

**Why it failed, and what would not.** The gate asks *"how sure was the model about the
word it wrote?"* — a question whose answer is almost always "certain". The discriminative
question is the other one: *"how much does the model dislike the word we want to put
there?"* — i.e. score the **candidate term** over that span under the model and compare it
against the tokens actually emitted. That is what FluidAudio approximates with a second
CTC model (badly, with a weak English-only 110M — see ``bias_fluid.py``), and what we
could do properly with the same TDT that produced the transcript, via forced alignment
through the joint. It is real work, and it is unproven; this script is kept as the thing
that will price it, and as the record of why the cheap version does not work.

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
