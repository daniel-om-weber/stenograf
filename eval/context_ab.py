"""Two automated follow-ups to the window-length study — no human judging.

Experiment 1 — context A/B (the causal test). Every short window (<8 s) from
the ``windows.py`` decode is re-decoded by the SAME Parakeet twice with more
left context, keeping only the words inside the original span:

- B1 "raw":    the contiguous preceding ``CTX_RAW_S`` of channel audio
               (often silence on a mic channel — tests whether mere audio
               length matters);
- B2 "splice": the last ``CTX_PREV_S`` of the *previous decode window*
               spliced on (real speech across a long gap — what a
               context-carry fix in pack_windows would actually ship).

Same model + same audio, only context differs, so A≠B is windowing-caused by
construction — the intrinsically-hard-audio confound that model juries can't
escape doesn't apply. Where a variant changes the text, the Whisper pivot
referees textually (no new Whisper decode): both variants come from the same
Parakeet, so style-anchor bias mostly cancels and the referee only breaks the
tie. Long windows (≥8 s) are sampled as the control arm — context should
change little there.

Experiment 3 — tail-jitter probe (added with cut-overlap decoding). The
greedy TDT decode is knife-edge unstable at the slice tail: the same span
decodes completely or drops ~10 trailing words on a millisecond-level bound
shift. For every *cut-ended* window (``cut_end`` in the windows.py record)
the probe decodes the slice three times with the end bound shifted ±50 ms
and counts the kept words, in two arms: "bare" = the pre-fix slice (ends at
the window end, no right clip) and "shipped" = the product's cut-overlap
decode (OVERHANG_S past the cut, words beyond the cut dropped, plus the
speech-coverage skip retry — an overhang-only arm measured WORSE than bare:
the overhang re-rolls the knife-edge rather than removing it, and a skip
can land inside the kept region). The kept-word-count range across the
three decodes is the instability measure — expect ≈0 for the shipped arm.

Experiment 2 — VAD-drop check. Disagreement sites covered by NO decode window
("none" in window-report.md) are spans where the pivot heard words but the
pipeline decoded nothing. Decoding each span (±0.5 s pad) with Parakeet
directly — no VAD gate — classifies them: Parakeet hears similar words →
real speech the VAD dropped; hears nothing → pivot hallucination; hears
something else → disputed. Confirmed drops shorter than ~0.3 s point at the
``min_speech = 0.25`` floor specifically.

Prerequisites: windows.py + transcribe.py --backend whisper have run.
Output: eval/out/context-ab.md (+ stdout). Decode cost: a few minutes.

Usage:
    uv run --group eval eval/context_ab.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import jiwer
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from adjudicate import PIVOT, WINDOWED, bucket_of, build_sites, load_windows  # noqa: E402
from common import OUT_DIR, load_manifest  # noqa: E402
from score import normalize  # noqa: E402
from windows import read_mono16k  # noqa: E402

from stenograf.asr import create_backend  # noqa: E402
from stenograf.audio import SAMPLE_RATE, sample_index  # noqa: E402
from stenograf.vad import (  # noqa: E402
    SPEECH_HOLE_S,
    SpeechSegment,
    Window,
    bare_slice,
    decode_slice,
    speech_hole,
)

CTX_RAW_S = 15.0
CTX_PREV_S = 10.0
SHORT_S = 8.0  # A/B every window below this; sample above as control
CONTROL_PER_SEGMENT = 8
PAD_NONE_S = 0.5
MATCH_WER = 0.6  # exp 2: hyp-vs-pivot WER at or below this = "same speech"
JITTER_S = 0.05  # exp 3: slice-end bound shift (±)
SEED = 20260719


def transcribe_words(asr, clip: np.ndarray, language, t0: float) -> list[dict]:
    """Decode a clip; word times shifted to absolute segment seconds."""
    return [
        {"text": w.text, "start": w.start + t0, "end": w.end + t0}
        for seg in asr.transcribe(clip, language)
        for w in seg.words
    ]


def words_in(words: list[dict], start: float, end: float) -> list[dict]:
    return [w for w in words if start - 0.05 <= (w["start"] + w["end"]) / 2 <= end + 0.05]


def text_of(words: list[dict]) -> str:
    return normalize(" ".join(w["text"] for w in words))


def referee(a: str, b: str, ref: str) -> str:
    """Which variant the pivot sides with: 'A', 'B', 'tie', or 'no-ref'."""
    if not ref:
        return "no-ref"
    wer_a = jiwer.wer(ref, a) if a else 1.0
    wer_b = jiwer.wer(ref, b) if b else 1.0
    if wer_a == wer_b:
        return "tie"
    return "B" if wer_b < wer_a else "A"


class Arm:
    """One context variant's tally within a bucket."""

    def __init__(self) -> None:
        self.n = 0
        self.changed = 0
        self.verdicts = {"A": 0, "B": 0, "tie": 0, "no-ref": 0}

    def row(self) -> str:
        pct = f"{self.changed / self.n:.0%}" if self.n else "—"
        return (
            f"{self.n} | {self.changed} ({pct}) | {self.verdicts['B']} "
            f"| {self.verdicts['A']} | {self.verdicts['tie']} | {self.verdicts['no-ref']}"
        )


def run_context_ab(asr, segments, rng) -> tuple[list[str], list[dict]]:
    tallies: dict[tuple[str, str], Arm] = {}
    examples: list[dict] = []
    for segment, samples, record, whisper_words in segments:
        spans = [(w["start"], w["end"]) for w in record["windows"]]
        rec_words = [w for seg in record["segments"] for w in seg["words"]]
        short_idx = [i for i, (a, b) in enumerate(spans) if b - a < SHORT_S]
        long_idx = [i for i, (a, b) in enumerate(spans) if b - a >= SHORT_S]
        control = sorted(rng.sample(long_idx, min(CONTROL_PER_SEGMENT, len(long_idx))))
        for i in short_idx + control:
            start, end = spans[i]
            bucket = "control" if i in control else bucket_of(end - start)
            a_text = text_of(words_in(rec_words, start, end))
            ref = text_of(words_in(whisper_words, start, end))
            arms: list[tuple[str, np.ndarray, float]] = []
            ctx_a = max(0.0, start - CTX_RAW_S)
            if ctx_a < start:
                arms.append(("raw", samples[sample_index(ctx_a) : sample_index(end)], ctx_a))
            if i > 0:
                p_start, p_end = spans[i - 1]
                lo = max(p_start, p_end - CTX_PREV_S)
                ctx = samples[sample_index(lo) : sample_index(p_end)]
                clip = np.concatenate([ctx, samples[sample_index(start) : sample_index(end)]])
                arms.append(("splice", clip, start - len(ctx) / 16000.0))
            for arm_name, clip, t0 in arms:
                b_text = text_of(words_in(transcribe_words(asr, clip, None, t0), start, end))
                arm = tallies.setdefault((bucket, arm_name), Arm())
                arm.n += 1
                if b_text == a_text:
                    continue
                arm.changed += 1
                verdict = referee(a_text, b_text, ref)
                arm.verdicts[verdict] += 1
                if verdict in ("A", "B") and len(examples) < 24:
                    examples.append(
                        {
                            "segment": segment.id,
                            "t": f"{start:.0f}s ({end - start:.1f}s window, {arm_name})",
                            "A": a_text,
                            "B": b_text,
                            "ref": ref,
                            "verdict": verdict,
                        }
                    )
    lines = [
        "## Experiment 1 — context A/B on the same model",
        "",
        "A = the window as the product decodes it; B = same window with left",
        "context added. Referee = Whisper pivot text over the same span.",
        "",
        "| Bucket | context | windows | B≠A | pivot sides B | sides A | tie | no-ref |",
        "|---|---|---|---|---|---|---|---|",
    ]
    order = {"<3s": 0, "3-8s": 1, "control": 2}
    for (bucket, arm_name), arm in sorted(
        tallies.items(), key=lambda kv: (order.get(kv[0][0], 9), kv[0][1])
    ):
        lines.append(f"| {bucket} | {arm_name} | {arm.row()} |")
    return lines, examples


def run_jitter(asr, segments) -> list[str]:
    header = ["## Experiment 3 — tail-jitter probe on cut-ended windows", ""]
    tallies: dict[str, list[int]] = {"bare": [], "shipped": []}
    retries = holes_left = 0
    for _segment, samples, record, _ in segments:
        wins = record["windows"]
        if wins and "cut_end" not in wins[0]:
            return header + [
                "windows.py records predate cut classification — re-run windows.py --force."
            ]
        duration = len(samples) / SAMPLE_RATE
        for wrec in wins:
            if wrec["cut_end"] is None:
                continue
            win = Window(
                wrec["start"],
                wrec["end"],
                wrec["cut_start"],
                wrec["cut_end"],
                speech=tuple(SpeechSegment(s["start"], s["end"]) for s in wrec["speech"]),
            )
            ctx, hi, keep_lo, keep_hi = decode_slice(win)
            bare_ctx, bare_end = bare_slice(win)

            def kept(lo_t, base_end, hi_bound, *, _s=samples, _d=duration, _lo=keep_lo):
                clip = _s[sample_index(lo_t) : sample_index(min(_d, base_end))]
                return [
                    w
                    for w in transcribe_words(asr, clip, None, lo_t)
                    if _lo <= (w["start"] + w["end"]) / 2 < hi_bound
                ]

            def hole_of(words, _win=win):
                return speech_hole([(w["start"], w["end"]) for w in words], _win.speech)

            bare_counts, shipped_counts = [], []
            for shift in (-JITTER_S, 0.0, JITTER_S):
                bare_counts.append(len(kept(bare_ctx, bare_end + shift, math.inf)))
                # The shipped path (pipeline._decode_one): overlap decode +
                # skip retry on the pre-change slice, better coverage wins.
                words = kept(ctx, hi + shift, keep_hi)
                hole = hole_of(words)
                if hole > SPEECH_HOLE_S:
                    retries += 1
                    retry = kept(bare_ctx, bare_end + shift, keep_hi)
                    if hole_of(retry) < hole:
                        words = retry
                    if hole_of(words) > SPEECH_HOLE_S:
                        holes_left += 1
                shipped_counts.append(len(words))
            tallies["bare"].append(max(bare_counts) - min(bare_counts))
            tallies["shipped"].append(max(shipped_counts) - min(shipped_counts))
    lines = header + [
        "Every cut-ended window decoded 3× with the window-end bound shifted ±50 ms;",
        "the kept-word-count range across the three decodes measures tail",
        "instability (0 = the bound shift changes nothing). bare = the pre-fix",
        "slice; shipped = overlap + speech-coverage skip retry (the product decode).",
        "",
        "| arm | windows | unstable (range>0) | Σ range | max range |",
        "|---|---|---|---|---|",
    ]
    for arm, ranges in tallies.items():
        unstable = sum(1 for r in ranges if r > 0)
        worst = max(ranges, default=0)
        lines.append(f"| {arm} | {len(ranges)} | {unstable} | {sum(ranges)} | {worst} |")
    lines.append("")
    lines.append(
        f"Shipped arm: {retries} tail-hole retries across {len(tallies['shipped']) * 3} decodes; "
        f"{holes_left} holes left after retry."
    )
    if not tallies["bare"]:
        lines.append("No cut-ended windows in any segment — nothing to probe.")
    return lines


def run_vad_drop(asr, segments) -> list[str]:
    counts = {"dropped-speech": 0, "hallucination": 0, "disputed": 0}
    tiny = 0  # confirmed drops short enough to blame min_speech
    dropped_s = 0.0
    examples: list[str] = []
    for segment, samples, record, _ in segments:
        spans = [(w["start"], w["end"]) for w in record["windows"]]
        sites = build_sites(segment.id, [WINDOWED, PIVOT], spans)
        for site in sites:
            if site["win_len"] is not None:
                continue
            a, b = site["span"]
            clip = samples[sample_index(max(0.0, a - PAD_NONE_S)) : sample_index(b + PAD_NONE_S)]
            if len(clip) == 0:
                continue
            hyp = text_of(transcribe_words(asr, clip, None, 0.0))
            pivot_text = normalize(
                next(v["text"] for v in site["variants"] if PIVOT in v["models"])
            )
            if not hyp:
                kind = "hallucination"
            elif pivot_text and jiwer.wer(pivot_text, hyp) <= MATCH_WER:
                kind = "dropped-speech"
            else:
                kind = "disputed"
            counts[kind] += 1
            if kind == "dropped-speech":
                dropped_s += b - a
                if b - a < 0.3:
                    tiny += 1
            if kind != "hallucination" and len(examples) < 15:
                examples.append(
                    f"- `{segment.id}` @ {a:.0f}s ({b - a:.1f}s, {kind}): "
                    f"pipeline=∅, parakeet-no-vad=“{hyp}”, pivot=“{pivot_text}”"
                )
    lines = [
        "## Experiment 2 — the 'no window' sites (VAD drops vs pivot hallucination)",
        "",
        "Each pivot-only span decoded by Parakeet directly, no VAD gate:",
        "",
        f"- **{counts['dropped-speech']} confirmed VAD drops** — real speech the "
        f"pipeline never transcribed ({dropped_s:.1f}s total; {tiny} under 0.3s, "
        "i.e. the min_speech=0.25 floor)",
        f"- {counts['hallucination']} pivot hallucinations (Parakeet hears nothing there)",
        f"- {counts['disputed']} disputed (both hear speech, different words)",
        "",
        "Examples (non-hallucination):",
        *examples,
    ]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", help="comma-separated segment ids (default: all)")
    args = parser.parse_args()
    wanted = set(args.segments.split(",")) if args.segments else None

    loaded = []
    for segment in load_manifest():
        if wanted is not None and segment.id not in wanted:
            continue
        windowed = load_windows(segment.id)
        whisper_path = OUT_DIR / PIVOT / f"{segment.id}.json"
        if windowed is None or not whisper_path.exists() or not segment.wav_path.exists():
            continue
        record = json.loads((OUT_DIR / WINDOWED / f"{segment.id}.json").read_text())
        whisper_words = [
            w for seg in json.loads(whisper_path.read_text())["segments"] for w in seg["words"]
        ]
        loaded.append((segment, read_mono16k(segment.wav_path), record, whisper_words))
    if not loaded:
        print("nothing to test — run windows.py and the whisper pass first", file=sys.stderr)
        return 1

    asr = create_backend("parakeet")
    asr.load()
    rng = random.Random(SEED)

    exp1, examples = run_context_ab(asr, loaded, rng)
    exp3 = run_jitter(asr, loaded)
    exp2 = run_vad_drop(asr, loaded)

    lines = ["# Context A/B + VAD-drop report", ""] + exp1 + [""]
    if examples:
        lines += ["Referee-decided examples:", ""]
        for ex in examples:
            lines += [
                f"- `{ex['segment']}` @ {ex['t']} → pivot sides **{ex['verdict']}**",
                f"  - A: “{ex['A']}”",
                f"  - B: “{ex['B']}”",
                f"  - ref: “{ex['ref']}”",
            ]
        lines += [""]
    lines += exp3 + [""] + exp2
    report = "\n".join(lines) + "\n"
    out = OUT_DIR / "context-ab.md"
    out.write_text(report)
    print(report)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
