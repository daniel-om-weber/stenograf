"""Label-free evaluation of the live pass (PLAN.md §5 Phase 2, Task 1 acceptance).

The live pass is provisional text that the finalize pass replaces on stop, so its
reference is finalize's own full-attention ``generate()`` output on the same
audio — *not* a human transcript (PLAN.md §5 "Live-pass evaluation"). This drives
``stenograf.live.LiveDecoder`` over a raw clip in simulated real time and reports
the three label-free metrics:

1. Agreement with finalize — WER of the committed-live transcript vs a full
   ``finalize_channel`` pass on the same audio (the live-degradation number).
2. Commit monotonicity — a committed word must never be reordered or rewritten by
   a later decode; any violation is a bug.
3. Commit latency — audio-arrival → commit time, per committed word.

Unlike the other eval/ scripts this one imports the shipped package: it is
testing the real ``LiveDecoder``, not a Phase-0 candidate model.

Usage:
    uv run --group eval eval/live.py                        # de-1 + en-1 wavs
    uv run --group eval eval/live.py --source "examples/x.mov" --start 60 --dur 300
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import jiwer
from common import AUDIO_DIR
from score import normalize

from stenograf import models
from stenograf.asr.base import Word
from stenograf.asr.parakeet import ParakeetMLXBackend
from stenograf.audio import SAMPLE_RATE, load_audio
from stenograf.live import LiveDecoder, WindowedLiveDecoder
from stenograf.pipeline import finalize_channel
from stenograf.vad import SileroVAD

DEFAULT_SOURCES = [AUDIO_DIR / "de-1.wav", AUDIO_DIR / "en-1.wav"]


def _pctl(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def stream(
    clip, decoder: LiveDecoder, feed_chunk: float
) -> tuple[list[Word], list[float], int]:
    """Feed the clip in real-time-sized chunks; return committed words, per-word
    commit latencies (feed commits only), and the count force-committed at flush."""
    committed: list[Word] = []
    latencies: list[float] = []
    step = int(feed_chunk * SAMPLE_RATE)
    for pos in range(0, len(clip), step):
        chunk = clip[pos : pos + step]
        update = decoder.feed(chunk, pos / SAMPLE_RATE)
        fed_end = (pos + len(chunk)) / SAMPLE_RATE
        for word in update.committed:
            committed.append(word)
            latencies.append(fed_end - word.end)
    flushed = decoder.flush().committed
    committed.extend(flushed)
    return committed, latencies, len(flushed)


def count_monotonicity_violations(committed: list[Word]) -> int:
    """Committed words must be non-decreasing in start time (append-only stream)."""
    violations = 0
    last = float("-inf")
    for word in committed:
        if word.start + 1e-6 < last:
            violations += 1
        last = max(last, word.start)
    return violations


def evaluate(source: Path, start: float, dur: float | None, feed_chunk: float,
             asr: ParakeetMLXBackend, vad: SileroVAD,
             decode_interval: float | None, mode: str) -> dict:
    samples = load_audio(source)
    lo = int(start * SAMPLE_RATE)
    hi = len(samples) if dur is None else min(len(samples), lo + int(dur * SAMPLE_RATE))
    clip = samples[lo:hi]
    duration = len(clip) / SAMPLE_RATE

    print(f"\n=== {source.name}  [{start:.0f}s..{start + duration:.0f}s]  ({duration:.0f}s) ===")

    t0 = time.monotonic()
    ref_entries = finalize_channel(clip, asr=asr, language=None, vad=vad, diarizer=None)
    # Compare word stream to word stream: entry.text is the segment-level
    # rendering, which spaces number-adjacent tokens differently from the word
    # tokens ("und 15.7." vs "und15.7.") — a rendering artifact, not a decode
    # difference, and the reuse path consumes words.
    reference = " ".join(w.text for e in ref_entries for w in e.words)
    finalize_s = time.monotonic() - t0

    t0 = time.monotonic()
    if mode == "window":
        decoder = WindowedLiveDecoder(asr, vad=vad)
    else:
        decoder = LiveDecoder(asr, vad=vad, decode_interval=decode_interval)
    committed, latencies, flushed = stream(clip, decoder, feed_chunk)
    live_s = time.monotonic() - t0
    hypothesis = " ".join(w.text for w in committed)

    wer = jiwer.wer(normalize(reference), normalize(hypothesis)) if reference else 0.0
    violations = count_monotonicity_violations(committed)

    print(f"  finalize: {len(reference.split()):>5} words   ({duration / finalize_s:.0f}x RT)")
    print(f"  live:     {len(hypothesis.split()):>5} words committed "
          f"({flushed} at flush)   ({duration / live_s:.0f}x RT, {decoder.decodes} decodes)")
    print(f"  1. agreement WER vs finalize : {wer:.1%}")
    print(f"  2. commit monotonicity       : {violations} violation(s)")
    print(f"  3. commit latency            : median {statistics.median(latencies):.2f}s  "
          f"p90 {_pctl(latencies, 0.9):.2f}s  max {max(latencies, default=0.0):.2f}s")
    return {"source": source.name, "wer": wer, "violations": violations,
            "median_latency": statistics.median(latencies) if latencies else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", action="append", type=Path,
                    help="Audio/video clip (repeatable); default: de-1 + en-1 eval wavs.")
    ap.add_argument("--start", type=float, default=0.0, help="Clip start (s).")
    ap.add_argument("--dur", type=float, default=None, help="Clip duration (s); default: all.")
    ap.add_argument("--feed-chunk", type=float, default=1.0,
                    help="Simulated arrival chunk (s) — sets the caption cadence.")
    ap.add_argument("--decode-interval", default="none",
                    help="LiveDecoder decode_interval: seconds, or 'none' for utterance "
                         "mode (decode only at VAD endpoints).")
    ap.add_argument("--mode", choices=["live", "window"], default="window",
                    help="'window' = WindowedLiveDecoder (the product default: finalize-"
                         "identical windows); 'live' = LiveDecoder at --decode-interval.")
    args = ap.parse_args()
    decode_interval = None if args.decode_interval == "none" else float(args.decode_interval)

    sources = args.source or DEFAULT_SOURCES
    missing = [s for s in sources if not s.exists()]
    if missing:
        print(f"missing clip(s): {', '.join(str(m) for m in missing)}\n"
              "extract eval audio first, or pass --source PATH", file=sys.stderr)
        return 1

    asr = ParakeetMLXBackend()
    asr.load()
    vad = SileroVAD(models.fetch(models.SILERO_VAD))

    summary = [
        evaluate(s, args.start, args.dur, args.feed_chunk, asr, vad, decode_interval, args.mode)
        for s in sources
    ]

    print("\n=== summary ===")
    for row in summary:
        print(f"  {row['source']:<16} WER {row['wer']:.1%}   "
              f"violations {row['violations']}   median latency {row['median_latency']:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
