"""Windowed finalize-path decode with per-window provenance (verification tier).

Decodes eval segments exactly the way a meeting's finalize pass does — Silero
VAD → ``pack_windows`` → one batch Parakeet decode per window (the same code
path ``pipeline._decode`` runs, and byte-identical audio slices) — and records
*which window produced which words*. That provenance is what the plain Phase 0
``transcribe.py --backend parakeet`` run cannot give: it feeds the model the
whole segment at once, so its output says nothing about how the product's
windowing affects accuracy.

Output: eval/out/parakeet-win/<id>.json in the transcribe.py record shape,
plus ``windows``: the padded [start, end] spans actually decoded (with their
cut_start/cut_end classification, see stenograf.vad.Window), in seconds on
the segment clock. adjudicate.py joins disagreement sites against these
spans to slice model errors by window length (the short-utterance study).

The backend runs WITHOUT the user glossary/biasing on purpose: the study
isolates windowing, and the Whisper pivot has no glossary either.

Usage:
    uv run --group eval eval/windows.py                      # all extracted segments
    uv run --group eval eval/windows.py --segments de-0714-mic,de-0714-sys
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import OUT_DIR, load_manifest  # noqa: E402

from stenograf import models  # noqa: E402
from stenograf.asr import create_backend  # noqa: E402
from stenograf.audio import SAMPLE_RATE, sample_index, to_float32  # noqa: E402
from stenograf.pipeline import _clip_window, _shift  # noqa: E402
from stenograf.vad import SileroVAD, decode_slice, pack_windows  # noqa: E402

BACKEND_DIR = "parakeet-win"


def read_mono16k(path: Path) -> np.ndarray:
    import wave

    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1 or w.getframerate() != SAMPLE_RATE:
            raise ValueError(f"{path} is not mono 16 kHz — re-run extract.py")
        frames = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return to_float32(frames)


def decode_windowed(asr, vad: SileroVAD, samples: np.ndarray, language):
    """pipeline._decode, with the window spans kept.

    The decode arithmetic — the slice bounds and keep interval
    (``decode_slice``) and the word clipping (``_clip_window``/``_shift``) —
    is imported from the package, never re-implemented here: this script's
    whole point is to record what the *real* decode path produces, so any
    local copy of that logic silently rots (it did once: the 2026-07-19
    context-carry fix landed in the package while this loop still decoded
    bare windows).
    """
    duration = len(samples) / SAMPLE_RATE
    windows = pack_windows(vad.speech_segments(samples), duration)
    segments = []
    for i, win in enumerate(windows):
        start, end = win.start, win.end
        ctx, hi, keep_lo, keep_hi = decode_slice(win)
        window = samples[sample_index(ctx) : sample_index(min(hi, duration))]
        for seg in asr.transcribe(window, language):
            clipped = _clip_window(_shift(seg, ctx), keep_lo, keep_hi)
            if clipped is None:
                continue
            segments.append(
                {
                    "text": clipped.text,
                    "start": clipped.start,
                    "end": clipped.end,
                    "words": [
                        {"text": w.text, "start": w.start, "end": w.end} for w in clipped.words
                    ],
                }
            )
        print(f"    window {i + 1}/{len(windows)} [{start:7.1f}s–{end:7.1f}s]", end="\r")
    segments.sort(key=lambda s: s["start"])
    return segments, [
        {"start": w.start, "end": w.end, "cut_start": w.cut_start, "cut_end": w.cut_end}
        for w in windows
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", help="comma-separated segment ids (default: all)")
    parser.add_argument("--force", action="store_true", help="re-decode existing records")
    args = parser.parse_args()

    wanted = set(args.segments.split(",")) if args.segments else None
    segments = [s for s in load_manifest() if wanted is None or s.id in wanted]
    if wanted:
        missing = wanted - {s.id for s in segments}
        if missing:
            print(f"unknown segment ids: {sorted(missing)}", file=sys.stderr)
            return 1
    segments = [s for s in segments if s.wav_path.exists()]
    if not segments:
        print("no extracted segments — run extract.py first", file=sys.stderr)
        return 1

    out_dir = OUT_DIR / BACKEND_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    asr = create_backend("parakeet")
    asr.load()
    vad = SileroVAD(models.fetch(models.SILERO_VAD))
    load_s = time.perf_counter() - t0
    print(f"loaded parakeet ({asr.model_id}) + VAD in {load_s:.1f}s")

    for segment in segments:
        out_path = out_dir / f"{segment.id}.json"
        if out_path.exists() and not args.force:
            print(f"[ok]   {segment.id}: already decoded")
            continue
        samples = read_mono16k(segment.wav_path)
        audio_s = len(samples) / SAMPLE_RATE
        t0 = time.perf_counter()
        segs, windows = decode_windowed(asr, vad, samples, segment.language)
        wall_s = time.perf_counter() - t0
        record = {
            "segment": segment.id,
            "backend": BACKEND_DIR,
            "model": asr.model_id,
            "language": segment.language,
            "detected_language": None,
            "audio_s": round(audio_s, 2),
            "wall_s": round(wall_s, 2),
            "speed_x_rt": round(audio_s / wall_s, 1),
            "load_s": round(load_s, 1),
            "text": " ".join(s["text"] for s in segs),
            "segments": segs,
            "windows": windows,
        }
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        lens = [w["end"] - w["start"] for w in windows]
        short = sum(1 for length in lens if length < 3.0)
        print(
            f"[new]  {segment.id}: {audio_s:.0f}s audio in {wall_s:.1f}s "
            f"({record['speed_x_rt']}x RT), {len(windows)} windows "
            f"({short} under 3s, median {np.median(lens):.1f}s)"
            if lens
            else f"[new]  {segment.id}: no speech windows"
        )
        import mlx.core as mx

        mx.clear_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
