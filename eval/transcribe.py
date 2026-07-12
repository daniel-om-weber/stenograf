"""Run one ASR backend over the extracted eval segments.

Usage:
    uv run --group eval eval/transcribe.py --backend parakeet
    uv run --group eval eval/transcribe.py --backend whisper --segments smoke-online
    uv run --group eval eval/transcribe.py --backend whisper --language de ...

One backend per process, on purpose: peak-memory readings stay attributable and
model runtimes can't interfere. Writes eval/out/<backend>/<segment>.json.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time

from backends import BACKENDS
from common import OUT_DIR, load_manifest, wav_duration


def peak_rss_mb() -> float:
    # macOS reports ru_maxrss in bytes (Linux: KiB) — this harness is macOS-only.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def mlx_peak_mb() -> float | None:
    try:
        import mlx.core as mx

        return mx.get_peak_memory() / 1e6
    except (ImportError, AttributeError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=sorted(BACKENDS))
    parser.add_argument("--segments", help="comma-separated segment ids (default: all)")
    parser.add_argument("--language", help="override manifest language for all segments")
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
        print("no extracted segments to transcribe — run extract.py first", file=sys.stderr)
        return 1

    backend = BACKENDS[args.backend]()
    out_dir = OUT_DIR / backend.name
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    backend.load()
    load_s = time.perf_counter() - t0
    print(f"loaded {backend.name} ({backend.model_id}) in {load_s:.1f}s")

    for segment in segments:
        language = args.language or segment.language
        audio_s = wav_duration(segment.wav_path)
        t0 = time.perf_counter()
        result = backend.transcribe(segment.wav_path, language)
        wall_s = time.perf_counter() - t0
        record = {
            "segment": segment.id,
            "backend": backend.name,
            "model": backend.model_id,
            "language": language,
            "detected_language": result["detected_language"],
            "audio_s": round(audio_s, 2),
            "wall_s": round(wall_s, 2),
            "speed_x_rt": round(audio_s / wall_s, 1),
            "load_s": round(load_s, 1),
            "peak_rss_mb": round(peak_rss_mb()),
            "mlx_peak_mb": round(peak) if (peak := mlx_peak_mb()) else peak,
            "text": result["text"],
            "segments": result["segments"],
        }
        segment.hyp_path(backend.name).write_text(
            json.dumps(record, ensure_ascii=False, indent=2)
        )
        print(
            f"[{segment.id}] {audio_s:.0f}s audio in {wall_s:.1f}s "
            f"({record['speed_x_rt']}x RT, peak {record['peak_rss_mb']} MB)"
        )
        try:
            import mlx.core as mx

            mx.clear_cache()  # return per-segment buffers instead of hoarding them
        except ImportError:
            pass
        preview = result["text"][:160].replace("\n", " ")
        print(f"    {preview}{'…' if len(result['text']) > 160 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
