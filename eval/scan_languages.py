"""Quick language/content scan over every recording in examples/.

Transcribes two 30 s probes per recording (at 25% and 75% of its length) with
Whisper tiny — good enough for a de/en call and a content peek, and helps pick
which stretches to hand-correct as references.

Usage:
    uv run --group eval eval/scan_languages.py

Writes eval/out/lid_scan.json and prints a summary.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import mlx_whisper
from common import EXAMPLES_DIR, OUT_DIR

PROBE_MODEL = "mlx-community/whisper-tiny"
AUDIO_SUFFIXES = {".mov", ".m4a", ".mp3", ".mp4", ".wav"}


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(out.stdout.strip())


def main() -> int:
    results = []
    recordings = sorted(p for p in EXAMPLES_DIR.iterdir() if p.suffix.lower() in AUDIO_SUFFIXES)
    with tempfile.TemporaryDirectory() as tmp:
        snippet = Path(tmp) / "probe.wav"
        for path in recordings:
            duration = probe_duration(path)
            probes = []
            for fraction in (0.25, 0.75):
                start = duration * fraction
                subprocess.run(
                    ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                     "-ss", str(start), "-t", "30", "-i", str(path),
                     "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(snippet)],
                    check=True,
                )
                raw = mlx_whisper.transcribe(str(snippet), path_or_hf_repo=PROBE_MODEL)
                probes.append(
                    {
                        "at_s": round(start),
                        "language": raw.get("language"),
                        "preview": raw["text"].strip()[:120],
                    }
                )
            languages = {p["language"] for p in probes}
            results.append(
                {
                    "file": path.name,
                    "duration_s": round(duration),
                    "language": languages.pop() if len(languages) == 1 else "mixed?",
                    "probes": probes,
                }
            )
            print(f"{path.name}  [{results[-1]['duration_s']}s]  → {results[-1]['language']}")
            for p in probes:
                print(f"    @{p['at_s']}s ({p['language']}): {p['preview']}")

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "lid_scan.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nwrote {OUT_DIR / 'lid_scan.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
