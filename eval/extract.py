"""Cut manifest segments out of the recordings in examples/ with ffmpeg.

Usage:
    uv run --group eval eval/extract.py            # extract missing segments
    uv run --group eval eval/extract.py --force    # re-extract everything
    uv run --group eval eval/extract.py --full "my-meeting-recording.mov"
                                                   # full-length WAV (for listening
                                                   # and picking segment bounds)

Output: mono 16 kHz s16 WAV in eval/audio/ (gitignored — private content).
"""

from __future__ import annotations

import argparse
import sys

from common import AUDIO_DIR, EXAMPLES_DIR, EvalSegment, load_manifest, to_wav16k


def ffmpeg_extract(segment: EvalSegment) -> None:
    to_wav16k(segment.source_path, segment.wav_path, start=segment.start, end=segment.end)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-extract existing segments")
    parser.add_argument("--full", metavar="FILENAME", help="extract one full recording instead")
    args = parser.parse_args()

    AUDIO_DIR.mkdir(exist_ok=True)

    if args.full:
        source = EXAMPLES_DIR / args.full
        if not source.exists():
            print(f"not found: {source}", file=sys.stderr)
            return 1
        out = AUDIO_DIR / f"full-{source.stem}.wav"
        to_wav16k(source, out)
        print(f"wrote {out}")
        return 0

    for segment in load_manifest():
        if not segment.source_path.exists():
            print(f"[skip] {segment.id}: missing source {segment.source}", file=sys.stderr)
            continue
        if segment.wav_path.exists() and not args.force:
            print(f"[ok]   {segment.id}: already extracted")
            continue
        ffmpeg_extract(segment)
        print(f"[new]  {segment.id}: {segment.source} [{segment.start:.0f}s–{segment.end:.0f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
