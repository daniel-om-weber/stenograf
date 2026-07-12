"""Produce diarization hypotheses (and bootstrap references) for DER scoring.

Phase 3, Task 0d. Runs stenograf's real diarizer + finalize pass over the
extracted eval segments and writes, per segment, into ``eval/out/diar/``:

- ``<id>.rttm`` — the raw diarization turns (the DER hypothesis), and
- ``<id>.words.json`` — the finalized transcript's words with their speaker
  labels (the word-attribution hypothesis).

Then ``eval/der.py`` scores those against the hand-labelled references in
``eval/refs/<id>.rttm``.

Because hand-labelling from scratch is slow, ``--bootstrap`` instead writes the
diarizer's turns to ``eval/refs/<id>.draft.rttm`` as a *starting point to
correct while listening* (mirroring the ``<id>.draft.txt`` transcript
convention) — never score against a draft, it flatters the model that produced
it. Correct it against the audio and rename to ``<id>.rttm``.

Usage::

    uv run eval/diarize.py                       # hypotheses for all segments
    uv run eval/diarize.py --segments de-1,de-2  # a subset
    uv run eval/diarize.py --num-speakers 3      # force the count (default: estimate)
    uv run eval/diarize.py --bootstrap           # seed refs/<id>.draft.rttm
"""

from __future__ import annotations

import argparse
import json
import sys

from common import OUT_DIR, REFS_DIR, load_manifest, read_pcm16
from rttm import Turn, write_rttm

from stenograf import models
from stenograf.asr.parakeet import ParakeetMLXBackend
from stenograf.config import Language
from stenograf.diarization.sherpa import SherpaOnnxDiarizer
from stenograf.pipeline import finalize_channel
from stenograf.vad import SileroVAD


def _build_diarizer(*, sherpa_only: bool):
    """The production stack (stenodiar for estimated counts when built) unless
    ``--sherpa-only`` pins the baseline — mirrors ``cli._load_diarizer``."""
    from stenograf.diarization.speakrs import (
        DiarizerHelperNotFoundError,
        SpeakrsCliDiarizer,
        find_stenodiar,
    )

    sherpa = SherpaOnnxDiarizer()
    if sherpa_only:
        return sherpa
    try:
        find_stenodiar()
    except DiarizerHelperNotFoundError:
        print("stenodiar not built — falling back to sherpa estimate mode", file=sys.stderr)
        return sherpa
    return SpeakrsCliDiarizer(sherpa)


def _words_json(entries) -> dict:
    words = []
    for entry in entries:
        if entry.words:
            words += [
                {"text": w.text, "start": w.start, "end": w.end, "speaker": entry.speaker}
                for w in entry.words
            ]
        else:  # a wordless backend — keep the entry as one coarse span
            words.append(
                {"text": entry.text, "start": entry.start, "end": entry.end,
                 "speaker": entry.speaker}
            )
    return {"words": words}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", help="comma-separated segment ids (default: all)")
    parser.add_argument(
        "--num-speakers", type=int, help="force the speaker count (default: estimate)"
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="write diarizer turns to refs/<id>.draft.rttm to hand-correct, not hypotheses",
    )
    parser.add_argument(
        "--sherpa-only",
        action="store_true",
        help="skip the stenodiar helper even if built (measure the sherpa baseline)",
    )
    args = parser.parse_args()

    wanted = set(args.segments.split(",")) if args.segments else None
    segments = [
        s
        for s in load_manifest()
        if (wanted is None or s.id in wanted) and s.wav_path.exists()
    ]
    if not segments:
        print("no extracted segments — run eval/extract.py first", file=sys.stderr)
        return 1

    diarizer = _build_diarizer(sherpa_only=args.sherpa_only)
    asr = vad = None
    if not args.bootstrap:  # the transcript (word attribution) is only needed for hypotheses
        asr = ParakeetMLXBackend()
        asr.load()
        vad = SileroVAD(models.fetch(models.SILERO_VAD))
    out_dir = OUT_DIR / "diar"
    out_dir.mkdir(parents=True, exist_ok=True)

    for segment in segments:
        pcm = read_pcm16(segment.wav_path)
        language = Language(segment.language) if segment.language else None

        turns = diarizer.diarize(pcm, args.num_speakers)
        rttm_turns = [Turn(t.speaker, t.start, t.end) for t in turns]
        n_spk = len({t.speaker for t in turns})

        if args.bootstrap:
            draft = REFS_DIR / f"{segment.id}.draft.rttm"
            write_rttm(draft, rttm_turns, segment.id)
            print(f"[{segment.id}] {len(turns)} turns, {n_spk} speakers → {draft.name} (draft)")
            continue

        write_rttm(out_dir / f"{segment.id}.rttm", rttm_turns, segment.id)
        entries = finalize_channel(
            pcm, asr=asr, language=language, vad=vad,
            diarizer=diarizer, num_speakers=args.num_speakers,
        )
        (out_dir / f"{segment.id}.words.json").write_text(
            json.dumps(_words_json(entries), ensure_ascii=False, indent=2)
        )
        print(
            f"[{segment.id}] {len(turns)} turns, {n_spk} speakers, "
            f"{sum(len(e.words) or 1 for e in entries)} words → out/diar/"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
