"""Produce diarization hypotheses (and bootstrap references) for DER scoring.

Runs stenograf's real diarizer + finalize pass over the
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
    uv run eval/diarize.py --ami                 # the AMI/ICSI channels (eval/ami.py)

The AMI mode covers the corpus channels ``eval/ami.py fetch`` built: per-channel
known speaker counts (the production topology's signal), plus a per-cluster
embedding file for the re-ID trials. Each channel is diarized once —
``finalize_channel`` gets the turns back through a frozen diarizer.
"""

from __future__ import annotations

import argparse
import json
import sys

from common import OUT_DIR, REFS_DIR, load_manifest, read_pcm16
from rttm import Turn, write_rttm

from stenograf import assets
from stenograf.asr.parakeet import ParakeetMLXBackend
from stenograf.config import Language
from stenograf.diarization.base import Diarizer, SpeakerTurn
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
                {
                    "text": entry.text,
                    "start": entry.start,
                    "end": entry.end,
                    "speaker": entry.speaker,
                }
            )
    return {"words": words}


class _FrozenDiarizer(Diarizer):
    """Replays one precomputed result so ``finalize_channel`` does not diarize
    the same audio a second time."""

    def __init__(self, turns: list[SpeakerTurn]) -> None:
        self._turns = turns

    def diarize(self, samples, num_speakers=None) -> list[SpeakerTurn]:
        return self._turns


def run_ami(channel_ids: set[str] | None = None) -> None:
    """Diarize + finalize every corpus channel; write hypotheses for both scorers.

    Per channel into ``out/diar/ami/``: ``<id>.rttm`` (DER), ``<id>.words.json``
    (word attribution), ``<id>.emb.json`` (cluster embeddings for the re-ID
    trials). Speaker counts are the known per-channel truth.

    Count-1 channels take the production path: ``finalize_channel`` never runs
    the diarizer for ``num_speakers=1`` (everything is ``S0`` on VAD/segment
    spans), so the mic hypothesis is those entry spans, and its re-ID embedding
    is computed from them the way a count>1 cluster's would be. What that
    bypass costs against a diarizer that *is* run on a solo channel is
    ``solo_arms.py``. Known-count channels likewise mirror the shipped path:
    diarized at k+1 and folded back (``pipeline.fold_excess_clusters``), so the
    hypotheses, embeddings and re-ID trials all reflect what a user gets."""
    import time

    import ami

    from stenograf.diarization.sherpa import cluster_embeddings
    from stenograf.pipeline import fold_excess_clusters

    channels = [c for c in ami.load_channels() if channel_ids is None or c.id in channel_ids]
    if not channels:
        raise SystemExit("no corpus channels — run `eval/ami.py fetch` first")

    diarizer = SherpaOnnxDiarizer()
    asr = ParakeetMLXBackend()
    asr_loaded = False
    vad = SileroVAD(assets.fetch(assets.SILERO_VAD))
    out_dir = OUT_DIR / "diar" / "ami"
    out_dir.mkdir(parents=True, exist_ok=True)

    for channel in channels:
        started = time.monotonic()
        pcm = read_pcm16(channel.wav_path)
        solo = channel.num_speakers == 1
        turns: list[SpeakerTurn] = []
        embeddings = {}
        if not solo:
            result = diarizer.diarize_with_embeddings(pcm, channel.num_speakers + 1)
            folded = fold_excess_clusters(result.turns, result.embeddings, channel.num_speakers)
            turns, embeddings = list(folded[0]), folded[1]

        if not asr_loaded:
            # After the diarization peak, so the resident MLX weights are not
            # stacked on top of it — that margin is what decided which loop
            # channel's process got killed back when sherpa leaked.
            asr.load()
            asr_loaded = True
        entries = finalize_channel(
            pcm,
            asr=asr,
            language=Language("en"),
            vad=vad,
            diarizer=None if solo else _FrozenDiarizer(turns),
            num_speakers=channel.num_speakers,
        )
        if solo:
            # Solo hypothesis activity = decoded word spans merged with the same
            # gap rule the references use — entry spans bridge every internal
            # pause (measured +7.7 pts false alarm on ES2003a.mic), raw word
            # spans count natural word gaps as missed (+10.4 pts).
            spans = ami.merge_spans(
                [(w.start, w.end) for e in entries for w in e.words], ami.MERGE_GAP_S
            )
            turns = [SpeakerTurn("S0", s, e) for s, e in spans]
            embeddings = cluster_embeddings(turns, pcm, diarizer.embed)

        write_rttm(
            out_dir / f"{channel.id}.rttm",
            [Turn(t.speaker, t.start, t.end) for t in turns],
            channel.id,
        )
        (out_dir / f"{channel.id}.emb.json").write_text(
            json.dumps({k: [float(x) for x in v] for k, v in embeddings.items()})
        )
        (out_dir / f"{channel.id}.words.json").write_text(
            json.dumps(_words_json(entries), ensure_ascii=False, indent=2)
        )
        print(
            f"[{channel.id}] {len(turns)} turns, "
            f"{sum(len(e.words) or 1 for e in entries)} words, "
            f"{time.monotonic() - started:.0f}s"
        )


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
    parser.add_argument(
        "--ami",
        action="store_true",
        help="run the AMI/ICSI corpus channels instead of the manifest segments",
    )
    args = parser.parse_args()

    if args.ami:
        run_ami(set(args.segments.split(",")) if args.segments else None)
        return 0

    wanted = set(args.segments.split(",")) if args.segments else None
    segments = [
        s for s in load_manifest() if (wanted is None or s.id in wanted) and s.wav_path.exists()
    ]
    if not segments:
        print("no extracted segments — run eval/extract.py first", file=sys.stderr)
        return 1

    diarizer = _build_diarizer(sherpa_only=args.sherpa_only)
    asr = vad = None
    if not args.bootstrap:  # the transcript (word attribution) is only needed for hypotheses
        asr = ParakeetMLXBackend()
        asr.load()
        vad = SileroVAD(assets.fetch(assets.SILERO_VAD))
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
            pcm,
            asr=asr,
            language=language,
            vad=vad,
            diarizer=diarizer,
            num_speakers=args.num_speakers,
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
