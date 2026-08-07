"""What the count-1 diarizer bypass buys on single-speaker channels.

``finalize_channel`` never runs a diarizer when the channel's speaker count is
1: the entries are the ASR/VAD spans, all labelled ``S0``. This scores that
shipped behavior against the two arms it replaces, on every single-speaker
corpus channel:

- ``k1``  — the same diarizer forced to one cluster.
- ``est`` — the diarizer left to estimate the count, which is what a user who
  never states a count gets. On a machine with the stenodiar helper this is
  speakrs' estimator (the shipped estimate path); ``--no-helper`` measures
  the loop threshold cut instead, the stenodiar-less estimate path.

Word times are read back from the matrix's own hypotheses, so no arm re-runs
ASR — only the diarization and the word→turn merge are recomputed, which is
also what keeps the arms comparable. Run ``eval/ami.py run`` first.

    uv run --group eval eval/solo_arms.py [--segments ES2003a.mic,...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common import OUT_DIR, read_pcm16
from der import AttributionScore, DiarizationScore, Word, score_attribution, score_der
from rttm import Turn, parse_rttm

from stenograf.asr.base import Word as PipelineWord
from stenograf.pipeline import merge_words_turns


def _load_word_times(path: Path) -> list[PipelineWord]:
    record = json.loads(path.read_text())
    return [PipelineWord(text=w["text"], start=w["start"], end=w["end"]) for w in record["words"]]


def _attribution(words: list[PipelineWord], turns, ref: list[Turn]) -> AttributionScore:
    entries = merge_words_turns(list(words), list(turns))
    labelled = [
        Word(w.text, w.start, w.end, entry.speaker) for entry in entries for w in entry.words
    ]
    return score_attribution(labelled, ref)


def _arm(
    turns, words: list[PipelineWord], ref: list[Turn]
) -> tuple[DiarizationScore, AttributionScore, int]:
    hyp = [Turn(t.speaker, t.start, t.end) for t in turns]
    return score_der(ref, hyp), _attribution(words, turns, ref), len({t.speaker for t in hyp})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", help="comma-separated channel ids (default: every solo one)")
    parser.add_argument(
        "--no-helper",
        action="store_true",
        help="estimate with the loop threshold cut even if stenodiar is built",
    )
    args = parser.parse_args()

    import ami
    from diarize import _build_diarizer

    from stenograf.diarization.loop import OwnDiarizer

    wanted = set(args.segments.split(",")) if args.segments else None
    channels = [
        c
        for c in ami.load_channels()
        if c.num_speakers == 1 and (wanted is None or c.id in wanted)
    ]
    if not channels:
        raise SystemExit("no solo corpus channels — run `eval/ami.py fetch` first")

    hyp_dir = OUT_DIR / "diar" / "ami"
    sherpa = OwnDiarizer()
    estimator = _build_diarizer(no_helper=args.no_helper)
    estimator_name = type(estimator).__name__

    rows: list[tuple[str, dict[str, tuple[DiarizationScore, AttributionScore, int]]]] = []
    for channel in channels:
        words_path = hyp_dir / f"{channel.id}.words.json"
        shipped_path = hyp_dir / f"{channel.id}.rttm"
        if not (words_path.exists() and shipped_path.exists()):
            print(f"  no matrix output for {channel.id} — run eval/ami.py run", file=sys.stderr)
            continue
        ref = parse_rttm(channel.ref_path)
        words = _load_word_times(words_path)
        pcm = read_pcm16(channel.wav_path)

        arms = {"shipped": _arm(parse_rttm(shipped_path), words, ref)}
        arms["k1"] = _arm(sherpa.diarize(pcm, num_speakers=1), words, ref)
        arms["est"] = _arm(estimator.diarize(pcm, num_speakers=None), words, ref)
        rows.append((channel.id, arms))
        print(
            f"[{channel.id}] "
            + "  ".join(
                f"{name} DER {s.der:.1%} words {a.accuracy:.1%} k={k}"
                for name, (s, a, k) in arms.items()
            ),
            flush=True,
        )

    lines = [
        "# Solo-channel arms — is the count-1 diarizer bypass right?",
        "",
        f"Estimator: {estimator_name}. `shipped` = no diarizer (ASR/VAD spans, one label);",
        "`k1` = diarizer forced to one cluster; `est` = diarizer estimating the count.",
        "",
        "| Channel | shipped DER | k1 DER | est DER | shipped words | k1 words "
        "| est words | est k |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for channel_id, arms in rows:
        lines.append(
            f"| {channel_id} | "
            + " | ".join(f"{arms[n][0].der:.1%}" for n in ("shipped", "k1", "est"))
            + " | "
            + " | ".join(f"{arms[n][1].accuracy:.1%}" for n in ("shipped", "k1", "est"))
            + f" | {arms['est'][2]} |"
        )
    if rows:
        lines.append(
            "| **mean** | "
            + " | ".join(
                f"{sum(a[n][0].der for _, a in rows) / len(rows):.1%}"
                for n in ("shipped", "k1", "est")
            )
            + " | "
            + " | ".join(
                f"{sum(a[n][1].accuracy for _, a in rows) / len(rows):.1%}"
                for n in ("shipped", "k1", "est")
            )
            + f" | {sum(a['est'][2] for _, a in rows) / len(rows):.1f} |"
        )
    report = OUT_DIR / "diar-solo-arms.md"
    report.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
