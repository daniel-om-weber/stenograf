"""Does diarizing a known-count channel at k+1 (recovered by naming) beat k?

Over-clustering measured +0.67 DIR over balanced and under-clustering −2.04
in the research record — splits are recoverable at the naming stage, merges
are not. This tests the shippable version of that bias on our own harness:
every multi-speaker loop channel diarized at its known count k (the cached
matrix hypotheses) versus k+1, with merge-at-naming (clusters over threshold
on the same profile take one name) as the recovery. Arms:

- ``k``        — the shipped behavior (cached; the matrix baseline).
- ``k+1``      — one extra cluster, no recovery (what an unprofiled channel
  would read).
- ``k+1 name`` — the extra cluster merged back where a profile catches it
  (AMI sessions b–d; ICSI has no galleries and session ``a`` enrolls them).

Word times come from the matrix hypotheses; k+1 turns and embeddings are
cached under ``out/diar/ami-k1/``. Run ``eval/ami.py run`` first.

    uv run --group eval eval/kplus1.py [--segments ES2003b.loop,...]
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from common import OUT_DIR, read_pcm16
from der import Word, score_attribution, score_der
from rttm import Turn, parse_rttm, write_rttm
from split_recovery import _load_word_times, _resolve

from stenograf.pipeline import merge_words_turns

K1_DIR = OUT_DIR / "diar" / "ami-k1"


def _attribution(words, turns: list[Turn], relabel: dict[str, str], ref):
    relabelled = [Turn(relabel.get(t.speaker, t.speaker), t.start, t.end) for t in turns]
    entries = merge_words_turns(list(words), relabelled)
    labelled = [Word(w.text, w.start, w.end, e.speaker) for e in entries for w in e.words]
    return score_attribution(labelled, ref)


def _k1_result(channel, diarizer) -> tuple[list[Turn], dict[str, np.ndarray]]:
    rttm_path = K1_DIR / f"{channel.id}.rttm"
    emb_path = K1_DIR / f"{channel.id}.emb.json"
    if rttm_path.exists() and emb_path.exists():
        embeddings = {
            k: np.asarray(v, dtype=np.float32)
            for k, v in json.loads(emb_path.read_text()).items()
        }
        return parse_rttm(rttm_path), embeddings
    result = diarizer.diarize_with_embeddings(
        read_pcm16(channel.wav_path), channel.num_speakers + 1
    )
    turns = [Turn(t.speaker, t.start, t.end) for t in result.turns]
    K1_DIR.mkdir(parents=True, exist_ok=True)
    write_rttm(rttm_path, turns, channel.id)
    emb_path.write_text(
        json.dumps({k: [float(x) for x in v] for k, v in result.embeddings.items()})
    )
    return turns, result.embeddings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", help="comma-separated channel ids (default: every loop one)")
    args = parser.parse_args()

    import ami

    from stenograf.diarization.sherpa import SherpaOnnxDiarizer

    wanted = set(args.segments.split(",")) if args.segments else None
    channels = [
        c
        for c in ami.load_channels()
        if c.num_speakers > 1 and (wanted is None or c.id in wanted)
    ]
    if not channels:
        raise SystemExit("no multi-speaker corpus channels — run `eval/ami.py fetch` first")

    diarizer = SherpaOnnxDiarizer()
    galleries = ami.build_galleries(diarizer.embed)
    hyp_dir = OUT_DIR / "diar" / "ami"

    rows = []
    for channel in channels:
        words_path = hyp_dir / f"{channel.id}.words.json"
        if not words_path.exists():
            print(f"  no matrix output for {channel.id} — run eval/ami.py run", file=sys.stderr)
            continue
        ref = parse_rttm(channel.ref_path)
        words = _load_word_times(words_path)
        k_turns = parse_rttm(hyp_dir / f"{channel.id}.rttm")
        k1_turns, k1_emb = _k1_result(channel, diarizer)
        gallery = galleries.get(channel.group) if channel.session != ami.ENROLL_SESSION else None

        arms = {
            "k": (_attribution(words, k_turns, {}, ref), score_der(ref, k_turns)),
            "k+1": (_attribution(words, k1_turns, {}, ref), score_der(ref, k1_turns)),
        }
        merged_n = 0
        if gallery:
            mapping = _resolve(k1_emb, gallery, exclusive=False)
            merged_n = len(mapping) - len(set(mapping.values()))
            relabelled = [Turn(mapping.get(t.speaker, t.speaker), t.start, t.end) for t in k1_turns]
            arms["k+1 name"] = (
                _attribution(words, k1_turns, mapping, ref),
                score_der(ref, relabelled),
            )
        rows.append((channel, arms, merged_n))
        print(
            f"[{channel.id}] "
            + "  ".join(f"{name} {a.accuracy:.1%}/{d.der:.1%}" for name, (a, d) in arms.items())
            + (f"  ({merged_n} merged)" if gallery else "  (no gallery)"),
            flush=True,
        )

    lines = [
        "# k vs k+1 on known-count channels (attribution / DER)",
        "",
        "`k+1 name` folds same-profile clusters together before scoring;",
        "*merged* counts cluster pairs the naming actually folded.",
        "",
        "| Channel | k | k+1 | k+1 name | merged |",
        "|---|---|---|---|---|",
    ]
    for channel, arms, merged_n in rows:
        named = arms.get("k+1 name")
        named_cell = f"{named[0].accuracy:.1%} / {named[1].der:.1%}" if named else "—"
        lines.append(
            f"| {channel.id} | {arms['k'][0].accuracy:.1%} / {arms['k'][1].der:.1%} "
            f"| {arms['k+1'][0].accuracy:.1%} / {arms['k+1'][1].der:.1%} "
            f"| {named_cell} | {merged_n} |"
        )
    report = OUT_DIR / "diar-kplus1.md"
    report.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
