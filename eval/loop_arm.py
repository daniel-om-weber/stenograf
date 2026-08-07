"""Run one clustering arm from the persisted freeze — no ASR, no segmentation.

Consumes ``loop_freeze.py``'s artifacts and the baseline's decoded words
(``out/diar/ami-loop/<id>.words.json`` — the words are diarization-independent:
VAD picks the decode windows, speakers are attached afterwards), and produces a
full arm directory in run_ami's exact formats via the production functions
themselves: ``_cluster`` → ``_assemble`` → ``cluster_embeddings`` →
``fold_excess_clusters`` → ``merge_words_turns``. Solo channels are copied from
the baseline (their path never touches the diarizer). Per-arm cost is
dominated by ``cluster_embeddings`` (the spans depend on the arm's turns, so
they cannot be frozen) — minutes, not the matrix's ~40.

``--check ami-loop`` byte-compares the produced files against the baseline:
the parity gate that the frozen path reproduces ``diarize.py --own-loop``
before any candidate arm is trusted.

Run::

    uv run --group eval eval/loop_arm.py --cluster ward
    uv run --group eval eval/loop_arm.py --cluster complete \
        --out-name ami-loop-frozen --check ami-loop
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time

import numpy as np
from common import OUT_DIR, read_pcm16
from diarize import _words_json
from loop_freeze import freeze_dir, load_emb_cache, save_emb_cache
from rttm import Turn, parse_rttm, write_rttm

from stenograf.asr.base import Word
from stenograf.audio import SAMPLE_RATE
from stenograf.diarization.loop import OwnDiarizer, _cluster
from stenograf.diarization.sherpa import cluster_embeddings
from stenograf.pipeline import fold_excess_clusters, merge_words_turns

BASELINE = "ami-loop"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster", required=True, help="complete | average | ward | nmesc")
    parser.add_argument(
        "--fold",
        default="production",
        help="production (pipeline.fold_excess_clusters as shipped) | "
        "fold_sweep.py's duration | maxpair | samevoice",
    )
    parser.add_argument(
        "--gate",
        type=float,
        default=None,
        help="samevoice pair-similarity gate (default: COLLAPSE_SIMILARITY)",
    )
    parser.add_argument(
        "--out-name", help=f"output dir under out/diar/ (default {BASELINE}-<cluster>)"
    )
    parser.add_argument("--segments", help="comma-separated channel ids (default: all)")
    parser.add_argument("--check", help="byte-compare outputs against this arm dir (parity gate)")
    parser.add_argument(
        "--shift-s", type=float, default=1.0, help="window stride in seconds (reference: 1.0)"
    )
    args = parser.parse_args()

    import ami

    out_name = args.out_name or f"{BASELINE}-{args.cluster}"
    out_dir = OUT_DIR / "diar" / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    base_dir = OUT_DIR / "diar" / BASELINE
    wanted = set(args.segments.split(",")) if args.segments else None
    channels = [c for c in ami.load_channels() if wanted is None or c.id in wanted]
    if not channels:
        raise SystemExit("no corpus channels — run `eval/ami.py fetch` first")

    root = freeze_dir(args.shift_s)
    diarizer = OwnDiarizer(
        cluster_method=args.cluster,
        shift=int(round(args.shift_s * SAMPLE_RATE)),
    )
    failures = 0
    for channel in channels:
        names = [f"{channel.id}.rttm", f"{channel.id}.emb.json", f"{channel.id}.words.json"]
        if channel.num_speakers == 1:
            for name in names:
                shutil.copy2(base_dir / name, out_dir / name)
            continue

        started = time.monotonic()
        freeze = root / f"{channel.id}.npz"
        if not freeze.exists():
            print(f"[{channel.id}] no freeze — run eval/loop_freeze.py first", file=sys.stderr)
            return 1
        data = np.load(freeze)
        pairs = [(int(a), int(b)) for a, b in data["pairs"]]
        clusters = _cluster(data["vectors"], 0.5, channel.num_speakers + 1, args.cluster)
        turns = diarizer._assemble(int(data["n"]), list(data["labels"]), pairs, clusters)

        embeddings = load_emb_cache(args.cluster, channel.id, root)
        if embeddings is None:
            embeddings = cluster_embeddings(turns, read_pcm16(channel.wav_path), diarizer.embed)
            save_emb_cache(args.cluster, channel.id, embeddings, root)
        if args.fold == "production":
            folded = fold_excess_clusters(turns, embeddings, channel.num_speakers)
            folded_turns, folded_emb = list(folded[0]), folded[1]
        else:
            from fold_sweep import fold_with_rule

            ref = parse_rttm(channel.ref_path)
            gate = {} if args.gate is None else {"gate": args.gate}
            folded_turns, folded_emb = fold_with_rule(
                args.fold, turns, embeddings, channel.num_speakers, ref, **gate
            )

        words = [
            Word(text=w["text"], start=w["start"], end=w["end"])
            for w in json.loads((base_dir / f"{channel.id}.words.json").read_text())["words"]
        ]
        entries = merge_words_turns(words, folded_turns)

        write_rttm(
            out_dir / names[0], [Turn(t.speaker, t.start, t.end) for t in folded_turns], channel.id
        )
        (out_dir / names[1]).write_text(
            json.dumps({k: [float(x) for x in v] for k, v in folded_emb.items()})
        )
        (out_dir / names[2]).write_text(
            json.dumps(_words_json(entries), ensure_ascii=False, indent=2)
        )
        print(f"[{channel.id}] {len(folded_turns)} turns, {time.monotonic() - started:.0f}s")

    if args.check:
        check_dir = OUT_DIR / "diar" / args.check
        for channel in channels:
            for suffix in (".rttm", ".emb.json", ".words.json"):
                name = f"{channel.id}{suffix}"
                same = (out_dir / name).read_bytes() == (check_dir / name).read_bytes()
                if not same:
                    failures += 1
                    print(f"PARITY FAIL {name}")
        print("parity: PASS" if failures == 0 else f"parity: {failures} files differ")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
