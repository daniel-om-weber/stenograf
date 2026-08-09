"""The pool gate: worker pool vs sequential reference, and vs the shipped baseline.

Two halves, run both before trusting a threading change to the diarization
loop (`PLAN-DIARIZATION-SPEED.md` step 2; numbers land in `eval/README.md`):

- **Half 1 — bit-exactness.** On one channel, the pooled loop at each
  ``--workers`` count must produce segmentation labels, (chunk, speaker)
  pairs, pair vectors and cluster embeddings **bit-identical** to
  ``workers=1`` (the sequential intra-op-1 reference). Exact equality, no
  tolerance: each ORT call is independent and single-threaded, so any
  difference is a bug, not float jitter. Also prints per-config wall-clock —
  the speedup table falls out for free.

- **Half 2 — no decision moved vs shipped.** The intra-op 8 → 1 move shifts
  embeddings by float reduction order (~3.4e-7 measured 2026-08-09), which
  parity can't see but a ward merge could. Re-runs every multi-speaker
  corpus channel through the pooled production path (``diarize_with_
  embeddings(k+1)`` + fold, exactly as ``diarize.py`` built the baseline)
  and byte-compares RTTMs against ``out/diar/<baseline>/`` while reporting
  embedding drift. Identical turns everywhere = no merge decision moved.

Run::

    uv run --group eval eval/pool_parity.py                 # both halves
    uv run --group eval eval/pool_parity.py --half 1 --workers 4,12
    uv run --group eval eval/pool_parity.py --half 2 --baseline ami-loop
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
from common import OUT_DIR, read_pcm16
from rttm import Turn, write_rttm

from stenograf.audio import to_float32
from stenograf.diarization.loop import OwnDiarizer
from stenograf.pipeline import fold_excess_clusters

DEFAULT_CHANNEL = "ES2003c.loop"
BASELINE = "ami-loop-ward-sv08"
"""The shipped config's measured artifact (`eval/README.md`: production
`fold_excess_clusters` byte-matches this arm through the harness, and the
pre-pool HEAD reproduced its RTTMs byte-exactly, verified 2026-08-09).
NOT `ami-loop` — that dir predates ward-by-default by six hours and no
current code reproduces it."""


def _bit_equal(a: list[np.ndarray], b: list[np.ndarray]) -> bool:
    return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def half1(channel_id: str, worker_counts: list[int]) -> bool:
    import ami

    channel = next(c for c in ami.load_channels() if c.id == channel_id)
    audio = to_float32(read_pcm16(channel.wav_path))
    print(f"[half 1] {channel_id}, {len(audio) / 16_000 / 60:.1f} min")

    results = {}
    for workers in [1, *worker_counts]:
        diarizer = OwnDiarizer(workers=workers)
        started = time.monotonic()
        labels = diarizer._chunk_labels(audio)
        seg_done = time.monotonic()
        pairs, vectors = diarizer._pair_embeddings(audio, labels)
        emb_done = time.monotonic()
        results[workers] = (labels, pairs, vectors)
        print(
            f"  workers={workers:2d}: segmentation {seg_done - started:6.1f} s, "
            f"embeddings {emb_done - seg_done:6.1f} s, "
            f"total {emb_done - started:6.1f} s ({len(pairs)} pairs)"
        )

    ok = True
    ref_labels, ref_pairs, ref_vectors = results[1]
    for workers in worker_counts:
        labels, pairs, vectors = results[workers]
        exact = (
            _bit_equal(labels, ref_labels)
            and pairs == ref_pairs
            and _bit_equal(vectors, ref_vectors)
        )
        print(f"  workers={workers}: {'BIT-EXACT vs workers=1' if exact else 'MISMATCH'}")
        ok &= exact
    return ok


def half2(baseline: str) -> bool:
    import ami

    baseline_dir = OUT_DIR / "diar" / baseline
    if not baseline_dir.exists():
        raise SystemExit(f"no baseline at {baseline_dir} — run diarize.py --ami first")
    out_dir = OUT_DIR / "diar" / f"{baseline}-pool-check"
    out_dir.mkdir(parents=True, exist_ok=True)

    diarizer = OwnDiarizer()
    identical = True
    drifts: list[float] = []
    for channel in ami.load_channels():
        if channel.num_speakers == 1:
            continue  # solo channels never touch the loop
        ref_rttm = baseline_dir / f"{channel.id}.rttm"
        if not ref_rttm.exists():
            print(f"[half 2] {channel.id}: no baseline rttm, skipped")
            continue
        started = time.monotonic()
        pcm = read_pcm16(channel.wav_path)
        result = diarizer.diarize_with_embeddings(pcm, channel.num_speakers + 1)
        turns, embeddings = fold_excess_clusters(
            result.turns, result.embeddings, channel.num_speakers
        )
        new_rttm = out_dir / f"{channel.id}.rttm"
        write_rttm(new_rttm, [Turn(t.speaker, t.start, t.end) for t in turns], channel.id)
        same = new_rttm.read_bytes() == ref_rttm.read_bytes()
        identical &= same

        drift = float("nan")
        ref_emb_path = baseline_dir / f"{channel.id}.emb.json"
        if ref_emb_path.exists():
            ref_emb = {
                k: np.asarray(v, dtype=np.float32)
                for k, v in json.loads(ref_emb_path.read_text()).items()
            }
            if ref_emb.keys() == embeddings.keys():
                drift = max(
                    (float(np.max(np.abs(embeddings[k] - ref_emb[k]))) for k in ref_emb),
                    default=0.0,
                )
                drifts.append(drift)
        print(
            f"[half 2] {channel.id}: {'turns identical' if same else 'TURNS DIFFER'}, "
            f"max emb drift {drift:.2e}, {time.monotonic() - started:.0f} s"
        )
    if drifts:
        print(f"[half 2] max embedding drift across channels: {max(drifts):.2e}")
    print(f"[half 2] {'PASS — no merge decision moved' if identical else 'FAIL'}")
    return identical


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--half", type=int, choices=(1, 2), default=None)
    parser.add_argument("--channel", default=DEFAULT_CHANNEL)
    parser.add_argument("--workers", default="4,12", help="pool sizes for half 1")
    parser.add_argument("--baseline", default=BASELINE)
    args = parser.parse_args()

    ok = True
    if args.half in (None, 1):
        ok &= half1(args.channel, [int(w) for w in args.workers.split(",")])
    if args.half in (None, 2):
        ok &= half2(args.baseline)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
