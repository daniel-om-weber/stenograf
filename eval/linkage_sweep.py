"""Known-count partitioner sweep on frozen embeddings: all linkages, all loops.

The Phase-B follow-up to `cluster_ab.py`'s bimodal NME-SC verdict: spectral
clustering fixed the confusion channels but broke others, and the p-sweep
probe showed its selection criterion inverting (`eval/README.md`). Before the
heavy PLDA/VBx fallback, the two untested cheap linkages get their
measurement — centroid (pyannote 3.1's shipped choice: Euclidean on unit
vectors) and Ward — against complete (the parity-verified reference) and
nmesc, on identical inputs: each channel's segmentation + (chunk, speaker)
embeddings are computed once and every partitioner re-scores from the freeze,
so a full arm costs seconds, not a matrix run. DER only (attribution tracked
DER faithfully in every measured arm so far; the ship candidate gets the full
matrix before shipping).

Run::

    uv run --group eval eval/linkage_sweep.py
"""

from __future__ import annotations

import sys

import numpy as np
from common import OUT_DIR, read_pcm16
from der import score_der
from rttm import Turn, parse_rttm

ARMS = ("complete", "centroid", "ward", "nmesc")


def main() -> int:
    import ami

    from stenograf.audio import to_float32
    from stenograf.diarization import loop as own_loop

    channels = [c for c in ami.load_channels() if c.num_speakers > 1]
    if not channels:
        print("no loop channels — run `eval/ami.py fetch` first", file=sys.stderr)
        return 1

    diarizer = own_loop.OwnDiarizer()
    rows = [
        "| channel | " + " | ".join(ARMS) + " |",
        "|---|" + "---|" * len(ARMS),
    ]
    totals: dict[str, list[float]] = {arm: [] for arm in ARMS}
    for channel in channels:
        audio = to_float32(read_pcm16(channel.wav_path))
        labels = diarizer._chunk_labels(audio)
        pairs, vectors = diarizer._pair_embeddings(audio, labels)
        stacked = np.stack(vectors)
        ref = parse_rttm(channel.ref_path)
        cells = []
        for arm in ARMS:
            clusters = own_loop._cluster(stacked, 0.5, channel.num_speakers + 1, arm)
            turns = diarizer._assemble(len(audio), labels, pairs, clusters)
            der = score_der(ref, [Turn(t.speaker, t.start, t.end) for t in turns]).der
            totals[arm].append(der)
            cells.append(f"{der:.1%}")
        rows.append(f"| {channel.id} | " + " | ".join(cells) + " |")
        print(rows[-1])

    rows.append(
        "| **mean** | "
        + " | ".join(f"**{np.mean(totals[arm]):.1%}**" for arm in ARMS)
        + " |"
    )
    lines = [
        "## Known-count partitioner sweep, frozen embeddings (linkage_sweep.py)",
        "",
        *rows,
    ]
    text = "\n".join(lines)
    print()
    print(text)
    out = OUT_DIR / "diar-linkage-sweep.md"
    out.write_text(text + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
