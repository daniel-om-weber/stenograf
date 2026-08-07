"""Loop parity: our owned diarization loop vs sherpa's, same models, raw turns.

`PLAN-DIARIZATION.md` step 4's first gate: before any clustering change, the
reimplemented loop (`stenograf.diarization.loop.OwnDiarizer`, built from
`eval/diarization-loop-spec.md`) must score DER within noise of sherpa's
`OfflineSpeakerDiarization` on every corpus loop channel, called exactly as
production calls it (raw `diarize(k+1)`, no fold). Bit-parity is not expected
— the spec records where the reference is internally inconsistent (§1.5) and
our embedding stream concatenates audio rather than feature frames — so the
gate is the score, plus a turn-level diff to see *where* any gap lives.

Runs each channel in-process, both arms back to back (the sherpa leak that
forced per-channel process isolation is fixed at the 1.13.4 floor)::

    uv run --group eval eval/loop_parity.py [--segments ES2003b.loop,...]
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
from common import OUT_DIR, read_pcm16
from der import score_der
from rttm import Turn, parse_rttm


def turn_diff(a, b) -> float:
    """Mean absolute boundary drift (s) between two turn lists after sorting —
    0 for identical output, large for structural disagreement; ∞ when the
    turn counts differ (structural, not boundary, disagreement)."""
    if len(a) != len(b):
        return float("inf")
    if not a:
        return 0.0
    pairs = zip(
        sorted(a, key=lambda t: (t.start, t.end)),
        sorted(b, key=lambda t: (t.start, t.end)),
        strict=True,
    )
    return float(np.mean([abs(x.start - y.start) + abs(x.end - y.end) for x, y in pairs]))


class _SherpaReference:
    """sherpa's ``OfflineSpeakerDiarization`` invoked directly from the
    installed package — production deleted its wrapper (the owned loop is the
    one path since 2026-08-07), but the parity gate must stay re-runnable
    against the original reference implementation."""

    def __init__(self) -> None:
        self._pipeline = None
        self._num_clusters = -1

    def diarize(self, samples, num_speakers):
        import sherpa_onnx

        from stenograf import assets
        from stenograf.audio import to_float32
        from stenograf.diarization.base import SpeakerTurn
        from stenograf.diarization.sherpa import _num_threads

        if self._pipeline is None or self._num_clusters != num_speakers:
            config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
                segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                    pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                        model=str(assets.fetch(assets.PYANNOTE_SEGMENTATION))
                    ),
                    num_threads=_num_threads(),
                ),
                embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                    model=str(assets.fetch(assets.SPEAKER_EMBEDDING)),
                    num_threads=_num_threads(),
                ),
                clustering=sherpa_onnx.FastClusteringConfig(
                    num_clusters=num_speakers, threshold=0.5
                ),
            )
            if self._pipeline is None:
                self._pipeline = sherpa_onnx.OfflineSpeakerDiarization(config)
            else:
                self._pipeline.set_config(config)  # models stay loaded
            self._num_clusters = num_speakers
        result = self._pipeline.process(to_float32(samples))
        return [
            SpeakerTurn(speaker=f"S{seg.speaker}", start=seg.start, end=seg.end)
            for seg in result.sort_by_start_time()
        ]


def main() -> int:
    import ami

    from stenograf.diarization.loop import OwnDiarizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", help="comma-separated channel ids (default: all loops)")
    args = parser.parse_args()
    wanted = set(args.segments.split(",")) if args.segments else None

    channels = [
        c
        for c in ami.load_channels()
        if c.num_speakers > 1 and (wanted is None or c.id in wanted)
    ]
    if not channels:
        print("no loop channels — run `eval/ami.py fetch` first", file=sys.stderr)
        return 1

    sherpa = _SherpaReference()
    own = OwnDiarizer(cluster_method="complete")  # parity is against the reference loop
    rows = [
        "| channel | sherpa DER | own DER | Δ | sherpa turns | own turns "
        "| boundary drift | sherpa s | own s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    deltas = []
    for channel in channels:
        pcm = read_pcm16(channel.wav_path)
        ref = parse_rttm(channel.ref_path)
        k = channel.num_speakers + 1  # production calls at k+1 (fold recovers)

        started = time.monotonic()
        sherpa_turns = sherpa.diarize(pcm, k)
        sherpa_s = time.monotonic() - started
        started = time.monotonic()
        own_turns = own.diarize(pcm, k)
        own_s = time.monotonic() - started

        def rttm(turns):
            return [Turn(t.speaker, t.start, t.end) for t in turns]

        sherpa_der = score_der(ref, rttm(sherpa_turns)).der
        own_der = score_der(ref, rttm(own_turns)).der
        drift = turn_diff(sherpa_turns, own_turns)
        deltas.append(own_der - sherpa_der)
        rows.append(
            f"| {channel.id} | {sherpa_der:.1%} | {own_der:.1%} "
            f"| {own_der - sherpa_der:+.1%} | {len(sherpa_turns)} | {len(own_turns)} "
            f"| {drift if drift != float('inf') else float('nan'):.3f} "
            f"| {sherpa_s:.0f} | {own_s:.0f} |"
        )
        print(rows[-1])

    lines = [
        "## Own loop vs sherpa, raw diarize(k+1) (loop_parity.py)",
        "",
        *rows,
        "",
        f"Mean ΔDER (own − sherpa): {np.mean(deltas):+.2%}; "
        f"max regression {max(deltas):+.2%} on {len(deltas)} channels.",
    ]
    text = "\n".join(lines)
    print()
    print(text)
    out = OUT_DIR / "diar-loop-parity.md"
    out.write_text(text + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
