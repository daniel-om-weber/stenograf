"""Score an ``--aec-dump`` triple: how much echo went, at what cost (PLAN-AEC.md §3).

Takes the mic/lpb/enh directory a ``steno start --aec-dump DIR`` run wrote and
reports the layer-0 signal metrics:

1. **ERLE** — energy the canceller removed from the mic while the far end was
   active. Only meaningful when nobody spoke locally (far-only scenario);
   during double-talk local speech dominates both sides and the number says
   nothing. Computed on energy over 10 ms frames, because AEC3's output is
   fractionally delayed and waveform correlation is useless on it.
2. **Residual level** — what the ASR actually receives during far-end activity
   (dBFS). The shipped canceller leaves ~−70 dBFS: inaudible, still decodeable.
3. **AECMOS** (Microsoft AEC Challenge metric, via ``speechmos``) — a learned
   score with the two axes that matter here: ``echo_mos`` (is echo gone) and
   ``deg_mos`` (did we damage the near-end speech getting there). Scored in
   20 s windows (the model's limit) and averaged. Pass the scenario when the
   rig knows it (``--scenario dt|st|nst``); without it the scenarioless model
   is used.

Usage:
    uv run --group eval eval/aec_score.py DUMP_DIR [--scenario dt] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
FRAME = SAMPLE_RATE // 100  # 10 ms, the AEC tick
FAR_ACTIVE_DBFS = -50.0
"""A 10 ms lpb frame above this counts as 'the speakers were playing'."""

AECMOS_WINDOW_S = 19.99  # the model truncates (and warns) at >= 20 s, so stay just under
AECMOS_MIN_TAIL_S = 5.0  # a shorter leftover window scores noise, drop it

TRIPLE = ("mic", "lpb", "enh")


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1 or w.getframerate() != SAMPLE_RATE or w.getsampwidth() != 2:
            raise SystemExit(f"{path} is not the mono 16 kHz int16 WAV --aec-dump writes")
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def frame_energies(x: np.ndarray) -> np.ndarray:
    """Mean-square energy per 10 ms frame, normalized to full scale."""
    n = x.size // FRAME
    frames = x[: n * FRAME].astype(np.float64).reshape(n, FRAME)
    return (frames**2).mean(axis=1) / 32768.0**2


def dbfs(energy: float) -> float:
    return 10 * np.log10(energy + 1e-12)


def signal_metrics(mic: np.ndarray, lpb: np.ndarray, enh: np.ndarray) -> dict:
    n = min(x.size // FRAME for x in (mic, lpb, enh))
    mic_e, lpb_e, enh_e = (frame_energies(x)[:n] for x in (mic, lpb, enh))
    active = dbfs_v(lpb_e) > FAR_ACTIVE_DBFS
    metrics = {
        "duration_s": round(n * FRAME / SAMPLE_RATE, 1),
        "far_active_s": round(int(active.sum()) * FRAME / SAMPLE_RATE, 1),
    }
    if not active.any():
        return metrics
    metrics["erle_db"] = round(dbfs(mic_e[active].mean()) - dbfs(enh_e[active].mean()), 1)
    metrics["mic_during_far_dbfs"] = round(dbfs(mic_e[active].mean()), 1)
    metrics["residual_dbfs"] = round(dbfs(enh_e[active].mean()), 1)
    return metrics


def dbfs_v(energy: np.ndarray) -> np.ndarray:
    return 10 * np.log10(energy + 1e-12)


def aecmos_metrics(mic: np.ndarray, lpb: np.ndarray, enh: np.ndarray, scenario: str | None) -> dict:
    from speechmos import aecmos  # deferred: imports onnxruntime + librosa

    n = min(mic.size, lpb.size, enh.size)
    window = int(AECMOS_WINDOW_S * SAMPLE_RATE)
    starts = list(range(0, n, window))
    echo, deg = [], []
    for start in starts:
        stop = min(start + window, n)
        if stop - start < AECMOS_MIN_TAIL_S * SAMPLE_RATE and echo:
            break  # a stub window scores noise; the full windows already cover the run
        sample = {
            "lpb": lpb[start:stop].astype(np.float32) / 32768.0,
            "mic": mic[start:stop].astype(np.float32) / 32768.0,
            "enh": enh[start:stop].astype(np.float32) / 32768.0,
        }
        scores = aecmos.run(sample, SAMPLE_RATE, talk_type=scenario)
        echo.append(scores["echo_mos"])
        deg.append(scores["deg_mos"])
    return {
        "aecmos_echo": round(float(np.mean(echo)), 2),
        "aecmos_deg": round(float(np.mean(deg)), 2),
        "aecmos_windows": len(echo),
        "aecmos_model": "scenarioless" if scenario is None else scenario,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dump", type=Path, help="directory a --aec-dump run wrote")
    parser.add_argument(
        "--scenario",
        choices=["dt", "st", "nst"],
        default=None,
        help="AECMOS talk type when the run's scenario is known: dt double-talk, "
        "st far-end single-talk, nst near-end single-talk [default: scenarioless]",
    )
    parser.add_argument("--json", type=Path, default=None, help="also write metrics as JSON")
    parser.add_argument(
        "--no-aecmos", action="store_true", help="signal metrics only (skips model inference)"
    )
    args = parser.parse_args()

    paths = {name: args.dump / f"{name}.wav" for name in TRIPLE}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit(f"not an --aec-dump directory, missing: {', '.join(missing)}")
    mic, lpb, enh = (read_wav(paths[name]) for name in TRIPLE)

    metrics = signal_metrics(mic, lpb, enh)
    if "erle_db" not in metrics:
        print(f"far end was never active above {FAR_ACTIVE_DBFS:.0f} dBFS — nothing to score")
        sys.exit(1)
    if not args.no_aecmos:
        metrics |= aecmos_metrics(mic, lpb, enh, args.scenario)

    lines = [
        f"duration          {metrics['duration_s']:>7.1f} s"
        f"  (far end active {metrics['far_active_s']:.1f} s)",
        f"mic during far    {metrics['mic_during_far_dbfs']:>7.1f} dBFS",
        f"residual (enh)    {metrics['residual_dbfs']:>7.1f} dBFS",
        f"ERLE              {metrics['erle_db']:>7.1f} dB"
        "   (far-only runs only; meaningless under double-talk)",
    ]
    if "aecmos_echo" in metrics:
        lines += [
            f"AECMOS echo       {metrics['aecmos_echo']:>7.2f} /5   (higher = less audible echo)",
            f"AECMOS degradation{metrics['aecmos_deg']:>7.2f} /5"
            "   (higher = near end better preserved)",
        ]
    print("\n".join(lines))

    if args.json:
        args.json.write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
