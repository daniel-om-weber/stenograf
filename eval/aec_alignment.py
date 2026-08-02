"""Is the far-end reference aligned with the echo it caused -- and by how much off?

The tool that found the Windows bug, kept because the same defect is possible on
any platform that stamps its two channels on *arrival*: ``SessionClock`` gives
each channel its own transport latency as a constant offset, and
``EchoCanceller`` pairs the channels by timestamp. When the tap's path is the
slower one, AEC3 is handed a reference that arrives *after* its own echo, and
since it aligns the near end against far-end **history** only, it cancels
nothing. Measured that way on Windows: 2.6 dB ERLE, and far-end speech
transcribed as the local speaker.

    steno start --local 1 --remote 1 --max-seconds 80 --aec-dump probe \\
        --out probe-meeting        # ...speech over the speakers, with pauses...
    uv run python eval/aec_alignment.py probe

It answers two questions and prints the constant to ship:

1. **Where the echo sits.** The dump's WAVs share the capture clock's t=0, so a
   negative mic-vs-reference lag is the pathological case -- physically the mic
   can only *trail* the speakers, so a lead can only be labelling.
2. **What correction fixes it.** The candidate is swept through the real
   :class:`~stenograf.aec.EchoCanceller` -- the shipped code path, fed the dump's
   own timestamps -- so the recommendation is measured rather than reasoned.

Run it on any platform whose capture provider opens one transport per channel
(Windows' WASAPI pair, Linux's two ``parec`` processes). macOS is exempt by
construction: its helper anchors both channels to one clock.

Everything printed is ASCII on purpose -- see ``eval/aec_echo_present.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from common import read_pcm16

from stenograf.aec import TICK_SAMPLES, EchoCanceller
from stenograf.capture.base import SAMPLE_RATE, AudioFrame, Channel

FAR_ACTIVE_DBFS = -50.0
FEED_SAMPLES = SAMPLE_RATE // 10
"""Replay chunk. Small enough that the mic backlog while it waits for a
corrected reference stays far inside the canceller's hold budget."""

SWEEP_MS = (0, 20, 40, 60, 80, 100, 150, 250)
SAFETY_FACTOR = 2.5
"""What to multiply a measured lag by when declaring ``far_end_lag_s``.

The error is one-sided: a reference that arrives early is what AEC3's delay
estimator searches, one that arrives late is unusable. Windows' sweep was flat
from 60 ms to 250 ms (15.1 -> 14.7 dB), so overshooting costs a fraction of a
decibel and undershooting costs the canceller -- and the true lag varies with the
driver's device period, which one machine cannot measure for every other.
"""

MIN_RECOMMENDED_S = 0.1
"""Floor under any nonzero recommendation, because **one dump is one sample**.

Each channel anchors at its own first frame, so the offset is fixed within a run
but re-rolled at every start: two 80 s runs on the same Windows machine, minutes
apart, measured 60 ms and 10-25 ms. Scaling the smaller of those by 2.5 would
have shipped 25 ms and left the canceller dead on the very next meeting.
"""


def envelope(signal: np.ndarray) -> np.ndarray:
    count = signal.size // TICK_SAMPLES
    frames = signal[: count * TICK_SAMPLES].astype(np.float64).reshape(count, TICK_SAMPLES)
    return np.sqrt((frames**2).mean(axis=1))


def best_lag(near: np.ndarray, far: np.ndarray, span: int = 100) -> tuple[int, float]:
    """Lag in ticks that maximizes correlation, and the correlation there.

    Positive means the mic trails the reference, which is the only physical case.
    """
    near, far = near - near.mean(), far - far.mean()
    scores: dict[int, float] = {}
    for lag in range(-span, span + 1):
        a = near[lag:] if lag >= 0 else near[: near.size + lag]
        b = far[: far.size - lag] if lag >= 0 else far[-lag:]
        count = min(a.size, b.size)
        if count < 100:
            continue
        scale = float(np.linalg.norm(a[:count]) * np.linalg.norm(b[:count]))
        scores[lag] = float(a[:count] @ b[:count]) / scale if scale else 0.0
    lag = max(scores, key=lambda key: scores[key])
    return lag, scores[lag]


def sample_lag(mic: np.ndarray, lpb: np.ndarray, span: int = SAMPLE_RATE // 5) -> float:
    """Lag in milliseconds over the loudest 8 s of far end, at sample resolution.

    Correlated through the FFT, not ``np.correlate``: the direct method is
    quadratic and this window is 128k samples, which took minutes.
    """
    window = 800 * TICK_SAMPLES
    if lpb.size <= window:
        near, far = mic.astype(np.float64), lpb.astype(np.float64)
    else:
        # Coarse search on the tick envelope — 100x fewer points, same answer for
        # "where was the far end loudest".
        loudness = envelope(lpb)
        smoothed = np.convolve(loudness, np.ones(800) / 800, "valid")
        start = int(np.argmax(smoothed)) * TICK_SAMPLES
        near = mic[start : start + window].astype(np.float64)
        far = lpb[start : start + window].astype(np.float64)
    near, far = near - near.mean(), far - far.mean()
    size = 1 << int(np.ceil(np.log2(near.size + far.size)))
    correlation = np.fft.irfft(np.fft.rfft(near, size) * np.conj(np.fft.rfft(far, size)), size)
    # irfft wraps negative lags to the tail; -span..+span spans the two ends.
    lags = np.concatenate([correlation[-span:], correlation[: span + 1]])
    return (int(np.argmax(np.abs(lags))) - span) / SAMPLE_RATE * 1000


def cancel(mic: np.ndarray, lpb: np.ndarray, far_end_lag_s: float) -> np.ndarray:
    """Replay the dump through the real canceller with one candidate correction.

    The dump carries the provider's own timestamps (``WavTee`` pads each file's
    head to its first frame), so replaying sample index n at n / SAMPLE_RATE
    reproduces the misalignment the canceller met live.
    """
    aec = EchoCanceller({Channel.MIC, Channel.SYSTEM}, far_end_lag_s=far_end_lag_s)
    out: list[np.ndarray] = []
    for start in range(0, min(mic.size, lpb.size) - FEED_SAMPLES + 1, FEED_SAMPLES):
        timestamp = start / SAMPLE_RATE
        piece = slice(start, start + FEED_SAMPLES)
        # The tap first, as it arrives live: the reference for an instant has to
        # be in hand before the mic frame covering it can be cancelled.
        aec.process(AudioFrame(Channel.SYSTEM, timestamp, lpb[piece]))
        out += [f.samples for f in aec.process(AudioFrame(Channel.MIC, timestamp, mic[piece]))]
    out += [f.samples for f in aec.drain() if f.channel is Channel.MIC]
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int16)


def erle_db(mic: np.ndarray, enh: np.ndarray, playing: np.ndarray) -> tuple[float, float]:
    """ERLE and residual dBFS over the ticks where the far end was active."""

    def energy(signal: np.ndarray) -> float:
        count = min(playing.size, signal.size // TICK_SAMPLES)
        frames = signal[: count * TICK_SAMPLES].astype(np.float64).reshape(count, TICK_SAMPLES)
        return float((frames[playing[:count]] ** 2).mean())

    residual = energy(enh)
    return (
        10 * np.log10(energy(mic) / (residual + 1e-12)),
        10 * np.log10(residual / 32768.0**2 + 1e-12),
    )


def declared_lag_s() -> float:
    """What the capturing provider already compensates, if anything.

    A dump records frames as the provider stamped them, *before* the canceller
    corrects its own copy -- so on a provider that declares a correction, the
    raw misalignment still shows up here and has to be accounted for.

    **Every shipped provider now declares zero**, because every one of them
    stamps both channels from a single device clock. Windows was the last
    holdout: it declared 0.15 s while its two WASAPI streams were arrival-
    stamped in-process, and the constant went away with the capture helper that
    replaced them (PLAN-CAPTURE-HELPER.md). Pass ``--declared`` to score a dump
    taken by an older build.
    """
    return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dump", type=Path, help="directory an --aec-dump run wrote")
    parser.add_argument(
        "--no-sweep", action="store_true", help="measure the lag only (skips 8 AEC3 replays)"
    )
    parser.add_argument(
        "--declared",
        type=float,
        default=None,
        help="far_end_lag_s the capturing provider declared (default: this platform's)",
    )
    arguments = parser.parse_args()
    declared = declared_lag_s() if arguments.declared is None else arguments.declared

    mic = read_pcm16(arguments.dump / "mic.wav")
    lpb = read_pcm16(arguments.dump / "lpb.wav")
    count = min(mic.size, lpb.size)
    mic, lpb = mic[:count], lpb[:count]
    near, far = envelope(mic), envelope(lpb)
    playing = far > 10 ** (FAR_ACTIVE_DBFS / 20) * 32768

    print(f"duration            {count / SAMPLE_RATE:.1f} s")
    print(f"far end playing     {playing.mean() * 100:.0f}% of frames")
    if playing.mean() < 0.2 or playing.all():
        print("\nINCONCLUSIVE: the far end must play for part of the run and rest for part")
        return 2

    lag, correlation = best_lag(near, far)
    print(f"mic vs reference    {lag * 10:+d} ms on the tick grid (correlation {correlation:.2f})")
    print(f"                    {sample_lag(mic, lpb):+.1f} ms at sample resolution")
    if correlation < 0.3:
        print("\nINCONCLUSIVE: mic and reference barely correlate, so there is no echo path to")
        print("  align. Run eval/aec_echo_present.py first -- raise the output volume.")
        return 2

    # Per-quarter numbers are diagnostics, not a verdict. Deliberately no drift
    # warning: over 20 s the peak is not prominent enough to separate a moving
    # clock from estimator noise -- one 80 s dump reported -60/-60/-60/-60 and
    # another +0/-250/-10/+0 with the same provider, and the second one is noise
    # (a quarter with almost no far-end activity correlates silence against
    # silence and reports a confident lag at whatever offset). Drift over a
    # meeting-length run is what the long AEC session exists to answer.
    quarter = near.size // 4
    print("per-quarter          ", end="")
    for index in range(4):
        piece = slice(index * quarter, (index + 1) * quarter)
        value, score = best_lag(near[piece], far[piece])
        print(f"{value * 10:+d} ms (r={score:.2f})  ", end="")
    print("\n                     low r means the estimate is noise; trust the whole-run figure")

    needed = -lag * 10 / 1000  # seconds of correction the measurement asks for
    print(f"declared correction {declared:.2f}s (this provider's far_end_lag_s)")

    if lag >= 0:
        print("\nPASS: the reference arrives before its own echo, which is what AEC3 needs.")
        print("  This provider needs no far_end_lag_s. Confirm with eval/aec_score.py.")
        return 0
    if declared >= needed:
        print(f"\nPASS: the reference arrives {needed * 1000:.0f} ms after its own echo in the")
        print("  dump -- which is the raw provider timeline, and the declared correction")
        print("  already covers it. The canceller reads the reference that much further")
        print("  ahead, so this dump is expected to look misaligned. Judge the result by")
        print("  eval/aec_score.py's ERLE and by Local-N lines in the transcript.")
        return 0

    print(f"\nFAIL: the reference arrives {needed * 1000:.0f} ms AFTER its own echo, and only")
    print(f"  {declared:.2f}s of that is declared.")
    print("  Physically impossible for a real echo path, so it is the timestamps: this")
    print("  provider's tap is the slower transport and both channels are stamped on")
    print("  arrival. AEC3 searches its far-end history backwards only, so it cancels")
    print("  nothing -- the transport must hand over device timestamps (native/stenocap).")
    if arguments.no_sweep:
        return 1

    print("\nSweeping the correction through the real canceller:")
    print(f"{'far_end_lag_s':>14}  {'ERLE':>8}  {'residual':>10}")
    best = (0.0, -np.inf)
    for shift_ms in SWEEP_MS:
        erle, residual = erle_db(mic, cancel(mic, lpb, shift_ms / 1000), playing)
        marker = "  <- shipped default" if shift_ms == 0 else ""
        print(f"{shift_ms / 1000:>13.2f}s  {erle:>5.1f} dB  {residual:>7.1f} dBFS{marker}")
        if erle > best[1]:
            best = (shift_ms / 1000, erle)

    recommended = max(round(needed * SAFETY_FACTOR, 3), MIN_RECOMMENDED_S)
    print(f"\nbest measured: {best[0]:.2f}s ({best[1]:.1f} dB ERLE)")
    print(f"RECOMMENDED far_end_lag_s = {recommended} ({SAFETY_FACTOR}x measured, floor "
          f"{MIN_RECOMMENDED_S})")
    print("  Deliberately past the best: the sweep plateaus, so overshooting costs a")
    print("  fraction of a decibel while undershooting costs the whole canceller. And this")
    print("  is one run -- the offset is re-rolled every time the two streams anchor, so")
    print("  measure two or three before trusting a number near the floor.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
