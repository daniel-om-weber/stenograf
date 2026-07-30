"""Is there an echo path at all? Run this *before* a long AEC measurement.

The one-minute check that would have saved a 33-minute run. On 2026-07-26 a
full-length speakers-not-headphones AEC session was captured on the Windows
notebook and scored 0.7 dB ERLE — which reads as a broken canceller and was
nothing of the sort: the microphone level was the same whether the speakers
were playing or silent, so no echo ever reached the mic and there was nothing
to cancel. ``eval/aec_score.py`` cannot tell those two apart, because ERLE is
undefined without an echo path; this script tests the precondition instead.

Point it at a short ``--aec-dump`` directory (60 s is plenty)::

    steno start --local 1 --remote 1 --max-seconds 60 --aec-dump /tmp/probe \\
        --out /tmp/probe-meeting
    # …with speech playing over the speakers…
    uv run python eval/aec_echo_present.py /tmp/probe

Everything this script *prints* is ASCII, unlike the rest of ``eval/``. That is
not a style choice: this is the one eval tool written to be run on Windows, and
a piped Python stdout there encodes as cp1252, which raises
``UnicodeEncodeError`` on the em-dash the house style would otherwise use — at
the end of the run, after the audio is gone. ``aec_score.py`` follows the same
rule for the same reason; the rest of the harness is macOS-only.

A **pass** means the mic hears the speakers well enough for the canceller to
have work to do. A **fail** means fix the room before measuring anything: raise
the output volume, and on Windows turn the capture endpoint's driver processing
off (Settings → System → Sound → the input device → Audio enhancements → Off) —
a Realtek-class APO doing its own echo suppression removes the very signal the
measurement is about, upstream of us.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from common import read_pcm16

SAMPLE_RATE = 16000
FRAME = SAMPLE_RATE // 100
FAR_ACTIVE_DBFS = -50.0
"""A 10 ms loopback frame above this counts as "the speakers were playing"."""

ECHO_MARGIN_DB = 6.0
"""How much louder the mic must be while the far end plays than while it is
silent. Six decibels is a quadrupling of energy — comfortably above room-noise
variation, and far below the 20–30 dB a speaker at conversational level puts
into a laptop's own microphone."""

MIN_FAR_FRACTION = 0.2
"""Below this the far end barely played and the comparison has no basis."""


def envelope(signal: np.ndarray) -> np.ndarray:
    count = signal.size // FRAME
    frames = signal[: count * FRAME].astype(np.float64).reshape(count, FRAME)
    return np.sqrt((frames**2).mean(axis=1))


def dbfs(level: float) -> float:
    return 20 * np.log10(level / 32768.0 + 1e-12)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dump", type=Path, help="directory a --aec-dump run wrote")
    arguments = parser.parse_args()

    mic = envelope(read_pcm16(arguments.dump / "mic.wav"))
    far = envelope(read_pcm16(arguments.dump / "lpb.wav"))
    count = min(mic.size, far.size)
    mic, far = mic[:count], far[:count]
    playing = far > 10 ** (FAR_ACTIVE_DBFS / 20) * 32768

    print(f"duration            {count / 100:.1f} s")
    print(f"far end playing     {playing.mean() * 100:.0f}% of frames")
    if playing.mean() < MIN_FAR_FRACTION or playing.all():
        print("\nINCONCLUSIVE: the far end must play for part of the run and rest for part")
        return 2

    loud, quiet = dbfs(mic[playing].mean()), dbfs(mic[~playing].mean())
    print(f"mic while playing   {loud:.1f} dBFS")
    print(f"mic while silent    {quiet:.1f} dBFS")
    print(f"echo above noise    {loud - quiet:.1f} dB  (need >= {ECHO_MARGIN_DB:.0f})")

    if loud - quiet >= ECHO_MARGIN_DB:
        print("\nPASS: the mic hears the speakers; an AEC measurement here means something")
        return 0
    print(
        "\nFAIL: no echo path, so ERLE would be undefined and a long run would measure "
        "nothing.\n  Raise the output volume first -- 40% on a laptop chassis was not "
        "enough and 90% was\n  (2026-07-26); if that is not it, turn the microphone's audio "
        "enhancements off, since a\n  driver APO suppressing echo upstream of us looks "
        "exactly like this."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
