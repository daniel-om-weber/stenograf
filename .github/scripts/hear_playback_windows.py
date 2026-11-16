"""The "Who spoke?" play button on Windows, heard back through VB-CABLE.

stenograf's player (waveOut) plays into the default output — the cable's
"CABLE Input", the runner's only endpoint once windows-sound-card.ps1 ran —
and the bundled ffmpeg records the cable's "CABLE Output" over DirectShow.
Two checks, the Linux one plus the one Linux cannot need:

1. a 4 s tone (a real sample's length) arrives whole — at least 3.6 s of it;
2. stop() cuts a playing sample short: the player reports done promptly and
   no more than ~1.5 s of tone reaches the recording.
"""

from __future__ import annotations

import re
import subprocess
import sys
import threading
import time

import numpy as np

from stenograf.audio import SAMPLE_RATE, ffmpeg_exe
from stenograf.playback import Playback

AMPLITUDE = 0.3
ABOVE = 1000
LOUD_FRACTION = 0.935  # a 0.3-amplitude sine sits above 1000 for ~93.5 % of its samples


def cable_output() -> str:
    listing = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
        capture_output=True,
        text=True,
    ).stderr
    print(listing)
    match = re.search(r'"([^"]*CABLE Output[^"]*)" \(audio\)', listing)
    if match is None:
        sys.exit("no 'CABLE Output' DirectShow device — did the driver install?")
    return match.group(1)


def record(device: str, seconds: float) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-f", "dshow",
         "-i", f"audio={device}", "-t", str(seconds), "-ac", "1", "-ar", str(SAMPLE_RATE),
         "-f", "s16le", "-"],  # fmt: skip
        stdout=subprocess.PIPE,
    )


def heard_seconds(raw: bytes) -> float:
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.int32)
    return float(np.count_nonzero(np.abs(samples) > ABOVE)) / (SAMPLE_RATE * LOUD_FRACTION)


def tone(seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (AMPLITUDE * 32767 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)


def play(clip: np.ndarray, stop_after: float | None = None) -> tuple[float, list[str | None]]:
    done, errors = threading.Event(), []
    started = time.monotonic()
    player = Playback(clip, lambda error: (errors.append(error), done.set())).start()
    if stop_after is not None:
        time.sleep(stop_after)
        player.stop()
    if not done.wait(30):
        sys.exit("the player never reported done")
    return time.monotonic() - started, errors


def main() -> None:
    device = cable_output()

    recorder = record(device, 7)
    time.sleep(1.5)  # DirectShow takes a moment to open the device
    elapsed, errors = play(tone(4))
    raw, _ = recorder.communicate()
    heard = heard_seconds(raw)
    print(f"whole sample: played in {elapsed:.2f} s, heard {heard:.2f} of 4 s, errors {errors}")
    assert errors == [None], errors
    assert heard >= 3.6, "the tone did not arrive whole"

    recorder = record(device, 6)
    time.sleep(1.5)
    elapsed, errors = play(tone(4), stop_after=1.0)
    raw, _ = recorder.communicate()
    heard = heard_seconds(raw)
    print(f"stopped sample: done after {elapsed:.2f} s, heard {heard:.2f} s, errors {errors}")
    assert errors == [None], errors
    assert elapsed < 2.5, "stop() did not end the sample promptly"
    assert heard <= 1.5, "the sample kept playing after stop()"


if __name__ == "__main__":
    main()
