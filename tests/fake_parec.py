"""Fake ``parec`` for LinuxCaptureProvider tests: raw s16le PCM on stdout.

Reads the ``--device=`` flag the provider passes to tell which channel it is
playing (``@DEFAULT_SOURCE@`` = mic, ``@DEFAULT_MONITOR@`` = system) and fills
every sample with a channel-distinct value so tests can assert routing.

Duration comes from the environment: ``FAKE_PAREC_SECONDS`` (default 0.4s,
then a clean EOF), overridable per channel via ``FAKE_PAREC_MIC_SECONDS`` /
``FAKE_PAREC_SYSTEM_SECONDS``; a negative value streams forever (until the
provider terminates the process, like real parec).
"""

import os
import sys
import time

RATE = 16_000

device = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--device=")), "?")
mic = "SOURCE" in device
value = 1 if mic else 2
sample = int(value).to_bytes(2, "little", signed=True)

key = "FAKE_PAREC_MIC_SECONDS" if mic else "FAKE_PAREC_SYSTEM_SECONDS"
seconds = float(os.environ.get(key, os.environ.get("FAKE_PAREC_SECONDS", "0.4")))

out = sys.stdout.buffer
if seconds < 0:
    while True:  # stream until terminated
        out.write(sample * (RATE // 10))
        out.flush()
        time.sleep(0.05)
out.write(sample * int(RATE * seconds))
out.flush()
# Linger so the sibling channel finishes writing before this exit triggers the
# provider's whole-capture teardown (real parec streams until terminated; only
# the fake's burst-write pattern can race it).
time.sleep(0.5)
