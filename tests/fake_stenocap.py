"""A stand-in for the native ``stenocap`` helper, for provider tests.

Speaks the same wire protocol (see stenograf.capture.helper): parses --mic /
--system flags and streams framed int16 PCM on stdout. With --forever it emits
until stopped (to test stop()); with --frames N it emits N frames per channel as
fast as it can and exits (to test the drain thread against a stalled consumer);
with --malformed it emits one valid frame, then a garbage header (to test that
stream desync reaches the consumer); otherwise it emits a few frames and exits
(to test natural end-of-stream). With --chatter it also logs status lines to
stderr the way the real helper does (format info at start, a WARNING, and
"stopped" at exit), to test the stderr routing.

**Two stop gestures, because the two real helpers have two.** By default it
takes SIGINT/SIGTERM, like the Swift helper; with --stop-on-stdin it also exits
when stdin reaches EOF, which is the only way the Windows helper can be stopped
(see HelperCaptureProvider._stop_signal).

With --devices it prints the device JSON for the requested channels and exits,
standing in for the preflight the Windows provider runs before a meeting.

Crash modes (the real helpers die like this when the platform's audio service is
wedged by a concurrent capture app): --die-at-start logs a FATAL line and exits 1
before any frame; --die-once MARKER does that only when MARKER doesn't exist yet
(creating it), so a respawn succeeds — the provider's retry path; --die-after N
emits N frames per channel, then crashes — a mid-meeting death. --die-at-start
also applies to --devices.
"""

import json
import os
import signal
import struct
import sys
import threading
import time

HEADER = struct.Struct("<BdI")
CODE = {"--mic": 0, "--system": 1}
FRAME_SAMPLES = 1600  # 0.1 s at 16 kHz


def emit(code: int, index: int) -> None:
    payload = struct.pack(f"<{FRAME_SAMPLES}h", *([index + 1] * FRAME_SAMPLES))
    sys.stdout.buffer.write(HEADER.pack(code, index * 0.1, FRAME_SAMPLES))
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def log(message: str) -> None:
    print(f"fake-stenocap: {message}", file=sys.stderr, flush=True)


def main() -> None:
    chatter = "--chatter" in sys.argv[1:]

    def bail(*_: object) -> None:
        if chatter:  # the real helper logs "stopped" on its way out
            log("stopped")
        sys.exit(0)

    signal.signal(signal.SIGINT, bail)
    signal.signal(signal.SIGTERM, bail)
    channels = [CODE[a] for a in sys.argv[1:] if a in CODE]
    forever = "--forever" in sys.argv[1:]

    if "--devices" in sys.argv[1:]:
        if "--die-at-start" in sys.argv[1:]:
            log("FATAL: no default system device, check Windows sound settings")
            sys.exit(1)
        wanted = [flag for flag, code in CODE.items() if code in channels] or list(CODE)
        names = {f.removeprefix("--"): f"Fake {f.removeprefix('--')} device" for f in wanted}
        print(json.dumps(names), flush=True)
        return

    if "--stop-on-stdin" in sys.argv[1:]:
        # The Windows stop gesture: the parent closes the pipe, nothing is ever
        # sent on it. A daemon thread so the emit loop below stays the same —
        # and os._exit, because sys.exit off the main thread only ends the
        # thread and would leave the helper running. os.read, not
        # sys.stdin.buffer.read: a natural exit (fixed frame count, crash
        # modes) would otherwise abort interpreter shutdown on the buffer
        # lock this thread holds while blocked.
        def on_eof() -> None:
            while os.read(0, 4096):
                pass
            if chatter:
                log("stopped")
            sys.stderr.flush()
            os._exit(0)

        threading.Thread(target=on_eof, daemon=True).start()

    if chatter:
        log("mic format: 48000.0 Hz, 1 ch")
        log("WARNING channel 0 drifted 12 ms from wall clock")
    total = 3
    if "--frames" in sys.argv[1:]:
        total = int(sys.argv[sys.argv.index("--frames") + 1])

    if "--die-at-start" in sys.argv[1:]:
        log("FATAL: system tap unavailable — another app is capturing?")
        sys.exit(1)

    if "--die-once" in sys.argv[1:]:
        marker = sys.argv[sys.argv.index("--die-once") + 1]
        if not os.path.exists(marker):
            open(marker, "w").close()
            log("FATAL: flaky start (first attempt)")
            sys.exit(1)

    die_after = None
    if "--die-after" in sys.argv[1:]:
        die_after = int(sys.argv[sys.argv.index("--die-after") + 1])

    if "--malformed" in sys.argv[1:]:
        emit(channels[0] if channels else 0, 0)
        sys.stdout.buffer.write(HEADER.pack(7, 0.0, 0))  # 7 is no channel code
        sys.stdout.buffer.flush()
        return

    index = 0
    while True:
        for code in channels:
            emit(code, index)
        index += 1
        if die_after is not None and index >= die_after:
            log("FATAL: stream died mid-meeting")
            sys.exit(1)
        if not forever and index >= total:
            if chatter:
                log("stopped")
            return
        if forever:
            time.sleep(0.02)


if __name__ == "__main__":
    main()
