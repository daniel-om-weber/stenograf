"""A stand-in for the native ``stenocap`` helper, for provider tests.

Speaks the same wire protocol (see stenograf.capture.macos): parses --mic /
--system flags and streams framed int16 PCM on stdout. With --forever it emits
until SIGINT (to test stop()); with --frames N it emits N frames per channel as
fast as it can and exits (to test the drain thread against a stalled consumer);
with --malformed it emits one valid frame, then a garbage header (to test that
stream desync reaches the consumer); otherwise it emits a few frames and exits
(to test natural end-of-stream). With --chatter it also logs status lines to
stderr the way the real helper does (format info at start, a WARNING, and
"stopped" at exit), to test the stderr routing.
"""

import signal
import struct
import sys
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
    if chatter:
        log("mic format: 48000.0 Hz, 1 ch")
        log("WARNING channel 0 drifted 12 ms from wall clock")
    total = 3
    if "--frames" in sys.argv[1:]:
        total = int(sys.argv[sys.argv.index("--frames") + 1])

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
        if not forever and index >= total:
            if chatter:
                log("stopped")
            return
        if forever:
            time.sleep(0.02)


if __name__ == "__main__":
    main()
