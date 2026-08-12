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
--list-inputs prints the microphone inventory the picker lists, and
--mic-device pins the mic channel to one of them by id or name — resolved with
the rule the real helpers use (trim, NFC-normalize, id first then exact name),
and FATAL when it names nothing.

**Unknown arguments are refused** (exit 2 with the usage line) and ``--help``
succeeds, exactly as the real helpers do: a Python that passes a flag its
helper binary predates must fail loudly rather than record the default
microphone while the UI says otherwise.

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
import unicodedata

HEADER = struct.Struct("<BdI")
CODE = {"--mic": 0, "--system": 1}
FRAME_SAMPLES = 1600  # 0.1 s at 16 kHz

INPUTS = [
    {"id": "fake-mic-builtin", "name": "Fake Built-in Microphone", "default": True},
    {"id": "fake-mic-usb", "name": "Fake USB Microphone", "default": False},
]
"""The inventory --list-inputs reports and --mic-device resolves against."""

USAGE = "usage: stenocap [--mic] [--system] [--mic-device ID] | --devices | --list-inputs"
FLAGS = (
    "--mic",
    "--system",
    "--devices",
    "--list-inputs",
    "-h",
    "--help",
    "--forever",
    "--malformed",
    "--chatter",
    "--stop-on-stdin",
    "--die-at-start",
)
VALUE_FLAGS = ("--frames", "--die-once", "--die-after", "--mic-device")
"""Flags that consume the next argument — including the test-only ones, whose
values would otherwise look like unknown arguments to the check below."""


def emit(code: int, index: int) -> None:
    payload = struct.pack(f"<{FRAME_SAMPLES}h", *([index + 1] * FRAME_SAMPLES))
    sys.stdout.buffer.write(HEADER.pack(code, index * 0.1, FRAME_SAMPLES))
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def log(message: str) -> None:
    print(f"fake-stenocap: {message}", file=sys.stderr, flush=True)


def parse(argv: list[str]) -> dict[str, object]:
    """argv as a dict of flags, refusing anything unrecognized (exit 2)."""
    parsed: dict[str, object] = {}
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg in FLAGS:
            parsed[arg] = True
        elif arg in VALUE_FLAGS:
            if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                log(f"{arg} needs a value")
                log(USAGE)
                sys.exit(2)
            parsed[arg] = argv[index + 1]
            index += 1
        else:
            log(f"unknown argument {arg}")
            log(USAGE)
            sys.exit(2)
        index += 1
    return parsed


def normalize(value: str) -> str:
    """The helpers' shared matching form: trimmed, NFC."""
    return unicodedata.normalize("NFC", value.strip())


def resolve_mic(pin: str) -> dict:
    """The device a --mic-device value names, or a FATAL exit."""
    wanted = normalize(pin)
    for device in INPUTS:
        if normalize(device["id"]) == wanted:
            return device
    named = [d for d in INPUTS if normalize(d["name"]) == wanted]
    if len(named) == 1:
        return named[0]
    if not named:
        log(
            f'FATAL: the selected microphone "{pin}" is not available — run `steno devices` '
            "to list what is connected, or pass --mic-device default"
        )
        sys.exit(1)
    log(f'FATAL: the microphone name "{pin}" matches {len(named)} connected devices')
    sys.exit(1)


def main() -> None:
    args = parse(sys.argv[1:])
    # --help succeeds, a bare invocation does not: the wheel smoke tests run
    # the first against the real binaries, and the fake states the contract.
    if "--help" in args or "-h" in args:
        log(USAGE)
        return
    chatter = "--chatter" in args

    def bail(*_: object) -> None:
        if chatter:  # the real helper logs "stopped" on its way out
            log("stopped")
        sys.exit(0)

    signal.signal(signal.SIGINT, bail)
    signal.signal(signal.SIGTERM, bail)
    channels = [CODE[flag] for flag in CODE if flag in args]
    forever = "--forever" in args
    pin = args.get("--mic-device")
    if isinstance(pin, str) and normalize(pin) == "default":
        pin = None  # the word every failure message offers as the way back
    if not channels and not ("--devices" in args or "--list-inputs" in args):
        log(f"{USAGE}  (at least one channel)")
        sys.exit(2)

    if "--list-inputs" in args:
        print(json.dumps(INPUTS), flush=True)
        return

    if "--devices" in args:
        if "--die-at-start" in args:
            log("FATAL: no default system device, check Windows sound settings")
            sys.exit(1)
        wanted = [flag for flag, code in CODE.items() if code in channels] or list(CODE)
        names = {f.removeprefix("--"): f"Fake {f.removeprefix('--')} device" for f in wanted}
        if "mic" in names and isinstance(pin, str):
            names["mic"] = resolve_mic(pin)["name"]
        print(json.dumps(names), flush=True)
        return

    if isinstance(pin, str):  # a pinned run resolves before it records anything
        resolve_mic(pin)

    if "--stop-on-stdin" in args:
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
    total = int(args.get("--frames", 3))  # type: ignore[arg-type]

    if "--die-at-start" in args:
        log("FATAL: system tap unavailable — another app is capturing?")
        sys.exit(1)

    if "--die-once" in args:
        marker = str(args["--die-once"])
        if not os.path.exists(marker):
            open(marker, "w").close()
            log("FATAL: flaky start (first attempt)")
            sys.exit(1)

    die_after = int(args["--die-after"]) if "--die-after" in args else None  # type: ignore[arg-type]

    if "--malformed" in args:
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
