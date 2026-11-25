# Native helpers

One helper lives here — **stenocap**, which captures — speaking a simple pipe
to the Python core under the rule **meeting audio never touches disk**. It has
two implementations, `stenocap-macos/` (Swift) and `stenocap/` (Rust, one
crate with a WASAPI backend for Windows and a PulseAudio-protocol backend for
Linux), speaking one wire protocol. Three platform wheels carry it (macOS
arm64, win_amd64, manylinux_2_28 x86_64); the Linux floor is stenocap's own,
since it links only libpulse, and `release.yml` asserts the highest glibc
symbol on the built wheel. Platforms with no tagged wheel (musl, Linux arm64)
install the pure `py3-none-any` wheel and have no live capture. Plus
`appbundle/`, which is not a helper at all but the sources for the macOS
desktop launcher `Stenograf.app`. It breaks this directory's convention on
purpose: its product is *committed* (`src/stenograf/assets/Stenograf.app`)
rather than built at install time, because macOS pins the app's microphone
grant to the bundle's exact bytes. Read `appbundle/README.md` before touching
anything in it.

`stenocap-macos/` holds **stenocap**, the ad-hoc-signed Swift binary that feeds
live audio to the Python core. It captures

- **system audio** via a Core Audio process tap (`AudioHardwareCreateProcessTap`,
  macOS 14.4+) — the remote participants in a call, and
- **microphone** via AVAudioEngine,

resamples both to mono 16 kHz int16 (AVAudioConverter), and streams them as
framed PCM on **stdout**. Audio is never written to disk — the helper only
streams it. `spike/` is the original throwaway proof that the tap + mic APIs
work; `stenocap-macos/` is the production version that adds resampling, framing,
and a clean lifecycle.

Echo cancellation is **not** done here. The helper once had a `--aec` flag
backed by Voice Processing IO (`setVoiceProcessingEnabled` on the input node).
Measured on macOS 26, two things killed it:

- **The mic goes dead.** VPIO reports the input as 7 or 9 channels (it varies
  per run) with no channel layout anyone configured, and AVAudioConverter then
  yields nothing: with VPIO on, no playback, and someone talking, the helper
  emitted *zero frames*; the same run without VPIO captured speech normally.
  Chromium drives VPIO as a raw AudioUnit with an explicitly forced mono format
  for this reason, so the API is likely usable — `AVAudioEngine`'s wrapper is not.
- **The system channel collapses.** With VPIO on, the process tap measured
  −64.5 dBFS while audio played, against −28.2 dBFS for the identical run
  without it. VPIO either ducks other applications' audio (WWDC23 "What's new in
  voice processing") or retargets the default output to its own private
  aggregate device. Either way it guts the remote-participant channel, which is
  the signal this tool exists to transcribe.

Echo is cancelled on the Python side instead, using the system channel as the
far-end reference — which is also Chrome's default (`kSystemLoopbackAsAecReference`
enabled, `kEnforceSystemEchoCancellation` disabled) despite shipping both.

**A pinned microphone takes one of two other paths here, and neither is
AVAudioEngine.** Measured on macOS 26.5.1 (2026-08-12): retargeting the engine's
input node with `kAudioOutputUnitProperty_CurrentDevice` while the tap's
aggregate runs wedges `engine.start()` in the HAL and it never returns (the main
thread waits forever on the shared client IO thread), and opening the device
with its own IO proc beside the aggregate wedges identically. Retargeting also
posts `AVAudioEngineConfigurationChange` 9–13 ms after each build, so the
supervisor rebuilt itself in a loop — nine rebuilds in ten seconds. So:

- **mic alone**: an `AudioDeviceIOProc` straight on the chosen device.
- **mic with system audio**: the device joins the tap's aggregate as a
  drift-compensated sub-device, and its audio arrives in the same buffer list,
  ahead of the tap's, on one clock. That aggregate delivers *nothing* until
  something renders on its output device (25 s of silence produced not one
  buffer; speech at second 10 started the flow at second 12), so a silent
  keep-alive IO proc holds the output device awake for the run — zeros, nothing
  audible, and nothing the tap can capture.

The unpinned path is untouched by all of this: it is still AVAudioEngine
following the default input, with the measured AirPods rebuild behaviour.

One consequence worth knowing before comparing echo-cancellation numbers: with
the keep-alive running, the system channel delivers *continuously*. Unpinned,
it goes quiet whenever nothing plays (measured 2026-08-12: zero frames in 5 s
of silence, against continuous frames on the pinned run). A reference that
never stops is the better shape for the canceller, but it means a pinned run
and an unpinned one are not the same experiment.

`stenocap/` holds the **Rust implementation**, one crate with a backend per
platform. On Windows it captures through WASAPI: the mic from the default
capture endpoint, system audio from *loopback* on the default render endpoint.
On Linux it opens one PulseAudio-protocol record stream per channel — the mic
from `@DEFAULT_SOURCE@`, system audio from `@DEFAULT_MONITOR@` — which serves
both sound servers, since PipeWire ships `pipewire-pulse` precisely so pulse
clients need no second path (libpulse is the one system library it links).
Neither platform signs anything — Windows gates the microphone through the
per-user privacy consent store rather than per binary, Linux gates nothing —
and neither needs a resampler: `AUTOCONVERTPCM | SRC_DEFAULT_QUALITY` has
Windows' audio engine deliver mono 16 kHz server-side, and the pulse server
resamples per stream the same way. `--devices` prints what each channel would
record from, as JSON, for the CLI's preflight.

It exists because of where its timestamps come from, and that is the whole
story. On Windows, `IAudioCaptureClient::GetBuffer` reports `pu64QPCPosition`
for every packet — the machine-wide performance counter, in 100-ns units, for
the packet's first sample. On Linux, the server accounts per stream how old
the sample at the read position is (`pa_stream_get_latency`, refreshed every
second), so each chunk is stamped `CLOCK_MONOTONIC − latency` — for a monitor
the server folds in the sink's own latency, so its stamps mark when the audio
becomes audible, the same render-side semantics as loopback's QPC stamps
(measured band ±7 ms against a sample-count line, `stenocap/src/pulse.rs`).
Three Windows traps, all measured (`eval/wasapi_timestamps.py` re-runs the
evidence in twelve seconds) and all silent if got wrong:

- **`pu64DevicePosition` is not a sample index.** It counts *device* frames,
  upstream of the resampler `AUTOCONVERTPCM` inserts, so it advances at 47,999/s
  against a stream that asked for 16,000. Only the QPC stamp is used.
- **The two taps' stamps mean different things, and that is correct.** The mic's
  marks capture and trails receipt (+11.1 ms); the loopback tap's is render-side
  and *leads* it (−9.6 ms), marking when the audio reached the endpoint — when
  the echo was born, which is the instant the canceller wants. Both are on one
  counter, so neither needs correcting.
- **Loopback delivers nothing at all while nothing renders** — no silent
  packets, just silence in the literal sense. The helper synthesizes that
  silence on a timer, stamped from the same counter, which is what lets a
  meeting where the far end never speaks still have a reference to cancel
  against.

**Type-checking the other platforms from a Mac.** Homebrew's cargo is first on
`PATH` and carries only the host target, which makes this look impossible; the
rustup toolchain has the others. `cargo check --locked --tests --target
x86_64-pc-windows-msvc` needs nothing else, and the Linux target additionally
needs a stub `libpulse.pc` on `PKG_CONFIG_PATH` with `PKG_CONFIG_ALLOW_CROSS=1`
(nothing is linked, so the stub's paths need not exist). Two borrow-checker
errors in `pulse.rs` reached a "CI is the only compiler" review this way
(2026-08-12) — the check costs seconds and CI costs a round trip.

## Wire protocol

stdout carries frames only (status and errors go to stderr), little-endian:

    frame = channel:u8  timestamp:f64  count:u32  samples:count×i16

`channel` is 0 for mic, 1 for system; `timestamp` is seconds since capture start
on a clock **shared by both channels**; `samples` is mono 16 kHz int16 PCM.
Channels are selected with argv flags (`--mic`, `--system`). Both
implementations speak exactly this; the consumer of both is
`stenograf.capture.helper`.

Argv, identical on both:

    stenocap [--mic] [--system] [--mic-device ID]  # capture
    stenocap --devices [--mic-device ID]           # what this run would record
    stenocap --list-inputs                         # every microphone, JSON array

`--mic-device` records the mic channel from one device instead of the OS
default. Its value is a platform-stable id (`kAudioDevicePropertyDeviceUID`,
a WASAPI endpoint id, a pulse source name) or the device's exact name; both
sides of the comparison are trimmed and NFC-normalized, the id is tried first,
and a name two connected devices share is an error naming both ids rather than
a guess. A value that names nothing connected is FATAL after a three-second
grace (a USB interface can enumerate a moment after wake) — never a silent fall
back to the default, which is the wrong-microphone recording this flag exists to
prevent. The system channel is not selectable: it follows the default output by
design. **Unknown arguments exit 2**, so a newer caller can never have its
device choice ignored by an older binary.

**Stopping differs, and only because the platforms do.** The Swift helper takes
SIGINT/SIGTERM, flushes and exits 0. The Rust helper stops when **stdin reaches
EOF** on both its platforms: Windows has no signal a parent can aim at one child
(`CTRL_C_EVENT` goes to a whole process group), while closing a pipe needs no
console, no process group and no handler, and cannot arrive before the helper is
ready to see it. On Linux it additionally *ignores* SIGINT — a terminal's Ctrl+C
hits the whole process group, and the parent must drain the pipe before the
helper lets go.

The shared clock matters, and it is the reason this protocol exists at all. The
mic and the system tap are separate devices with separate transports: on macOS
they start hundreds of milliseconds apart (the tap is already running while
AVAudioEngine opens the mic), and on Windows the loopback tap is a longer path
than the microphone. Stamping frames when they *arrive* therefore gives each
channel its own hidden offset, and the echo canceller — which pairs the two by
timestamp — is handed a reference that does not line up with the echo it must
remove. Measured on Windows, that was the difference between 2.6 dB and 13.7 dB
ERLE. So each implementation stamps from the OS's own clock for both taps: Mach
host time on macOS, `pu64QPCPosition` on Windows, and `CLOCK_MONOTONIC` minus
the server's stream latency on Linux.

## Build

    sh stenocap-macos/build.sh          # macOS
    powershell -File stenocap/build.ps1 # Windows (rustup + VS Build Tools)
    sh stenocap/build.sh                # Linux (rustup + libpulse headers)

In every case the binary is a build artifact (gitignored), and the Python side
finds it the same three ways: next to its sources in the checkout, a packaged
`stenograf/bin/stenocap[.exe]` in a wheel, or the `STENOGRAF_CAPTURE_HELPER`
environment override.

The macOS build compiles + ad-hoc signs `stenocap-macos/stenocap` (swiftc; no
Apple Developer account needed — DECISIONS.md's history, `git log --follow -p
DECISIONS.md`). TCC usage
strings are embedded from `stenocap-macos/Info.plist`; on first run the terminal
is granted mic + system-audio permission once. The Windows and Linux builds are
plain `cargo build --release --locked`, and sign nothing.

Each is **mandatory** where it applies, which is why `hatch_build.py` fails the
wheel if it fails to build: without it there is no live capture on that
platform at all. Three wheels carry a helper — macOS arm64, win_amd64 and
manylinux_2_28 x86_64. On macOS every wheel build compiles it; off macOS the
hook builds it only when `STENOGRAF_BUNDLE_HELPERS` is set (the release build),
and otherwise produces the pure wheel with no helper.

Reference implementations this was built from:

- https://github.com/insidegui/AudioCap — canonical process-tap sample code
- https://stronglytyped.uk/articles/audiotee-capture-system-audio-output-macos —
  tap → stdout PCM streaming CLI
