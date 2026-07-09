# Native capture helper (macOS)

`helper/` holds **stenocap**, the ad-hoc-signed Swift binary that feeds live
audio to the Python core. It captures

- **system audio** via a Core Audio process tap (`AudioHardwareCreateProcessTap`,
  macOS 14.4+) — the remote participants in a call, and
- **microphone** via AVAudioEngine,

resamples both to mono 16 kHz int16 (AVAudioConverter), and streams them as
framed PCM on **stdout**. Audio is never written to disk — the helper only
streams it. `spike/` is the original throwaway proof that the tap + mic APIs
work; `helper/` is the production version that adds resampling, framing, and a
clean lifecycle.

Echo cancellation is **not** done here. The helper once had a `--aec` flag
backed by Voice Processing IO (`setVoiceProcessingEnabled`). Measured on macOS
26 it reported a 9-channel mic format and delivered an all-zero mic, while
driving the system tap to 6x real time — so it was removed. Echo is cancelled on
the Python side, using the system channel as the far-end reference. (VPIO
does reference the whole output-device mix, so it cancels other apps' audio in
principle; it also ducks that audio, which would attenuate the very remote
speech we transcribe. Chrome, which ships both, defaults to loopback-fed AEC3
for the same reason.)

## Wire protocol

stdout carries frames only (status and errors go to stderr), little-endian:

    frame = channel:u8  timestamp:f64  count:u32  samples:count×i16

`channel` is 0 for mic, 1 for system; `timestamp` is seconds since capture start
on a clock **shared by both channels**; `samples` is mono 16 kHz int16 PCM.
Channels are selected with argv flags (`--mic`, `--system`); stopping is a
SIGINT/SIGTERM, on which the helper flushes and exits 0. The consumer is
`stenograf.capture.macos`.

The shared clock matters. The mic and the tap are separate devices that start
hundreds of milliseconds apart (the tap is already running while AVAudioEngine
opens the mic). Each channel is therefore anchored to the Mach host time of its
first buffer, not to its own sample count — otherwise both would claim `t=0` for
audio captured far apart, and the echo canceller would align the far-end
reference against the wrong instant.

## Build

    sh helper/build.sh

Compiles + ad-hoc signs `helper/stenocap` (swiftc; no Apple Developer account
needed — PLAN.md "Deployment & distribution"). The binary is a build artifact
(gitignored). The Python side finds it via `native/helper/stenocap` in the
source tree, a packaged `stenograf/bin/stenocap` in a wheel, or the
`STENOGRAF_CAPTURE_HELPER` environment override. TCC usage strings are embedded
from `helper/Info.plist`; on first run the terminal is granted mic + system-audio
permission once.

Reference implementations this was built from:

- https://github.com/insidegui/AudioCap — canonical process-tap sample code
- https://stronglytyped.uk/articles/audiotee-capture-system-audio-output-macos —
  tap → stdout PCM streaming CLI
