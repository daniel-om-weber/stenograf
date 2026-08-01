# Native helpers

Two optional binaries live here — stenocap is macOS-only, stenodiar is built
for every platform stenograf ships a wheel for — both speaking simple pipes to
the Python core and both honoring the same rule: **meeting audio never touches
disk** — plus
`appbundle/`, which is not a helper at all but the sources for the macOS
desktop launcher `Stenograf.app`. It breaks this directory's convention on
purpose: its product is *committed* (`src/stenograf/assets/Stenograf.app`)
rather than built at install time, because macOS pins the app's microphone
grant to the bundle's exact bytes. Read `appbundle/README.md` before touching
anything in it.

`stenodiar/` holds the **diarization helper**, a Rust CLI around
[speakrs](https://github.com/avencera/speakrs) (the pyannote community-1
pipeline with VBx clustering, CoreML). It exists because sherpa's threshold
clustering cannot *estimate* a speaker count (it over-splits absurdly), while
VBx can; stenograf routes estimate-mode diarization through it and known
counts through sherpa (see `stenograf/diarization/speakrs.py`). Raw mono
16 kHz s16le PCM goes in on stdin, JSON speaker turns come out on stdout.
Build with `stenodiar/build.sh` (needs a Rust toolchain: `brew install rust`);
on Windows use `stenodiar/build.ps1` (rustup + VS Build Tools; CPU/ORT default,
CUDA a manual `--features cuda` opt-in); no signing needed — it touches no
TCC-guarded resource. The first run per
machine downloads models and compiles them for CoreML (minutes; `--warmup`
does it eagerly). Without the binary, stenograf silently falls back to
sherpa-only diarization.

Unlike stenocap it is not macOS-only, and users never build it: all three
platform wheels carry it (macOS arm64 with CoreML, manylinux_2_39 x86_64 and
win_amd64 with ORT CPU), because at ~15 MB compressed the bytes are cheaper
than a documented download nobody performs — the alternative that was
considered and dropped. Platforms without one of those wheels (musl, Linux
arm64, **anything older than glibc 2.39**) install the pure `py3-none-any`
wheel and keep the sherpa-only fallback.

The Linux binary is the delicate one: it must start on distros it was not
built on. Two things fix its floor, and only one of them is our choice.

- **glibc 2.39, not ours.** The onnxruntime static library `ort-sys`
  downloads references `__isoc23_strtol` and `__libc_single_threaded` — glibc
  2.38 symbols — so an older image cannot link stenodiar at all
  (manylinux_2_28 was tried, and fails at the link step, not at runtime).
  `release.yml` therefore pins `ubuntu-24.04` (glibc 2.39) and tags the wheel
  to match; measured, the binary itself needs only 2.38 and GLIBCXX_3.4.30.
- **OpenSSL, ours.** `stenodiar/Cargo.toml` vendors it statically so the
  helper never depends on the user's OpenSSL version. Note the flag reaches
  only the *runtime* graph: `ort-sys`'s build script pulls its own
  `openssl-sys` through a build-dependency, which cargo resolves with a
  separate feature set, so the build machine still needs `libssl-dev` (and a
  full `perl`, which is what compiles the vendored copy).

`release.yml` asserts all of it on the built wheel — highest glibc symbol,
libstdc++ against the runner's own, and no `libssl` in `ldd`.

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

    sh stenocap-macos/build.sh

Compiles + ad-hoc signs `stenocap-macos/stenocap` (swiftc; no Apple Developer
account needed — PLAN.md "Deployment & distribution"). The binary is a build
artifact (gitignored). The Python side finds it via
`native/stenocap-macos/stenocap` in the source tree, a packaged
`stenograf/bin/stenocap` in a wheel, or the `STENOGRAF_CAPTURE_HELPER`
environment override. TCC usage strings are embedded from
`stenocap-macos/Info.plist`; on first run the terminal is granted mic +
system-audio permission once.

Reference implementations this was built from:

- https://github.com/insidegui/AudioCap — canonical process-tap sample code
- https://stronglytyped.uk/articles/audiotee-capture-system-audio-output-macos —
  tap → stdout PCM streaming CLI
