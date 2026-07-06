# Native capture helper (macOS)

Planned Phase 1 component: a small ad-hoc-signed Swift binary that captures

- **system audio** via a Core Audio process tap (`AudioHardwareCreateProcessTap`,
  macOS 14.4+) — whole-system or per-app, and
- **microphone** via AVAudioEngine, with Voice Processing IO echo cancellation
  when playing through speakers in hybrid meetings,

and streams both as separate channels of mono 16 kHz int16 PCM over stdout /
a Unix socket to the Python core. Audio is never written to disk.

Reference implementations to build from:

- https://github.com/insidegui/AudioCap — canonical process-tap sample code
- https://stronglytyped.uk/articles/audiotee-capture-system-audio-output-macos —
  tap → stdout PCM streaming CLI

Build (once implemented): SwiftPM, compiled and ad-hoc signed by the wheel
build hook in CI; a dev-mode fallback compiles locally when Xcode Command Line
Tools are present.
