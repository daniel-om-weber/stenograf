#!/bin/sh
# Build the stenocap capture helper and drop the binary next to this script,
# where stenograf's dev fallback looks for it (mirrors build.ps1, the Windows
# twin). Needs a Rust toolchain (rustup) and the libpulse development headers
# (package libpulse-dev / pulseaudio-libs-devel, depending on distro). No
# signing: Linux gates nothing per binary — unlike macOS, where the signature
# *is* the microphone grant.
#
# The wheel build hook runs this script too (hatch_build.py), and unlike
# stenodiar a failure here fails the wheel: without this binary there is no
# live capture at all on Linux.
set -e
cd "$(dirname "$0")"

cargo build --release --locked
cp target/release/stenocap stenocap

echo "built: $(pwd)/stenocap"
