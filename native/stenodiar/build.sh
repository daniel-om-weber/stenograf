#!/bin/sh
# Build the stenodiar diarization helper and drop the binary next to this
# script, where stenograf's dev fallback looks for it (mirrors helper/stenocap).
# Needs a Rust toolchain (brew install rust). No signing: stenodiar touches no
# TCC-guarded resource, so an unsigned binary is fine.
set -e
cd "$(dirname "$0")"

cargo build --release
cp target/release/stenodiar stenodiar

echo "built: $(pwd)/stenodiar"
