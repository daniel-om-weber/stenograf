#!/bin/sh
# Build the stenodiar diarization helper and drop the binary next to this
# script, where stenograf's dev fallback looks for it (mirrors helper/stenocap).
# Needs a Rust toolchain (brew install rust). No signing: stenodiar touches no
# TCC-guarded resource, so an unsigned binary is fine.
set -e
cd "$(dirname "$0")"

# CoreML is a cargo feature, not the default: this script is the macOS build
# (Windows/Linux build with --no-default-features or --features cuda directly).
cargo build --release --features coreml
cp target/release/stenodiar stenodiar

echo "built: $(pwd)/stenodiar"
