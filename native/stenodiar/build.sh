#!/bin/sh
# Build the stenodiar diarization helper and drop the binary next to this
# script, where stenograf's dev fallback looks for it (mirrors stenocap-macos).
# Needs a Rust toolchain (brew install rust / rustup). No signing: stenodiar touches no
# TCC-guarded resource, so an unsigned binary is fine.
#
# The wheel build hook runs this script too (hatch_build.py), so what it
# produces on Linux is what users install — hence the portability flags below.
# Windows has its own twin, build.ps1.
set -e
cd "$(dirname "$0")"

case "$(uname -s)" in
Darwin)
    # CoreML is a cargo feature, not the default.
    cargo build --release --locked --features coreml
    ;;
*)
    # Linux: speakrs' native CPU backend, the path the wheel ships (--features
    # cuda is a manual GPU opt-in). OpenSSL is linked in statically (Cargo.toml
    # vendors it). The binary's glibc symbol floor comes from the build
    # machine, which is why release.yml pins ubuntu-24.04 and asserts the
    # binary needs nothing newer than that runner provides.
    cargo build --release --locked
    STRIP=1
    ;;
esac
cp target/release/stenodiar stenodiar

# The wheel carries the binary on every install; debug symbols for a helper
# whose failures arrive as a stderr line are not worth their bytes. Linux only:
# on arm64 macOS the linker's ad-hoc signature is part of the binary, and strip
# invalidates it — the stripped binary is killed on launch.
if [ -n "${STRIP:-}" ]; then
    strip stenodiar
fi

echo "built: $(pwd)/stenodiar"
