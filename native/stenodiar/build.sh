#!/bin/sh
# Build the stenodiar diarization helper and drop the binary next to this
# script, where stenograf's dev fallback looks for it (mirrors helper/stenocap).
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
    # Linux: ORT CPU, the path the wheel ships (--features cuda is a manual
    # GPU opt-in). onnxruntime and OpenSSL are both linked in statically (ort
    # emits link-lib=static=onnxruntime; Cargo.toml vendors OpenSSL), so the
    # only external libraries left are libc/libm/libgcc/libstdc++ — all four
    # on manylinux's allowed list. libstdc++ cannot be made static from here:
    # ort emits a plain `cargo:rustc-link-lib=stdc++`, and an explicit -lstdc++
    # beats -static-libstdc++. Its symbol floor comes from the build machine
    # instead, which is why release.yml pins ubuntu-24.04 and asserts the
    # binary needs nothing newer than that runner provides.
    cargo build --release --locked
    STRIP=1
    ;;
esac
cp target/release/stenodiar stenodiar

# 61 MB → 52 MB, and the wheel carries it on every install; debug symbols for a
# helper whose failures arrive as a stderr line are not worth that. Linux only:
# on arm64 macOS the linker's ad-hoc signature is part of the binary, and strip
# invalidates it — the stripped binary is killed on launch.
if [ -n "${STRIP:-}" ]; then
    strip stenodiar
fi

echo "built: $(pwd)/stenodiar"
