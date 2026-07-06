#!/bin/sh
# Build + ad-hoc sign the capture spike. The __info_plist section embeds the
# TCC usage descriptions — without it the system-audio prompt never appears.
set -e
cd "$(dirname "$0")"

swiftc -swift-version 5 -O main.swift -o tap-spike \
  -framework CoreAudio -framework AudioToolbox -framework AVFoundation \
  -Xlinker -sectcreate -Xlinker __TEXT -Xlinker __info_plist -Xlinker Info.plist

codesign --force --sign - tap-spike

echo "--- signature ---"
codesign -dv tap-spike 2>&1 | sed -n '1,5p'
echo "built: $(pwd)/tap-spike"
