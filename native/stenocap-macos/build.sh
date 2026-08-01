#!/bin/sh
# Build + ad-hoc sign the stenocap capture helper. The __info_plist section
# embeds the TCC usage descriptions — without it the system-audio and mic
# prompts never appear. Ad-hoc signing (codesign -s -) is all that's needed;
# no Apple Developer account (PLAN.md "Deployment & distribution").
set -e
cd "$(dirname "$0")"

# Explicit deployment target: without it the binary inherits the build host's
# OS as its minimum (a helper built on macOS 26 refuses to launch on 14/15),
# and the wheel is tagged macosx_14_0. 14.4 = the Core Audio process-tap floor.
swiftc -swift-version 5 -O -target arm64-apple-macos14.4 main.swift -o stenocap \
  -framework CoreAudio -framework AudioToolbox -framework AVFoundation \
  -Xlinker -sectcreate -Xlinker __TEXT -Xlinker __info_plist -Xlinker Info.plist

codesign --force --sign - stenocap

echo "--- signature ---"
codesign -dv stenocap 2>&1 | sed -n '1,3p'
echo "built: $(pwd)/stenocap"
