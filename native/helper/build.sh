#!/bin/sh
# Build + ad-hoc sign the stenocap capture helper. The __info_plist section
# embeds the TCC usage descriptions — without it the system-audio and mic
# prompts never appear. Ad-hoc signing (codesign -s -) is all that's needed;
# no Apple Developer account (PLAN.md "Deployment & distribution").
set -e
cd "$(dirname "$0")"

swiftc -swift-version 5 -O main.swift -o stenocap \
  -framework CoreAudio -framework AudioToolbox -framework AVFoundation \
  -Xlinker -sectcreate -Xlinker __TEXT -Xlinker __info_plist -Xlinker Info.plist

codesign --force --sign - stenocap

echo "--- signature ---"
codesign -dv stenocap 2>&1 | sed -n '1,3p'
echo "built: $(pwd)/stenocap"
