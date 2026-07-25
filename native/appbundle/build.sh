#!/bin/sh
# Rebuild the committed Stenograf.app template. Read README.md first — running
# this changes the bundle's cdhash, and the cdhash *is* everyone's microphone
# grant. It exists to be auditable, not to be run.
#
# Unlike native/helper/build.sh, the product is committed (under
# src/stenograf/assets/) rather than gitignored: the bundle every user gets has
# to be the same bytes, so it is built once, here, and never on their machine.
set -e
cd "$(dirname "$0")"
ROOT=$(cd ../.. && pwd)
APP="$ROOT/src/stenograf/assets/Stenograf.app"
ICONSET=$(mktemp -d)/Stenograf.iconset
trap 'rm -rf "$(dirname "$ICONSET")"' EXIT

echo "--- icon ---"
# PySide6 rather than a new image toolchain; it is already the GUI's dependency.
uv run --with PySide6-Essentials python render_icon.py \
  icon.svg "$ICONSET" "$ROOT/src/stenograf/assets/icon.png"
iconutil --convert icns --output "$(dirname "$ICONSET")/Stenograf.icns" "$ICONSET"

echo "--- launcher stub ---"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
# Universal: the binary is frozen for good, so it has to keep working on the
# Intel Macs this project still supports (each slice carries its own cdhash,
# and each is pinned by TCC on the machine that runs it). 11.0 is the floor
# arm64 allows; the plist's LSMinimumSystemVersion is the real product floor.
clang -arch arm64 -arch x86_64 -mmacosx-version-min=11.0 -O2 -Wall -Wextra -Werror \
  main.c -framework CoreFoundation -o "$APP/Contents/MacOS/Stenograf"

cp Info.plist "$APP/Contents/Info.plist"
cp "$(dirname "$ICONSET")/Stenograf.icns" "$APP/Contents/Resources/Stenograf.icns"

# Signed here, once, and the signature is committed with the rest: the Info.plist
# and the icon are sealed into the executable's cdhash, so re-signing on the
# user's machine would make every install a different app to TCC.
codesign --force --sign - "$APP"
codesign --verify --strict --verbose=2 "$APP"

echo "--- frozen identity ---"
for arch in arm64 x86_64; do
  printf '%s: ' "$arch"
  codesign -d --verbose=4 --arch "$arch" "$APP" 2>&1 | sed -n 's/^CDHash=/cdhash /p'
done
echo "built: $APP"
echo
echo "tests/test_shortcut.py pins a hash of this tree — update it deliberately,"
echo "and only if you accept re-prompting every user who already granted access."
