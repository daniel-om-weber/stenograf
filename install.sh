#!/bin/sh
# stenograf installer — one command sets up everything:
#
#   curl -fsSL https://raw.githubusercontent.com/daniel-om-weber/stenograf/main/install.sh | sh
#
# Installs uv if missing, installs stenograf as a uv tool, then runs
# `steno setup` (permission prompts, desktop launcher, model downloads).
# Safe to re-run: every step is idempotent and re-running upgrades stenograf.
set -eu

if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
    echo "installing uv (https://docs.astral.sh/uv/) ..."
    curl -fsSL https://astral.sh/uv/install.sh | sh
fi
UV=$(command -v uv || true)
[ -n "$UV" ] || UV="$HOME/.local/bin/uv"

"$UV" tool install --upgrade stenograf

# A freshly created uv bin dir isn't on this shell's PATH yet — ask uv where it is.
STENO=$(command -v steno || true)
[ -n "$STENO" ] || STENO="$("$UV" tool dir --bin)/steno"

"$STENO" setup

echo ""
echo "stenograf is installed. Start it from the desktop launcher above,"
echo "or run: steno"
