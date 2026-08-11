#!/bin/sh
# stenograf installer — one command sets up everything:
#
#   curl -fsSL https://raw.githubusercontent.com/daniel-om-weber/stenograf/main/install.sh | sh
#
# Installs uv if missing, installs stenograf as a uv tool, then runs
# `steno setup` (permission prompts, desktop launcher, model downloads).
# Safe to re-run: every step is idempotent and re-running upgrades stenograf.
set -eu

# Git Bash, MSYS2 and Cygwin run this script happily on Windows: uv, the wheel
# and the launcher all land in the native AppData tree, so it looks installed.
# One such install could not start the desktop app (2026-08-11, one report, the
# cause never isolated) — and nothing about this route is tested, which is
# reason enough on its own. Refused rather than repaired: Windows has its own
# installer, and this file is what the README's first code block runs.
case "$(uname -s 2>/dev/null || echo unknown)" in
    MINGW* | MSYS* | CYGWIN*)
        echo "This is the macOS/Linux installer and you are on Windows." >&2
        echo "" >&2
        echo "Close this shell, open PowerShell, and run:" >&2
        echo "" >&2
        echo '  powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/daniel-om-weber/stenograf/main/install.ps1 | iex"' >&2
        echo "" >&2
        echo "If this shell already installed stenograf, that command upgrades" >&2
        echo "it in place and keeps the environment it built. Rebuild instead:" >&2
        echo "" >&2
        echo "  uv tool install --force stenograf" >&2
        exit 1
        ;;
esac

# WSL is a real Linux and this installer belongs there — a note, not a refusal.
# The catch is which audio it can reach: WSLg's PulseAudio monitor carries what
# plays *inside* WSL, so a meeting running on the Windows side records as
# silence, and the user would find that out from the transcript.
if grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null; then
    echo "note: this looks like WSL. Live capture reaches only audio playing" >&2
    echo "inside WSL — for a meeting held on the Windows side, install there" >&2
    echo "with install.ps1 instead." >&2
    echo "" >&2
fi

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
