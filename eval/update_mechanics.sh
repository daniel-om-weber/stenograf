#!/bin/sh
# What `uv` actually does when stenograf is upgraded — the measurements the
# update design rests on. Runs in about a minute with a warm uv cache.
#
# Everything happens in a throwaway tool directory: the real install under
# `uv tool dir` is never touched. uv's shared download cache is used (and
# appended to) unless noted, which is why the timings are a lower bound.
#
#   sh eval/update_mechanics.sh [OLD] [NEW]      # default 0.6.0 0.7.0
#
# The throwaway directory must sit on the same file system as uv's cache, or uv
# copies every file instead of linking it and the timings mean nothing. /tmp is
# a RAM disk on many Linux systems, so name a directory there:
#
#   UPDATE_MECH_DIR=~/.cache sh eval/update_mechanics.sh
set -eu

OLD=${1:-0.6.0}
NEW=${2:-0.7.0}
S=$(mktemp -d "${UPDATE_MECH_DIR:-${TMPDIR:-/tmp}}/steno-update.XXXXXX")
trap 'rm -rf "$S"' EXIT
cd "$S"
export UV_TOOL_DIR="$S/tools" UV_TOOL_BIN_DIR="$S/bin"
ENV="$S/tools/stenograf"
echo "$(uname -sm), $(uv --version), scratch on $(df -P "$S" | tail -1 | cut -d' ' -f1)," \
     "cache on $(df -P "$(uv cache dir)" | tail -1 | cut -d' ' -f1)"

quiet() { grep -vE "^ [+~-] |not on your PATH" || true; }
now() { python3 -c 'import time; print(time.time())'; }
# /usr/bin/time is not installed everywhere (Arch has none by default).
timed() { t0=$(now); "$@"; t1=$(now); echo "$t1 $t0" | awk '{printf "real %.2f s\n", $1 - $2}'; }
receipt() { grep -vE "install-path|entrypoints|^\]" "$ENV/uv-receipt.toml" | cut -c1-160; }
version() { "$S/bin/steno" --version; }

echo "== 1. an upgrade under a running process leaves it holding two versions"
uv tool install --python 3.13 "stenograf==$OLD" 2>&1 | quiet | tail -1
cat > runner.py <<'EOF'
import importlib.metadata as md, os, sys, time
import stenograf
while not os.path.exists(sys.argv[1]):
    time.sleep(0.2)
import stenograf.gui.app  # loaded only now, from whatever is on disk
print("in memory", stenograf.__version__, "| on disk", md.version("stenograf"))
EOF
"$ENV/bin/python" runner.py "$S/go" > runner.log 2>&1 &
RUNNER=$!
sleep 3
timed uv tool install --upgrade --python 3.13 stenograf 2>&1 | quiet | tail -4
touch go
wait "$RUNNER" || true
cat runner.log

echo "== 2. a pinned install is not moved by 'uv tool upgrade'"
uv tool install --force --python 3.13 "stenograf==$OLD" >/dev/null 2>&1
uv tool upgrade stenograf 2>&1 | quiet | tail -1
version

echo "== 3. --exclude-newer and --constraints are remembered in the receipt"
uv tool install --force --python 3.13 stenograf --exclude-newer 2026-08-10T00:00:00Z >/dev/null 2>&1
receipt
uv tool upgrade stenograf 2>&1 | quiet | tail -1
uv pip freeze --python "$ENV/bin/python" > freeze.txt
uv tool install --force --python 3.13 stenograf --constraints freeze.txt >/dev/null 2>&1
echo "constraint lines in the receipt: $(grep -c specifier "$ENV/uv-receipt.toml")"
uv tool install --upgrade --python 3.13 stenograf >/dev/null 2>&1
echo "after the installer's own command:"
receipt
version

echo "== 4. an unreachable index fails before anything is changed"
uv tool install --upgrade --refresh --python 3.13 --default-index http://127.0.0.1:9/simple stenograf 2>&1 | tail -1 || true
version

echo "== 5. park the old environment, install fresh into the same path, roll back"
uv tool install --force --python 3.13 "stenograf==$OLD" >/dev/null 2>&1
mv "$ENV" "$S/tools/.stenograf-previous"
echo "uv's view of a parked environment INSIDE the tool directory:"
uv tool list 2>&1 | head -2
mv "$S/tools/.stenograf-previous" "$S/stenograf-previous"
timed uv tool install --force --python 3.13 stenograf 2>&1 | quiet | tail -4
version
rm -rf "$ENV"
mv "$S/stenograf-previous" "$ENV"
version
"$ENV/bin/python" -c "import stenograf.gui.app; print('the rolled-back environment imports the app')"
uv tool list 2>&1 | head -1

echo "== 6. a trial run from the cache leaves the install alone; then it installs offline"
timed uv tool run --python 3.13 --from "stenograf==$NEW" steno --version 2>&1 | quiet | tail -4
version
uv tool install --offline --force --python 3.13 stenograf 2>&1 | quiet | tail -1
version
receipt

echo "== 7. the base interpreter, as seen from inside a tool environment"
"$ENV/bin/python" -c "import sys; print(sys._base_executable); print(sum(1 for _ in __import__('pathlib').Path(sys.prefix).rglob('*')), 'files in the environment')"

echo "== 8. an install killed halfway leaves a folder with no receipt"
uv tool install --force --python 3.13 "stenograf==$OLD" >/dev/null 2>&1
mv "$ENV" "$S/stenograf-previous"
UV_LINK_MODE=copy uv tool install --force --python 3.13 stenograf >/dev/null 2>&1 &
UV_PID=$!
# Kill as soon as packages start landing: a fixed sleep is too long on a fast
# disk, where the whole copy takes under a second.
tries=0
while [ "$tries" -lt 400 ] && [ "$(ls "$ENV"/lib/python3.13/site-packages 2>/dev/null | wc -l)" -le 5 ]; do
    sleep 0.02
    tries=$((tries + 1))
done
kill -9 "$UV_PID" 2>/dev/null || echo "(the install finished before the kill)"
wait "$UV_PID" 2>/dev/null || true
if [ -f "$ENV/uv-receipt.toml" ]; then echo "receipt: present"; else echo "receipt: MISSING"; fi
du -sh "$ENV" 2>/dev/null | cut -f1
uv tool list 2>&1 | head -1

echo "== 9. the installer's command in that state, and with the folder gone"
uv tool install --upgrade --python 3.13 stenograf 2>&1 | quiet | tail -1 || true
rm -rf "$ENV"
echo "-- folder gone, entry points dangling:"
uv tool install --upgrade --python 3.13 stenograf 2>&1 | quiet | tail -1 || true
echo "-- the same with --force:"
uv tool install --upgrade --force --python 3.13 stenograf 2>&1 | quiet | tail -1
version

echo "== 10. from an EMPTY cache, with a small package (no warm-cache excuse)"
export UV_CACHE_DIR="$S/cache"
uv tool install --offline --python 3.13 httpie 2>&1 | tail -1 || true
uv tool run --python 3.13 --from "httpie==3.2.3" http --version 2>&1 | quiet | tail -1
echo "-- offline, unpinned: the newest version IN THE CACHE, not on the index"
uv tool install --offline --python 3.13 httpie 2>&1 | quiet | tail -1
"$S/bin/http" --version
echo "-- offline with a lower bound the cache cannot meet:"
uv tool install --offline --force --python 3.13 "httpie>=3.2.4" 2>&1 | grep -iE "error|unsatisfiable|not found|×" | head -1 || true
echo "-- a lower bound in the receipt does not stop a later upgrade:"
uv tool install --offline --force --python 3.13 "httpie>=3.2.3" >/dev/null 2>&1
grep requirements "$S/tools/httpie/uv-receipt.toml"
uv tool upgrade httpie 2>&1 | quiet | tail -1
"$S/bin/http" --version

echo "== 11. one lock file: shared by every process, exclusive for the updater"
cat > locks.py <<'EOF'
import fcntl, subprocess, sys

path = sys.argv[1]
HOLD = ("import fcntl, sys, time; f = open(sys.argv[1], 'a'); "
        "fcntl.flock(f, fcntl.LOCK_SH); print('held', flush=True); time.sleep(120)")
PROBE = ("import fcntl, sys; f = open(sys.argv[1], 'a')\n"
         "try:\n    fcntl.flock(f, fcntl.LOCK_SH | fcntl.LOCK_NB); print('granted')\n"
         "except OSError:\n    print('refused')")

def holder():
    process = subprocess.Popen([sys.executable, "-c", HOLD, path], stdout=subprocess.PIPE)
    process.stdout.readline()
    return process

def exclusive():
    handle = open(path, "a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return handle
    except OSError:
        handle.close()
        return None

first, second = holder(), holder()
print("exclusive while two processes hold it shared:", "granted (BAD)" if exclusive() else "refused")
first.kill(); first.wait()  # SIGKILL: a crash, nothing gets to clean up
print("exclusive after one of them was killed:      ", "granted (BAD)" if exclusive() else "refused")
second.kill(); second.wait()
held = exclusive()
print("exclusive after both were killed:            ", "granted" if held else "refused (BAD)")
probe = subprocess.run([sys.executable, "-c", PROBE, path], capture_output=True, text=True)
print("shared while the updater holds it exclusively:", probe.stdout.strip())
EOF
"$ENV/bin/python" locks.py "$S/stenograf.lock"
if [ "$(uname -s)" = Linux ]; then echo "lock file on: $(stat -f -c %T "$S")"; fi

echo "== 12. a detached child outlives the process that started it"
cat > parent.py <<'EOF'
import os, subprocess, sys
CHILD = "import sys, time; time.sleep(3); open(sys.argv[1], 'w').write('alive')"
subprocess.Popen([sys._base_executable, "-I", "-c", CHILD, sys.argv[1]], start_new_session=True,
                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                 cwd=os.path.dirname(sys.argv[1]))
os._exit(0)
EOF
"$ENV/bin/python" parent.py "$S/plain.txt"
sleep 5
echo "started from a plain process:  $(cat "$S/plain.txt" 2>/dev/null || echo DEAD)"
if command -v systemd-run >/dev/null 2>&1 && systemctl --user is-system-running >/dev/null 2>&1; then
    # A Linux desktop starts each app in its own systemd scope; this is that shape.
    systemd-run --user --scope --quiet "$ENV/bin/python" parent.py "$S/scope.txt"
    sleep 5
    echo "started from a systemd scope:  $(cat "$S/scope.txt" 2>/dev/null || echo DEAD)"
fi
