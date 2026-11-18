#!/bin/bash
# Net CPU energy of one runner on macOS: mean CPU power during the run minus
# idle, times wall time. Needs passwordless `sudo powermetrics` and a quiet
# machine. `bg` runs at background QoS, which macOS keeps on efficiency cores.
#
#   eval/embedders/energy.sh <label> <bg|fg> <manifest.json> <runner command...>
#   eval/embedders/energy.sh b6 bg M.json uv run eval/embedders/redimnet.py \
#       --checkpoint redimnet2-b6-vb2-lm --device cpu --threads 1
set -u
label=$1; qos=$2; manifest=$3; shift 3
tmp=$(mktemp -d)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
mean() { awk '/^CPU Power/ {s+=$3; n++} END {print s/n}' "$1"; }
sudo -n /usr/bin/powermetrics --samplers cpu_power -i 1000 -n 10 > "$tmp/idle.txt" 2>/dev/null
sudo -n /usr/bin/powermetrics --samplers cpu_power -i 500 > "$tmp/run.txt" 2>/dev/null & pm=$!
t0=$(date +%s.%N)
if [ "$qos" = bg ]; then taskpolicy -b "$@" --manifest "$manifest" --out "$tmp/e.npz" > "$tmp/e.log" 2>&1
else "$@" --manifest "$manifest" --out "$tmp/e.npz" > "$tmp/e.log" 2>&1; fi
t1=$(date +%s.%N)
kill $pm 2>/dev/null; wait $pm 2>/dev/null
uv run python - <<PY
import numpy as np
z = np.load("$tmp/e.npz")
wall, audio = $t1 - $t0, float(z["audio_seconds"])
idle, run = $(mean "$tmp/idle.txt"), $(mean "$tmp/run.txt")
joules = (run - idle) / 1000 * wall
print(f"$label $qos: idle {idle:.0f} mW, run {run:.0f} mW, wall {wall:.1f} s "
      f"(embedding {float(z['seconds_cpu']):.1f} s), {audio:.0f} s of speech -> "
      f"{joules:.1f} J, {joules / audio * 60:.2f} J per minute of speech")
PY
rm -rf "$tmp"
