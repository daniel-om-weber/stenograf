"""Repeatable echo-path runs on real hardware (PLAN-AEC.md §3, layer 1).

Drives the *real* pipeline — native capture helper, speakers, mic, AEC — through
one scenario and scores both layers:

- **far-only**: plays a speech WAV out the speakers while nobody talks. Every
  ``Local-N`` line of ≥3 words in the resulting transcript is leaked echo; the
  pass criterion is zero. This is the number that decides whether a canceller
  change actually helps.
- **near-only**: speakers stay silent while the local side talks (read a fixed
  script aloud, or play it from a second device at a fixed position). Measures
  collateral damage: compare the transcript against a ``--no-aec`` run of the
  same script — the words must match.
- **double-talk**: both at once — the case suppression must not eat.

Each run lands in ``eval/out/aec/<scenario>-<stamp>/`` with the meeting output,
the ``--aec-dump`` triple, and ``rig.json`` (layer-0 signal metrics via
``aec_score`` + the layer-1 line counts). Compare runs, don't stare at one:
``--no-aec`` records the uncancelled baseline with the same scenario.

Reproducibility notes: set the system output volume to a fixed level (75%
unless you are probing the loud-speaker nonlinearity at 100%), keep the lid
angle and room constant, and use the same source clip across runs. The far-end
clip loops for the whole capture.

Usage:
    uv run --group eval eval/aec_rig.py far-only [--seconds 60] [--no-aec]
    uv run --group eval eval/aec_rig.py near-only
    uv run --group eval eval/aec_rig.py double-talk --source eval/audio/en-1.wav
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

import aec_score
from common import AUDIO_DIR

EVAL_DIR = Path(__file__).parent
OUT_DIR = EVAL_DIR / "out" / "aec"
DEFAULT_SOURCE = AUDIO_DIR / "en-1.wav"

SCENARIOS = {
    # scenario -> (plays far end, AECMOS talk type, operator instruction)
    "far-only": (True, "st", "Stay quiet. The speakers do the talking."),
    "near-only": (False, "nst", "Speakers silent. Read the script aloud now."),
    "double-talk": (True, "dt", "Talk over the speakers for the whole run."),
}

ECHO_MIN_WORDS = 3  # session.py's dedup floor: shorter matches are agreement, not echo


def steno_command(run_dir: Path, dump_dir: Path, seconds: float, aec: bool) -> list[str]:
    cmd = [
        sys.executable,
        "-c",
        "from stenograf.cli import main; main()",
        "start",
        "--no-live",
        "--plain",
        "--no-archive",
        "--out",
        str(run_dir),
        "--aec-dump",
        str(dump_dir),
        "--max-seconds",
        str(seconds),
    ]
    if not aec:
        cmd.append("--no-aec")
    return cmd


def play_far_end(source: Path, stop: threading.Event) -> None:
    """Loop the clip out the default output until told to stop."""
    while not stop.is_set():
        player = subprocess.Popen(["afplay", str(source)])
        while player.poll() is None:
            if stop.wait(0.5):
                player.terminate()
                return


def run_capture(cmd: list[str], far_source: Path | None) -> int:
    """Run steno, starting far-end playback once capture is actually up."""
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    stop = threading.Event()
    player: threading.Thread | None = None
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            print(f"  steno: {line.rstrip()}")
            if far_source is not None and player is None and line.startswith("capturing:"):
                player = threading.Thread(target=play_far_end, args=(far_source, stop))
                player.start()
        return proc.wait()
    finally:
        stop.set()
        if player is not None:
            player.join()
        if proc.poll() is None:
            proc.terminate()


def local_lines(run_dir: Path) -> tuple[list, list]:
    """(local ≥3-word lines, all entries) from the run's transcript JSON."""
    transcripts = sorted(run_dir.glob("*.transcript.json"))
    if not transcripts:
        raise SystemExit(f"no transcript JSON in {run_dir} — did the run fail?")
    from stenograf.transcript import Transcript

    transcript = Transcript.from_json(transcripts[-1].read_text())
    leaked = [
        e
        for e in transcript.entries
        if e.speaker.startswith("Local") and len(e.text.split()) >= ECHO_MIN_WORDS
    ]
    return leaked, transcript.entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scenario", choices=sorted(SCENARIOS))
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"far-end speech WAV to play [default: {DEFAULT_SOURCE}]",
    )
    parser.add_argument("--seconds", type=float, default=60.0, help="capture length [60]")
    parser.add_argument(
        "--no-aec",
        dest="aec",
        action="store_false",
        help="record the uncancelled baseline of the same scenario",
    )
    args = parser.parse_args()

    plays_far, talk_type, instruction = SCENARIOS[args.scenario]
    if plays_far and not args.source.exists():
        raise SystemExit(
            f"{args.source} not found — pass --source; any 16 kHz speech WAV works "
            "(eval/audio/ is gitignored, see eval/README.md)"
        )
    if shutil.which("afplay") is None and plays_far:
        raise SystemExit("afplay not found — this rig drives macOS speakers")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    label = args.scenario if args.aec else f"{args.scenario}-noaec"
    run_dir = OUT_DIR / f"{label}-{stamp}"
    dump_dir = run_dir / "dump"
    run_dir.mkdir(parents=True)

    print(f"scenario: {args.scenario}  (aec {'on' if args.aec else 'OFF — baseline'})")
    print(f">>> {instruction}")
    print(f">>> {args.seconds:.0f} s once capture starts; results in {run_dir}")

    code = run_capture(
        steno_command(run_dir, dump_dir, args.seconds, args.aec),
        args.source if plays_far else None,
    )
    if code != 0:
        raise SystemExit(f"steno exited with {code}")

    leaked, entries = local_lines(run_dir)
    mic, lpb, enh = (aec_score.read_wav(dump_dir / f"{n}.wav") for n in aec_score.TRIPLE)
    metrics = aec_score.signal_metrics(mic, lpb, enh)
    if "erle_db" in metrics:
        metrics |= aec_score.aecmos_metrics(mic, lpb, enh, talk_type)
    metrics |= {
        "scenario": args.scenario,
        "aec": args.aec,
        "local_lines_3w": len(leaked),
        "entries_total": len(entries),
        "started": stamp,
        "seconds": args.seconds,
    }
    (run_dir / "rig.json").write_text(json.dumps(metrics, indent=2) + "\n")

    print()
    for key in ("erle_db", "residual_dbfs", "aecmos_echo", "aecmos_deg"):
        if key in metrics:
            print(f"{key:>14}: {metrics[key]}")
    print(f"{'local lines':>14}: {len(leaked)} of {len(entries)} entries (≥{ECHO_MIN_WORDS} words)")
    if args.scenario == "far-only":
        verdict = "PASS — no echo reached the transcript" if not leaked else "FAIL — leaked echo:"
        print(f"{'far-only':>14}: {verdict}")
        for entry in leaked:
            print(f"{'':>16}[{entry.start:6.1f}s] {entry.speaker}: {entry.text}")
    print(f"{'stored':>14}: {run_dir}")


if __name__ == "__main__":
    main()
