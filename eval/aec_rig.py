"""Repeatable echo-path runs on real hardware (layer 1).

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

Reproducibility notes: keep the output volume, the lid angle and the room
constant, and use the same source clip across runs. The far-end clip loops for
the whole capture. ``--volume`` fixes the level for you and restores it
afterwards (Windows only) -- treat it as part of the measurement, not a
preference: 40 % on a laptop chassis puts *no* echo into the microphone and
cost a 33-minute run, 90 % puts it 22 dB above its noise floor.

macOS and Windows are both driven; the Windows-only pieces (SAPI source clip,
``winsound`` playback, endpoint volume) live in ``aec_windows.py``. On Windows
with no ``--source`` the far end is synthesized on the spot, because
``eval/audio/`` is gitignored and a fresh checkout has no speech in it. Linux
is not wired up: ``parec``-side far-end alignment is an open question in
PLAN.md and wants its own session.

Usage (macOS):
    uv run --group eval eval/aec_rig.py far-only [--seconds 60] [--no-aec]
    uv run --group eval eval/aec_rig.py near-only
    uv run --group eval eval/aec_rig.py double-talk --source eval/audio/en-1.wav

Usage (Windows) -- **plain ``uv run``, no ``--group eval``**, which cannot
resolve here: it pulls mlx (no win_amd64 wheel) and a second onnxruntime
flavor, and this platform allows exactly one. Everything the rig needs beyond
the package itself is numpy, which is a main dependency; the one casualty is
the learned AECMOS score, and the run says so instead of dying:
    uv run python eval/aec_rig.py far-only --seconds 60 --volume 90
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

WINDOWS = sys.platform == "win32"
if WINDOWS:
    import aec_windows

AEC_OUT_DIR = Path(__file__).parent / "out" / "aec"
DEFAULT_SOURCE = AUDIO_DIR / ("far-en.wav" if WINDOWS else "en-1.wav")

SCENARIOS = {
    # scenario -> (plays far end, AECMOS talk type, operator instruction)
    "far-only": (True, "st", "Stay quiet. The speakers do the talking."),
    "near-only": (False, "nst", "Speakers silent. Read the script aloud now."),
    "double-talk": (True, "dt", "Talk over the speakers for the whole run."),
}

ECHO_MIN_WORDS = 3  # session.py's dedup floor: shorter matches are agreement, not echo


def steno_command(
    run_dir: Path, dump_dir: Path, seconds: float, aec: bool, live: bool
) -> list[str]:
    cmd = [
        sys.executable,
        "-c",
        "from stenograf.cli import main; main()",
        "start",
        "--live" if live else "--no-live",
        # No --no-archive: [archive] was renamed to [output] (settings.py) and
        # the flag went with it. --out already keeps the run out of the
        # meetings folder, which is all it was ever here for.
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


def player_command(source: Path) -> list[str]:
    """The one-shot "play this WAV out the default device" command."""
    if WINDOWS:
        return aec_windows.player_command(source)
    return ["afplay", str(source)]


def play_far_end(source: Path, stop: threading.Event) -> None:
    """Loop the clip out the default output until told to stop."""
    command = player_command(source)
    while not stop.is_set():
        player = subprocess.Popen(command)
        while player.poll() is None:
            if stop.wait(0.5):
                player.terminate()
                return


def run_capture(cmd: list[str], far_source: Path | None) -> int:
    """Run steno, starting far-end playback once capture is actually up."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        # Not bare text=True: that decodes with the *locale* encoding, which is
        # cp1252 on a German Windows, while click hands us UTF-8. The captions
        # this relays are meeting speech, so a non-ASCII byte is the norm and
        # not the exception -- it killed a run on the very first status line.
        encoding="utf-8",
        errors="replace",
        bufsize=1,
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
    # "*transcript.json", not "*.transcript.json": with --out (which the rig
    # always passes) the meeting writes a bare transcript.json, and only a
    # date-named folder gets the <slug>. prefix.
    transcripts = sorted(run_dir.glob("*transcript.json"))
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
    if WINDOWS:
        # Belt to the ASCII discipline's braces. A piped stdout here is cp1252,
        # and this script relays steno's own output as well as printing its
        # own; one unmappable character would otherwise end a finished run in
        # a traceback, with the audio already gone and unrepeatable.
        sys.stdout.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scenario", choices=sorted(SCENARIOS))
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help=f"far-end speech WAV to play [default: {DEFAULT_SOURCE}, "
        "synthesized on Windows if absent]",
    )
    parser.add_argument(
        "--volume",
        type=float,
        default=None,
        metavar="PERCENT",
        help="set the master output volume for the run and restore it after "
        "(Windows only). 90 is the level measured to establish an echo path on "
        "a laptop chassis; 40 established none at all.",
    )
    parser.add_argument("--seconds", type=float, default=60.0, help="capture length [60]")
    parser.add_argument(
        "--no-aec",
        dest="aec",
        action="store_false",
        help="record the uncancelled baseline of the same scenario",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="run the live captions pass during capture — puts the real MLX "
        "inference load on the machine, the regime that produced the historical "
        "leaked lines (batch capture is idle by comparison)",
    )
    args = parser.parse_args()

    plays_far, talk_type, instruction = SCENARIOS[args.scenario]
    source = args.source or DEFAULT_SOURCE
    if plays_far:
        if not WINDOWS and shutil.which("afplay") is None:
            raise SystemExit("afplay not found - this rig drives macOS or Windows speakers")
        if not source.exists():
            if args.source is not None or not WINDOWS:
                raise SystemExit(
                    f"{source} not found - pass --source; any 16 kHz speech WAV works "
                    "(eval/audio/ is gitignored, see eval/README.md)"
                )
            print(f"synthesizing far end ({aec_windows.VOICE}) -> {source}")
            print(f"  {aec_windows.synthesize(source):.1f} s of speech with silences")

    if args.volume is not None:
        if not WINDOWS:
            raise SystemExit("--volume is implemented for Windows only")
        restore_volume = aec_windows.output_volume()
        aec_windows.set_output_volume(args.volume / 100)
        print(f"output volume {restore_volume * 100:.0f}% -> {args.volume:.0f}% (restored after)")
    else:
        restore_volume = None

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    label = args.scenario if args.aec else f"{args.scenario}-noaec"
    if args.live:
        label += "-live"
    run_dir = AEC_OUT_DIR / f"{label}-{stamp}"
    dump_dir = run_dir / "dump"
    run_dir.mkdir(parents=True)

    mode = "live" if args.live else "batch"
    print(f"scenario: {args.scenario}  ({mode}, aec {'on' if args.aec else 'OFF - baseline'})")
    print(f">>> {instruction}")
    print(f">>> {args.seconds:.0f} s once capture starts; results in {run_dir}")

    try:
        code = run_capture(
            steno_command(run_dir, dump_dir, args.seconds, args.aec, args.live),
            source if plays_far else None,
        )
    finally:
        if restore_volume is not None:
            aec_windows.set_output_volume(restore_volume)
            print(f"output volume restored to {restore_volume * 100:.0f}%")
    if code != 0:
        raise SystemExit(f"steno exited with {code}")

    leaked, entries = local_lines(run_dir)
    mic, lpb, enh = (aec_score.read_wav(dump_dir / f"{n}.wav") for n in aec_score.TRIPLE)
    metrics = aec_score.signal_metrics(mic, lpb, enh)
    if "erle_db" in metrics:
        try:
            metrics |= aec_score.aecmos_metrics(mic, lpb, enh, talk_type)
        except ImportError:
            # speechmos lives in the `eval` dependency group, which cannot
            # resolve on Windows at all: it pulls mlx (no win_amd64 wheel) and
            # a second onnxruntime flavor, which this platform forbids. ERLE,
            # the residual and the leaked-line count are what decide anything
            # here; throwing away a finished capture over the optional learned
            # score would be the worse trade.
            metrics["aecmos"] = "unavailable (speechmos not installed)"
            print("aecmos: skipped, speechmos not installed")
    metrics |= {
        "scenario": args.scenario,
        "aec": args.aec,
        "live": args.live,
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
    # ASCII only from here down: a piped stdout on a German Windows encodes as
    # cp1252, where U+2265 has no mapping -- this line used to raise
    # UnicodeEncodeError at the end of the run, with the audio already gone.
    print(f"{'local lines':>14}: {len(leaked)} of {len(entries)} ({ECHO_MIN_WORDS}+ words)")
    if args.scenario == "far-only":
        verdict = "PASS - no echo reached the transcript" if not leaked else "FAIL - leaked echo:"
        print(f"{'far-only':>14}: {verdict}")
        for entry in leaked:
            print(f"{'':>16}[{entry.start:6.1f}s] {entry.speaker}: {entry.text}")
    print(f"{'stored':>14}: {run_dir}")


if __name__ == "__main__":
    main()
