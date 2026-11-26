"""The two-voice check (`stenograf.diarization.split`) measured: harness, own meetings, cost.

    uv run --group eval eval/two_voice.py harness [--workers 8]     # ~19 min, M4 Max
    uv run eval/two_voice.py cases ANSWERS                           # ~3 min
    uv run eval/two_voice.py meeting DIR mic|system [--skip NAME,..] # ~2 min per 30 min
    uv run eval/two_voice.py energy DIR mic|system on|off            # ~5 min per 30 min

``harness`` builds four channels per AMI/ICSI meeting from the corpus harness's
crosstalk-masked wavs (``eval/ami.py fetch``): ``mic`` (one headset), ``loop`` (the
others mixed), ``room`` (mic + loop) and ``phone`` (mic clean + loop through a rough
speakerphone path: 300–3400 Hz band, −6 dB, a 150 ms noise tail). Nemotron runs as
production does (no count); for every label with a minute of turns it records the
reference speakers' shares of the label and both checks' statistics — the voice split
on turns, and the rerun on *turns*, because the corpus has no transcript entries, the
spans production reruns on. ``cases`` reads the blind listening page's key
(``eval/out/splitcheck-listen/key.json``, private) and ANSWERS (``"1:Z 2:E …"``) and
compares the rerun on turns with the rerun on entries per case. ``meeting`` runs a stored
meeting's channel through ``finalize_channel`` with its words reused, one pass against
the stream fed in 0.5 s blocks, and times the stop with the split on and off.
``energy`` feeds the stream at background QoS with the per-turn embeddings on or off
and reads the process's CPU energy (``proc_pid_rusage``, the kernel's model).
Numbers: ``eval/README.md``.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from common import AUDIO_DIR, OUT_DIR, REFS_DIR, read_pcm16
from rttm import parse_rttm

from stenograf.diarization import split as sp

SR = 16000
AMI = AUDIO_DIR / "ami"
AMI_REFS = REFS_DIR / "ami"
OUT = OUT_DIR / "two-voice"
MEETINGS = Path.home() / "Documents" / "Meetings"
KINDS = ("mic", "loop", "room", "phone")
_models: dict = {}


def _voice_fires(n: int, cos: float, minor: float) -> bool:
    return n >= sp.MIN_VOICE_TURNS and minor >= sp.MIN_MINOR_SHARE and cos < sp.MAX_CENTROID_COSINE


def _diarizer():
    if "d" not in _models:
        from stenograf.diarization.nemotron import NemotronDiarizer

        _models["d"] = NemotronDiarizer()
    return _models["d"]


def _phone(x: np.ndarray) -> np.ndarray:
    from scipy.signal import butter, sosfilt

    y = sosfilt(butter(4, [300, 3400], "bandpass", fs=SR, output="sos"), x) * 0.5
    n = int(0.15 * SR)
    ir = np.exp(-np.arange(n) / (0.04 * SR)) * np.random.default_rng(1).standard_normal(n)
    ir[0] = 1.0
    ir /= np.sqrt((ir**2).sum())
    return np.convolve(y, ir)[: len(y)].astype(np.float32)


def _channel(meeting: str, kind: str):
    mic = read_pcm16(AMI / f"{meeting}.mic.wav").astype(np.float32) / 32768
    ref_mic = parse_rttm(AMI_REFS / f"{meeting}.mic.rttm")
    if kind == "mic":
        return mic, ref_mic
    loop = read_pcm16(AMI / f"{meeting}.loop.wav").astype(np.float32) / 32768
    ref_loop = parse_rttm(AMI_REFS / f"{meeting}.loop.rttm")
    if kind == "loop":
        return loop, ref_loop
    if kind == "phone":
        loop = _phone(loop)
    mix = np.zeros(max(len(mic), len(loop)), np.float32)
    mix[: len(mic)] += mic
    mix[: len(loop)] += loop
    return mix / max(1.0, float(np.abs(mix).max())), list(ref_mic) + list(ref_loop)


def _one_channel(job: tuple[str, str]) -> list[dict]:
    from stenograf.diarization.nemotron import turns_from_probs

    meeting, kind = job
    audio, ref = _channel(meeting, kind)
    d = _diarizer()
    turns = turns_from_probs(d.probabilities(audio))
    frames = len(audio) // sp.HOP + 1
    names = sorted({t.speaker for t in ref})
    active = np.zeros((frames, len(names)), bool)
    for t in ref:
        active[int(t.start * 100) : int(t.end * 100), names.index(t.speaker)] = True
    alone = np.where(active.sum(1) == 1, active.argmax(1), -1)
    rows = []
    for label in sorted({t.speaker for t in turns}):
        own = [t for t in turns if t.speaker == label]
        speech = sum(t.end - t.start for t in own)
        if speech < sp.MIN_SPEAKER_S:
            continue
        mine = np.zeros(frames, bool)
        for t in own:
            mine[int(t.start * 100) : int(t.end * 100)] = True
        counts = np.bincount(alone[mine & (alone >= 0)], minlength=len(names))
        shares = np.sort(counts / max(counts.sum(), 1))[::-1]
        vectors = [d.embedder.embed(audio[a:b]) for a, b in sp.voice_crops(own)]
        vectors = [v for v in vectors if v is not None]
        cos, minor = sp.voice_split(np.array(vectors)) if len(vectors) >= 2 else (1.0, 0.0)
        spans = [sp.to_span(t.start, t.end) for t in sorted(own, key=lambda t: t.start)]
        spans = [s for s in spans if s[1] - s[0] >= sp.MIN_RERUN_SPAN]
        share = sp.second_voice(d.probabilities(sp.joined(audio, spans)), spans)[0]
        rows.append({
            "meeting": meeting, "kind": kind, "label": label, "speech_s": round(speech),
            "ref_shares": [round(float(x), 3) for x in shares[:3]],
            "n": len(vectors), "cos": round(cos, 3), "minor": round(minor, 3),
            "rerun": round(share, 3),
        })
    print(f"{meeting}.{kind}: {len(rows)} labels", file=sys.stderr, flush=True)
    return rows


def harness(workers: int) -> None:
    meetings = sorted(p.name.split(".")[0] for p in AMI.glob("*.mic.wav"))
    jobs = sorted(((m, k) for m in meetings for k in KINDS),
                  key=lambda j: -(AMI / f"{j[0]}.mic.wav").stat().st_size)
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for rs in ex.map(_one_channel, jobs) for r in rs]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "harness.json").write_text(json.dumps(rows, indent=0))
    def voice(r):
        return _voice_fires(r["n"], r["cos"], r["minor"])

    def rerun(r):
        return r["rerun"] >= sp.SECOND_VOICE_SHARE

    print("kind   labels merges | voice split TP FP | rerun on turns TP FP | both TP FP")
    for kind in (*KINDS, "all"):
        rs = [r for r in rows if kind in ("all", r["kind"])]
        merged = [r for r in rs if r["ref_shares"][1:2] and r["ref_shares"][1] >= 0.15]
        cells = []
        for rule in (voice, rerun, lambda r: voice(r) and rerun(r)):
            tp = sum(map(rule, merged))
            cells.append(f"{tp:3d} {sum(map(rule, rs)) - tp:3d}")
        print(f"{kind:6s} {len(rs):6d} {len(merged):6d} | " + " | ".join(cells))


def cases(answers: str) -> None:
    key = json.loads((OUT_DIR / "splitcheck-listen" / "key.json").read_text())
    ear = dict(a.split(":") for a in answers.split() if ":" in a)
    rows = []
    for case in key:
        if str(case["case"]) not in ear:
            continue
        rows.append(_case(case) | {"ear": ear[str(case["case"])]})
        print(rows[-1], file=sys.stderr, flush=True)
    for name, field in (("turns", "rerun_turns"), ("entries", "rerun_entries")):
        fires = [r for r in rows if r["voice"] and r[field] >= sp.SECOND_VOICE_SHARE]
        two = sum(r["ear"] == "Z" for r in fires)
        print(f"voice split on turns + rerun on {name}: two people caught {two}/"
              f"{sum(r['ear'] == 'Z' for r in rows)}, one person or no voice split "
              f"{len(fires) - two}/{sum(r['ear'] in 'EK' for r in rows)}")


def _case(case: dict) -> dict:
    from stenograf.audio import load_audio_channels
    from stenograf.diarization.nemotron import turns_from_probs

    folder = MEETINGS / case["meeting"]
    entries = json.loads((folder / "transcript.json").read_text())["entries"]
    chans = load_audio_channels(folder / "audio.opus")
    audio = chans[-1] if case["channel"] == "system" else chans[0]
    d = _diarizer()
    turns = turns_from_probs(d.probabilities(audio))
    mine = [(e["start"], e["end"]) for e in entries if e["speaker"] == case["speaker"]]
    overlap: dict[str, float] = {}
    for t in turns:
        for a, b in mine:
            shared = max(0.0, min(t.end, b) - max(t.start, a))
            overlap[t.speaker] = overlap.get(t.speaker, 0.0) + shared
    label = max(overlap, key=lambda k: overlap[k])
    own = [t for t in turns if t.speaker == label]
    vectors = [d.embedder.embed(audio[a:b]) for a, b in sp.voice_crops(own)]
    vectors = [v for v in vectors if v is not None]
    cos, minor = sp.voice_split(np.array(vectors)) if len(vectors) >= 2 else (1.0, 0.0)

    def rerun(spans):
        spans = sorted(s for s in spans if s[1] - s[0] >= sp.MIN_RERUN_SPAN)
        return round(sp.second_voice(d.probabilities(sp.joined(audio, spans)), spans)[0], 3)

    return {
        "case": case["case"], "label": label,
        "voice": _voice_fires(len(vectors), cos, minor),
        "cos": round(cos, 3), "minor": round(minor, 3),
        "rerun_turns": rerun([sp.to_span(t.start, t.end) for t in own]),
        "rerun_entries": rerun([sp.to_span(a, b) for a, b in mine]),
    }


def meeting(folder: Path, channel: str, skip: set[str]) -> None:
    from stenograf.asr.base import Word
    from stenograf.audio import load_audio_channels
    from stenograf.diarization.nemotron import NemotronDiarizer
    from stenograf.pipeline import finalize_channel

    chans = load_audio_channels(folder / "audio.opus")
    audio = chans[0] if channel == "mic" else chans[-1]
    entries = json.loads((folder / "transcript.json").read_text())["entries"]
    words = tuple(Word(w["text"], w["start"], w["end"])
                  for e in entries if e["speaker"] not in skip for w in e.get("words", []))

    def finalize(diarizer):
        voices: dict = {}
        t0 = time.time()
        out = finalize_channel(
            audio, asr=None, language=None, diarizer=diarizer,  # type: ignore[arg-type]
            on_embeddings=voices.update, precomputed_words=words,
        )
        return out, voices, time.time() - t0

    def streamed(split_on: bool):
        d = NemotronDiarizer()
        d.embedder.load()
        stream = d.stream()
        for n in range(8000, len(audio) + 1, 8000):
            stream.advance(lambda lo, hi: audio[lo:hi], n)
        resumed = stream.resumed()
        if not split_on:
            resumed.split_speakers = lambda *a, **k: None  # type: ignore[method-assign]
        out = finalize(resumed)
        stream.close()
        return out, stream._loop.steps if stream._loop else 0

    def speech(es):
        s: dict = {}
        for e in es:
            s[e.speaker] = s.get(e.speaker, 0.0) + e.end - e.start
        return {k: round(v) for k, v in sorted(s.items())}

    one, one_voices, one_s = finalize(NemotronDiarizer())
    (_, _, stop_off), _ = streamed(False)
    (got, voices, stop_on), steps = streamed(True)
    same = [(e.speaker, e.start, e.end) for e in one] == [(e.speaker, e.start, e.end) for e in got]
    same = same and set(one_voices) == set(voices)
    same = same and all(np.array_equal(one_voices[k], voices[k]) for k in voices)
    print(f"{folder.name} {channel}: {len(audio) / SR / 60:.1f} min, one pass {one_s:.1f} s")
    print(f"stop: {stop_off:.2f} s without the split, {stop_on:.2f} s with it")
    print(f"{steps} streamed steps")
    print(f"speakers {speech(got)}; streamed == one pass, bit for bit: {same}")


def energy(folder: Path, channel: str, arm: str) -> None:
    from stenograf.audio import load_audio_channels
    from stenograf.cpu import background_plan, enter_background
    from stenograf.diarization.nemotron import NemotronDiarizer

    libproc = ctypes.CDLL("/usr/lib/libproc.dylib")

    def joules() -> tuple[float, float]:
        buf = (ctypes.c_uint64 * 64)()
        assert libproc.proc_pid_rusage(os.getpid(), 6, ctypes.byref(buf)) == 0
        return buf[42] / 1e9, buf[43] / 1e9  # rusage_info_v6: ri_energy_nj, ri_penergy_nj

    chans = load_audio_channels(folder / "audio.opus")
    audio = chans[0] if channel == "mic" else chans[-1]
    d = NemotronDiarizer()
    d.embedder.load()
    stream = d.stream()
    if arm == "off":
        stream._turn_voices_failed = True

    def feed():
        enter_background(background_plan())
        for n in range(8000, len(audio) + 1, 8000):
            stream.advance(lambda lo, hi: audio[lo:hi], n)

    e0, p0 = joules()
    t0 = time.time()
    worker = threading.Thread(target=feed)
    worker.start()
    worker.join()
    e1, p1 = joules()
    hours = len(audio) / SR / 3600
    print(f"{arm}: {time.time() - t0:.0f} s, {e1 - e0:.1f} J ({p1 - p0:.1f} J on P-cores), "
          f"{(e1 - e0) / hours:.1f} J per audio-hour, {len(stream._turn_voices)} turns embedded")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("harness")
    h.add_argument("--workers", type=int, default=8)
    c = sub.add_parser("cases")
    c.add_argument("answers")
    m = sub.add_parser("meeting")
    m.add_argument("folder", type=Path)
    m.add_argument("channel", choices=("mic", "system"))
    m.add_argument("--skip", default="", help="speakers whose words are on the other channel")
    e = sub.add_parser("energy")
    e.add_argument("folder", type=Path)
    e.add_argument("channel", choices=("mic", "system"))
    e.add_argument("arm", choices=("on", "off"))
    args = parser.parse_args()
    if args.cmd == "harness":
        harness(args.workers)
    elif args.cmd == "cases":
        cases(args.answers)
    elif args.cmd == "meeting":
        meeting(args.folder, args.channel, set(filter(None, args.skip.split(","))))
    else:
        energy(args.folder, args.channel, args.arm)


if __name__ == "__main__":
    main()
