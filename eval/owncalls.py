"""Two-speaker diarization on the user's own recorded calls, labelled by channel.

Every AMI number is English headset audio. A recorded one-to-one online call
already carries its own reference: the stored ``audio.opus`` keeps the mic left
and the system tap right (``recording.py``), both after echo cancellation
(the recorder sits downstream of ``EchoCancellingProvider``). Speech on the
system channel is the remote person; speech on the mic is the user. Summing the
two gives a two-speaker channel whose who-spoke-when is known without anyone
labelling it — the same construction as the harness's headset mixes, on real
German calls through a laptop mic and a call codec.

What it is not: stenograf never diarizes this mix (the channels are separate in
production), and a laptop mic against a call codec separates by channel colour
alone — a diarizer can sit at the ceiling on it. ``pairs`` builds the harder
set: two remote people from different calls, both through the system tap,
taking turns — close to the product's two-remote-people channel. Neither covers
a shared room mic, which has no free reference.

Selection: a meeting qualifies when the calendar event it overlaps has exactly
one participant besides the user and both channels carry at least
``MIN_SIDE_S`` of speech. Attendees are read from a temp copy of Calendar's
store (the ``cal-agenda`` rule: never the live database). Residual echo is
measured per call as the share of remote speech during which the mic VAD also
fires; calls above ``MAX_COACTIVITY`` are dropped as echo-contaminated labels.
All outputs — audio mixes, references, the meeting list — stay in
``eval/out/owncalls/`` (gitignored): they are private recordings.

    uv run --group eval eval/owncalls.py select [--limit N]
    uv run --group eval eval/owncalls.py build
    uv run eval/nemotron_arm.py run --device webgpu --out eval/out/owncalls \
        --audio-dir eval/out/owncalls/audio
    uv run --group eval eval/owncalls.py score
    uv run --group eval eval/owncalls.py pairs        # then nemotron / score
    # with --set pairs (nemotron: --out/--audio-dir eval/out/owncalls/pairs[/audio])
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import heldout
import numpy as np
from der import score_der
from rttm import Turn, parse_rttm

MEETINGS = Path.home() / "Documents" / "Meetings"
OUT = Path(__file__).resolve().parent / "out" / "owncalls"
AUDIO = OUT / "audio"
CALENDAR = Path.home() / "Library" / "Group Containers" / "group.com.apple.calendar"
MIN_CALL_S = 600.0
MIN_SIDE_S = 120.0
MAX_COACTIVITY = 0.35
"""Share of remote speech the mic VAD also flags. Real crosstalk-free conversation
overlaps a few to ~15 %; far above that the mic is hearing the remote side."""
DECLINED = 3
_EPOCH = datetime(2001, 1, 1)  # Core Data reference date


def _calendar_copy() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="owncalls-cal-"))
    for suffix in ("", "-wal", "-shm"):
        src = CALENDAR / f"Calendar.sqlitedb{suffix}"
        if src.exists():
            shutil.copy2(src, tmp / src.name)
    return tmp / "Calendar.sqlitedb"


def _events(db: sqlite3.Connection, start: datetime, end: datetime) -> list[dict]:
    """Non-all-day events overlapping [start, end] (local naive times) with their
    non-self, non-declined participant count."""
    agenda = subprocess.run(
        ["cal-agenda", "--since", f"{start:%Y-%m-%d}", "--until", f"{end:%Y-%m-%d}", "--json"],
        capture_output=True,
        text=True,
    )
    if agenda.returncode not in (0, 1):
        raise RuntimeError(agenda.stderr.strip())
    out = []
    for e in json.loads(agenda.stdout or "[]"):
        if e["all_day"]:
            continue
        s = datetime.fromisoformat(e["start"]).replace(tzinfo=None)
        t = datetime.fromisoformat(e["end"]).replace(tzinfo=None)
        overlap = (min(t, end) - max(s, start)).total_seconds()
        if overlap <= 0:
            continue
        rows = db.execute(
            "select lower(coalesce(email,'')), is_self, status from Participant where owner_id=?",
            (e["event_id"],),
        ).fetchall()
        self_mails = {m for m, me, _ in rows if me}
        others = {m for m, me, st in rows if m and not me and st != DECLINED} - self_mails
        partner = hashlib.sha256(min(others).encode()).hexdigest()[:8] if others else None
        out.append(
            {
                "event_id": e["event_id"],
                "overlap": overlap,
                "others": len(others),
                "partner": partner,
            }
        )
    return out


def _speech_by_side(transcript: Path) -> tuple[float, float, float]:
    data = json.loads(transcript.read_text())
    local = remote = end = 0.0
    for e in data.get("entries", []):
        span = e.get("end", 0.0) - e.get("start", 0.0)
        end = max(end, e.get("end", 0.0))
        if str(e.get("speaker", "")).startswith("Local"):
            local += span
        else:
            remote += span
    return local, remote, end


def cmd_select(limit: int | None) -> int:
    db = sqlite3.connect(f"file:{_calendar_copy()}?mode=ro", uri=True)
    picked, reasons = [], {}
    for folder in sorted(MEETINGS.glob("meeting-*")):
        names = ("audio.opus", "audio.wav", "audio.opus.wav")
        audio = next((folder / n for n in names if (folder / n).exists()), None)
        transcript = folder / "transcript.json"
        if audio is None or not transcript.exists():
            reasons["no audio/transcript"] = reasons.get("no audio/transcript", 0) + 1
            continue
        local, remote, dur = _speech_by_side(transcript)
        if dur < MIN_CALL_S or local < MIN_SIDE_S or remote < MIN_SIDE_S:
            reasons["short or one-sided"] = reasons.get("short or one-sided", 0) + 1
            continue
        start = datetime.strptime(folder.name[:23], "meeting-%Y%m%d-%H%M%S")
        events = _events(db, start, start + timedelta(seconds=dur))
        best = max(events, key=lambda e: e["overlap"], default=None)
        if best is None or best["overlap"] < 0.5 * dur:
            reasons["no matching event"] = reasons.get("no matching event", 0) + 1
            continue
        if best["others"] != 1:
            key = "no attendee list" if best["others"] == 0 else "more than one other person"
            reasons[key] = reasons.get(key, 0) + 1
            continue
        picked.append(
            {
                "id": folder.name,
                "audio": str(audio),
                "duration_s": round(dur, 1),
                "partner": best["partner"],
            }
        )
    if limit:
        picked = picked[-limit:]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "meetings.json").write_text(json.dumps(picked, indent=1))
    hours = sum(p["duration_s"] for p in picked) / 3600
    print(f"{len(picked)} one-to-one calls ({hours:.1f} h) → {OUT / 'meetings.json'}")
    print("excluded:", reasons)
    return 0


def _decode(path: str) -> tuple[np.ndarray, np.ndarray]:
    from stenograf.audio import ffmpeg_exe

    raw = subprocess.run(
        [
            ffmpeg_exe(),
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            path,
            "-ac",
            "2",
            "-ar",
            "16000",
            "-f",
            "s16le",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    stereo = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
    return stereo[:, 0].copy(), stereo[:, 1].copy()


def _activity(segments, n: int) -> np.ndarray:
    grid = np.zeros(n, bool)
    for s in segments:
        grid[int(s.start * 100) : int(s.end * 100)] = True
    return grid


def cmd_build() -> int:
    from stenograf import assets
    from stenograf.vad import SileroVAD

    vad = SileroVAD(assets.fetch(assets.SILERO_VAD))
    meetings = json.loads((OUT / "meetings.json").read_text())
    AUDIO.mkdir(parents=True, exist_ok=True)
    channels, report = [], []
    for m in meetings:
        mic, system = _decode(m["audio"])
        n = len(mic) // 160 + 1
        local = vad.speech_segments(mic.astype(np.float32) / 32768)
        remote = vad.speech_segments(system.astype(np.float32) / 32768)
        loc, rem = _activity(local, n), _activity(remote, n)
        coactivity = float((loc & rem).sum() / max(rem.sum(), 1))
        row = {
            "id": m["id"],
            "hours": round(len(mic) / 16000 / 3600, 2),
            "coactivity": round(coactivity, 3),
            "local_s": round(loc.sum() / 100),
            "remote_s": round(rem.sum() / 100),
        }
        report.append(row)
        if coactivity > MAX_COACTIVITY:
            print(f"{m['id']}: dropped, mic co-active on {coactivity:.0%} of remote speech")
            continue
        mix = mic.astype(np.float32) + system.astype(np.float32)
        peak = float(np.abs(mix).max()) or 1.0
        if peak > 32767:
            mix *= 32767 / peak
        heldout_wav = AUDIO / f"{m['id']}.wav"
        _write_wav(heldout_wav, mix.astype(np.int16))
        with (OUT / f"{m['id']}.rttm").open("w") as f:
            for who, segs in (("LOCAL", local), ("REMOTE", remote)):
                for s in segs:
                    f.write(
                        f"SPEAKER {m['id']} 1 {s.start:.2f} {s.end - s.start:.2f} "
                        f"<NA> <NA> {who} <NA> <NA>\n"
                    )
        channels.append({"id": m["id"], "group": "own", "session": "", "num_speakers": 2})
        print(f"{m['id']}: {row['hours']} h, co-activity {coactivity:.0%}", flush=True)
    (OUT / "channels.json").write_text(json.dumps(channels, indent=1))
    (OUT / "build_report.json").write_text(json.dumps(report, indent=1))
    print(f"{len(channels)} of {len(meetings)} calls built → {OUT / 'channels.json'}")
    return 0


def _write_wav(path: Path, pcm: np.ndarray) -> None:
    import wave

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm.tobytes())


def cmd_score(root: Path) -> int:
    channels = json.loads((root / "channels.json").read_text())
    systems: dict[str, dict[str, dict]] = {}
    for path in sorted(root.glob("turns_*.json")):
        systems[path.stem.removeprefix("turns_")] = json.loads(path.read_text())
    for probs in sorted(root.glob("probs_*")):
        tag = probs.name.removeprefix("probs_")
        systems[f"{tag} + count cap"] = {
            c["id"]: {
                "turns": heldout.count_capped(
                    np.load(probs / f"{c['id']}.npy").astype(np.float32), 2
                )
            }
            for c in channels
            if (probs / f"{c['id']}.npy").exists()
        }
    rows = {}
    for name, result in systems.items():
        for c in channels:
            cid = c["id"]
            if cid not in result:
                continue
            ref = parse_rttm(root / f"{cid}.rttm")
            hyp = result[cid]["turns"]
            der = score_der(ref, [Turn(t["speaker"], t["start"], t["end"]) for t in hyp])
            speech: dict[str, float] = {}
            for t in hyp:
                speech[t["speaker"]] = speech.get(t["speaker"], 0.0) + t["end"] - t["start"]
            total = sum(speech.values()) or 1.0
            rows.setdefault(name, []).append(
                {
                    "id": cid,
                    "attr": heldout.attribution(ref, hyp),
                    "der": der.der,
                    "conf": der.confusion,
                    "found": sum(v > 0.01 * total for v in speech.values()),
                    "secs": result[cid].get("secs"),
                    "audio_s": result[cid].get("audio_s"),
                }
            )
    shared = set.intersection(*({r["id"] for r in rs} for rs in rows.values())) if rows else set()
    print(f"{len(shared)} calls scored by every system\n")
    print(f"{'system':40s} attribution mean (min)   confusion   count 2 (≥1 %)   s/audio-h")
    for name, rs in rows.items():
        rs = [r for r in rs if r["id"] in shared]
        a = [r["attr"] for r in rs]
        timed = [r for r in rs if r["secs"] is not None and r["audio_s"]]
        speed = (
            sum(r["secs"] for r in timed) / (sum(r["audio_s"] for r in timed) / 3600)
            if timed
            else float("nan")
        )
        print(
            f"{name:40s} {100 * np.mean(a):5.1f} % ({100 * min(a):5.1f})"
            f"      {100 * np.mean([r['conf'] for r in rs]):5.1f} %"
            f"     {sum(r['found'] == 2 for r in rs):3d}/{len(rs)}"
            f"          {speed:5.1f}"
        )
    (root / "score.json").write_text(json.dumps(rows, indent=1, default=float))
    return 0


PAIR_SPEECH_S = 600.0
PAIR_GAP_S = 0.25


def cmd_pairs() -> int:
    """Two remote people through the same call path, taking turns.

    The call mixes above separate trivially: one side is a laptop mic, the other a
    call codec, so a diarizer can split them by channel colour alone. Here both
    voices are system-tap audio from two different one-to-one calls (different
    calendar partners), cut at their own VAD boundaries and laid end to end in
    alternation with a 0.25 s gap — no overlap, exact labels, one channel
    character. Each partner's longest call is the source; every partner appears
    in four dialogues (ring neighbours at distance 1 and 3). The two voices still
    come from different calls, so some call-to-call channel difference remains."""
    from stenograf import assets
    from stenograf.vad import SileroVAD

    vad = SileroVAD(assets.fetch(assets.SILERO_VAD))
    meetings = json.loads((OUT / "meetings.json").read_text())
    longest: dict[str, dict] = {}
    for m in meetings:
        if m["partner"] and m["duration_s"] > longest.get(m["partner"], {}).get("duration_s", 0):
            longest[m["partner"]] = m
    people = sorted(longest)
    voices = {}
    for p in people:
        _, system = _decode(longest[p]["audio"])
        segments = vad.speech_segments(system.astype(np.float32) / 32768)
        voices[p] = [system[int(s.start * 16000) : int(s.end * 16000)] for s in segments]
        print(f"partner {p}: {len(segments)} speech runs", flush=True)
    root = OUT / "pairs"
    (root / "audio").mkdir(parents=True, exist_ok=True)
    n, pairs = len(people), set()
    for i in range(n):
        for step in (1, 3):
            pairs.add(tuple(sorted((people[i], people[(i + step) % n]))))
    gap = np.zeros(int(PAIR_GAP_S * 16000), np.int16)
    channels, offset = [], {p: 0 for p in people}
    for a, b in sorted(pairs):
        cid = f"pair-{a}-{b}"
        pieces, lines, t, spoken = [], [], 0.0, 0.0
        turn = 0
        while spoken < PAIR_SPEECH_S:
            who = (a, b)[turn % 2]
            runs = voices[who]
            if not runs:
                break
            run = runs[offset[who] % len(runs)]
            offset[who] += 1
            pieces += [run, gap]
            dur = len(run) / 16000
            lines.append(f"SPEAKER {cid} 1 {t:.2f} {dur:.2f} <NA> <NA> {who} <NA> <NA>\n")
            t += dur + PAIR_GAP_S
            spoken += dur
            turn += 1
        _write_wav(root / "audio" / f"{cid}.wav", np.concatenate(pieces))
        (root / f"{cid}.rttm").write_text("".join(lines))
        channels.append({"id": cid, "group": "pairs", "session": "", "num_speakers": 2})
    (root / "channels.json").write_text(json.dumps(channels, indent=1))
    print(f"{len(channels)} dialogues from {n} partners → {root / 'channels.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select").add_argument("--limit", type=int)
    sub.add_parser("build")
    sub.add_parser("pairs")
    sub.add_parser("score").add_argument("--set", choices=["calls", "pairs"], default="calls")
    args = parser.parse_args()
    if args.cmd == "select":
        return cmd_select(args.limit)
    if args.cmd == "build":
        return cmd_build()
    if args.cmd == "pairs":
        return cmd_pairs()
    return cmd_score(OUT / "pairs" if args.set == "pairs" else OUT)


if __name__ == "__main__":
    raise SystemExit(main())
