"""Speaker-embedding models compared where they are used: naming speakers.

Since Nemotron 3 diarizes, the embedding model sits in one seat only — it turns
a finished cluster into a voiceprint that is cosine-matched against saved
profiles. A candidate therefore no longer needs a diarization matrix: every
model embeds the same frozen items (``build`` writes them once), and the
scorer runs the open-set naming rule over each model's vectors. The models run
in their reference implementations (``eval/embedders/``) so a conversion bug
cannot decide the comparison.

Four trial sets, one identity rule — every trial item is scored twice: as a
known trial against its group's full gallery, and as a stranger trial against
the gallery with its own identity removed. Every person is a stranger exactly
once per item, rather than one fixed unenrolled participant per group (the
harness convention, which leaves six strangers on AMI).

- ``ami-headset`` / ``ami-room`` / ``ami-cross``: six AMI series (the corpus
  harness's four plus the held-out ES2004/TS3003), four people each, enrolled
  from session a, trials from b–d. Reference word spans with overlap removed,
  so the embedder alone is measured. ``room`` is the Array1-01 far-field mic
  for enrollment and trial alike (the hybrid room meeting); ``cross`` enrolls
  from the headset and tries on the room mic.
- ``ami-nemotron``: production clusters — Nemotron 3 with the true count
  applied, on the held-out channels — enrolled from session a's clusters as
  ``steno profiles assign`` would (multi-entry profiles score by median).
- ``owncalls``: the user's recorded one-to-one German calls, labelled by
  channel and calendar (``owncalls.py``). Each partner enrolls from the first
  half of their first call; trials are the second half (same call) and every
  later call (cross call). The user's own mic side is identity ``self``.
  Private: everything stays under ``eval/out/`` (gitignored).

Trials per (identity, session): the full cluster, capped, plus ``EXCERPTS``
random excerpts per duration in ``DURATIONS`` — consecutive clean spans from a
random start, as a short cluster would hold them. Each span is embedded
separately and combined by production's rule (spans ≥ 0.5 s, duration-weighted
mean).

    uv run --group eval eval/reid_models.py fetch     # AMI room-mic WAVs
    uv run --group eval eval/reid_models.py build     # items + trials
    (run each model in eval/embedders/ on out/reid_models/manifest.json;
     eval/embedders/shard.py splits it across one-thread processes)
    uv run --group eval eval/reid_models.py check-cap # after base: the cap is safe
    uv run --group eval eval/reid_models.py score     # → out/reid_models/report.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import ami
import heldout
import numpy as np
from common import OUT_DIR
from rttm import parse_rttm

OUT = OUT_DIR / "reid_models"
EMB_DIR = OUT / "emb"
GROUPS = {**ami.AMI_GROUPS, **heldout.GROUPS}
ENROLL = "a"
FULL_CAP_S = 120.0
"""Clean speech per full cluster or enrollment. Production embeds the whole
cluster; past two minutes the mean voiceprint has stopped moving, and the cap
keeps a 1.5-hour AMI speaker from costing one forward pass per span for every
model. ``check-cap`` measures that claim on the shipped model."""
DURATIONS = (2.0, 3.0, 5.0, 10.0)
EXCERPTS = 10
MIN_SPAN_S = 0.5  # stenograf.diarization.embedding.MIN_EMBED_SECONDS
ROOM = "Array1-01"
SEED = 20261117


# -- spans -----------------------------------------------------------------


def _clean(
    by_speaker: dict[str, list[tuple[float, float]]],
) -> dict[str, list[tuple[float, float]]]:
    from stenograf.diarization.base import SpeakerTurn
    from stenograf.diarization.embedding import _overlap_spans, _subtract_spans

    turns = {k: [SpeakerTurn(k, s, e) for s, e in v] for k, v in by_speaker.items()}
    overlap = _overlap_spans(turns)
    return {k: _subtract_spans(sorted(v), overlap) for k, v in by_speaker.items()}


CAPS = {"first30": 30.0, "first60": 60.0, "first120": 120.0, "even120": 120.0, "whole": None}
"""How much of a cluster becomes its voiceprint (``build-caps``): the first N
seconds of clean speech, as embedding during the meeting would bank them;
``even120``, the main run's sampling across the whole meeting; or all of it."""
_cap_mode = "even120"


def _cap_even(
    spans: list[tuple[float, float]], cap: float = FULL_CAP_S
) -> list[tuple[float, float]]:
    """Every k-th span (k fixed by the total) until ``cap`` seconds are held —
    or the cluster cut as ``_cap_mode`` names."""
    spans = [s for s in spans if s[1] - s[0] >= MIN_SPAN_S] or spans
    if _cap_mode == "whole":
        return spans
    if _cap_mode.startswith("first"):
        picked, held = [], 0.0
        for s, e in spans:
            take = min(e - s, CAPS[_cap_mode] - held)
            picked.append((s, s + take))
            held += take
            if held >= CAPS[_cap_mode] - 1e-6:
                break
        return picked
    total = sum(e - s for s, e in spans)
    if total <= cap:
        return spans
    stride = total / cap
    picked, held = [], 0.0
    for i in np.arange(0, len(spans), stride):
        s, e = spans[int(i)]
        picked.append((s, e))
        held += e - s
        if held >= cap:
            break
    return picked


def _excerpts(
    spans: list[tuple[float, float]], seconds: float, n: int, key: str
) -> list[list[tuple[float, float]]]:
    """``n`` short clusters of ``seconds`` clean speech: consecutive spans from a
    seeded random start, the last one trimmed. Starts too late to fill the
    budget are not drawn."""
    rng = np.random.default_rng(int(hashlib.sha1(f"{SEED}{key}".encode()).hexdigest()[:8], 16))
    lengths = np.array([e - s for s, e in spans])
    tail = np.cumsum(lengths[::-1])[::-1]  # speech from span i to the end
    starts = np.flatnonzero(tail >= seconds)
    if len(starts) == 0:
        return []
    out = []
    for i in rng.choice(starts, size=min(n, len(starts)), replace=False):
        held, cluster = 0.0, []
        for s, e in spans[i:]:
            take = min(e - s, seconds - held)
            cluster.append((s, s + take))
            held += take
            if held >= seconds - 1e-6:
                break
        out.append(cluster)
    return out


# -- building ----------------------------------------------------------------


class Builder:
    def __init__(self, excerpts: bool = True) -> None:
        self.items: dict[str, dict] = {}
        self.sets: dict[str, dict] = {}
        self.excerpts = excerpts

    def item(self, iid: str, wav: Path, spans: list[tuple[float, float]]) -> str | None:
        if not spans:
            return None
        self.items[iid] = {
            "id": iid,
            "wav": str(wav),
            "spans": [[round(s, 3), round(e, 3)] for s, e in spans],
        }
        return iid

    def new_set(self, name: str) -> dict:
        self.sets[name] = {"profiles": {}, "groups": {}, "trials": []}
        return self.sets[name]

    def trials(self, sset: dict, prefix: str, wav: Path, identity: str, spans, tag: str) -> None:
        full = self.item(f"{prefix}/full", wav, _cap_even(spans))
        if full:
            sset["trials"].append(
                {"item": full, "identity": identity, "duration": "full", "tag": tag}
            )
        for d in DURATIONS if self.excerpts else ():
            for k, cluster in enumerate(_excerpts(spans, d, EXCERPTS, f"{prefix}/{d}")):
                iid = self.item(f"{prefix}/{d:g}s{k}", wav, cluster)
                if iid:
                    sset["trials"].append(
                        {"item": iid, "identity": identity, "duration": f"{d:g}s", "tag": tag}
                    )


def _ami_meeting(meeting: str) -> dict[str, tuple[Path, list[tuple[float, float]]]]:
    """name → (headset wav, clean reference spans) for one AMI meeting."""
    annotations = ami.RAW_DIR / "annotations"
    meetings = ami.parse_meetings_xml(annotations / "corpusResources" / "meetings.xml")
    raw = {}
    wavs = {}
    for agent, (headset, name) in meetings[meeting].items():
        spans = ami.parse_words_xml(annotations / "words" / f"{meeting}.{agent}.words.xml")
        if spans:
            raw[name] = ami.merge_spans(spans)
            wavs[name] = ami.RAW_DIR / meeting / f"{meeting}.Headset-{headset}.wav"
    clean = _clean(raw)
    return {n: (wavs[n], clean[n]) for n in raw}


def _room_wav(meeting: str) -> Path:
    return ami.RAW_DIR / meeting / f"{meeting}.{ROOM}.wav"


def build_ami(b: Builder) -> None:
    head, room, cross = b.new_set("ami-headset"), b.new_set("ami-room"), b.new_set("ami-cross")
    for group, sessions in GROUPS.items():
        for session in sessions:
            meeting = group + session
            for name, (wav, spans) in sorted(_ami_meeting(meeting).items()):
                for sset in (head, room, cross):
                    sset["groups"][name] = group
                if session == ENROLL:
                    h = b.item(f"ami/{meeting}/{name}/hs/enroll", wav, _cap_even(spans))
                    r = b.item(
                        f"ami/{meeting}/{name}/room/enroll", _room_wav(meeting), _cap_even(spans)
                    )
                    head["profiles"][name] = [h]
                    cross["profiles"][name] = [h]
                    room["profiles"][name] = [r]
                    continue
                b.trials(head, f"ami/{meeting}/{name}/hs", wav, name, spans, "cross-session")
                b.trials(
                    room,
                    f"ami/{meeting}/{name}/room",
                    _room_wav(meeting),
                    name,
                    spans,
                    "cross-session",
                )
    cross["trials"] = list(room["trials"])


def build_nemotron(b: Builder) -> None:
    import heldout_reid

    from stenograf.diarization.base import SpeakerTurn

    arms = heldout_reid.arm_turns()
    turns_by_channel = arms.get("Nemotron, count applied after")
    if not turns_by_channel:
        print("ami-nemotron: no count-applied Nemotron turns on disk — skipped")
        return
    sset = b.new_set("ami-nemotron")
    channels = {c["id"]: c for c in json.loads((heldout.OUT_DIR / "channels.json").read_text())}
    from rttm import Turn

    for cid, turns in sorted(turns_by_channel.items()):
        c = channels[cid]
        ref = parse_rttm(ami.AMI_REFS_DIR / f"{cid}.rttm")
        if c["num_speakers"] == 1:
            # a count-1 channel is one channel-long turn; production embeds the
            # transcript entries, which the reference segments stand in for
            turns = [SpeakerTurn("S0", t.start, t.end) for t in ref]
        by_cluster: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for t in turns:
            by_cluster[t.speaker].append((t.start, t.end))
        clean = _clean(by_cluster)
        wav = ami.CHANNELS_DIR / f"{cid}.wav"
        for label, spans in sorted(clean.items()):
            true = ami.dominant_speaker([Turn(label, s, e) for s, e in by_cluster[label]], ref)
            if true is None or not spans:
                continue
            sset["groups"][true] = c["group"]
            if c["session"] == ENROLL:
                iid = b.item(f"nem/{cid}/{label}/enroll", wav, _cap_even(spans))
                if iid:
                    sset["profiles"].setdefault(true, []).append(iid)
            else:
                b.trials(sset, f"nem/{cid}/{label}", wav, true, spans, "cross-session")


def _who(partner: str) -> str:
    return "p" + hashlib.sha1(partner.encode()).hexdigest()[:8]


def build_owncalls(b: Builder) -> None:
    import owncalls

    meetings = json.loads((owncalls.OUT / "meetings.json").read_text())
    built = {c["id"] for c in json.loads((owncalls.OUT / "channels.json").read_text())}
    sides = owncalls.OUT / "sides"
    sides.mkdir(exist_ok=True)
    sset = b.new_set("owncalls")
    seen: set[str] = set()
    for m in sorted(meetings, key=lambda m: m["id"]):
        if m["id"] not in built:
            continue
        paths = {s: sides / f"{m['id']}.{s}.wav" for s in ("mic", "sys")}
        if not all(p.exists() for p in paths.values()):
            mic, system = owncalls._decode(m["audio"])
            owncalls._write_wav(paths["mic"], mic)
            owncalls._write_wav(paths["sys"], system)
        ref = parse_rttm(owncalls.OUT / f"{m['id']}.rttm")
        raw = {
            side: [(t.start, t.end) for t in ref if t.speaker == label]
            for side, label in (("mic", "LOCAL"), ("sys", "REMOTE"))
        }
        clean = _clean(raw)
        mid = max(e for spans in raw.values() for _, e in spans) / 2
        for side, identity in (("mic", "self"), ("sys", _who(m["partner"]))):
            sset["groups"][identity] = "own"
            spans = clean[side]
            first = [s for s in spans if s[1] <= mid]
            second = [s for s in spans if s[0] >= mid]
            prefix = f"own/{m['id']}/{side}"
            if identity not in seen:
                seen.add(identity)
                iid = b.item(f"{prefix}/enroll", paths[side], _cap_even(first))
                if iid:
                    sset["profiles"][identity] = [iid]
                b.trials(sset, f"{prefix}/h2", paths[side], identity, second, "same-call")
            else:
                b.trials(sset, prefix, paths[side], identity, spans, "cross-call")


def cmd_fetch() -> int:
    for group, sessions in GROUPS.items():
        for session in sessions:
            meeting = group + session
            dst = _room_wav(meeting)
            url = f"{ami.MIRROR}/AMICorpusMirror/amicorpus/{meeting}/audio/{dst.name}"
            heldout._polite_download(url, dst)
            print(f"[{meeting}] {dst.name}", flush=True)
    return 0


def cmd_build() -> int:
    b = Builder()
    build_ami(b)
    build_nemotron(b)
    build_owncalls(b)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps({"items": list(b.items.values())}))
    (OUT / "trials.json").write_text(json.dumps(b.sets, indent=1))
    spans = sum(len(i["spans"]) for i in b.items.values())
    audio = sum(e - s for i in b.items.values() for s, e in i["spans"]) / 3600
    print(f"{len(b.items)} items, {spans} spans, {audio:.1f} h of audio → {OUT / 'manifest.json'}")
    for name, sset in b.sets.items():
        by = defaultdict(int)
        for t in sset["trials"]:
            by[t["duration"]] += 1
        print(f"  {name}: {len(sset['profiles'])} profiles, trials {dict(by)}")
    return 0


def cmd_build_caps() -> int:
    """Full clusters and enrollments again, once per entry of ``CAPS``."""
    global _cap_mode
    items, sets = [], {}
    for mode in CAPS:
        _cap_mode = mode
        b = Builder(excerpts=False)
        build_ami(b)
        build_nemotron(b)
        build_owncalls(b)
        items += [{**i, "id": f"{i['id']}@{mode}"} for i in b.items.values()]
        for sset in b.sets.values():
            sset["profiles"] = {k: [f"{e}@{mode}" for e in v] for k, v in sset["profiles"].items()}
            sset["trials"] = [{**t, "item": f"{t['item']}@{mode}"} for t in sset["trials"]]
        sets[mode] = b.sets
    _cap_mode = "even120"
    out = OUT / "caps"
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps({"items": items}))
    (out / "trials.json").write_text(json.dumps(sets))
    for mode in CAPS:
        hours = (
            sum(e - s for i in items if i["id"].endswith("@" + mode) for s, e in i["spans"]) / 3600
        )
        print(f"{mode}: {hours:.1f} h")
    return 0


def cmd_score_caps() -> int:
    """Per model and cut: does the voiceprint still name the same people?"""
    out = OUT / "caps"
    items = {i["id"]: i for i in json.loads((out / "manifest.json").read_text())["items"]}
    sets = json.loads((out / "trials.json").read_text())
    lines = [
        "# How much of a cluster the voiceprint needs",
        "",
        "Full-cluster trials, profile and trial cut the same way. Macro over "
        + ", ".join(PRIMARY_SETS)
        + ". `at T`: DIR / stranger-accept at the model's own all-duration "
        "threshold from the main run. `cosine`: to the whole-cluster vector.",
        "",
        "| model | cut | DIR@1% | EER | at T: DIR / FAR | worst margin | cosine median (min) |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in sorted((out / "emb").glob("*.npz")):
        vecs = load_vectors(p, items)
        thr = CAP_THRESHOLDS.get(p.stem)
        for mode in CAPS:
            cells = defaultdict(list)
            for name in PRIMARY_SETS:
                key = "owncalls" if name == "owncalls-remote" else name
                rows = trial_scores(sets[mode][key], vecs)
                if name == "owncalls-remote":
                    rows = [r for r in rows if r["identity"] != "self"]
                k = metrics(rows, thr)
                cells["dir"].append(k["dir@0.01"])
                cells["eer"].append(k["eer"])
                if thr is not None:
                    cells["fd"].append(k["fixed"][0])
                    cells["ff"].append(k["fixed"][1])
                cells["margin"].append(min(r["target"] - r["stranger"] for r in rows))
            cos = [
                float(v @ vecs[i.rsplit("@", 1)[0] + "@whole"])
                for i, v in vecs.items()
                if i.endswith("@" + mode)
            ]
            fixed = (
                f"{100 * np.mean(cells['fd']):.1f} / {100 * np.mean(cells['ff']):.1f}"
                if thr is not None
                else "—"
            )
            lines.append(
                f"| {p.stem} | {mode} | {100 * np.mean(cells['dir']):.1f} "
                f"| {100 * np.mean(cells['eer']):.2f} | {fixed} "
                f"| {min(cells['margin']):+.3f} | {np.median(cos):.4f} ({min(cos):.4f}) |"
            )
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


CAP_THRESHOLDS = {"base": 0.616, "redimnet2-b6-vb2-lm": 0.621}
"""Each model's one threshold over every duration (1 % macro stranger accepts,
main run, 2026-11-18) — the operating point a cut has to survive."""


# -- scoring -----------------------------------------------------------------


def load_vectors(path: Path, items: dict[str, dict]) -> dict[str, np.ndarray]:
    """item id → production's cluster vector: spans ≥ 0.5 s preferred,
    duration-weighted mean, L2-normalised. Items whose every span failed are
    absent (the trial then counts as unnamed/unscored)."""
    data = np.load(path, allow_pickle=False)
    emb, offsets, ids = data["emb"], data["offsets"], data["item_ids"]
    out = {}
    for i, iid in enumerate(ids):
        rows = emb[offsets[i] : offsets[i + 1]]
        spans = items[str(iid)]["spans"]
        dur = np.array([e - s for s, e in spans])
        ok = np.isfinite(rows).all(axis=1)
        long = ok & (dur >= MIN_SPAN_S)
        use = long if long.any() else ok
        if not use.any():
            continue
        v = np.average(rows[use], axis=0, weights=dur[use])
        out[str(iid)] = v / np.linalg.norm(v)
    return out


def trial_scores(sset: dict, vecs: dict[str, np.ndarray]) -> list[dict]:
    """Per trial: true-profile score, best other-in-group score and identity.

    ``stranger`` = max score against the group's gallery without the trial's own
    identity; ``top_ok`` = the full-gallery winner is the true identity."""
    groups: dict[str, list[str]] = defaultdict(list)
    for ident in sset["profiles"]:
        groups[sset["groups"][ident]].append(ident)
    rows = []
    for t in sset["trials"]:
        v = vecs.get(t["item"])
        ident = t["identity"]
        gallery = groups.get(sset["groups"].get(ident, ""), [])
        if ident not in sset["profiles"] or v is None:
            continue
        scores = {}
        for g in gallery:
            entries = [vecs[e] for e in sset["profiles"][g] if e in vecs]
            if entries:
                scores[g] = float(np.median([v @ e for e in entries]))
        if ident not in scores or len(scores) < 2:
            continue
        others = {g: s for g, s in scores.items() if g != ident}
        best_other = max(others.values())
        rows.append(
            {
                **t,
                "target": scores[ident],
                "stranger": best_other,
                "nontargets": list(others.values()),
                "top_ok": scores[ident] > best_other,
            }
        )
    return rows


def eer(target: np.ndarray, nontarget: np.ndarray) -> float:
    scores = np.concatenate([target, nontarget])
    labels = np.concatenate([np.ones(len(target)), np.zeros(len(nontarget))])
    order = np.argsort(-scores, kind="stable")
    labels = labels[order]
    tp = np.cumsum(labels) / len(target)
    fp = np.cumsum(1 - labels) / len(nontarget)
    fnr = 1 - tp
    i = np.argmin(np.abs(fnr - fp))
    return float((fnr[i] + fp[i]) / 2)


def far_threshold(stranger: np.ndarray, far: float) -> float:
    """Smallest threshold at which at most ``far`` of strangers reach it."""
    s = np.sort(stranger)[::-1]
    k = int(np.floor(far * len(s)))
    return float(s[k]) + 1e-9 if k < len(s) else float(s[-1])


def dir_at(rows: list[dict], thr: float) -> tuple[float, float]:
    """(detection-and-identification rate, false-accept rate) at ``thr``."""
    known = np.array([r["top_ok"] and r["target"] >= thr for r in rows])
    fa = np.array([r["stranger"] >= thr for r in rows])
    return float(known.mean()), float(fa.mean())


def metrics(rows: list[dict], fixed: float | None) -> dict:
    target = np.array([r["target"] for r in rows])
    non = np.array([s for r in rows for s in r["nontargets"]])
    stranger = np.array([r["stranger"] for r in rows])
    out = {
        "n": len(rows),
        "ids": len({r["identity"] for r in rows}),
        "eer": eer(target, non),
        "rank1": float(np.mean([r["top_ok"] for r in rows])),
    }
    for far in (0.01, 0.05):
        thr = far_threshold(stranger, far)
        out[f"dir@{far:g}"] = dir_at(rows, thr)[0]
        out[f"thr@{far:g}"] = thr
    if fixed is not None:
        out["fixed"] = dir_at(rows, fixed)
    return out


def bootstrap(rows_a: list[dict], rows_b: list[dict], stat, n: int = 1000) -> tuple[float, float]:
    """95 % interval of stat(b) − stat(a), resampling identities (trials of one
    person are correlated, so the person is the unit). Rows must align."""
    rng = np.random.default_rng(SEED)
    ids = sorted({r["identity"] for r in rows_a})
    idx = defaultdict(list)
    for i, r in enumerate(rows_a):
        idx[r["identity"]].append(i)
    deltas = []
    for _ in range(n):
        pick = rng.choice(len(ids), size=len(ids), replace=True)
        sel = [i for p in pick for i in idx[ids[p]]]
        deltas.append(stat([rows_b[i] for i in sel]) - stat([rows_a[i] for i in sel]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(lo), float(hi)


PRIMARY_SETS = ("ami-headset", "ami-room", "ami-cross", "ami-nemotron", "owncalls-remote")
"""The decision metric's sets, weighted equally: online call through a codec,
headset, room mic, enrollment and trial on different mics, and production
clusters. ``owncalls-remote`` drops the user's own mic side — one laptop mic
in every call, so its trials would mostly measure that mic."""


class Arrays:
    """One model × set × duration's trials as arrays, for fast resampling."""

    def __init__(self, rows: list[dict]) -> None:
        self.target = np.array([r["target"] for r in rows])
        self.stranger = np.array([r["stranger"] for r in rows])
        self.top_ok = np.array([r["top_ok"] for r in rows])
        people = sorted({r["identity"] for r in rows})
        self.people = [
            np.array([i for i, r in enumerate(rows) if r["identity"] == p]) for p in people
        ]


def _macro(arrs: list[Arrays], picks: list[np.ndarray] | None, far: float) -> tuple[float, float]:
    """(macro DIR, threshold) at the one threshold whose macro stranger-accept
    rate is ``far``: each set weighs the same however many trials it holds."""
    sets = []
    for k, a in enumerate(arrs):
        sel = slice(None) if picks is None else picks[k]
        sets.append((a.target[sel], np.sort(a.stranger[sel]), a.top_ok[sel]))
    cands = np.unique(np.concatenate([s for _, s, _ in sets])) + 1e-9
    fa = np.mean([1 - np.searchsorted(s, cands, side="left") / len(s) for _, s, _ in sets], axis=0)
    ok = np.flatnonzero(fa <= far)
    thr = float(cands[ok[0]]) if len(ok) else float(cands[-1])
    dirs = [float(np.mean(top & (t >= thr))) for t, _, top in sets]
    return float(np.mean(dirs)), thr


def _resample(arrs: list[Arrays], rng) -> list[np.ndarray]:
    picks = []
    for a in arrs:
        chosen = rng.choice(len(a.people), size=len(a.people), replace=True)
        picks.append(np.concatenate([a.people[c] for c in chosen]))
    return picks


def summary(results: dict[str, dict[str, list[dict]]], far: float = 0.01) -> list[str]:
    """The pre-registered decision metric (``out/reid_models/decision_rule.md``)."""

    def select(m: str, name: str, dur: str) -> list[dict]:
        if name == "owncalls-remote":
            rows = [r for r in results[m]["owncalls"] if r["identity"] != "self"]
        else:
            rows = results[m].get(name, [])
        return [r for r in rows if r["duration"] == dur]

    models = list(results)
    base = "base" if "base" in results else models[0]
    durs = ["2s", "3s", "5s", "10s", "full"]
    arrays = {(m, d): [Arrays(select(m, s, d)) for s in PRIMARY_SETS] for m in models for d in durs}
    lines = [
        "",
        "## Summary: one threshold for every set (the decision metric)",
        "",
        f"Macro DIR over {', '.join(PRIMARY_SETS)} at each model's single threshold "
        f"where the macro stranger-accept rate is {100 * far:g} %. Brackets: 95 % "
        f"interval of the paired difference to `{base}`, resampling people within each "
        "set (1000 draws). 'short' = mean of 2 s and 3 s.",
        "",
        "| model | " + " | ".join(durs) + " | short |",
        "|---|" + "---|" * (len(durs) + 1),
    ]
    rng = np.random.default_rng(SEED)
    draws = 1000
    picks = {d: [_resample(arrays[(base, d)], rng) for _ in range(draws)] for d in durs}
    for m in models:
        cells, short_pt, short_deltas = [], [], np.zeros(draws)
        for d in durs:
            point, _ = _macro(arrays[(m, d)], None, far)
            cell = f"{100 * point:.1f}"
            if m != base:
                deltas = np.array(
                    [
                        _macro(arrays[(m, d)], p, far)[0] - _macro(arrays[(base, d)], p, far)[0]
                        for p in picks[d]
                    ]
                )
                lo, hi = np.percentile(deltas, [2.5, 97.5])
                cell += f" [{100 * lo:+.1f}, {100 * hi:+.1f}]"
                if d in ("2s", "3s"):
                    short_deltas += deltas / 2
            if d in ("2s", "3s"):
                short_pt.append(point)
            cells.append(cell)
        short = f"{100 * np.mean(short_pt):.1f}"
        if m != base:
            lo, hi = np.percentile(short_deltas, [2.5, 97.5])
            short += f" [{100 * lo:+.1f}, {100 * hi:+.1f}]"
        lines.append(f"| {m} | " + " | ".join(cells) + f" | {short} |")

    for d in ("2s", "3s", "full"):
        lines += [
            "",
            f"### Per set at the one threshold · {d}",
            "",
            "| model | threshold | " + " | ".join(PRIMARY_SETS) + " | stranger-accept per set |",
            "|---|---|" + "---|" * (len(PRIMARY_SETS) + 1),
        ]
        for m in models:
            _, thr = _macro(arrays[(m, d)], None, far)
            dirs = [float(np.mean(a.top_ok & (a.target >= thr))) for a in arrays[(m, d)]]
            fas = [float(np.mean(a.stranger >= thr)) for a in arrays[(m, d)]]
            lines.append(
                f"| {m} | {thr:.3f} | "
                + " | ".join(f"{100 * x:.1f}" for x in dirs)
                + " | "
                + " / ".join(f"{100 * x:.1f}" for x in fas)
                + " |"
            )

    lines += [
        "",
        "### Calibration drift · 2 s",
        "",
        "Each set's own 1 %-FAR threshold; a wide spread means one fixed threshold "
        "over-accepts in some domains and under-names in others.",
        "",
        "| model | " + " | ".join(PRIMARY_SETS) + " | spread |",
        "|---|" + "---|" * (len(PRIMARY_SETS) + 1),
    ]
    for m in models:
        thrs = [far_threshold(a.stranger, far) for a in arrays[(m, "2s")]]
        lines.append(
            f"| {m} | " + " | ".join(f"{t:.3f}" for t in thrs) + f" | {max(thrs) - min(thrs):.3f} |"
        )
    return lines


def cmd_score(models: list[str] | None) -> int:
    items = {i["id"]: i for i in json.loads((OUT / "manifest.json").read_text())["items"]}
    sets = json.loads((OUT / "trials.json").read_text())
    paths = sorted(EMB_DIR.glob("*.npz"))
    if models:
        paths = [p for p in paths if p.stem in models]
    if not paths:
        print(f"no embeddings in {EMB_DIR}", file=sys.stderr)
        return 1
    results: dict[str, dict[str, list[dict]]] = {}
    speed = {}
    for p in paths:
        vecs = load_vectors(p, items)
        meta = np.load(p, allow_pickle=False)
        speed[p.stem] = float(meta["seconds_cpu"]) / float(meta["audio_seconds"]) * 10
        results[p.stem] = {name: trial_scores(sset, vecs) for name, sset in sets.items()}
        missing = len(items) - len(vecs)
        print(f"{p.stem}: {len(vecs)} items embedded, {missing} missing", flush=True)

    # One threshold per model for every condition, as production has one: the
    # point where 1 % of all stranger trials, pooled over every set and
    # duration, would be named.
    fixed = {}
    for m, by_set in results.items():
        pooled = np.array([r["stranger"] for rows in by_set.values() for r in rows])
        fixed[m] = far_threshold(pooled, 0.01)

    base = "base" if "base" in results else None
    lines = [
        "# Speaker-embedding models in the naming seat",
        "",
        "Open-set naming per trial set and cluster duration. DIR = detection-and-"
        "identification rate (named, and named right) at the threshold where 1 % "
        "(5 %) of stranger trials would be named; EER over all trial × profile "
        "pairs; rank-1 = closed-set top match. 'one threshold' = DIR / FAR at each "
        "model's single threshold (1 % FAR pooled over everything). Brackets: 95 % "
        "interval of the difference to `base`, resampling people.",
        "",
        "| model | one threshold | s per 10 s audio |",
        "|---|---|---|",
    ]
    for m in results:
        lines.append(f"| {m} | {fixed[m]:.3f} | {speed[m]:.3f} |")
    lines += summary(results)
    lines += ["", "## Detail per set, tag and duration"]
    for name in sets:
        for tag in sorted({t["tag"] for t in sets[name]["trials"]}):
            for dur in ["full", *[f"{d:g}s" for d in DURATIONS]]:
                lines += [
                    "",
                    f"## {name} · {tag} · {dur}",
                    "",
                    "| model | trials (people) | DIR@1% | DIR@5% | EER | rank-1 "
                    "| one threshold: DIR / FAR | thr@1% |",
                    "|---|---|---|---|---|---|---|---|",
                ]
                for m, by_set in results.items():
                    rows = [r for r in by_set[name] if r["tag"] == tag and r["duration"] == dur]
                    if not rows:
                        continue
                    k = metrics(rows, fixed[m])
                    ci = ""
                    if base and m != base:
                        brows = [
                            r
                            for r in results[base][name]
                            if r["tag"] == tag and r["duration"] == dur
                        ]
                        if [r["item"] for r in brows] == [r["item"] for r in rows]:
                            lo, hi = bootstrap(
                                brows, rows, lambda rs: metrics(rs, None)["dir@0.01"], n=300
                            )
                            ci = f" [{100 * lo:+.0f}, {100 * hi:+.0f}]"
                    d, f = k["fixed"]
                    lines.append(
                        f"| {m} | {k['n']} ({k['ids']}) | {100 * k['dir@0.01']:.1f}{ci} "
                        f"| {100 * k['dir@0.05']:.1f} | {100 * k['eer']:.2f} "
                        f"| {100 * k['rank1']:.1f} | {100 * d:.1f} / {100 * f:.1f} "
                        f"| {k['thr@0.01']:.3f} |"
                    )
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    (OUT / "scores.json").write_text(json.dumps({"fixed": fixed, "speed": speed}, indent=1))
    print(f"→ {OUT / 'report.md'}")
    return 0


def cmd_check_cap() -> int:
    """Shipped model, ami-headset enrollments and full trials: capped vs every
    clean span. The cap is safe if naming and target scores do not move."""
    import subprocess

    items = {i["id"]: i for i in json.loads((OUT / "manifest.json").read_text())["items"]}
    sset = json.loads((OUT / "trials.json").read_text())["ami-headset"]
    sset["trials"] = [t for t in sset["trials"] if t["duration"] == "full"]
    wanted = {e for v in sset["profiles"].values() for e in v} | {t["item"] for t in sset["trials"]}
    spans_of: dict[str, list] = {}
    for group, sessions in GROUPS.items():
        for session in sessions:
            meeting = group + session
            for name, (_, spans) in _ami_meeting(meeting).items():
                kind = "enroll" if session == ENROLL else "full"
                spans_of[f"ami/{meeting}/{name}/hs/{kind}"] = [
                    s for s in spans if s[1] - s[0] >= MIN_SPAN_S
                ] or spans
    uncapped = []
    for iid in sorted(wanted):
        uncapped.append(
            {**items[iid], "spans": [[round(s, 3), round(e, 3)] for s, e in spans_of[iid]]}
        )
    man = OUT / "manifest_uncapped.json"
    man.write_text(json.dumps({"items": uncapped}))
    out = OUT / "emb_uncapped_base.npz"
    subprocess.run(
        [
            "uv",
            "run",
            "--group",
            "eval",
            "eval/embedders/base.py",
            "--manifest",
            str(man),
            "--out",
            str(out),
        ],
        check=True,
    )
    capped = load_vectors(EMB_DIR / "base.npz", items)
    full = load_vectors(out, {i["id"]: i for i in uncapped})
    hours = sum(e - s for i in uncapped for s, e in i["spans"]) / 3600
    drift = [float(capped[i] @ full[i]) for i in wanted if i in capped and i in full]
    print(
        f"uncapped: {hours:.1f} h; capped-vs-uncapped cosine "
        f"min {min(drift):.4f} median {np.median(drift):.4f}"
    )
    for label, vecs in (("capped", capped), ("uncapped", full)):
        rows = trial_scores(sset, vecs)
        k = metrics(rows, 0.56)
        d, f = k["fixed"]
        print(
            f"{label}: n={k['n']} EER {100 * k['eer']:.2f} rank-1 {100 * k['rank1']:.1f} "
            f"DIR@1% {100 * k['dir@0.01']:.1f} at 0.56 DIR/FAR {100 * d:.1f}/{100 * f:.1f} "
            f"mean target {np.mean([r['target'] for r in rows]):.3f}"
        )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    sub.add_parser("build")
    sub.add_parser("check-cap")
    sub.add_parser("build-caps")
    sub.add_parser("score-caps")
    s = sub.add_parser("score")
    s.add_argument("--models", nargs="*")
    a = p.parse_args()
    if a.cmd == "fetch":
        return cmd_fetch()
    if a.cmd == "build":
        return cmd_build()
    if a.cmd == "check-cap":
        return cmd_check_cap()
    if a.cmd == "build-caps":
        return cmd_build_caps()
    if a.cmd == "score-caps":
        return cmd_score_caps()
    return cmd_score(a.models)


if __name__ == "__main__":
    sys.exit(main())
