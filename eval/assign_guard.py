"""Where should ``steno profiles assign`` refuse a voice that disagrees with the profile?

``assign`` adds a meeting speaker's voice to a named profile. Two mistakes
poison a profile: a *phantom* — a crosstalk cluster that carries an on-channel
person's label, the held-out FIO084 case (cosine 0.046, the profile's later
scores 0.84 → 0.41) — and a *wrong person*, a real voice given someone else's
name. A guard refuses an entry whose mean cosine to the profile's stored
entries is below some bar. The bar must let *genuine* entries through, above
all the ones the command exists for: speakers re-ID left unnamed, whose
similarity is below the naming threshold by construction.

Scored on AMI's held-out meetings, Nemotron's library turns with no count and
with the true count applied, embeddings as production computes them:

- **profiles** — per participant, from the raw session headset (clean), and as
  ``assign`` would build them: each session-a cluster labelled with its
  dominant reference speaker, assigned in channel order (mic, loop);
- **entries** — every cluster of sessions b–d against its labelled speaker's
  profile, and every genuine one against the other participants' profiles
  (wrong person). A cluster is a phantom when ≥ 60 % of its speech lies inside
  a participant who is *not* on the channel (``heldout.explained_extras``'
  crosstalk rule — overlap with the labelled speaker alone misses it, since
  crosstalk co-occurs with the wearer's own speech), genuine when it is not and
  ≥ 50 % of it is the labelled speaker's. Genuine clusters are also scored from
  their first 3 s and 2 s of clean speech: a person who said little, the
  regime where re-ID leaves voices unnamed and a correction is most needed.

Last, what admitting an entry *does*: each short genuine entry and each phantom
is added to its person's profile in turn, and that person's full-length
genuine clusters from the *other* sessions are re-scored — named at the shipped
threshold before and after, by the entry's own similarity to the profile.

    uv run --group eval eval/assign_guard.py
"""

from __future__ import annotations

import json
import sys

import ami
import heldout
import numpy as np
from common import read_pcm16
from naming_gate import clean_spans, embed_spans, truncate_spans
from rttm import Turn, parse_rttm

from stenograf.diarization.base import SpeakerTurn
from stenograf.diarization.embedding import VoiceEmbedder
from stenograf.diarization.nemotron import turns_from_probs

OUT = heldout.OUT_DIR
PROBS = OUT / "probs_nemotron_library"
NAMING = 0.56
PURE = 0.5
SHORT = (3.0, 2.0)
BINS = ((-1.0, 0.30), (0.30, 0.45), (0.45, NAMING), (NAMING, 1.01))
BARS = (0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.56)


def share(cluster: list[SpeakerTurn], ref: list[Turn]) -> float:
    """Share of the cluster's speech inside ``ref``'s turns."""
    end = max([t.end for t in cluster] + [t.end for t in ref])
    n = int(end * 100) + 2
    mine, theirs = np.zeros(n, bool), np.zeros(n, bool)
    for t in cluster:
        mine[int(t.start * 100) : int(t.end * 100)] = True
    for t in ref:
        theirs[int(t.start * 100) : int(t.end * 100)] = True
    return float((mine & theirs).sum() / max(mine.sum(), 1))


def off_channel(cid: str, ref: list[Turn]) -> list[Turn]:
    """The meeting's speech by participants not on this channel."""
    meeting, on = cid.split(".")[0], {t.speaker for t in ref}
    return [
        t
        for kind in ("mic", "loop")
        for t in parse_rttm(ami.AMI_REFS_DIR / f"{meeting}.{kind}.rttm")
        if t.speaker not in on
    ]


def headset_profiles(meeting: str, embedder: VoiceEmbedder) -> dict[str, np.ndarray]:
    annotations = ami.RAW_DIR / "annotations"
    meetings = ami.parse_meetings_xml(annotations / "corpusResources" / "meetings.xml")
    out = {}
    for agent, (headset, name) in meetings[meeting].items():
        spans = ami.merge_spans(
            ami.parse_words_xml(annotations / "words" / f"{meeting}.{agent}.words.xml")
        )
        if not spans:
            continue
        turns = [SpeakerTurn(name, s, e) for s, e in spans]
        wav = ami.RAW_DIR / meeting / f"{meeting}.Headset-{headset}.wav"
        out[name] = embedder.cluster_embeddings(turns, read_pcm16(wav))[name]
    return out


def clusters(channel: dict, capped: bool, embedder: VoiceEmbedder) -> list[dict]:
    cid = channel["id"]
    probs = np.load(PROBS / f"{cid}.npy").astype(np.float32)
    turns = turns_from_probs(probs, channel["num_speakers"] if capped else None)
    cache = OUT / ("assign_guard_capped" if capped else "collapse_embeddings") / f"{cid}.npz"
    cache.parent.mkdir(exist_ok=True)
    if cache.exists():
        with np.load(cache) as stored:
            vectors = {k: stored[k] for k in stored.files}
    else:
        vectors = embedder.cluster_embeddings(turns, read_pcm16(ami.CHANNELS_DIR / f"{cid}.wav"))
        np.savez(cache, **vectors)
    ref = parse_rttm(ami.AMI_REFS_DIR / f"{cid}.rttm")
    outside = off_channel(cid, ref)
    by: dict[str, list[SpeakerTurn]] = {}
    for t in turns:
        by.setdefault(t.speaker, []).append(t)
    spans = clean_spans(by)
    pcm = None
    out = []
    for label, cl in sorted(by.items()):
        who = ami.dominant_speaker([Turn(t.speaker, t.start, t.end) for t in cl], ref)
        if who is None or label not in vectors:
            continue
        if share(cl, outside) >= heldout.EXPLAINED_SHARE:
            kind = "phantom"
        elif share(cl, [t for t in ref if t.speaker == who]) >= PURE:
            kind = "genuine"
        else:
            kind = "mixed"
        short = {}
        if kind == "genuine" and channel["session"] != "a":
            if pcm is None:
                pcm = read_pcm16(ami.CHANNELS_DIR / f"{cid}.wav")
            for budget in SHORT:
                short[budget] = embed_spans(
                    pcm, truncate_spans(spans[label], budget), embedder.embed
                )
        out.append(
            {
                "id": f"{cid}/{label}",
                "who": who,
                "session": channel["session"],
                "kind": kind,
                "vec": vectors[label],
                "short": short,
            }
        )
    return out


def mean_cos(vec: np.ndarray, entries: list[np.ndarray]) -> float:
    return float(np.mean([vec @ e for e in entries]))


def refused(scores: list[float], bar: float) -> str:
    return f"{sum(s < bar for s in scores)}/{len(scores)}"


def admit_effect(later: list[tuple[str, dict]], profiles: dict) -> None:
    """Admit each candidate entry alone; re-score its person's other-session clusters."""
    rows: dict[tuple[str, int], list[tuple[int, int, float]]] = {}
    for group, x in later:
        prof = profiles[group]
        if x["who"] not in prof:
            continue
        entries = prof[x["who"]]
        others = [
            y
            for g, y in later
            if g == group
            and y["who"] == x["who"]
            and y["kind"] == "genuine"
            and y["session"] != x["session"]
        ]
        targets = {
            "full": [y["vec"] for y in others],
            "3 s": [y["short"][3.0] for y in others if y["short"].get(3.0) is not None],
        }
        if not targets["full"]:
            continue
        candidates = [("phantom", x["vec"])] if x["kind"] == "phantom" else []
        if x["kind"] == "genuine":
            candidates += [(f"{b:g} s genuine", v) for b, v in x["short"].items() if v is not None]
        for kind, vec in candidates:
            sim = mean_cos(vec, entries)
            b = next(i for i, (lo, hi) in enumerate(BINS) if lo <= sim < hi)
            row = []
            for group_targets in targets.values():
                before = [mean_cos(t, entries) for t in group_targets]
                after = [mean_cos(t, [*entries, vec]) for t in group_targets]
                row += [sum(s >= NAMING for s in before), sum(s >= NAMING for s in after)]
            rows.setdefault((kind, b), []).append(tuple(row))
    print(
        "  admitting one entry: its person's other-session clusters named before → after"
        " (full length; first 3 s)"
    )
    for (kind, b), rs in sorted(rows.items()):
        lo, hi = BINS[b]
        total = [sum(r[i] for r in rs) for i in range(4)]
        print(
            f"    {kind:14s} entry sim {max(lo, 0):.2f}–{min(hi, 1):.2f}: {len(rs):3d} entries, "
            f"full {total[0]} → {total[1]}; 3 s {total[2]} → {total[3]}"
        )


def main() -> int:
    channels = json.loads((OUT / "channels.json").read_text())
    embedder = VoiceEmbedder()
    groups = sorted({c["group"] for c in channels})
    headset = {g: headset_profiles(g + "a", embedder) for g in groups}
    for capped in (False, True):
        arm = "count applied" if capped else "no count"
        sequential: list[tuple[str, str, float]] = []
        built: dict[str, dict[str, list[np.ndarray]]] = {g: {} for g in groups}
        later: list[tuple[str, dict]] = []
        for c in channels:
            group = c["group"]
            found = clusters(c, capped, embedder)
            if c["session"] != "a":
                later += [(group, x) for x in found]
                continue
            if c["id"].endswith(".duo"):
                continue  # the duo mixes the same headsets the loop does
            for x in found:
                entries = built[group].setdefault(x["who"], [])
                if entries:
                    sequential.append((x["id"], x["kind"], mean_cos(x["vec"], entries)))
                entries.append(x["vec"])
        print(f"\n== Nemotron, {arm} ==")
        print("session-a assign order (mic, then loop), each later entry vs the profile so far:")
        for cid, kind, s in sequential:
            print(f"  {cid:22s} {kind:8s} {s:6.3f}")
        for source, profiles in (
            ("headset profile", {g: {n: [v] for n, v in p.items()} for g, p in headset.items()}),
            ("session-a cluster profile", built),
        ):
            cols: dict[str, list[float]] = {
                "genuine": [],
                "3 s": [],
                "2 s": [],
                "phantom": [],
                "wrong": [],
            }
            for group, x in later:
                prof = profiles[group]
                if x["who"] in prof:
                    entries = prof[x["who"]]
                    if x["kind"] == "phantom":
                        cols["phantom"].append(mean_cos(x["vec"], entries))
                    elif x["kind"] == "genuine":
                        cols["genuine"].append(mean_cos(x["vec"], entries))
                        for budget, label in zip(SHORT, ("3 s", "2 s"), strict=True):
                            if x["short"].get(budget) is not None:
                                cols[label].append(mean_cos(x["short"][budget], entries))
                if x["kind"] == "genuine":
                    cols["wrong"] += [
                        mean_cos(x["vec"], e) for n, e in prof.items() if n != x["who"]
                    ]
            print(f"\n{source}: entries refused at each bar")
            print("  " + f"{'bar':>5s}" + "".join(f"{k:>10s}" for k in cols))
            for bar in BARS:
                print(
                    "  " + f"{bar:5.2f}" + "".join(f"{refused(v, bar):>10s}" for v in cols.values())
                )
            print(
                "  "
                + "; ".join(
                    f"{k} {min(v):.3f}–{max(v):.3f}" if v else f"{k} none" for k, v in cols.items()
                )
            )
            admit_effect(later, profiles)
    return 0


if __name__ == "__main__":
    sys.exit(main())
