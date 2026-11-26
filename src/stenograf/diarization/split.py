"""A speaker the diarizer found who is really two people, split in two.

Nemotron can put two far-end voices heard through a phone speaker into one
slot beside a clean near voice — the meeting that showed it phoned two people
in turn and got one speaker. Two checks must agree before a speaker is split,
because each alone also fires on single voices:

- the **voice split**, on the diarizer's turns: up to :data:`VOICE_TURNS` turns
  of at least :data:`VOICE_TURN_S`, evenly over the meeting and cropped to
  their middle :data:`VOICE_CROP_S`, embedded and cut in two by spherical
  2-means. Centroids below :data:`MAX_CENTROID_COSINE` with the smaller half
  holding at least :data:`MIN_MINOR_SHARE` say two voices;
- the **rerun**, on the transcript's entries, only for a speaker the voice
  split flagged: the diarizer again on that speaker's entries alone, joined
  with :data:`RERUN_GAP_S` of silence. A second slot holding at least
  :data:`SECOND_VOICE_SHARE` of the activity says two voices, and the slot an
  entry's frames lean to says whose it is. Entries, not turns: the diarizer's
  own turns run a few seconds each, and on them the rerun missed the meeting
  that motivated this.

The entries that move take with them the parts of the speaker's turns they
cover: a diarizer that merged two people does not end a turn where one hands
over to the other.

Every threshold was fixed before the data that scores it; the scores (the
AMI/ICSI harness with synthetic speakerphone channels, a blind listening test
on the user's own meetings) and the turns-against-entries comparison are in
``eval/README.md`` ("Two people in one speaker"). Nothing here knows the
diarizer or the transcript: the embeddings and the rerun arrive as callbacks,
the entries as spans.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from stenograf.audio import SAMPLE_RATE
from stenograf.diarization.base import SpeakerTurn
from stenograf.log import logger

FRAMES_PER_SECOND = 100
HOP = SAMPLE_RATE // FRAMES_PER_SECOND

MIN_SPEAKER_S = 60.0
VOICE_TURN_S = 2.0
VOICE_CROP_S = 6.0
VOICE_TURNS = 60
MIN_VOICE_TURNS = 10
MIN_MINOR_SHARE = 0.15
MAX_CENTROID_COSINE = 0.7
RERUN_GAP_S = 0.4
MIN_RERUN_SPAN_S = 0.5
"""An entry shorter than this stays out of the rerun, and with its speaker."""
SECOND_VOICE_SHARE = 0.15

VOICE_TURN = int(VOICE_TURN_S * SAMPLE_RATE)
RERUN_GAP = int(RERUN_GAP_S * SAMPLE_RATE)
MIN_RERUN_SPAN = int(MIN_RERUN_SPAN_S * SAMPLE_RATE)

Span = tuple[int, int]
"""Samples ``[lo, hi)`` of the channel, on 10 ms frames."""

Voices = Callable[[list[Span]], list[np.ndarray | None]]
"""``voices(crops)``: an L2-normalized embedding per crop, ``None`` where the
crop does not embed."""

Rerun = Callable[[np.ndarray], np.ndarray]
"""``rerun(audio)``: the diarizer's (frames, slots) probabilities of ``audio``."""


@dataclass(frozen=True)
class SpeakerCheck:
    """What the checks measured for one speaker, kept for the log and the eval."""

    label: str
    speech_s: float
    turns_embedded: int = 0
    centroid_cosine: float | None = None
    minor_share: float | None = None
    second_share: float | None = None
    split_as: str | None = None


@dataclass(frozen=True)
class SplitResult:
    speakers: list[str]
    """Each segment's speaker after the split, in the order given."""
    turns: list[SpeakerTurn]
    halves: dict[str, str]
    """New label → the label it was split from."""
    checks: list[SpeakerCheck]


def to_span(start: float, end: float) -> Span:
    """Seconds to samples on the 10 ms frame grid turns already sit on."""
    return (
        round(start * FRAMES_PER_SECOND) * HOP,
        round(end * FRAMES_PER_SECOND) * HOP,
    )


def voice_crop(span: Span) -> Span:
    """The middle :data:`VOICE_CROP_S` of a span, or the span if shorter."""
    lo, hi = span
    half = int(VOICE_CROP_S * SAMPLE_RATE) // 2
    if hi - lo <= 2 * half:
        return span
    mid = (lo + hi) // 2
    return mid - half, mid + half


def voice_crops(turns: list[SpeakerTurn]) -> list[Span]:
    """Up to :data:`VOICE_TURNS` turns of at least :data:`VOICE_TURN_S`, evenly
    over the meeting, each cut by :func:`voice_crop`."""
    spans = sorted(to_span(t.start, t.end) for t in turns)
    spans = [s for s in spans if s[1] - s[0] >= VOICE_TURN]
    if len(spans) > VOICE_TURNS:
        picks = np.linspace(0, len(spans) - 1, VOICE_TURNS).round().astype(int)
        spans = [spans[i] for i in picks]
    return [voice_crop(s) for s in spans]


def voice_split(vectors: np.ndarray) -> tuple[float, float]:
    """(centroid cosine, smaller half's share) of the best of ten spherical
    2-means cuts of L2-normalized ``vectors``; (1.0, 0.0) when none cuts."""
    x = np.asarray(vectors, np.float64)
    rng = np.random.default_rng(0)
    best: tuple[float, np.ndarray] | None = None
    # numpy 2.2.6's Accelerate matmul on macOS raised divide/overflow/invalid
    # flags on these finite unit vectors (2026-11-26); every input is finite
    # (embed returns None otherwise), so the flags carry no information.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for _ in range(10):
            centroids = x[rng.choice(len(x), 2, replace=False)]
            group = np.zeros(len(x), bool)
            for _ in range(30):
                group = (x @ centroids[0]) >= (x @ centroids[1])
                if group.all() or not group.any():
                    break
                centroids = np.array([_unit(x[group].mean(0)), _unit(x[~group].mean(0))])
            if group.all() or not group.any():
                continue
            score = float((x[group] @ centroids[0]).sum() + (x[~group] @ centroids[1]).sum())
            if best is None or score > best[0]:
                best = (score, group.copy())
        if best is None:
            return 1.0, 0.0
        group = best[1]
        cosine = float(_unit(x[group].mean(0)) @ _unit(x[~group].mean(0)))
    return cosine, float(min(group.mean(), 1 - group.mean()))


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def joined(audio: np.ndarray, spans: list[Span]) -> np.ndarray:
    """``spans`` of ``audio`` back to back, each followed by :data:`RERUN_GAP`
    samples of silence — the recording the rerun hears."""
    gap = np.zeros(RERUN_GAP, np.float32)
    parts = [piece for lo, hi in spans for piece in (audio[lo:hi], gap)]
    return np.concatenate(parts) if parts else np.zeros(0, np.float32)


def second_voice(probs: np.ndarray, spans: list[Span]) -> tuple[float, list[int | None]]:
    """The second most active slot's share of the rerun's activity inside the
    spans of :func:`joined`, and per span 1 when its frames lean to that slot,
    0 when to the most active one, ``None`` when neither speaks in it."""
    active = probs > 0.5
    bounds, at = [], 0
    for lo, hi in spans:
        bounds.append((min(at // HOP, len(active)), min((at + hi - lo) // HOP, len(active))))
        at += hi - lo + RERUN_GAP
    if active.ndim != 2 or active.shape[1] < 2:
        return 0.0, [None] * len(spans)
    counts = np.zeros(active.shape[1], np.int64)
    for a, b in bounds:
        counts += active[a:b].sum(0)
    if counts.sum() == 0:
        return 0.0, [None] * len(spans)
    order = np.argsort(-counts, kind="stable")
    share = float(counts[order[1]] / counts.sum())
    sides: list[int | None] = []
    for a, b in bounds:
        first, second = active[a:b][:, order[:2]].sum(0)
        sides.append(None if first + second == 0 else int(second > first))
    return share, sides


def split_two_voice_speakers(
    turns: list[SpeakerTurn],
    segments: list[tuple[str, float, float]],
    audio: np.ndarray,
    *,
    voices: Voices,
    rerun: Rerun,
    num_speakers: int | None = None,
) -> SplitResult:
    """Split every speaker both checks call two people.

    ``segments`` are the transcript's entries as ``(speaker, start, end)``,
    labelled with ``turns``' labels. Every speaker with :data:`MIN_SPEAKER_S`
    of turns is voice-split; one that splits is rerun on its entries, and the
    entries the rerun gives its second slot move to a new label, with the
    parts of the turns they cover. A stated count is never exceeded."""
    by_label: dict[str, list[SpeakerTurn]] = {}
    for t in turns:
        by_label.setdefault(t.speaker, []).append(t)
    speakers = [label for label, _, _ in segments]
    room = None if not num_speakers else num_speakers - len(by_label)
    next_slot = 1 + max((int(label[1:]) for label in by_label), default=-1)
    halves: dict[str, str] = {}
    checks: list[SpeakerCheck] = []
    for label in sorted(by_label, key=lambda label: int(label[1:])):
        own = by_label[label]
        speech = sum(t.end - t.start for t in own)
        if speech < MIN_SPEAKER_S or (room is not None and room <= 0):
            continue
        check = _voice_check(SpeakerCheck(label, speech), own, voices)
        if _two_voices(check):
            new = f"S{next_slot}"
            check, moved = _rerun_check(check, label, segments, audio, rerun)
            if moved:
                next_slot += 1
                for i in moved:
                    speakers[i] = new
                halves[new] = label
                check = replace(check, split_as=new)
                if room is not None:
                    room -= 1
        checks.append(check)
        logger.info(
            "two-voice check %s: %.0f s, %d turns embedded, centroid cosine %s, minor %s, "
            "rerun second voice %s%s",
            label,
            speech,
            check.turns_embedded,
            _fmt(check.centroid_cosine),
            _fmt(check.minor_share),
            _fmt(check.second_share),
            f", split as {check.split_as}" if check.split_as else "",
        )
    if not halves:
        return SplitResult(speakers, list(turns), {}, checks)
    return SplitResult(speakers, _follow(turns, segments, speakers, halves), halves, checks)


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _voice_check(check: SpeakerCheck, turns: list[SpeakerTurn], voices: Voices) -> SpeakerCheck:
    vectors = [v for v in voices(voice_crops(turns)) if v is not None]
    if len(vectors) < MIN_VOICE_TURNS:
        return replace(check, turns_embedded=len(vectors))
    cosine, minor = voice_split(np.array(vectors))
    return replace(
        check, turns_embedded=len(vectors), centroid_cosine=cosine, minor_share=minor
    )


def _two_voices(check: SpeakerCheck) -> bool:
    return (
        check.turns_embedded >= MIN_VOICE_TURNS
        and check.minor_share is not None
        and check.minor_share >= MIN_MINOR_SHARE
        and check.centroid_cosine is not None
        and check.centroid_cosine < MAX_CENTROID_COSINE
    )


def _rerun_check(
    check: SpeakerCheck,
    label: str,
    segments: list[tuple[str, float, float]],
    audio: np.ndarray,
    rerun: Rerun,
) -> tuple[SpeakerCheck, list[int]]:
    """The rerun's share, and the indices of the segments that move."""
    picked = sorted(
        (to_span(start, end), i)
        for i, (speaker, start, end) in enumerate(segments)
        if speaker == label
    )
    picked = [(span, i) for span, i in picked if span[1] - span[0] >= MIN_RERUN_SPAN]
    if not picked:
        return replace(check, second_share=0.0), []
    spans = [span for span, _ in picked]
    share, sides = second_voice(rerun(joined(audio, spans)), spans)
    check = replace(check, second_share=share)
    if share < SECOND_VOICE_SHARE:
        return check, []
    return check, [i for (_, i), side in zip(picked, sides, strict=True) if side == 1]


def _follow(
    turns: list[SpeakerTurn],
    segments: list[tuple[str, float, float]],
    speakers: list[str],
    halves: dict[str, str],
) -> list[SpeakerTurn]:
    """Cut each turn of a split speaker at the edges of the segments that
    moved: the parts inside a moved segment go with it, the rest stays.
    Segments are a channel's entries, in time order and disjoint; their edges
    are rounded onto the turns' 10 ms grid."""
    moved: dict[str, list[tuple[float, float, str]]] = {}
    for (was, start, end), now in zip(segments, speakers, strict=True):
        if now != was and halves.get(now) == was:
            moved.setdefault(was, []).append((round(start, 2), round(end, 2), now))
    for cuts in moved.values():
        cuts.sort()
    out: list[SpeakerTurn] = []
    for t in turns:
        cuts = moved.get(t.speaker)
        if not cuts:
            out.append(t)
            continue
        at = t.start
        k = max(0, bisect_right([c[0] for c in cuts], t.start) - 1)
        while k < len(cuts) and cuts[k][0] < t.end:
            start, end, now = cuts[k]
            lo, hi = max(start, at), min(end, t.end)
            if hi > lo:
                if lo > at:
                    out.append(SpeakerTurn(t.speaker, at, lo))
                out.append(SpeakerTurn(now, lo, hi))
                at = hi
            k += 1
        if at < t.end:
            out.append(SpeakerTurn(t.speaker, at, t.end))
    out.sort(key=lambda t: (t.start, t.speaker))
    return out
