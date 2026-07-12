"""The finalize pass: VAD windows → batch ASR → diarization → merged entries.

This is the accuracy core (PLAN.md §2). It operates on one channel of mono
16 kHz PCM; the meeting orchestrator runs it per channel (mic / system) and
interleaves the results. ``steno transcribe`` runs it on a file.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import replace
from typing import Protocol

import numpy as np

from stenograf.asr.base import ASRBackend, Segment, Word
from stenograf.audio import SAMPLE_RATE, sample_index
from stenograf.config import Language, MeetingProfile, ResolvedParameters, resolve_value
from stenograf.diarization.base import Diarizer, SpeakerTurn
from stenograf.glossary import DEFAULT_THRESHOLD, apply_glossary
from stenograf.lid import detect_language
from stenograf.transcript import Transcript, TranscriptEntry
from stenograf.vad import SileroVAD, pack_windows

MAX_ENTRY_GAP = 1.5
"""Silence (s) between words of one speaker that still reads as one entry."""

_RAW_CLUSTER = re.compile(r"S\d+")
"""Raw diarization cluster label (``S0``, ``S1``…), as emitted by
:class:`~stenograf.diarization.base.SpeakerTurn` and :func:`merge_words_turns`.
:func:`relabel_speakers` only renumbers labels of this shape; anything else (a
re-ID profile name) is already final and passes through untouched."""


class SpeakerResolver(Protocol):
    """Maps a run's per-cluster voice embeddings to persistent speaker names.

    Structurally satisfied by :class:`stenograf.profiles.SpeakerReID`; kept as a
    Protocol so the accuracy core need not depend on the profile store.
    """

    def resolve(self, embeddings: dict[str, np.ndarray]) -> dict[str, str]: ...


def finalize_channel(
    samples,
    *,
    asr: ASRBackend,
    language: Language | None,
    vad: SileroVAD | None = None,
    diarizer: Diarizer | None = None,
    num_speakers: int | None = None,
    reid: SpeakerResolver | None = None,
    on_progress=None,
    precomputed_words: tuple[Word, ...] | None = None,
) -> list[TranscriptEntry]:
    """Transcribe one channel; returns entries with raw ``S<n>`` speaker labels.

    ``diarizer=None`` or ``num_speakers=1`` attributes everything to ``S0``.
    ``on_progress`` is called as ``on_progress(stage: str, done: int, total: int)``.

    With ``reid`` given (and diarization running), the diarizer additionally emits
    a per-cluster voice embedding and ``reid`` maps matched clusters to persistent
    speaker-profile names; those entries carry the profile name instead of ``S<n>``
    (cross-meeting re-ID, PLAN.md §2). Unmatched clusters keep their ``S<n>`` label
    for the caller to template.

    ``precomputed_words`` skips the VAD+ASR stage entirely: the words (absolute
    session times) come from the live window pass, whose decodes are
    finalize-identical (:class:`~stenograf.live.WindowedLiveDecoder`); only
    diarization and merging run here. An empty tuple means the channel had no
    speech. ``asr``/``vad``/``language`` are ignored in that case.
    """
    if precomputed_words is not None:
        if diarizer is None or num_speakers == 1:
            return group_words(sorted(precomputed_words, key=lambda w: w.start), "S0")
        words = list(precomputed_words)
        segments: list[Segment] = []
    else:
        duration = len(samples) / SAMPLE_RATE
        if vad is not None:
            windows = pack_windows(vad.speech_segments(samples), duration)
        else:
            windows = [(0.0, duration)] if duration > 0 else []

        segments = []
        for i, (start, end) in enumerate(windows):
            if on_progress is not None:
                on_progress("asr", i, len(windows))
            window = samples[sample_index(start) : sample_index(end)]
            segments.extend(_shift(seg, start) for seg in asr.transcribe(window, language))
        segments.sort(key=lambda seg: seg.start)

        if diarizer is None or num_speakers == 1:
            return [
                TranscriptEntry(
                    speaker="S0",
                    text=seg.text,
                    start=seg.start,
                    end=seg.end,
                    words=seg.words,
                )
                for seg in segments
            ]

        words = [word for seg in segments for word in seg.words]
    if not words and segments:
        # A backend that emits text but no word timestamps (a contract
        # violation for diarized use — see ASRBackend) would otherwise drop the
        # whole transcript here. Fall back to attributing each segment as a unit
        # by its time span rather than losing the text.
        words = [Word(text=seg.text, start=seg.start, end=seg.end) for seg in segments]
    if not words:
        # No speech on this channel: nothing to diarize, so skip it. Diarizing
        # here is not just wasted work — sherpa can raise on empty/near-silent
        # input forced to num_clusters > 1, and that exception would otherwise
        # sink the whole meeting's finalize (a silent remote or a dead second
        # mic is reachable in hybrid mode).
        return []

    if on_progress is not None:
        on_progress("diarization", 0, 1)
    if reid is not None:
        result = diarizer.diarize_with_embeddings(samples, num_speakers)
        turns = result.turns
        names = reid.resolve(result.embeddings)
    else:
        turns = diarizer.diarize(samples, num_speakers)
        names = {}
    entries = merge_words_turns(words, turns)
    if names:
        entries = [
            replace(e, speaker=names[e.speaker]) if e.speaker in names else e for e in entries
        ]
    return entries


def finalize_file(
    samples,
    *,
    profile: MeetingProfile,
    asr: ASRBackend,
    vad: SileroVAD | None = None,
    diarizer: Diarizer | None = None,
    num_speakers: int | None = None,
    reid: SpeakerResolver | None = None,
    glossary_threshold: float | None = None,
    on_progress=None,
) -> Transcript:
    """One mixed audio stream → a finished transcript (``steno transcribe``).

    Runs the same accuracy core a meeting's stop runs (:func:`finalize_channel`)
    followed by the same post-steps :meth:`MeetingRecorder.finalize` applies —
    display relabel, glossary snap, language detection, parameter provenance —
    so a file transcribe and a live meeting produce the same artifact shape.
    One un-split stream has no local/remote model, so speakers get the neutral
    ``Speaker <n>`` template and provenance is recorded under a single
    ``"audio"`` channel (PLAN.md §5 3b). ``profile.language`` is the *given*
    language (``None`` = detect); the returned transcript carries the resolved
    one."""
    entries = relabel_speakers(
        finalize_channel(
            samples,
            asr=asr,
            language=profile.language,
            vad=vad,
            diarizer=diarizer,
            num_speakers=num_speakers,
            reid=reid,
            on_progress=on_progress,
        )
    )
    threshold = DEFAULT_THRESHOLD if glossary_threshold is None else glossary_threshold
    entries = apply_glossary(
        entries,
        glossary=profile.glossary,
        attendee_names=profile.attendee_names,
        threshold=threshold,
    )
    language = profile.language
    if language is None:
        language = detect_language(" ".join(e.text for e in entries))
    parameters = ResolvedParameters(
        language=resolve_value(profile.language, language),
        speakers={"audio": resolve_value(num_speakers, len({e.speaker for e in entries}))},
    )
    return Transcript(language=language, profile=profile, entries=entries, parameters=parameters)


def _shift(seg: Segment, offset: float) -> Segment:
    return replace(
        seg,
        start=seg.start + offset,
        end=seg.end + offset,
        words=tuple(replace(w, start=w.start + offset, end=w.end + offset) for w in seg.words),
    )


def merge_words_turns(
    words: list[Word],
    turns: list[SpeakerTurn],
    *,
    max_gap: float = MAX_ENTRY_GAP,
) -> list[TranscriptEntry]:
    """Assign each word a speaker and group runs into transcript entries.

    A word takes the speaker of the turn covering its midpoint. Inside
    overlapping turns the largest-overlap turn wins and the entry is flagged
    provisional; words outside every turn take the nearest turn's speaker.
    """
    ordered = sorted(words, key=lambda w: w.start)
    return _group_runs(ordered, lambda word: _assign(word, turns), max_gap)


def group_words(
    words: list[Word], speaker: str, *, max_gap: float = MAX_ENTRY_GAP
) -> list[TranscriptEntry]:
    """Group one un-diarized speaker's words into entries, split on gaps > max_gap.

    The live checkpoint (Option B, PLAN.md §3) turns a channel's committed live
    words into readable entries the same way :func:`merge_words_turns` groups a
    diarization turn — one entry per continuous run of speech — but with no
    speaker assignment: every word is attributed to ``speaker`` (a channel-coarse
    ``Local``/``Remote`` label, since the live pass does not diarize). Words must
    already be in time order.
    """
    return _group_runs(words, lambda _: (speaker, False), max_gap)


def _group_runs(
    words: list[Word],
    assign: Callable[[Word], tuple[str, bool]],
    max_gap: float,
) -> list[TranscriptEntry]:
    """Close-run-on-gap grouping shared by the diarized and un-diarized paths.

    ``assign`` gives each word its ``(speaker, provisional)``; a run closes when
    the speaker changes or the silence to the next word exceeds ``max_gap``, and
    an entry is provisional if any word in its run was. Words must be in time
    order.
    """
    entries: list[TranscriptEntry] = []
    run: list[Word] = []
    run_speaker = ""
    run_provisional = False

    def close_run() -> None:
        nonlocal run, run_provisional
        if run:
            entries.append(
                TranscriptEntry(
                    speaker=run_speaker,
                    text=" ".join(w.text for w in run),
                    start=run[0].start,
                    end=run[-1].end,
                    provisional=run_provisional,
                    words=tuple(run),
                )
            )
        run = []
        run_provisional = False

    for word in words:
        speaker, provisional = assign(word)
        if run and (speaker != run_speaker or word.start - run[-1].end > max_gap):
            close_run()
        run.append(word)
        run_speaker = speaker
        run_provisional = run_provisional or provisional
    close_run()
    return entries


def _assign(word: Word, turns: list[SpeakerTurn]) -> tuple[str, bool]:
    if not turns:
        return "S0", False
    midpoint = (word.start + word.end) / 2
    covering = [t for t in turns if t.start <= midpoint < t.end]
    if len(covering) == 1:
        return covering[0].speaker, False
    if covering:  # overlapping speech
        best = max(covering, key=lambda t: min(t.end, word.end) - max(t.start, word.start))
        return best.speaker, True
    nearest = min(turns, key=lambda t: max(t.start - midpoint, midpoint - t.end))
    return nearest.speaker, False


def relabel_speakers(
    entries: list[TranscriptEntry], template: str = "Speaker {n}"
) -> list[TranscriptEntry]:
    """Map raw ``S<n>`` cluster labels to display names, numbered by first
    appearance. Labels that are not raw cluster labels — a speaker-profile name
    assigned by re-ID — are already final and pass through unchanged (so a
    matched "Daniel" is not renumbered into ``Local-1``)."""
    mapping: dict[str, str] = {}
    result = []
    for entry in entries:
        label = entry.speaker
        if _RAW_CLUSTER.fullmatch(label):
            if label not in mapping:
                mapping[label] = template.format(n=len(mapping) + 1)
            label = mapping[label]
        result.append(replace(entry, speaker=label))
    return result
