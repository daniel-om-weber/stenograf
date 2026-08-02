"""The finalize pass: VAD windows → batch ASR → diarization → merged entries.

This is the accuracy core. It operates on one channel of mono
16 kHz PCM; the meeting orchestrator runs it per channel (mic / system) and
interleaves the results. ``steno transcribe`` runs it on a file.
"""

from __future__ import annotations

import re
import time
from bisect import bisect_right
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from stenograf.session import MeetingResult
    from stenograf.view import LiveView

from stenograf.asr.base import ASRBackend, Segment, Word
from stenograf.audio import SAMPLE_RATE, sample_index
from stenograf.config import Language, MeetingProfile, ResolvedParameters, resolve_value
from stenograf.diarization.base import Diarizer, SpeakerTurn
from stenograf.glossary import DEFAULT_THRESHOLD, apply_glossary
from stenograf.lid import detect_language
from stenograf.transcript import Transcript, TranscriptEntry
from stenograf.vad import SileroVAD, claim_start, context_start, pack_windows

MAX_ENTRY_GAP = 1.5
"""Silence (s) between words of one speaker that still reads as one entry."""

STAGE_ASR = "asr"
STAGE_DIARIZATION = "diarization"
"""The two ``on_progress`` stage names (``on_progress(stage, done, total)``)."""

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
    (cross-meeting re-ID). Unmatched clusters keep their ``S<n>`` label
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
        segments = _decode(samples, asr=asr, language=language, vad=vad, on_progress=on_progress)
        if diarizer is None or num_speakers == 1:
            return [
                TranscriptEntry(
                    speaker="S0", text=seg.text, start=seg.start, end=seg.end, words=seg.words
                )
                for seg in segments
            ]
        words = [word for seg in segments for word in seg.words]
    return _attribute(
        samples,
        words,
        segments,
        diarizer=diarizer,
        num_speakers=num_speakers,
        reid=reid,
        on_progress=on_progress,
    )


def _decode(
    samples,
    *,
    asr: ASRBackend,
    language: Language | None,
    vad: SileroVAD | None,
    on_progress,
) -> list[Segment]:
    """VAD-window the channel and batch-decode each window into segments."""
    duration = len(samples) / SAMPLE_RATE
    if vad is not None:
        windows = pack_windows(vad.speech_segments(samples), duration)
    else:
        windows = [(0.0, duration)] if duration > 0 else []
    segments: list[Segment] = []
    previous_end = 0.0
    for i, (start, end) in enumerate(windows):
        if on_progress is not None:
            on_progress(STAGE_ASR, i, len(windows))
        # Short windows decode with contiguous left context (vad.context_start),
        # and every window claims the words it owns — its span plus a pre-roll
        # for a late VAD onset (vad.claim_start); the rest of the re-read
        # context is dropped below. Mirrored operation for operation by
        # WindowedLiveDecoder._decode_window (reuse guarantee).
        ctx = context_start(start, end)
        keep_from = claim_start(start, previous_end)
        previous_end = end
        window = samples[sample_index(ctx) : sample_index(end)]
        for seg in asr.transcribe(window, language):
            clipped = _clip_context(_shift(seg, ctx), keep_from)
            if clipped is not None:
                segments.append(clipped)
    segments.sort(key=lambda seg: seg.start)
    return segments


def _clip_context(seg: Segment, keep_from: float) -> Segment | None:
    """Drop the words a context-carried decode re-read but does not own.

    A word belongs to the window containing its midpoint (the rule speaker
    attribution uses), where a window reaches ``keep_from`` — its padded start
    less a pre-roll, floored at the previous window's end (``vad.claim_start``).
    Most segments lie wholly inside the window and pass through untouched
    (keeping the model's own sentence text); a segment straddling the boundary
    is rebuilt from its kept words, and one entirely inside the context
    disappears — its window already emitted it.
    """
    kept = tuple(w for w in seg.words if (w.start + w.end) / 2 >= keep_from)
    if len(kept) == len(seg.words):
        return seg
    if not kept:
        return None
    return Segment(
        text=" ".join(w.text for w in kept), start=kept[0].start, end=kept[-1].end, words=kept
    )


def _attribute(
    samples,
    words: list[Word],
    segments: list[Segment],
    *,
    diarizer: Diarizer,
    num_speakers: int | None,
    reid: SpeakerResolver | None,
    on_progress,
) -> list[TranscriptEntry]:
    """Diarize the channel and merge the decoded words with the speaker turns."""
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
        on_progress(STAGE_DIARIZATION, 0, 1)
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
    ``"audio"`` channel. ``profile.language`` is the *given*
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
    index = _TurnIndex(turns)
    return _group_runs(ordered, lambda word: _assign(word, index), max_gap)


def group_words(
    words: list[Word], speaker: str, *, max_gap: float = MAX_ENTRY_GAP
) -> list[TranscriptEntry]:
    """Group one un-diarized speaker's words into entries, split on gaps > max_gap.

    The live checkpoint (Option B) turns a channel's committed live
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


class _TurnIndex:
    """Turn lookup by word midpoint for :func:`merge_words_turns`.

    Assigning every word by scanning the full turn list is O(words × turns) —
    quadratic in meeting length. Sorting the turns by start once, with a
    running max of end, bounds each lookup to the few turns that can still
    reach the midpoint.
    """

    def __init__(self, turns: list[SpeakerTurn]) -> None:
        self._turns = sorted(turns, key=lambda t: t.start)
        self._starts = [t.start for t in self._turns]
        # Running max of end over the sorted prefix (and the turn holding it):
        # once it drops to the midpoint no earlier turn can cover it, and when
        # nothing covers, that turn is the nearest one ending at or before it.
        self._max_end: list[float] = []
        self._max_end_turn: list[SpeakerTurn] = []
        for t in self._turns:
            if not self._max_end or t.end > self._max_end[-1]:
                self._max_end.append(t.end)
                self._max_end_turn.append(t)
            else:
                self._max_end.append(self._max_end[-1])
                self._max_end_turn.append(self._max_end_turn[-1])

    def covering(self, midpoint: float) -> list[SpeakerTurn]:
        """All turns with ``start <= midpoint < end``, in start order."""
        found = []
        i = bisect_right(self._starts, midpoint) - 1
        while i >= 0 and self._max_end[i] > midpoint:
            if self._turns[i].end > midpoint:
                found.append(self._turns[i])
            i -= 1
        found.reverse()
        return found

    def nearest(self, midpoint: float) -> SpeakerTurn | None:
        """The turn closest to an uncovered midpoint (``None`` with no turns).

        With nothing covering the midpoint, every turn starting at or before
        it also ends at or before it — the candidate is the latest such end —
        and every other turn starts after it — the candidate is the earliest
        such start.
        """
        hi = bisect_right(self._starts, midpoint)
        before = self._max_end_turn[hi - 1] if hi else None
        after = self._turns[hi] if hi < len(self._turns) else None
        if before is None or after is None:
            return before or after
        return before if midpoint - before.end <= after.start - midpoint else after


def _assign(word: Word, turns: _TurnIndex) -> tuple[str, bool]:
    midpoint = (word.start + word.end) / 2
    covering = turns.covering(midpoint)
    if len(covering) == 1:
        return covering[0].speaker, False
    if covering:  # overlapping speech
        best = max(covering, key=lambda t: min(t.end, word.end) - max(t.start, word.start))
        return best.speaker, True
    nearest = turns.nearest(midpoint)
    if nearest is None:
        return "S0", False
    return nearest.speaker, False


def resolve_split_channels(
    audio_file: Path, mode: str
) -> tuple[tuple[np.ndarray, np.ndarray] | None, float | None]:
    """Decide mixed vs per-channel transcription for a recorded file.

    ``mode`` is ``auto``/``mix``/``split`` (the CLI's ``--channels``; the app
    always passes ``auto``). Returns ``(pcms, correlation)``: ``pcms`` is the
    ``(left, right)`` float32 pair when the file should be transcribed as two
    voice channels, ``None`` for the classic mixed stream. ``correlation`` is
    the envelope correlation whenever ``auto`` examined a 2-channel file (for
    the caller to explain its decision), ``None`` when no decision was needed
    or the split was forced. Raises :class:`ValueError` when a forced split
    meets audio without exactly 2 channels.
    """
    from stenograf.audio import (
        audio_channel_count,
        channels_look_independent,
        load_audio_channels,
    )

    count = audio_channel_count(audio_file)
    if mode == "split" and count != 2:
        raise ValueError(
            f"--channels split needs 2-channel audio; {audio_file.name} has {count} channel(s)"
        )
    if count != 2 or mode == "mix":
        return None, None
    left, right = load_audio_channels(audio_file)
    if mode == "split":
        return (left, right), None
    independent, correlation = channels_look_independent(left, right)
    return ((left, right) if independent else None), correlation


def transcribe_split_channels(
    left: np.ndarray,
    right: np.ndarray,
    *,
    profile: MeetingProfile,
    view: LiveView,
    use_reid: bool = True,
    reid_threshold: float | None = None,
    glossary_threshold: float | None = None,
    asr_backend: str | None = None,
    asr_provider: str | None = None,
    asr_boost: float | None = None,
    profile_store: Path | None = None,
) -> tuple[MeetingResult, float]:
    """Transcribe two voice channels through the meeting finalize.

    This is the exact pipeline a live meeting runs on stop — per-channel ASR
    and diarization with the channel's speaker count, cross-channel echo-text
    dedup (armed conservatively: the recording's canceller state is unknown),
    glossary, one interleaved Local-N/Remote-N transcript — just fed from a
    file instead of a capture session. Returns ``(result, elapsed)``; the
    :class:`~stenograf.session.MeetingResult` carries the per-channel speaker
    counts for reporting, ``elapsed`` the processing seconds (clocked after
    model load, so a first-run weight download never masquerades as
    transcription speed).

    ``view`` receives the loader progress and per-channel status lines — the
    CLI passes an echoing view, the app its meeting-screen view; a click echo
    has no terminal to land on in a GUI process.
    """
    from stenograf import loaders
    from stenograf.audio import to_int16
    from stenograf.capture.base import AudioFrame, Channel
    from stenograf.session import MeetingRecorder, SessionStore, plan_channels

    plans = plan_channels(profile)
    asr, vad, diarizer = loaders.load_backends(
        need_diarizer=any(p.num_speakers != 1 for p in plans),
        asr_backend=asr_backend,
        asr_provider=asr_provider,
        glossary=profile.glossary,
        attendee_names=profile.attendee_names,
        boost=asr_boost,
        announce=view.status,
    )
    reid = None
    if diarizer is not None:  # re-ID relabels diarized speakers only
        reid = loaders.load_reid(
            enabled=use_reid,
            threshold=reid_threshold,
            store_path=profile_store or profile.speaker_profile_store,
        )
        if reid is not None:
            view.status(f"re-ID: {len(reid.store.for_model(reid.model))} profile(s) active")
    recorder = MeetingRecorder(
        profile,
        asr=asr,
        vad=vad,
        diarizer=diarizer,
        reid=reid,
        glossary_threshold=glossary_threshold,
    )
    store = SessionStore({Channel.MIC, Channel.SYSTEM})
    store.append(AudioFrame(Channel.MIC, 0.0, to_int16(left)))
    store.append(AudioFrame(Channel.SYSTEM, 0.0, to_int16(right)))
    started = time.monotonic()
    result = recorder.finalize(store, plans, view=view)
    return result, time.monotonic() - started


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
