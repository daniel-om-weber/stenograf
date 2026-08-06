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
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from stenograf.session import MeetingResult
    from stenograf.view import LiveView

from stenograf.asr.base import ASRBackend, Segment, Word
from stenograf.audio import SAMPLE_RATE, l2_normalize, sample_index
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

ProgressCallback = Callable[[str, int, int], None]
"""``on_progress(stage, done, total)`` — ``stage`` is one of the two above."""

_RAW_CLUSTER = re.compile(r"S\d+")
"""Raw diarization cluster label (``S0``, ``S1``…), as emitted by
:class:`~stenograf.diarization.base.SpeakerTurn` and :func:`merge_words_turns`.
:func:`relabel_speakers` only renumbers labels of this shape; anything else (a
re-ID profile name) is already final and passes through untouched."""


class SpeakerResolver(Protocol):
    """Maps a run's per-cluster voice embeddings to persistent speaker names.

    Structurally satisfied by :class:`stenograf.voiceprints.SpeakerReID`; kept as a
    Protocol so the accuracy core need not depend on the profile store.
    """

    def resolve(self, embeddings: dict[str, np.ndarray]) -> dict[str, str]: ...


def finalize_channel(
    samples: np.ndarray,
    *,
    asr: ASRBackend,
    language: Language | None,
    vad: SileroVAD | None = None,
    diarizer: Diarizer | None = None,
    num_speakers: int | None = None,
    reid: SpeakerResolver | None = None,
    on_progress: ProgressCallback | None = None,
    on_embeddings: Callable[[dict[str, np.ndarray]], None] | None = None,
    precomputed_words: tuple[Word, ...] | None = None,
) -> list[TranscriptEntry]:
    """Transcribe one channel; returns entries with raw ``S<n>`` speaker labels.

    ``diarizer=None`` or ``num_speakers=1`` attributes everything to ``S0``.
    ``on_progress`` is called as ``on_progress(stage: str, done: int, total: int)``.
    ``on_embeddings`` receives the diarized clusters' voice embeddings under
    the same labels the returned entries carry — the enrollment material for
    "this speaker is …" corrections after the meeting. Never called when no
    diarizer ran: a solo channel has no cluster embedding.

    Diarization always runs with per-cluster voice embeddings: a known count is
    requested one high and folded back (:func:`fold_excess_clusters`), an
    estimated one may collapse to a single voice (:func:`collapse_single_voice`).
    With ``reid`` given, matched clusters carry persistent speaker-profile names
    instead of ``S<n>`` (cross-meeting re-ID) — several clusters resolving to the
    same profile is the over-split recovery, not an error. Unmatched clusters
    keep their ``S<n>`` label for the caller to template.

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
        on_embeddings=on_embeddings,
    )


def _decode(
    samples: np.ndarray,
    *,
    asr: ASRBackend,
    language: Language | None,
    vad: SileroVAD | None,
    on_progress: ProgressCallback | None,
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
    samples: np.ndarray,
    words: list[Word],
    segments: list[Segment],
    *,
    diarizer: Diarizer,
    num_speakers: int | None,
    reid: SpeakerResolver | None,
    on_progress: ProgressCallback | None,
    on_embeddings: Callable[[dict[str, np.ndarray]], None] | None = None,
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
    requested = num_speakers if num_speakers is None else num_speakers + 1
    result = diarizer.diarize_with_embeddings(samples, requested)
    turns, embeddings = result.turns, result.embeddings
    if num_speakers is not None:
        turns, embeddings = fold_excess_clusters(turns, embeddings, num_speakers)
    else:
        turns, embeddings = collapse_single_voice(turns, embeddings)
    names = reid.resolve(embeddings) if reid is not None else {}
    entries = merge_words_turns(words, turns)
    if names:
        entries = [
            replace(e, speaker=names[e.speaker]) if e.speaker in names else e for e in entries
        ]
    if on_embeddings is not None:
        # Only labels the entries carry: a cluster that attracted no words is
        # invisible in the transcript, so it must not appear in the meeting's
        # voiceprint sidecar either (and would leak an untemplated raw label).
        visible = {e.speaker for e in entries}
        on_embeddings(
            {
                label: vector
                for label, vector in _entry_embeddings(turns, embeddings, names).items()
                if label in visible
            }
        )
    return entries


def _entry_embeddings(
    turns: list[SpeakerTurn],
    embeddings: dict[str, np.ndarray],
    names: dict[str, str],
) -> dict[str, np.ndarray]:
    """Cluster embeddings under the labels the entries carry.

    Several clusters resolving to one profile name (merge-at-naming) merge to
    their duration-weighted mean — the same rule :func:`fold_excess_clusters`
    uses for a merged cluster's embedding."""
    durations: dict[str, float] = {}
    for t in turns:
        durations[t.speaker] = durations.get(t.speaker, 0.0) + (t.end - t.start)
    sums: dict[str, np.ndarray] = {}
    for cluster, vector in embeddings.items():
        label = names.get(cluster, cluster)
        weighted = vector * durations.get(cluster, 0.0)
        sums[label] = sums[label] + weighted if label in sums else weighted
    return {
        label: l2_normalize(total)
        for label, total in sums.items()
        if float(np.linalg.norm(total)) > 0.0
    }


COLLAPSE_SIMILARITY = 0.6
"""An estimated-count channel collapses to one speaker when every pair of its
cluster embeddings is at least this similar. Measured 2026-08-06 on the corpus
harness with overlap-clean embeddings (``eval/README.md``): channels that
really hold one voice but were split 2–3 ways sit at min pairwise cosine
0.73–0.98 (10/10 recovered), true two-speaker channels at ≤0.39 (0/24 falsely
collapsed), larger groups at ≤0.16. 0.6 sits in the empty band, deliberately
nearer its solo edge: the false collapse (two real speakers merged) is the
unrecoverable failure, so it gets the larger margin. Only the
everything-is-one-voice collapse is safe at all: merging individual similar
pairs on a multi-speaker channel measurably destroys attribution
(cross-speaker cluster means reach 0.95)."""


def collapse_single_voice(
    turns: list[SpeakerTurn],
    embeddings: dict[str, np.ndarray],
) -> tuple[list[SpeakerTurn], dict[str, np.ndarray]]:
    """Undo an estimator's split of a single voice into several "speakers".

    Count estimation splits a genuinely single-speaker channel 2–3 ways on
    every corpus channel measured (a monologue chopped across "Speaker 1/2/3",
    −22 pts word attribution — the cost of nobody stating a count). When *all*
    clusters are mutually at least :data:`COLLAPSE_SIMILARITY` similar, they
    are one voice: everything folds to one cluster. Any cluster without an
    embedding, or any pair under the bar, leaves the result untouched.
    """
    labels = {t.speaker for t in turns}
    if len(labels) < 2 or not labels <= embeddings.keys():
        return turns, embeddings
    if any(
        float(embeddings[a] @ embeddings[b]) < COLLAPSE_SIMILARITY
        for a, b in combinations(sorted(labels), 2)
    ):
        return turns, embeddings
    return fold_excess_clusters(turns, embeddings, 1)


def fold_excess_clusters(
    turns: list[SpeakerTurn],
    embeddings: dict[str, np.ndarray],
    num_speakers: int,
) -> tuple[list[SpeakerTurn], dict[str, np.ndarray]]:
    """Fold the smallest cluster into its most-similar partner until
    ``num_speakers`` remain.

    A known-count channel is diarized at one cluster *over* the stated count,
    then folded back here: the spare cluster gives the clustering room to keep
    two genuinely different voices apart that an exact-count run would fuse
    (measured 2026-08-06 on the corpus harness: +2.3 pts mean word attribution,
    one channel +20.8, worst channel −0.3 — ``eval/README.md``), and the fold
    returns exactly the count the user stated, so no phantom speaker appears.
    The spare is identified by *duration* — the smallest cluster — and only its
    partner by similarity, so a wrong partner choice can never cost more than
    the spare's own speech. Folding the globally most-similar pair instead
    measured 7 pts worse with overlap-clean embeddings (2026-08-06): the true
    pairing is rarely the max pair, and what made max-pair look right before
    was overlap-inflated similarity around the tiny spare — luck, not
    mechanism. The merged cluster keeps the longer side's label; its embedding
    is the duration-weighted mean. Clusters without an embedding cannot be
    folded; if too few clusters have one, the remainder is returned unfolded
    (real backends always embed — see ``DiarizationResult.embeddings``).
    """
    turns = list(turns)
    embeddings = dict(embeddings)
    while len({t.speaker for t in turns}) > num_speakers:
        durations: dict[str, float] = {}
        for t in turns:
            durations[t.speaker] = durations.get(t.speaker, 0.0) + (t.end - t.start)
        embedded = [c for c in durations if c in embeddings]
        if len(embedded) < 2:
            return turns, embeddings
        spare = min(embedded, key=lambda c: durations[c])
        partner = max(
            (c for c in embedded if c != spare),
            key=lambda c: float(embeddings[c] @ embeddings[spare]),
        )
        keep, fold = (
            (partner, spare) if durations[partner] >= durations[spare] else (spare, partner)
        )
        turns = [replace(t, speaker=keep) if t.speaker == fold else t for t in turns]
        merged = embeddings[keep] * durations[keep] + embeddings[fold] * durations[fold]
        embeddings[keep] = l2_normalize(merged)
        del embeddings[fold]
    return turns, embeddings


def assemble_transcript(
    entries: list[TranscriptEntry],
    *,
    profile: MeetingProfile,
    language: Language | None,
    parameters_for: Callable[[Language | None], ResolvedParameters],
    glossary_threshold: float | None = None,
    on_language: Callable[[Language], None] | None = None,
) -> Transcript:
    """The shared finalize post-steps: glossary snap → language resolve →
    parameter provenance → :class:`Transcript`.

    Applied by :meth:`MeetingRecorder.finalize` and :func:`finalize_file`
    alike, so a file transcribe and a live meeting produce the same artifact
    shape. The glossary snap runs on the authoritative transcript only
    (checkpoints stay raw). ``language`` is the *configured* language
    (``None`` = detect over the finalized text; a detection fires
    ``on_language``); ``parameters_for`` builds the provenance record from the
    resolved language, since the two callers record speakers differently
    (per-channel counts vs one ``"audio"`` channel).
    """
    threshold = DEFAULT_THRESHOLD if glossary_threshold is None else glossary_threshold
    entries = apply_glossary(
        entries,
        glossary=profile.glossary,
        attendee_names=profile.attendee_names,
        threshold=threshold,
    )
    resolved = language
    if resolved is None:
        resolved = detect_language(" ".join(e.text for e in entries))
        if resolved is not None and on_language is not None:
            on_language(resolved)
    return Transcript(
        language=resolved,
        profile=profile,
        entries=entries,
        parameters=parameters_for(resolved),
    )


def finalize_file(
    samples: np.ndarray,
    *,
    profile: MeetingProfile,
    asr: ASRBackend,
    vad: SileroVAD | None = None,
    diarizer: Diarizer | None = None,
    num_speakers: int | None = None,
    reid: SpeakerResolver | None = None,
    glossary_threshold: float | None = None,
    on_progress: ProgressCallback | None = None,
    on_embeddings: Callable[[dict[str, np.ndarray]], None] | None = None,
) -> Transcript:
    """One mixed audio stream → a finished transcript (``steno transcribe``).

    Runs the same accuracy core a meeting's stop runs (:func:`finalize_channel`)
    followed by the same post-steps (:func:`assemble_transcript`) —
    display relabel, glossary snap, language detection, parameter provenance —
    so a file transcribe and a live meeting produce the same artifact shape.
    One un-split stream has no local/remote model, so speakers get the neutral
    ``Speaker <n>`` template and provenance is recorded under a single
    ``"audio"`` channel. ``profile.language`` is the *given*
    language (``None`` = detect); the returned transcript carries the resolved
    one."""
    captured: dict[str, np.ndarray] = {}
    raw = finalize_channel(
        samples,
        asr=asr,
        language=profile.language,
        vad=vad,
        diarizer=diarizer,
        num_speakers=num_speakers,
        reid=reid,
        on_progress=on_progress,
        on_embeddings=captured.update if on_embeddings is not None else None,
    )
    mapping = raw_label_map(raw, "Speaker {n}")
    entries = [replace(e, speaker=mapping.get(e.speaker, e.speaker)) for e in raw]
    if on_embeddings is not None and captured:
        on_embeddings({mapping.get(label, label): v for label, v in captured.items()})
    detected = len({e.speaker for e in entries})
    return assemble_transcript(
        entries,
        profile=profile,
        language=profile.language,
        glossary_threshold=glossary_threshold,
        parameters_for=lambda resolved: ResolvedParameters(
            language=resolve_value(profile.language, resolved),
            speakers={"audio": resolve_value(num_speakers, detected)},
        ),
    )


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
    asr_ep: str | None = None,
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
        asr_ep=asr_ep,
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
    mapping = raw_label_map(entries, template)
    return [replace(e, speaker=mapping.get(e.speaker, e.speaker)) for e in entries]


def raw_label_map(entries: list[TranscriptEntry], template: str) -> dict[str, str]:
    """The raw-``S<n>``-to-display-name mapping :func:`relabel_speakers` applies,
    exposed so a channel's cluster embeddings can be relabeled in lockstep with
    its entries (the meeting voiceprints must carry the labels the user reads)."""
    mapping: dict[str, str] = {}
    for entry in entries:
        if _RAW_CLUSTER.fullmatch(entry.speaker) and entry.speaker not in mapping:
            mapping[entry.speaker] = template.format(n=len(mapping) + 1)
    return mapping
