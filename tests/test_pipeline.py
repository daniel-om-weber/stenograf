import numpy as np
import pytest
from conftest import EmbeddingDiarizer, FakeASR, FakeDiarizer, GermanASR, RaisingDiarizer

from stenograf.asr.base import Segment, Word
from stenograf.audio import SAMPLE_RATE
from stenograf.config import Language, MeetingProfile, Provenance
from stenograf.diarization.base import SpeakerTurn
from stenograf.pipeline import (
    collapse_single_voice,
    finalize_channel,
    finalize_file,
    fold_excess_clusters,
    group_words,
    merge_words_turns,
    relabel_speakers,
)
from stenograf.transcript import TranscriptEntry
from stenograf.vad import SpeechSegment


def word(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=start, end=end)


def turn(speaker: str, start: float, end: float) -> SpeakerTurn:
    return SpeakerTurn(speaker=speaker, start=start, end=end)


class TestMergeWordsTurns:
    def test_groups_consecutive_words_of_one_speaker(self):
        words = [word("guten", 0.0, 0.4), word("morgen", 0.5, 0.9)]
        entries = merge_words_turns(words, [turn("S0", 0.0, 1.0)])
        assert len(entries) == 1
        assert entries[0].text == "guten morgen"
        assert entries[0].speaker == "S0"
        assert not entries[0].provisional

    def test_splits_on_speaker_change(self):
        words = [word("hallo", 0.0, 0.4), word("hi", 2.0, 2.2)]
        turns = [turn("S0", 0.0, 1.0), turn("S1", 1.9, 2.5)]
        entries = merge_words_turns(words, turns)
        assert [e.speaker for e in entries] == ["S0", "S1"]

    def test_splits_on_long_gap_within_speaker(self):
        words = [word("eins", 0.0, 0.4), word("zwei", 5.0, 5.4)]
        entries = merge_words_turns(words, [turn("S0", 0.0, 6.0)])
        assert len(entries) == 2
        assert all(e.speaker == "S0" for e in entries)

    def test_overlap_flags_provisional(self):
        words = [word("beide", 1.0, 1.4)]
        turns = [turn("S0", 0.0, 2.0), turn("S1", 0.5, 3.0)]
        entries = merge_words_turns(words, turns)
        assert entries[0].provisional
        # Largest overlap with the word span wins; both cover it fully, so
        # the tie resolves deterministically to the first maximal turn.
        assert entries[0].speaker in {"S0", "S1"}

    def test_word_outside_turns_takes_nearest(self):
        words = [word("nachzügler", 4.0, 4.5)]
        turns = [turn("S0", 0.0, 1.0), turn("S1", 5.0, 6.0)]
        entries = merge_words_turns(words, turns)
        assert entries[0].speaker == "S1"
        assert not entries[0].provisional

    def test_no_turns_falls_back_to_single_speaker(self):
        entries = merge_words_turns([word("solo", 0.0, 0.5)], [])
        assert entries[0].speaker == "S0"

    def test_retains_word_timestamps_per_entry(self):
        words = [word("guten", 0.0, 0.4), word("morgen", 0.5, 0.9), word("hi", 5.0, 5.4)]
        entries = merge_words_turns(words, [turn("S0", 0.0, 6.0)])
        # Split on the long gap; each entry keeps exactly its own words.
        assert [w.text for w in entries[0].words] == ["guten", "morgen"]
        assert [(w.start, w.end) for w in entries[1].words] == [(5.0, 5.4)]


class TestGroupWords:
    def test_groups_one_speaker_and_splits_on_a_long_gap(self):
        words = [
            word("guten", 0.0, 0.4),
            word("morgen", 0.5, 0.9),  # small gap → same entry
            word("hallo", 3.0, 3.4),  # gap > max_gap → new entry
        ]
        entries = group_words(words, "Local", max_gap=1.5)
        assert [(e.speaker, e.text) for e in entries] == [
            ("Local", "guten morgen"),
            ("Local", "hallo"),
        ]
        assert (entries[0].start, entries[0].end) == (0.0, 0.9)
        assert (entries[1].start, entries[1].end) == (3.0, 3.4)

    def test_empty_words_yield_no_entries(self):
        assert group_words([], "Remote") == []

    def test_retains_word_timestamps(self):
        words = [word("guten", 0.0, 0.4), word("morgen", 0.5, 0.9)]
        entries = group_words(words, "Local")
        assert tuple((w.text, w.start, w.end) for w in entries[0].words) == (
            ("guten", 0.0, 0.4),
            ("morgen", 0.5, 0.9),
        )


def test_relabel_speakers_by_first_appearance():
    words = [word("b", 0.0, 0.1), word("a", 1.0, 1.1), word("b2", 2.0, 2.1)]
    turns = [turn("S7", 0.0, 0.5), turn("S2", 0.9, 1.5), turn("S7", 1.9, 2.5)]
    entries = relabel_speakers(merge_words_turns(words, turns))
    assert [e.speaker for e in entries] == ["Speaker 1", "Speaker 2", "Speaker 1"]


def test_relabel_speakers_preserves_reid_profile_names():
    # A re-ID'd cluster carries a profile name, not an S<n> label: it must pass
    # through unchanged, while the remaining raw clusters are still templated —
    # and the template numbering counts only the raw clusters.
    entries = [
        TranscriptEntry("Daniel", "a", 0.0, 0.5),
        TranscriptEntry("S3", "b", 1.0, 1.5),
        TranscriptEntry("Daniel", "c", 2.0, 2.5),
    ]
    relabeled = relabel_speakers(entries, "Local-{n}")
    assert [e.speaker for e in relabeled] == ["Daniel", "Local-1", "Daniel"]


def test_relabel_speakers_preserves_words():
    words = [word("b", 0.0, 0.1), word("a", 1.0, 1.1)]
    turns = [turn("S7", 0.0, 0.5), turn("S2", 0.9, 1.5)]
    entries = relabel_speakers(merge_words_turns(words, turns))
    assert [w.text for e in entries for w in e.words] == ["b", "a"]


class WordlessASR(FakeASR):
    """Emits segment text but no word timestamps (e.g. a Whisper/Voxtral path)."""

    name = "wordless"

    def transcribe(self, samples: np.ndarray, language) -> list[Segment]:
        self.calls.append(len(samples))
        return [Segment(text="ganzer satz", start=0.1, end=1.0, words=())]


class SilentASR(FakeASR):
    """Finds no speech — returns no segments for any window."""

    name = "silent"

    def transcribe(self, samples: np.ndarray, language) -> list[Segment]:
        self.calls.append(len(samples))
        return []


class WindowScriptASR(FakeASR):
    """One queued word list per decoded window, at times relative to the slice.

    The slice starts at ``vad.context_start``, not at the window — that offset
    is the whole point of the tests using this.
    """

    name = "window-script"

    def __init__(self, responses: list[list[Word]]) -> None:
        super().__init__()
        self._responses = responses

    def transcribe(self, samples: np.ndarray, language) -> list[Segment]:
        self.calls.append(len(samples))
        words = self._responses[min(len(self.calls) - 1, len(self._responses) - 1)]
        if not words:
            return []
        return [
            Segment(
                text=" ".join(x.text for x in words),
                start=words[0].start,
                end=words[-1].end,
                words=tuple(words),
            )
        ]


class ScriptedVAD:
    """Fixed speech runs, independent of the audio (no model, no inspection)."""

    def __init__(self, segments: list[SpeechSegment]) -> None:
        self._segments = segments

    def speech_segments(self, samples: np.ndarray) -> list[SpeechSegment]:
        return list(self._segments)


class TestWindowClaim:
    """Which decoded words a window keeps (vad.claim_start, via _clip_context)."""

    def test_a_late_onset_word_is_claimed_from_the_preroll(self):
        # Speech reported at 20.0–23.0 → padded window [19.85, 23.15], short, so
        # the slice starts 15 s earlier at 4.85 and slice-relative 15.0 is the
        # window's start. The speaker actually began ~0.2 s before the VAD
        # noticed: that word is in the slice, and nobody used to emit it.
        asr = WindowScriptASR(
            [
                [
                    word("kontext", 14.0, 14.3),  # 18.85–19.15: not ours
                    word("eine", 14.6, 14.9),  # 19.45–19.75: the clipped onset
                    word("frage", 15.5, 16.0),  # inside the window
                ]
            ]
        )
        entries = finalize_channel(
            np.zeros(SAMPLE_RATE * 60, dtype=np.float32),
            asr=asr,
            language=None,
            vad=ScriptedVAD([SpeechSegment(20.0, 23.0)]),
        )
        assert [w.text for e in entries for w in e.words] == ["eine", "frage"]
        assert abs(entries[0].start - 19.45) < 1e-6  # the recovered onset

    def test_touching_windows_never_claim_the_same_word_twice(self):
        # Two runs 0.2 s apart that the 30 s budget splits: their pads collide,
        # so window 2 = [29.15, 35.15] begins exactly where window 1's audio
        # ended. Window 2 is short, so its slice reaches back over the seam and
        # its decode sees the same word again — and only window 1 may keep it.
        asr = WindowScriptASR(
            [
                [word("naht", 28.9, 29.1)],  # window 1: [0, 29.15], no offset
                [word("naht", 14.75, 14.95)],  # window 2: slice starts at 14.15
            ]
        )
        entries = finalize_channel(
            np.zeros(SAMPLE_RATE * 60, dtype=np.float32),
            asr=asr,
            language=None,
            vad=ScriptedVAD([SpeechSegment(0.0, 29.0), SpeechSegment(29.2, 35.0)]),
        )
        assert len(asr.calls) == 2
        assert [w.text for e in entries for w in e.words] == ["naht"]


def test_diarizer_default_embeddings_are_empty():
    # The ABC default runs diarize() and returns no embeddings, so backends that
    # cannot embed (and every FakeDiarizer) keep working — re-ID treats a missing
    # embedding as "no match" rather than crashing.
    diarizer = FakeDiarizer([turn("S0", 0.0, 1.0)])
    result = diarizer.diarize_with_embeddings(np.zeros(SAMPLE_RATE, np.float32), num_speakers=1)
    assert [t.speaker for t in result.turns] == ["S0"]
    assert result.embeddings == {}
    assert diarizer.seen_num_speakers == 1  # the default forwards the count


class TestFinalizeChannel:
    def test_without_vad_or_diarizer_single_window_single_speaker(self):
        asr = FakeASR()
        samples = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
        entries = finalize_channel(samples, asr=asr, language=None)
        assert len(asr.calls) == 1
        assert entries[0].speaker == "S0"
        assert entries[0].text == "wort"

    def test_diarizer_receives_one_over_the_stated_count(self):
        # A known count is requested at k+1; fold_excess_clusters returns the
        # stated k afterwards, so the extra cluster never reaches the user.
        asr = FakeASR()
        diarizer = FakeDiarizer([turn("S1", 0.0, 2.0)])
        samples = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
        entries = finalize_channel(
            samples, asr=asr, language=None, diarizer=diarizer, num_speakers=3
        )
        assert diarizer.seen_num_speakers == 4
        assert entries[0].speaker == "S1"

    def test_num_speakers_one_skips_diarization(self):
        asr = FakeASR()
        diarizer = FakeDiarizer([turn("S1", 0.0, 2.0)])
        samples = np.zeros(SAMPLE_RATE, dtype=np.float32)
        entries = finalize_channel(
            samples, asr=asr, language=None, diarizer=diarizer, num_speakers=1
        )
        assert diarizer.seen_num_speakers == "unset"  # never called
        assert entries[0].speaker == "S0"

    def test_empty_audio_yields_no_entries(self):
        entries = finalize_channel(np.zeros(0, dtype=np.float32), asr=FakeASR(), language=None)
        assert entries == []


class TestFinalizeChannelReuse:
    """precomputed_words: the live window pass already decoded this channel."""

    def test_diarized_reuse_skips_asr(self):
        asr = FakeASR()
        diarizer = FakeDiarizer([turn("S1", 0.0, 2.0)])
        words = (Word("hallo", 0.2, 0.6), Word("welt", 0.8, 1.2))
        entries = finalize_channel(
            np.zeros(SAMPLE_RATE * 2, dtype=np.float32),
            asr=asr,
            language=None,
            diarizer=diarizer,
            num_speakers=2,
            precomputed_words=words,
        )
        assert asr.calls == []  # no re-decode — the whole point of reuse
        assert diarizer.seen_num_speakers == 3  # diarization still runs (at k+1)
        assert [e.text for e in entries] == ["hallo welt"]
        assert entries[0].speaker == "S1"

    def test_single_speaker_reuse_groups_words(self):
        asr = FakeASR()
        words = (Word("a", 0.1, 0.4), Word("b", 5.0, 5.4))  # gap > MAX_ENTRY_GAP
        entries = finalize_channel(
            np.zeros(SAMPLE_RATE * 6, dtype=np.float32),
            asr=asr,
            language=None,
            num_speakers=1,
            precomputed_words=words,
        )
        assert asr.calls == []
        assert [(e.speaker, e.text) for e in entries] == [("S0", "a"), ("S0", "b")]

    def test_empty_reuse_means_silent_channel(self):
        # The live pass saw no speech: no ASR, no diarization, no entries — and
        # crucially no fallback re-decode (empty tuple ≠ missing).
        asr = FakeASR()
        diarizer = FakeDiarizer([turn("S1", 0.0, 2.0)])
        entries = finalize_channel(
            np.zeros(SAMPLE_RATE * 2, dtype=np.float32),
            asr=asr,
            language=None,
            diarizer=diarizer,
            num_speakers=2,
            precomputed_words=(),
        )
        assert entries == []
        assert asr.calls == [] and diarizer.seen_num_speakers == "unset"

    def test_silent_channel_skips_diarization(self):
        # No speech → no words. Diarizing an empty channel is wasted work and can
        # throw (sherpa forced to num_clusters > 1 on near-silent input), which
        # would otherwise sink the whole meeting's finalize. Skip it instead.
        asr = SilentASR()
        diarizer = RaisingDiarizer()
        samples = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
        entries = finalize_channel(
            samples, asr=asr, language=None, diarizer=diarizer, num_speakers=2
        )
        assert entries == []
        assert diarizer.calls == 0

    def test_single_speaker_entries_retain_words(self):
        asr = FakeASR()
        samples = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
        entries = finalize_channel(samples, asr=asr, language=None)
        assert [w.text for w in entries[0].words] == ["wort"]

    def test_diarized_backend_without_word_timestamps_keeps_text(self):
        # A backend with no word timestamps must not silently drop the diarized
        # transcript; segments fall back to whole-unit attribution.
        asr = WordlessASR()
        diarizer = FakeDiarizer([turn("S1", 0.0, 2.0)])
        samples = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
        entries = finalize_channel(
            samples, asr=asr, language=None, diarizer=diarizer, num_speakers=2
        )
        assert len(entries) == 1
        assert entries[0].text == "ganzer satz"
        assert entries[0].speaker == "S1"


class TestFoldExcessClusters:
    def _unit(self, *components: float) -> np.ndarray:
        v = np.asarray(components, dtype=np.float32)
        return v / np.linalg.norm(v)

    def test_folds_smallest_cluster_into_its_most_similar_partner(self):
        # S1 is the spare (least speech); of S0/S2 it is closest to S0, and the
        # merge keeps the longer side's label.
        turns = [turn("S0", 0.0, 4.0), turn("S1", 4.0, 5.0), turn("S2", 5.0, 8.0)]
        embeddings = {
            "S0": self._unit(1.0, 0.0),
            "S1": self._unit(0.9, 0.44),
            "S2": self._unit(0.0, 1.0),
        }
        folded_turns, folded_emb = fold_excess_clusters(turns, embeddings, 2)
        assert [t.speaker for t in folded_turns] == ["S0", "S0", "S2"]
        assert set(folded_emb) == {"S0", "S2"}
        assert np.linalg.norm(folded_emb["S0"]) == pytest.approx(1.0)

    def test_spare_is_chosen_by_duration_not_similarity(self):
        # The two long clusters are the most similar pair by far, but only the
        # smallest cluster may fold — a wrong partner can then never cost more
        # than the spare's own speech.
        turns = [turn("S0", 0.0, 4.0), turn("S1", 4.0, 8.0), turn("S2", 8.0, 9.0)]
        embeddings = {
            "S0": self._unit(1.0, 0.0, 0.0),
            "S1": self._unit(0.98, 0.2, 0.0),  # near-duplicate of S0
            "S2": self._unit(0.0, 0.0, 1.0),  # the spare, similar to nothing
        }
        folded_turns, _ = fold_excess_clusters(turns, embeddings, 2)
        assert len({t.speaker for t in folded_turns}) == 2
        assert {"S0", "S1"} <= {t.speaker for t in folded_turns}  # long pair untouched

    def test_at_or_below_count_is_untouched(self):
        turns = [turn("S0", 0.0, 1.0), turn("S1", 1.0, 2.0)]
        embeddings = {"S0": self._unit(1.0, 0.0), "S1": self._unit(0.0, 1.0)}
        folded_turns, folded_emb = fold_excess_clusters(turns, embeddings, 2)
        assert folded_turns == turns
        assert folded_emb == embeddings

    def test_folds_repeatedly_down_to_count(self):
        turns = [turn(f"S{i}", float(i), float(i + 1)) for i in range(4)]
        embeddings = {f"S{i}": self._unit(1.0, 0.01 * i) for i in range(4)}
        folded_turns, _ = fold_excess_clusters(turns, embeddings, 1)
        assert len({t.speaker for t in folded_turns}) == 1

    def test_unembedded_clusters_cannot_fold(self):
        # One embedded cluster among three: no pair to compare, so the excess
        # stays — better an extra label than an arbitrary merge.
        turns = [turn("S0", 0.0, 1.0), turn("S1", 1.0, 2.0), turn("S2", 2.0, 3.0)]
        folded_turns, _ = fold_excess_clusters(turns, {"S0": self._unit(1.0, 0.0)}, 2)
        assert [t.speaker for t in folded_turns] == ["S0", "S1", "S2"]


class TestCollapseSingleVoice:
    def _unit(self, *components: float) -> np.ndarray:
        v = np.asarray(components, dtype=np.float32)
        return v / np.linalg.norm(v)

    def test_mutually_similar_clusters_collapse_to_one(self):
        turns = [turn("S0", 0.0, 3.0), turn("S1", 3.0, 4.0), turn("S0", 4.0, 6.0)]
        embeddings = {"S0": self._unit(1.0, 0.1), "S1": self._unit(1.0, 0.3)}  # cosine ~0.98
        collapsed_turns, collapsed_emb = collapse_single_voice(turns, embeddings)
        assert len({t.speaker for t in collapsed_turns}) == 1
        assert len(collapsed_emb) == 1

    def test_one_distant_pair_blocks_the_collapse(self):
        turns = [turn("S0", 0.0, 1.0), turn("S1", 1.0, 2.0), turn("S2", 2.0, 3.0)]
        embeddings = {
            "S0": self._unit(1.0, 0.1),
            "S1": self._unit(1.0, 0.3),
            "S2": self._unit(0.0, 1.0),  # a genuinely different voice
        }
        collapsed_turns, collapsed_emb = collapse_single_voice(turns, embeddings)
        assert collapsed_turns == turns
        assert collapsed_emb == embeddings

    def test_unembedded_cluster_blocks_the_collapse(self):
        turns = [turn("S0", 0.0, 1.0), turn("S1", 1.0, 2.0)]
        collapsed_turns, _ = collapse_single_voice(turns, {"S0": self._unit(1.0, 0.0)})
        assert collapsed_turns == turns

    def test_estimated_count_channel_collapses_in_finalize(self):
        # The estimator split one voice into two clusters; num_speakers=None
        # (nobody stated a count) triggers the collapse and one speaker remains.
        diarizer = EmbeddingDiarizer(
            [turn("S0", 0.0, 1.0), turn("S1", 1.0, 2.0)],
            {"S0": self._unit(1.0, 0.1), "S1": self._unit(1.0, 0.3)},
        )
        entries = finalize_channel(
            np.zeros(SAMPLE_RATE * 2, dtype=np.float32),
            asr=FakeASR(),
            language=None,
            diarizer=diarizer,
            num_speakers=None,
        )
        assert len({e.speaker for e in entries}) == 1


class MappingReID:
    """A fake SpeakerResolver: relabels clusters by a fixed lookup."""

    def __init__(self, names):
        self.names = names
        self.seen = None

    def resolve(self, embeddings):
        self.seen = dict(embeddings)
        return {c: n for c, n in self.names.items() if c in embeddings}


class TestFinalizeChannelReID:
    def _samples(self):
        return np.zeros(SAMPLE_RATE * 2, dtype=np.float32)

    def test_matched_cluster_takes_profile_name(self):
        diarizer = EmbeddingDiarizer(
            [turn("S0", 0.0, 2.0)], {"S0": np.array([1.0, 0.0], np.float32)}
        )
        reid = MappingReID({"S0": "Daniel"})
        entries = finalize_channel(
            self._samples(),
            asr=FakeASR(),
            language=None,
            diarizer=diarizer,
            num_speakers=2,
            reid=reid,
        )
        assert entries[0].speaker == "Daniel"
        # Took the embeddings path, not the plain diarize path.
        assert diarizer.embed_calls == 1
        assert diarizer.diarize_calls == 0
        # The resolver was handed the cluster embeddings.
        assert set(reid.seen) == {"S0"}

    def test_unmatched_cluster_keeps_raw_label(self):
        diarizer = EmbeddingDiarizer(
            [turn("S0", 0.0, 2.0)], {"S0": np.array([1.0, 0.0], np.float32)}
        )
        reid = MappingReID({})  # matches nothing
        entries = finalize_channel(
            self._samples(),
            asr=FakeASR(),
            language=None,
            diarizer=diarizer,
            num_speakers=2,
            reid=reid,
        )
        assert entries[0].speaker == "S0"

    def test_no_reid_still_takes_embeddings_path(self):
        # Even without re-ID the known-count path needs per-cluster embeddings:
        # they are what folds the k+1 request back to the stated count.
        diarizer = EmbeddingDiarizer([turn("S0", 0.0, 2.0)], {})
        finalize_channel(
            self._samples(),
            asr=FakeASR(),
            language=None,
            diarizer=diarizer,
            num_speakers=2,
        )
        assert diarizer.embed_calls == 1
        assert diarizer.diarize_calls == 0

    def test_reid_ignored_without_diarization(self):
        # num_speakers=1 → no diarization → re-ID never runs (nothing to resolve).
        diarizer = EmbeddingDiarizer([turn("S0", 0.0, 2.0)], {})
        reid = MappingReID({"S0": "Daniel"})
        entries = finalize_channel(
            self._samples(),
            asr=FakeASR(),
            language=None,
            diarizer=diarizer,
            num_speakers=1,
            reid=reid,
        )
        assert entries[0].speaker == "S0"
        assert diarizer.embed_calls == 0
        assert reid.seen is None


class TestFinalizeFile:
    """finalize_file assembles the same artifact shape a meeting's stop does."""

    def test_detects_language_and_records_audio_channel_provenance(self):
        profile = MeetingProfile(title="Planung")
        transcript = finalize_file(
            np.zeros(SAMPLE_RATE, dtype=np.float32), profile=profile, asr=GermanASR()
        )
        assert transcript.language is Language.GERMAN
        assert transcript.profile is profile  # the given (language=None) profile survives
        assert [e.speaker for e in transcript.entries] == ["Speaker 1"]
        assert transcript.parameters.language.provenance is Provenance.DETECTED
        speakers = transcript.parameters.speakers["audio"]
        assert speakers.value == 1 and speakers.provenance is Provenance.DETECTED

    def test_explicit_language_and_speaker_count_are_marked_explicit(self):
        profile = MeetingProfile(language=Language.ENGLISH)
        transcript = finalize_file(
            np.zeros(SAMPLE_RATE, dtype=np.float32),
            profile=profile,
            asr=FakeASR(),
            num_speakers=1,
        )
        assert transcript.language is Language.ENGLISH
        assert transcript.parameters.language.provenance is Provenance.EXPLICIT
        assert transcript.parameters.speakers["audio"].provenance is Provenance.EXPLICIT


class TestOnEmbeddings:
    """finalize_channel's embeddings out-channel: the enrollment material for
    `steno profiles assign` must carry the labels the entries carry."""

    def _samples(self):
        return np.zeros(SAMPLE_RATE * 2, dtype=np.float32)

    def test_labels_match_entries_named_and_raw(self):
        diarizer = EmbeddingDiarizer(
            [turn("S0", 0.0, 1.0), turn("S1", 1.2, 2.0)],
            {
                "S0": np.array([1.0, 0.0], np.float32),
                "S1": np.array([0.0, 1.0], np.float32),
            },
        )
        captured: dict[str, np.ndarray] = {}
        entries = finalize_channel(
            self._samples(),
            asr=FakeASR(),
            language=None,
            diarizer=diarizer,
            num_speakers=2,
            reid=MappingReID({"S0": "Daniel"}),
            precomputed_words=(word("hallo", 0.2, 0.6), word("welt", 1.4, 1.8)),
            on_embeddings=captured.update,
        )
        assert set(captured) == {e.speaker for e in entries} == {"Daniel", "S1"}
        np.testing.assert_allclose(captured["Daniel"], [1.0, 0.0])
        np.testing.assert_allclose(captured["S1"], [0.0, 1.0])

    def test_a_cluster_without_words_stays_out_of_the_sidecar(self):
        # FakeASR decodes one word (midpoint in S0): cluster S1 is invisible in
        # the transcript, so its embedding must not surface as an assignable label.
        diarizer = EmbeddingDiarizer(
            [turn("S0", 0.0, 1.0), turn("S1", 1.2, 2.0)],
            {
                "S0": np.array([1.0, 0.0], np.float32),
                "S1": np.array([0.0, 1.0], np.float32),
            },
        )
        captured: dict[str, np.ndarray] = {}
        entries = finalize_channel(
            self._samples(),
            asr=FakeASR(),
            language=None,
            diarizer=diarizer,
            num_speakers=2,
            on_embeddings=captured.update,
        )
        assert {e.speaker for e in entries} == {"S0"}
        assert set(captured) == {"S0"}

    def test_merge_at_naming_merges_by_duration(self):
        # Two clusters resolving to one profile merge like fold_excess_clusters
        # merges: duration-weighted mean, re-normalized. S0 speaks 3 s, S1 1 s.
        diarizer = EmbeddingDiarizer(
            [turn("S0", 0.0, 3.0), turn("S1", 4.0, 5.0)],
            {
                "S0": np.array([1.0, 0.0], np.float32),
                "S1": np.array([0.0, 1.0], np.float32),
            },
        )
        captured: dict[str, np.ndarray] = {}
        finalize_channel(
            np.zeros(SAMPLE_RATE * 5, dtype=np.float32),
            asr=FakeASR(),
            language=None,
            diarizer=diarizer,
            num_speakers=2,
            reid=MappingReID({"S0": "Daniel", "S1": "Daniel"}),
            precomputed_words=(word("hallo", 0.2, 0.6), word("welt", 4.4, 4.8)),
            on_embeddings=captured.update,
        )
        assert set(captured) == {"Daniel"}
        expected = np.array([3.0, 1.0]) / np.hypot(3.0, 1.0)
        np.testing.assert_allclose(captured["Daniel"], expected, rtol=1e-6)

    def test_never_called_without_diarization(self):
        called = []
        entries = finalize_channel(
            self._samples(),
            asr=FakeASR(),
            language=None,
            diarizer=EmbeddingDiarizer([turn("S0", 0.0, 2.0)], {}),
            num_speakers=1,
            on_embeddings=lambda e: called.append(e),
        )
        assert entries and not called

    def test_finalize_file_relabels_embedding_keys(self):
        # The sidecar must show "Speaker N", not raw S<n> — the labels the
        # user reads in a file transcription.
        diarizer = EmbeddingDiarizer(
            [turn("S0", 0.0, 1.0), turn("S1", 1.2, 2.0)],
            {
                "S0": np.array([1.0, 0.0], np.float32),
                "S1": np.array([0.0, 1.0], np.float32),
            },
        )
        captured: dict[str, np.ndarray] = {}
        transcript = finalize_file(
            np.zeros(SAMPLE_RATE * 2, dtype=np.float32),
            profile=MeetingProfile(),
            asr=FakeASR(),
            diarizer=diarizer,
            num_speakers=2,
            on_embeddings=captured.update,
        )
        assert set(captured) == {e.speaker for e in transcript.entries} <= {
            "Speaker 1",
            "Speaker 2",
        }
