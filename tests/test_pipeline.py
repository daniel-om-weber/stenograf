import numpy as np

from stenograf.asr.base import ASRBackend, Segment, Word
from stenograf.audio import SAMPLE_RATE
from stenograf.diarization.base import Diarizer, SpeakerTurn
from stenograf.pipeline import (
    finalize_channel,
    group_words,
    merge_words_turns,
    relabel_speakers,
)


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


def test_relabel_speakers_by_first_appearance():
    words = [word("b", 0.0, 0.1), word("a", 1.0, 1.1), word("b2", 2.0, 2.1)]
    turns = [turn("S7", 0.0, 0.5), turn("S2", 0.9, 1.5), turn("S7", 1.9, 2.5)]
    entries = relabel_speakers(merge_words_turns(words, turns))
    assert [e.speaker for e in entries] == ["Speaker 1", "Speaker 2", "Speaker 1"]


class FakeASR(ASRBackend):
    name = "fake"

    def __init__(self):
        self.calls: list[int] = []

    def load(self) -> None:
        pass

    def transcribe(self, samples: np.ndarray, language) -> list[Segment]:
        self.calls.append(len(samples))
        # One word per call, timestamped relative to the window start.
        return [
            Segment(
                text="wort",
                start=0.1,
                end=0.5,
                words=(Word(text="wort", start=0.1, end=0.5),),
            )
        ]

    def unload(self) -> None:
        pass


class WordlessASR(ASRBackend):
    """Emits segment text but no word timestamps (e.g. a Whisper/Voxtral path)."""

    name = "wordless"

    def load(self) -> None:
        pass

    def transcribe(self, samples: np.ndarray, language) -> list[Segment]:
        return [Segment(text="ganzer satz", start=0.1, end=1.0, words=())]

    def unload(self) -> None:
        pass


class SilentASR(ASRBackend):
    """Finds no speech — returns no segments for any window."""

    name = "silent"

    def load(self) -> None:
        pass

    def transcribe(self, samples: np.ndarray, language) -> list[Segment]:
        return []

    def unload(self) -> None:
        pass


class FakeDiarizer(Diarizer):
    def __init__(self, turns):
        self.turns = turns
        self.seen_num_speakers = None

    def diarize(self, samples, num_speakers=None):
        self.seen_num_speakers = num_speakers
        return self.turns


class RaisingDiarizer(Diarizer):
    """Fails on every call — stands in for a backend that throws on odd input."""

    def __init__(self):
        self.called = False

    def diarize(self, samples, num_speakers=None):
        self.called = True
        raise RuntimeError("diarizer exploded")


class TestFinalizeChannel:
    def test_without_vad_or_diarizer_single_window_single_speaker(self):
        asr = FakeASR()
        samples = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
        entries = finalize_channel(samples, asr=asr, language=None)
        assert len(asr.calls) == 1
        assert entries[0].speaker == "S0"
        assert entries[0].text == "wort"

    def test_diarizer_receives_speaker_count_and_labels_entries(self):
        asr = FakeASR()
        diarizer = FakeDiarizer([turn("S1", 0.0, 2.0)])
        samples = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
        entries = finalize_channel(
            samples, asr=asr, language=None, diarizer=diarizer, num_speakers=3
        )
        assert diarizer.seen_num_speakers == 3
        assert entries[0].speaker == "S1"

    def test_num_speakers_one_skips_diarization(self):
        asr = FakeASR()
        diarizer = FakeDiarizer([turn("S1", 0.0, 2.0)])
        samples = np.zeros(SAMPLE_RATE, dtype=np.float32)
        entries = finalize_channel(
            samples, asr=asr, language=None, diarizer=diarizer, num_speakers=1
        )
        assert diarizer.seen_num_speakers is None  # never called
        assert entries[0].speaker == "S0"

    def test_empty_audio_yields_no_entries(self):
        entries = finalize_channel(
            np.zeros(0, dtype=np.float32), asr=FakeASR(), language=None
        )
        assert entries == []

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
        assert not diarizer.called

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
