import numpy as np

from stenograf.asr.base import ASRBackend, Segment, Word
from stenograf.audio import SAMPLE_RATE
from stenograf.diarization.base import Diarizer, SpeakerTurn
from stenograf.pipeline import finalize_channel, merge_words_turns, relabel_speakers


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


class FakeDiarizer(Diarizer):
    def __init__(self, turns):
        self.turns = turns
        self.seen_num_speakers = None

    def diarize(self, samples, num_speakers=None):
        self.seen_num_speakers = num_speakers
        return self.turns


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
