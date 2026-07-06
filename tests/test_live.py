import numpy as np

from stenograf.asr.base import ASRBackend, Segment, Word
from stenograf.audio import SAMPLE_RATE
from stenograf.live import LiveDecoder
from stenograf.vad import SpeechSegment


def w(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=start, end=end)


def pcm(duration: float) -> np.ndarray:
    """A block of silence; the scripted ASR ignores samples, only its length matters."""
    return np.zeros(int(duration * SAMPLE_RATE), dtype=np.float32)


class ScriptedASR(ASRBackend):
    """Returns a queued word list per decode; the last entry repeats when exhausted.

    Word times are relative to the window start. Tests keep the total fed audio
    short (< left_context) so the window never trims — its start stays at t=0 and
    scripted relative times equal absolute session times.
    """

    name = "scripted"

    def __init__(self, responses: list[list[Word]]):
        self._responses = responses
        self.calls = 0

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def transcribe(self, samples, language):
        words = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        if not words:
            return []
        return [Segment(text=" ".join(x.text for x in words), start=words[0].start,
                        end=words[-1].end, words=tuple(words))]


class FakeVAD:
    """Returns a queued list of speech segments per call (last entry repeats)."""

    def __init__(self, per_call: list[list[SpeechSegment]]):
        self._per_call = per_call
        self.calls = 0

    def speech_segments(self, samples):
        segs = self._per_call[min(self.calls, len(self._per_call) - 1)]
        self.calls += 1
        return segs


class TestLocalAgreement:
    def test_commits_prefix_two_decodes_agree(self):
        dec = LiveDecoder(
            ScriptedASR([
                [w("hallo", 0.1, 0.5), w("welt", 0.6, 0.9)],
                [w("hallo", 0.1, 0.5), w("welt", 0.6, 0.9), w("wie", 1.1, 1.4)],
            ]),
            grey_zone=0.0,
        )
        first = dec.feed(pcm(1.0), 0.0)
        assert first.committed == ()  # nothing to agree with yet
        assert dec.interim == "hallo welt"

        second = dec.feed(pcm(1.0), 1.0)
        assert [x.text for x in second.committed] == ["hallo", "welt"]
        assert dec.committed_text == "hallo welt"
        assert dec.interim == "wie"  # last word not yet confirmed by a second decode

    def test_grey_zone_holds_back_fresh_words(self):
        dec = LiveDecoder(
            ScriptedASR([
                [w("a", 0.1, 0.4), w("b", 1.2, 1.6)],
                [w("a", 0.1, 0.4), w("b", 1.2, 1.6), w("c", 1.7, 1.9)],
            ]),
            grey_zone=1.0,
        )
        dec.feed(pcm(1.5), 0.0)
        dec.feed(pcm(0.5), 1.5)  # audio_end 2.0, horizon 1.0
        # a and b both agree, but b ends at 1.6 — inside the grey zone — so held.
        assert dec.committed_text == "a"
        assert dec.interim == "b c"

    def test_disagreement_stops_commit(self):
        dec = LiveDecoder(
            ScriptedASR([
                [w("a", 0.1, 0.4), w("b", 0.5, 0.8)],
                [w("a", 0.1, 0.4), w("x", 0.5, 0.8)],
            ]),
            grey_zone=0.0,
        )
        dec.feed(pcm(1.0), 0.0)
        dec.feed(pcm(1.0), 1.0)
        assert dec.committed_text == "a"  # only the agreed prefix

    def test_match_is_case_and_punctuation_insensitive(self):
        dec = LiveDecoder(
            ScriptedASR([
                [w("Hallo", 0.1, 0.5)],
                [w("hallo,", 0.1, 0.5), w("welt", 0.6, 0.9)],
            ]),
            grey_zone=0.0,
        )
        dec.feed(pcm(1.0), 0.0)
        dec.feed(pcm(1.0), 1.0)
        # "Hallo" and "hallo," agree despite case/punctuation; committed keeps the
        # freshest surface form.
        assert dec.committed_text == "hallo,"

    def test_committed_words_carry_absolute_timestamps(self):
        dec = LiveDecoder(
            ScriptedASR([
                [w("eins", 0.2, 0.6)],
                [w("eins", 0.2, 0.6), w("zwei", 1.2, 1.5)],
            ]),
            grey_zone=0.0,
        )
        dec.feed(pcm(1.0), 0.0)
        dec.feed(pcm(1.0), 1.0)
        (word,) = dec.committed_words
        assert word.text == "eins"
        assert (word.start, word.end) == (0.2, 0.6)


class TestMonotonicity:
    def test_committed_stream_is_append_only_and_ordered(self):
        # A hypothesis that grows one confirmed word each decode.
        script = [
            [w("ich", 0.1, 0.3)],
            [w("ich", 0.1, 0.3), w("gehe", 0.4, 0.7)],
            [w("ich", 0.1, 0.3), w("gehe", 0.4, 0.7), w("nach", 0.8, 1.1)],
            [w("ich", 0.1, 0.3), w("gehe", 0.4, 0.7), w("nach", 0.8, 1.1), w("hause", 1.2, 1.5)],
        ]
        dec = LiveDecoder(ScriptedASR(script), grey_zone=0.0)
        seen: list[Word] = []
        for i in range(len(script)):
            update = dec.feed(pcm(0.5), i * 0.5)
            # Each delta only extends the transcript — never rewrites a prior word.
            assert list(dec.committed_words[: len(seen)]) == seen
            seen = list(dec.committed_words)
            for prev, nxt in zip(seen, seen[1:], strict=False):
                assert prev.start <= nxt.start
            assert all(x in dec.committed_words for x in update.committed)
        dec.flush()
        assert dec.committed_text == "ich gehe nach hause"


class TestFilterNew:
    def test_no_duplication_when_window_re_emits_committed(self):
        dec = LiveDecoder(
            ScriptedASR([
                [w("a", 0.1, 0.4), w("b", 0.5, 0.8)],
                [w("a", 0.1, 0.4), w("b", 0.5, 0.8), w("c", 0.9, 1.2)],
                [w("a", 0.1, 0.4), w("b", 0.5, 0.8), w("c", 0.9, 1.2), w("d", 1.3, 1.6)],
            ]),
            grey_zone=0.0,
        )
        dec.feed(pcm(1.0), 0.0)
        dec.feed(pcm(1.0), 1.0)  # commits a, b
        dec.feed(pcm(1.0), 2.0)  # window re-emits a,b,c,d
        assert dec.committed_text.startswith("a b c")
        assert dec.committed_text.count("a b") == 1  # committed words not duplicated

    def test_ngram_dedup_when_committed_word_reappears_with_drift(self):
        dec = LiveDecoder(
            ScriptedASR([
                [w("a", 0.1, 0.4), w("b", 0.5, 0.8)],
                [w("a", 0.1, 0.4), w("b", 0.5, 0.8)],  # commit a, b
                # b re-emitted with a drifted start past the time cutoff; the
                # n-gram cleanup must still drop it rather than duplicate.
                [w("b", 0.75, 1.0), w("c", 1.1, 1.4), w("c", 1.1, 1.4)],
            ]),
            grey_zone=0.0,
        )
        dec.feed(pcm(1.0), 0.0)
        dec.feed(pcm(1.0), 1.0)
        dec.feed(pcm(1.0), 2.0)
        assert dec.committed_text.count("b") == 1


class TestFlushAndReset:
    def test_flush_force_commits_pending_tail(self):
        dec = LiveDecoder(ScriptedASR([[w("a", 0.1, 0.4), w("b", 1.5, 1.9)]]), grey_zone=2.0)
        update = dec.feed(pcm(2.0), 0.0)
        assert update.committed == ()  # grey zone holds everything back
        assert dec.interim == "a b"

        flushed = dec.flush()
        assert [x.text for x in flushed.committed] == ["a", "b"]
        assert dec.committed_text == "a b"
        assert dec.interim == ""

    def test_reset_drops_tail_without_committing(self):
        dec = LiveDecoder(ScriptedASR([[w("a", 0.1, 0.4), w("b", 1.5, 1.9)]]), grey_zone=2.0)
        dec.feed(pcm(2.0), 0.0)
        assert dec.interim == "a b"
        dec.reset()
        assert dec.committed_words == ()
        assert dec.interim == ""


class TestVadGating:
    def test_no_decode_in_silence(self):
        asr = ScriptedASR([[w("x", 0.1, 0.4)]])
        dec = LiveDecoder(asr, vad=FakeVAD([[]]), grey_zone=0.0)
        update = dec.feed(pcm(1.0), 0.0)
        assert asr.calls == 0  # gated: no ASR ran on silence
        assert dec.decodes == 0
        assert update.committed == ()

    def test_decodes_when_speech_present(self):
        asr = ScriptedASR([
            [w("hallo", 0.1, 0.5)],
            [w("hallo", 0.1, 0.5), w("welt", 0.6, 0.9)],
        ])
        vad = FakeVAD([
            [SpeechSegment(0.0, 1.0)],  # speech runs to the live edge
            [SpeechSegment(0.0, 2.0)],
        ])
        dec = LiveDecoder(asr, vad=vad, grey_zone=0.0)
        dec.feed(pcm(1.0), 0.0)
        dec.feed(pcm(1.0), 1.0)
        assert dec.decodes == 2
        assert dec.committed_text == "hallo"

    def test_endpoint_silence_finalizes_utterance(self):
        asr = ScriptedASR([
            [w("hallo", 0.1, 0.5), w("welt", 0.6, 0.9)],
        ])
        # Speech ends at 0.9; the second feed is a pause past endpoint_silence.
        vad = FakeVAD([
            [SpeechSegment(0.0, 1.0)],
            [SpeechSegment(0.1, 0.9)],
        ])
        dec = LiveDecoder(asr, vad=vad, grey_zone=2.0, endpoint_silence=0.6)
        dec.feed(pcm(1.0), 0.0)  # grey zone holds "hallo welt" as interim
        update = dec.feed(pcm(1.0), 1.0)  # pause → force-commit the tail
        assert dec.committed_text == "hallo welt"
        assert [x.text for x in update.committed] == ["hallo", "welt"]
        assert dec.interim == ""
