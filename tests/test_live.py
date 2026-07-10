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
        return [
            Segment(
                text=" ".join(x.text for x in words),
                start=words[0].start,
                end=words[-1].end,
                words=tuple(words),
            )
        ]


class FakeVAD:
    """Returns a queued list of speech segments per call (last entry repeats)."""

    def __init__(self, per_call: list[list[SpeechSegment]]):
        self._per_call = per_call
        self.calls = 0

    def speech_segments(self, samples):
        segs = self._per_call[min(self.calls, len(self._per_call) - 1)]
        self.calls += 1
        return segs


class FakeVADStream:
    """Records pushes; returns a queued segment list per ``segments`` call."""

    def __init__(self, origin: float, per_call: list[list[SpeechSegment]]):
        self.origin = origin
        self.pushed_samples = 0
        self._per_call = per_call
        self.calls = 0

    def push(self, samples):
        self.pushed_samples += len(samples)

    def segments(self, min_end):
        segs = self._per_call[min(self.calls, len(self._per_call) - 1)]
        self.calls += 1
        return segs


class StreamingFakeVAD:
    """A VAD offering ``stream()`` — the decoder must never fall back to scans."""

    def __init__(self, per_call: list[list[SpeechSegment]]):
        self._per_call = per_call
        self.streams: list[FakeVADStream] = []

    def speech_segments(self, samples):
        raise AssertionError("the streaming path should be used, not a window scan")

    def stream(self, origin: float) -> FakeVADStream:
        s = FakeVADStream(origin, self._per_call)
        self.streams.append(s)
        return s


class TestLocalAgreement:
    def test_commits_prefix_two_decodes_agree(self):
        dec = LiveDecoder(
            ScriptedASR(
                [
                    [w("hallo", 0.1, 0.5), w("welt", 0.6, 0.9)],
                    [w("hallo", 0.1, 0.5), w("welt", 0.6, 0.9), w("wie", 1.1, 1.4)],
                ]
            ),
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
            ScriptedASR(
                [
                    [w("a", 0.1, 0.4), w("b", 1.2, 1.6)],
                    [w("a", 0.1, 0.4), w("b", 1.2, 1.6), w("c", 1.7, 1.9)],
                ]
            ),
            grey_zone=1.0,
            decode_interval=0.0,  # the 0.5 s second feed must still decode
        )
        dec.feed(pcm(1.5), 0.0)
        dec.feed(pcm(0.5), 1.5)  # audio_end 2.0, horizon 1.0
        # a and b both agree, but b ends at 1.6 — inside the grey zone — so held.
        assert dec.committed_text == "a"
        assert dec.interim == "b c"

    def test_disagreement_stops_commit(self):
        dec = LiveDecoder(
            ScriptedASR(
                [
                    [w("a", 0.1, 0.4), w("b", 0.5, 0.8)],
                    [w("a", 0.1, 0.4), w("x", 0.5, 0.8)],
                ]
            ),
            grey_zone=0.0,
        )
        dec.feed(pcm(1.0), 0.0)
        dec.feed(pcm(1.0), 1.0)
        assert dec.committed_text == "a"  # only the agreed prefix

    def test_match_is_case_and_punctuation_insensitive(self):
        dec = LiveDecoder(
            ScriptedASR(
                [
                    [w("Hallo", 0.1, 0.5)],
                    [w("hallo,", 0.1, 0.5), w("welt", 0.6, 0.9)],
                ]
            ),
            grey_zone=0.0,
        )
        dec.feed(pcm(1.0), 0.0)
        dec.feed(pcm(1.0), 1.0)
        # "Hallo" and "hallo," agree despite case/punctuation; committed keeps the
        # freshest surface form.
        assert dec.committed_text == "hallo,"

    def test_committed_words_carry_absolute_timestamps(self):
        dec = LiveDecoder(
            ScriptedASR(
                [
                    [w("eins", 0.2, 0.6)],
                    [w("eins", 0.2, 0.6), w("zwei", 1.2, 1.5)],
                ]
            ),
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
        dec = LiveDecoder(ScriptedASR(script), grey_zone=0.0, decode_interval=0.0)
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
            ScriptedASR(
                [
                    [w("a", 0.1, 0.4), w("b", 0.5, 0.8)],
                    [w("a", 0.1, 0.4), w("b", 0.5, 0.8), w("c", 0.9, 1.2)],
                    [w("a", 0.1, 0.4), w("b", 0.5, 0.8), w("c", 0.9, 1.2), w("d", 1.3, 1.6)],
                ]
            ),
            grey_zone=0.0,
        )
        dec.feed(pcm(1.0), 0.0)
        dec.feed(pcm(1.0), 1.0)  # commits a, b
        dec.feed(pcm(1.0), 2.0)  # window re-emits a,b,c,d
        assert dec.committed_text.startswith("a b c")
        assert dec.committed_text.count("a b") == 1  # committed words not duplicated

    def test_ngram_dedup_when_committed_word_reappears_with_drift(self):
        dec = LiveDecoder(
            ScriptedASR(
                [
                    [w("a", 0.1, 0.4), w("b", 0.5, 0.8)],
                    [w("a", 0.1, 0.4), w("b", 0.5, 0.8)],  # commit a, b
                    # b re-emitted with a drifted start past the time cutoff; the
                    # n-gram cleanup must still drop it rather than duplicate.
                    [w("b", 0.75, 1.0), w("c", 1.1, 1.4), w("c", 1.1, 1.4)],
                ]
            ),
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

    def test_drop_window_skips_forward_without_padding_silence(self):
        # Load-shed (Task 0f): after committing an early utterance, drop the window
        # and resume far ahead. The skipped span must be a caption *gap* — no silence
        # is padded across it, and the committed stream stays monotonic.
        asr = ScriptedASR(
            [
                [w("a", 0.1, 0.4), w("b", 0.6, 0.9)],
                [w("a", 0.1, 0.4), w("b", 0.6, 0.9)],  # commit a, b near t=0
                [w("c", 0.1, 0.4), w("d", 0.6, 0.9)],
                [w("c", 0.1, 0.4), w("d", 0.6, 0.9)],  # commit c, d near t=30
            ]
        )
        dec = LiveDecoder(asr, grey_zone=0.0)
        dec.feed(pcm(1.0), 0.0)
        dec.feed(pcm(1.0), 1.0)
        assert dec.committed_text == "a b"

        dec.drop_window()
        dec.feed(pcm(1.0), 30.0)  # a 29 s jump forward — no silence padded
        dec.feed(pcm(1.0), 31.0)

        starts = [x.start for x in dec.committed_words]
        assert [x.text for x in dec.committed_words] == ["a", "b", "c", "d"]
        assert starts == sorted(starts)  # append-only across the gap
        # "c" lands at its new offset (~30 s), proving the buffer origin was cleared
        # rather than the gap being padded (which would have kept it near ~1 s).
        assert 30.0 <= dec.committed_words[2].start < 31.0


class TestDecodeThrottle:
    def test_skips_decode_within_interval(self):
        asr = ScriptedASR(
            [
                [w("a", 0.1, 0.4)],
                [w("a", 0.1, 0.4), w("b", 1.1, 1.4)],
            ]
        )
        dec = LiveDecoder(asr, grey_zone=0.0, decode_interval=1.0)
        first = dec.feed(pcm(0.5), 0.0)
        assert dec.decodes == 1
        assert first.interim == "a"

        # Two feeds inside the interval: no decode, and the interim is preserved.
        held = dec.feed(pcm(0.25), 0.5)
        assert dec.decodes == 1
        assert held.committed == ()
        assert held.interim == "a"
        dec.feed(pcm(0.25), 0.75)
        assert dec.decodes == 1

        dec.feed(pcm(0.5), 1.0)  # a full interval of audio elapsed → decode
        assert dec.decodes == 2
        assert dec.committed_text == "a"

    def test_endpoint_bypasses_throttle(self):
        asr = ScriptedASR(
            [
                [w("hallo", 0.1, 0.5)],
                [w("hallo", 0.1, 0.5), w("welt", 0.6, 0.9)],
            ]
        )
        # Speech ends at 0.9; the second feed's trailing silence passes
        # endpoint_silence, which must force a decode despite the huge interval.
        vad = FakeVAD(
            [
                [SpeechSegment(0.0, 1.0)],
                [SpeechSegment(0.1, 0.9)],
            ]
        )
        dec = LiveDecoder(asr, vad=vad, grey_zone=2.0, endpoint_silence=0.6, decode_interval=60.0)
        dec.feed(pcm(1.0), 0.0)
        assert dec.decodes == 1
        update = dec.feed(pcm(1.0), 1.0)
        assert dec.decodes == 2  # utterance finalize ran inside the interval
        assert [x.text for x in update.committed] == ["hallo", "welt"]


class TestUtteranceMode:
    """decode_interval=None: no speculative decodes — the efficiency floor."""

    def test_one_decode_per_utterance_at_the_endpoint(self):
        asr = ScriptedASR(
            [
                [w("hallo", 0.1, 0.5), w("welt", 0.6, 0.9)],
            ]
        )
        vad = FakeVAD(
            [
                [SpeechSegment(0.0, 1.0)],  # speech runs to the live edge
                [SpeechSegment(0.1, 0.9)],  # feed 2: the tail has gone quiet
            ]
        )
        dec = LiveDecoder(asr, vad=vad, grey_zone=2.0, endpoint_silence=0.6, decode_interval=None)
        held = dec.feed(pcm(1.0), 0.0)
        assert dec.decodes == 0  # mid-utterance: no speculative decode at all
        assert held.committed == () and held.interim == ""

        update = dec.feed(pcm(1.0), 1.0)  # the pause closes the utterance
        assert dec.decodes == 1  # the whole utterance cost exactly one decode
        assert [x.text for x in update.committed] == ["hallo", "welt"]

    def test_unbroken_speech_flushes_at_window_cap(self):
        # No endpoint ever fires and no LocalAgreement tail exists, so the
        # overflow bound must fire on uncommitted *speech* or a monologue would
        # buffer without limit.
        asr = ScriptedASR([[w("mono", 0.2, 3.5)]])
        vad = FakeVAD(
            [
                [SpeechSegment(0.0, 1.0)],
                [SpeechSegment(0.0, 2.0)],
                [SpeechSegment(0.0, 3.0)],
                [SpeechSegment(0.0, 4.0)],
            ]
        )
        dec = LiveDecoder(asr, vad=vad, grey_zone=2.0, window_cap=3.0, decode_interval=None)
        for i in range(3):
            dec.feed(pcm(1.0), float(i))
        assert dec.decodes == 0
        update = dec.feed(pcm(1.0), 3.0)  # audio_end 4.0 > window_cap
        assert dec.decodes == 1
        assert [x.text for x in update.committed] == ["mono"]

    def test_silence_still_decodes_nothing(self):
        asr = ScriptedASR([[w("x", 0.1, 0.4)]])
        dec = LiveDecoder(asr, vad=FakeVAD([[]]), decode_interval=None)
        for i in range(5):
            dec.feed(pcm(1.0), float(i))
        assert asr.calls == 0 and dec.decodes == 0


class TestVadGating:
    def test_no_decode_in_silence(self):
        asr = ScriptedASR([[w("x", 0.1, 0.4)]])
        dec = LiveDecoder(asr, vad=FakeVAD([[]]), grey_zone=0.0)
        update = dec.feed(pcm(1.0), 0.0)
        assert asr.calls == 0  # gated: no ASR ran on silence
        assert dec.decodes == 0
        assert update.committed == ()

    def test_decodes_when_speech_present(self):
        asr = ScriptedASR(
            [
                [w("hallo", 0.1, 0.5)],
                [w("hallo", 0.1, 0.5), w("welt", 0.6, 0.9)],
            ]
        )
        vad = FakeVAD(
            [
                [SpeechSegment(0.0, 1.0)],  # speech runs to the live edge
                [SpeechSegment(0.0, 2.0)],
            ]
        )
        dec = LiveDecoder(asr, vad=vad, grey_zone=0.0)
        dec.feed(pcm(1.0), 0.0)
        dec.feed(pcm(1.0), 1.0)
        assert dec.decodes == 2
        assert dec.committed_text == "hallo"

    def test_streaming_vad_receives_appends_and_gap_padding(self):
        # Silence throughout: the stream must still see every appended sample,
        # including the silence synthesized across a feed gap, so its sample
        # clock stays aligned with the buffer timeline.
        asr = ScriptedASR([[w("x", 0.1, 0.4)]])
        vad = StreamingFakeVAD([[]])
        dec = LiveDecoder(asr, vad=vad, grey_zone=0.0)
        dec.feed(pcm(1.0), 0.0)
        dec.feed(pcm(1.0), 2.0)  # 1 s gap → 1 s of padding + 1 s chunk
        assert len(vad.streams) == 1
        assert vad.streams[0].origin == 0.0
        assert vad.streams[0].pushed_samples == 3 * SAMPLE_RATE
        assert asr.calls == 0  # silence still gates every decode

    def test_drop_window_rebuilds_the_stream_at_the_new_origin(self):
        # A load-shed jumps the timeline; the stream's sample clock cannot, so
        # the decoder must discard it and start a fresh one where audio resumes.
        vad = StreamingFakeVAD([[]])
        dec = LiveDecoder(ScriptedASR([[w("x", 0.1, 0.4)]]), vad=vad, grey_zone=0.0)
        dec.feed(pcm(1.0), 0.0)
        dec.drop_window()
        dec.feed(pcm(1.0), 30.0)
        assert [s.origin for s in vad.streams] == [0.0, 30.0]
        assert vad.streams[1].pushed_samples == SAMPLE_RATE  # no padding across the shed

    def test_endpoint_silence_finalizes_utterance(self):
        asr = ScriptedASR(
            [
                [w("hallo", 0.1, 0.5), w("welt", 0.6, 0.9)],
            ]
        )
        # Speech ends at 0.9; the second feed is a pause past endpoint_silence.
        vad = FakeVAD(
            [
                [SpeechSegment(0.0, 1.0)],
                [SpeechSegment(0.1, 0.9)],
            ]
        )
        dec = LiveDecoder(asr, vad=vad, grey_zone=2.0, endpoint_silence=0.6)
        dec.feed(pcm(1.0), 0.0)  # grey zone holds "hallo welt" as interim
        update = dec.feed(pcm(1.0), 1.0)  # pause → force-commit the tail
        assert dec.committed_text == "hallo welt"
        assert [x.text for x in update.committed] == ["hallo", "welt"]
        assert dec.interim == ""
