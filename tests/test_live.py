import wave
from pathlib import Path

import numpy as np
import pytest

from stenograf import models
from stenograf.asr.base import ASRBackend, Segment, Word
from stenograf.audio import SAMPLE_RATE, sample_index
from stenograf.live import WindowedLiveDecoder
from stenograf.vad import (
    DECODE_CONTEXT_S,
    SileroVAD,
    SpeechSegment,
    context_start,
    pack_windows,
)

# en-2 on purpose: it contains a >30 s unbroken speech run, so it exercises the
# oversized hard-split path on top of ordinary gap/budget packing.
_EVAL_WAV = Path(__file__).resolve().parent.parent / "eval" / "audio" / "en-2.wav"


def w(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=start, end=end)


def pcm(duration: float) -> np.ndarray:
    """A block of silence; the scripted ASR ignores samples, only its length matters."""
    return np.zeros(int(duration * SAMPLE_RATE), dtype=np.float32)


class ScriptedASR(ASRBackend):
    """Returns a queued word list per decode; the last entry repeats when exhausted.

    Word times are relative to the start of the decoded slice.
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


class PackingVADStream:
    """Scripted stream for the window pass: (completed, open) per feed/flush.

    The script is owned by the fake VAD and consumed across stream rebuilds —
    a real stream never re-emits runs it already delivered.
    """

    def __init__(self, owner: "PackingFakeVAD"):
        self._owner = owner
        self.pushed_samples = 0

    def push(self, samples):
        self.pushed_samples += len(samples)

    def take_completed(self):
        completed, _ = self._owner.current_step()
        return list(completed)

    def open_segment(self):
        _, open_seg = self._owner.current_step()
        self._owner.calls += 1  # called once per feed/flush, after take_completed
        return open_seg


class PackingFakeVAD:
    def __init__(self, script: list[tuple[list[SpeechSegment], SpeechSegment | None]]):
        self._script = script
        self.calls = 0
        self.streams: list[PackingVADStream] = []

    def current_step(self):
        return self._script[min(self.calls, len(self._script) - 1)]

    def speech_segments(self, samples):
        raise AssertionError("the window pass must not re-scan")

    def stream(self, origin: float) -> PackingVADStream:
        s = PackingVADStream(self)
        self.streams.append(s)
        return s


class TestWindowedDecoder:
    """The window pass: decode exactly the windows pack_windows would build."""

    def test_window_closes_max_gap_after_speech(self):
        # The window spans [0.85, 3.15] (speech 1.0–3.0 plus pad). Short window ⇒
        # the decode slice reaches back to the context start (clamped to the
        # buffer origin, 0.0 here), so scripted times are relative to 0.0 and
        # a word wholly inside the context region must NOT be committed.
        asr = ScriptedASR([[w("kontext", 0.1, 0.5), w("hallo", 1.0, 1.6), w("welt", 1.7, 2.6)]])
        vad = PackingFakeVAD(
            [
                ([SpeechSegment(1.0, 3.0)], None),  # run closed, silence follows
                ([], None),
                ([], None),
            ]
        )
        dec = WindowedLiveDecoder(asr, vad=vad, max_gap=5.0)
        dec.feed(pcm(4.0), 0.0)
        assert dec.decodes == 0  # 4.0 - 3.0 = 1 s of silence: window still open
        dec.feed(pcm(4.0), 4.0)
        assert dec.decodes == 0  # 5 s: not yet beyond max_gap
        update = dec.feed(pcm(1.0), 8.0)
        assert dec.decodes == 1  # 6 s of silence closed the window
        assert [x.text for x in update.committed] == ["hallo", "welt"]
        assert abs(update.committed[0].start - 1.0) < 1e-6

    def test_a_late_onset_word_is_claimed_from_the_preroll(self):
        # Same window [0.85, 3.15] and slice as above. The speaker began before
        # the VAD noticed: "ja" at 0.6–0.9 is inside the slice but ahead of the
        # padded start, and the window's pre-roll (vad.claim_start) is what
        # decides that it belongs here rather than nowhere. "kontext" is further
        # back than the pre-roll reaches and stays dropped.
        asr = ScriptedASR([[w("kontext", 0.1, 0.5), w("ja", 0.6, 0.9), w("hallo", 1.2, 1.6)]])
        vad = PackingFakeVAD(
            [
                ([SpeechSegment(1.0, 3.0)], None),
                ([], None),
                ([], None),
            ]
        )
        dec = WindowedLiveDecoder(asr, vad=vad, max_gap=5.0)
        dec.feed(pcm(4.0), 0.0)
        dec.feed(pcm(4.0), 4.0)
        update = dec.feed(pcm(1.0), 8.0)
        assert dec.decodes == 1
        assert [x.text for x in update.committed] == ["ja", "hallo"]

    def test_touching_windows_never_commit_the_same_word_twice(self):
        # The budget splits two runs 0.2 s apart, so the pads collide and window
        # 2 begins exactly where window 1's audio ended. Window 2 is short, so
        # its slice reaches back over the seam and its decode sees the same word
        # again, a hair later — and the floor at _decoded_to (vad.claim_start)
        # leaves it to window 1, whose audio it was.
        asr = ScriptedASR(
            [
                [w("naht", 28.9, 29.1)],  # window 1 = [0, 29.15]; slice at 0.0
                [w("naht", 14.8, 15.0)],  # window 2 = [29.15, 35.15]; slice at 14.15
            ]
        )
        vad = PackingFakeVAD(
            [
                ([SpeechSegment(0.0, 29.0)], None),
                ([SpeechSegment(29.2, 35.0)], None),  # 35 - 0 > 30 → window 1 closes
                ([], None),
            ]
        )
        dec = WindowedLiveDecoder(asr, vad=vad, max_window=30.0, max_gap=5.0)
        dec.feed(pcm(30.0), 0.0)
        first = dec.feed(pcm(6.0), 30.0)
        assert [x.text for x in first.committed] == ["naht"]
        flushed = dec.flush()
        assert dec.decodes == 2  # the seam word was decoded twice…
        assert flushed.committed == ()  # …and committed once
        assert [x.text for x in dec.committed_words] == ["naht"]

    def test_budget_split_matches_pack_windows(self):
        # Scripted times are relative to each decode's slice, which starts at
        # vad.context_start — the window's own start for these two long windows,
        # so 0.0 and 21.85 ([21.85, 35.15] is the second span).
        asr = ScriptedASR([[w("a", 1.0, 2.0)], [w("b", 3.0, 4.0)]])
        vad = PackingFakeVAD(
            [
                ([SpeechSegment(0.0, 20.0)], None),
                ([SpeechSegment(22.0, 35.0)], None),  # 35 - 0 > 30 → previous window closes
                ([], None),
            ]
        )
        dec = WindowedLiveDecoder(asr, vad=vad, max_window=30.0, max_gap=5.0)
        dec.feed(pcm(21.0), 0.0)
        assert dec.decodes == 0
        dec.feed(pcm(15.0), 21.0)
        assert dec.decodes == 1  # [0, 20] decoded; [22, 35] pends
        flushed = dec.flush()
        assert dec.decodes == 2  # the tail window decoded at end of stream
        assert [x.text for x in flushed.committed] == ["b"]  # the second window's

    def test_open_run_past_budget_closes_the_window_early(self):
        asr = ScriptedASR([[w("a", 1.0, 2.0)]])
        vad = PackingFakeVAD(
            [
                ([SpeechSegment(0.0, 5.0)], None),
                # An unbroken run is still open but already reaches past the shared
                # budget: nothing can join the pending window any more, so waiting
                # for the run to complete would only delay the caption.
                ([], SpeechSegment(6.0, 31.0)),
            ]
        )
        dec = WindowedLiveDecoder(asr, vad=vad, max_window=30.0, max_gap=5.0)
        dec.feed(pcm(6.0), 0.0)
        assert dec.decodes == 0
        dec.feed(pcm(25.0), 6.0)
        assert dec.decodes == 1  # [0, 5] decoded although the gap was only 1 s

    def test_oversized_run_is_hard_split_like_pack_windows(self):
        # sherpa's max_speech_duration is a soft bound; a 31 s unbroken run must
        # split at the max_window grid exactly as pack_windows would, with the
        # last (short) piece staying open for later runs to join.
        asr = ScriptedASR([[w("a", 1.0, 2.0)]])
        vad = PackingFakeVAD(
            [
                ([SpeechSegment(0.0, 31.0)], None),
                ([SpeechSegment(33.0, 34.0)], None),  # joins the 1 s tail piece
                ([], None),
            ]
        )
        dec = WindowedLiveDecoder(asr, vad=vad, max_window=30.0, max_gap=5.0)
        dec.feed(pcm(32.0), 0.0)
        assert dec.decodes == 1  # the [0, 30] piece decoded immediately
        dec.feed(pcm(3.0), 32.0)
        assert dec.decodes == 1  # [30, 31] + [33, 34] still pending together
        dec.flush()
        assert dec.decodes == 2

    def test_silence_costs_no_decodes_and_keeps_memory_bounded(self):
        asr = ScriptedASR([[w("x", 0.1, 0.4)]])
        dec = WindowedLiveDecoder(asr, vad=PackingFakeVAD([([], None)]))
        for i in range(20):
            dec.feed(pcm(1.0), float(i))
        assert asr.calls == 0 and dec.decodes == 0
        # Trimmed to the silence guard plus the retained decode context.
        assert dec.buffered_seconds <= DECODE_CONTEXT_S + 2.0
        assert dec.flush().committed == ()

    def test_drop_window_abandons_the_pending_window(self):
        asr = ScriptedASR([[w("x", 0.1, 0.4)]])
        vad = PackingFakeVAD([([SpeechSegment(0.5, 1.5)], None), ([], None)])
        dec = WindowedLiveDecoder(asr, vad=vad)
        dec.feed(pcm(2.0), 0.0)
        dec.drop_window()  # load-shed: the pending window is a caption gap now
        dec.feed(pcm(1.0), 30.0)
        assert dec.flush().committed == ()
        assert dec.decodes == 0
        assert len(vad.streams) == 2  # the stream was rebuilt at the new origin


class TestCaptionBuffer:
    """The retained-buffer behaviors, exercised through the decoder's surface."""

    def test_requires_a_streaming_vad(self):
        with pytest.raises(TypeError, match="streaming VAD"):
            WindowedLiveDecoder(ScriptedASR([[]]), vad=object())  # type: ignore[arg-type]

    def test_streaming_vad_receives_appends_and_gap_padding(self):
        # The stream must see every appended sample, including the silence
        # synthesized across a feed gap, so its sample clock stays aligned
        # with the buffer timeline.
        vad = PackingFakeVAD([([], None)])
        dec = WindowedLiveDecoder(ScriptedASR([[w("x", 0.1, 0.4)]]), vad=vad)
        dec.feed(pcm(1.0), 0.0)
        dec.feed(pcm(1.0), 2.0)  # 1 s gap → 1 s of padding + 1 s chunk
        assert len(vad.streams) == 1
        assert vad.streams[0].pushed_samples == 3 * SAMPLE_RATE

    def test_backwards_feed_raises(self):
        dec = WindowedLiveDecoder(ScriptedASR([[]]), vad=PackingFakeVAD([([], None)]))
        dec.feed(pcm(1.0), 0.0)
        with pytest.raises(ValueError, match="backwards"):
            dec.feed(pcm(1.0), 0.0)


class SliceRecorder(ASRBackend):
    """Records the exact sample arrays it is asked to decode; emits no words."""

    name = "slice-recorder"

    def __init__(self):
        self.slices: list[np.ndarray] = []

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def transcribe(self, samples, language):
        self.slices.append(np.asarray(samples).copy())
        return []


@pytest.mark.skipif(
    models.cached_path(models.SILERO_VAD) is None or not _EVAL_WAV.exists(),
    reason="needs the cached silero model and the eval audio",
)
def test_windowed_slices_are_byte_identical_to_the_batch_pass():
    # The finalize pass reuses the window pass's decodes verbatim, which is only
    # sound if both passes hand the model the very same bytes. This pins the
    # whole chain — streaming VAD, online packing, sample_index() slicing —
    # against pack_windows + the batch slice arithmetic on real speech.
    with wave.open(str(_EVAL_WAV)) as wv:
        raw = np.frombuffer(wv.readframes(wv.getnframes()), dtype=np.int16)
        if wv.getnchannels() == 2:
            raw = raw[::2]
    audio = raw[: 160 * SAMPLE_RATE].astype(np.float32) / 32768.0

    vad = SileroVAD(models.cached_path(models.SILERO_VAD))
    batch = pack_windows(vad.speech_segments(audio), len(audio) / SAMPLE_RATE)
    # Short windows decode from their context start — the batch slice the live
    # pass must reproduce byte for byte (pipeline._decode does the same).
    batch_slices = [audio[sample_index(context_start(a, b)) : sample_index(b)] for a, b in batch]
    assert len(batch_slices) >= 3, "the clip should pack several windows"

    asr = SliceRecorder()
    dec = WindowedLiveDecoder(asr, vad=vad)
    step = SAMPLE_RATE // 5  # ~200 ms live frames
    for pos in range(0, len(audio), step):
        dec.feed(audio[pos : pos + step], pos / SAMPLE_RATE)
    dec.flush()

    assert len(asr.slices) == len(batch_slices)
    for i, (live, ref) in enumerate(zip(asr.slices, batch_slices, strict=True)):
        assert live.shape == ref.shape, f"window {i} length differs"
        assert np.array_equal(live, ref), f"window {i} bytes differ"


