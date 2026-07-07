"""Live pass: a re-decode window with LocalAgreement-2 commits.

This is Phase 2, Task 1 (PLAN.md §5). It turns the batch ``ASRBackend`` into a
streaming captioner **without any new dependency and without parakeet-mlx's
incremental ``transcribe_stream`` API** — the Phase 2 spike measured that API as
garbage at small right-context and fragile otherwise (PLAN.md §2 "Live ASR").

Instead the decoder re-decodes a short trailing window over the model's full
``generate()`` path every time it is fed audio, and commits text with a
LocalAgreement-2 policy: a word becomes final ("committed", shown black) only
once two consecutive window decodes agree on it. The unstable tail stays
provisional ("interim", shown grey) and is replaced on the next decode. Because
every decode uses the same full-attention path as the finalize pass, live
captions land at finalize-grade accuracy (~10% WER, spike-measured), and the
finalize pass still replaces the whole live transcript on stop.

Window management (PLAN.md §2):

- The window is ``left_context`` seconds of already-committed audio (context for
  the model) plus the uncommitted tail, capped at ``window_cap`` seconds. As
  words commit, the committed audio drops out of the window, so it stays short
  (~a handful of seconds during normal speech) rather than growing unbounded.
- Silero VAD gates the work: with no fresh speech beyond the commit point the
  decoder does no ASR at all (~0% accelerator in silence). At an utterance
  boundary (a pause the VAD closes) the tail is force-committed and the window
  resets, so context never bleeds across utterances.

The commit stream is **append-only**: a committed word is never rewritten
(monotonicity — one of the label-free acceptance metrics). The decoder composes
``ASRBackend.transcribe`` and is model-agnostic, but the live pass needs word
timestamps, so in practice it runs Parakeet (the committed default).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from stenograf.asr.base import ASRBackend, Word
from stenograf.audio import SAMPLE_RATE, to_float32
from stenograf.config import Language
from stenograf.vad import SileroVAD, SpeechSegment

_WORD_KEY = re.compile(r"\W+", re.UNICODE)


@dataclass(frozen=True)
class StreamingUpdate:
    """The result of feeding one chunk: what newly committed, plus the grey tail.

    ``committed`` are the words finalized by *this* feed (empty when the decode
    agreed on nothing new or was skipped in silence); the full committed
    transcript lives on the decoder. ``interim`` is the current provisional tail
    — the best guess for the audio that has not settled yet, shown grey.
    """

    committed: tuple[Word, ...]
    interim: str

    @property
    def committed_text(self) -> str:
        return " ".join(w.text for w in self.committed)


class LiveDecoder:
    """Streaming captioner over a batch ``ASRBackend`` (re-decode + LocalAgreement-2).

    Feed mono 16 kHz PCM as it arrives; each :meth:`feed` re-decodes the trailing
    window and returns the words that just became final. Call :meth:`flush` at the
    end of the stream (or an utterance) to force-commit the tail, and :meth:`reset`
    to drop the window at a long silence without committing.

    Times are absolute session seconds: ``feed`` is given the timestamp of its
    first sample, and committed/interim word times are on that same clock so they
    line up with the finalize pass and the transcript store.
    """

    def __init__(
        self,
        asr: ASRBackend,
        *,
        vad: SileroVAD | None = None,
        language: Language | None = None,
        left_context: float = 4.0,
        window_cap: float = 15.0,
        grey_zone: float = 2.0,
        endpoint_silence: float = 0.6,
        pre_roll: float = 0.25,
        match_tolerance: float = 0.15,
    ) -> None:
        self._asr = asr
        self._vad = vad
        self._language = language
        self.left_context = left_context
        self.window_cap = window_cap
        # Words ending within grey_zone of the live edge are too fresh to commit
        # even if two decodes agree — they lack right-context and often change.
        self.grey_zone = grey_zone
        # Trailing silence (s) past the last speech that closes an utterance.
        self.endpoint_silence = endpoint_silence
        # Silence kept after a reset so a word starting right away is not clipped.
        self.pre_roll = pre_roll
        self.match_tolerance = match_tolerance

        self._buf = np.zeros(0, dtype=np.float32)
        self._buf_start: float | None = None  # absolute time of _buf[0]
        self._committed: list[Word] = []
        # Previous decode's uncommitted tail; LocalAgreement-2 commits the prefix
        # this agrees on with the next decode.
        self._buffer: list[Word] = []
        # Count of ASR decodes performed — a CPU proxy (should stay 0 in silence).
        self.decodes = 0

    # -- public API --------------------------------------------------------

    def feed(self, samples: np.ndarray, t_offset: float) -> StreamingUpdate:
        """Add audio at absolute time ``t_offset`` and re-decode the window."""
        if len(samples):
            self._append(samples, t_offset)
        if len(self._buf) == 0:
            return StreamingUpdate((), self._interim_text())

        audio_end = self._audio_end()
        speech = self._speech()
        if speech is not None:
            committed_end = self._committed_end()
            last_speech = speech[-1].end if speech else self._buf_start
            uncommitted_speech = any(s.end > committed_end + self.match_tolerance for s in speech)
            # An utterance boundary: the VAD-closed tail has gone quiet. Force the
            # pending words out (with the pause as right-context) and reset.
            if audio_end - last_speech >= self.endpoint_silence and (
                uncommitted_speech or self._buffer
            ):
                return StreamingUpdate(tuple(self._finalize_utterance(speech)), "")
            if not uncommitted_speech:
                # Idle silence between utterances: everything is committed, so drop
                # the buffered audio down to a pre-roll and keep memory bounded.
                self._reset_buf()
                return StreamingUpdate((), "")

        # Bound the window: if LocalAgreement stalls through more than window_cap
        # of unbroken speech, force the pending tail out (in order) rather than
        # growing the buffer or dropping un-transcribed audio. Rare — the spike
        # measured parakeet committing steadily (PLAN.md §2 Live ASR).
        keep_from = max(self._committed_end() - self.left_context, self._buf_start)
        if audio_end - keep_from > self.window_cap and self._buffer:
            return StreamingUpdate(tuple(self._finalize_utterance(speech)), "")

        words = self._decode()
        new = self._filter_new(words)
        committed = self._commit(new, audio_end)
        self._trim()
        return StreamingUpdate(tuple(committed), self._interim_text())

    def flush(self) -> StreamingUpdate:
        """Force-commit the pending tail (end of stream or utterance) and reset."""
        return StreamingUpdate(tuple(self._finalize_utterance(self._speech())), "")

    def reset(self) -> None:
        """Drop the window and pending tail at a long silence; keep committed text."""
        self._buffer = []
        self._reset_buf()

    def drop_window(self) -> None:
        """Abandon the retained window and pending tail without committing them.

        A live-pass **load-shed**. Unlike :meth:`reset`, this keeps no pre-roll and
        clears the buffer origin, so the next :meth:`feed` restarts clean at its own
        ``t_offset`` — no silence is padded across the skipped span. Committed
        history is left intact and still monotonic; the abandoned audio becomes a
        caption gap the finalize pass fills on stop. The worker calls this when
        inference has fallen so far behind that feeding the whole backlog would
        spiral (PLAN.md §5, Task 0f)."""
        self._buf = np.zeros(0, dtype=np.float32)
        self._buf_start = None
        self._buffer = []

    @property
    def committed_words(self) -> tuple[Word, ...]:
        return tuple(self._committed)

    @property
    def committed_text(self) -> str:
        return " ".join(w.text for w in self._committed)

    @property
    def interim(self) -> str:
        return self._interim_text()

    # -- windowing ---------------------------------------------------------

    def _append(self, samples: np.ndarray, t_offset: float) -> None:
        chunk = to_float32(np.asarray(samples)).reshape(-1)
        if self._buf_start is None:
            self._buf_start = float(t_offset)
            self._buf = chunk.copy()
            return
        gap = t_offset - self._audio_end()
        tol = self.match_tolerance
        if gap < -tol:
            raise ValueError(
                f"feed went backwards {-gap:.3f}s (t_offset {t_offset:.3f}s "
                f"< buffered end {self._audio_end():.3f}s); frames must arrive in order"
            )
        if gap > tol:  # a real gap since the last chunk → pad silence
            self._buf = np.concatenate([self._buf, np.zeros(round(gap * SAMPLE_RATE), np.float32)])
        self._buf = np.concatenate([self._buf, chunk])

    def _audio_end(self) -> float:
        return (self._buf_start or 0.0) + len(self._buf) / SAMPLE_RATE

    def _committed_end(self) -> float:
        return self._committed[-1].end if self._committed else (self._buf_start or 0.0)

    def _speech(self) -> list[SpeechSegment] | None:
        """VAD segments over the current buffer (absolute time), or None w/o a VAD."""
        if self._vad is None or len(self._buf) == 0:
            return None
        start = self._buf_start or 0.0
        return [
            SpeechSegment(s.start + start, s.end + start)
            for s in self._vad.speech_segments(self._buf)
        ]

    def _decode(self) -> list[Word]:
        """Re-decode the whole retained window; word times shifted to absolute."""
        self.decodes += 1
        start = self._buf_start or 0.0
        return [
            Word(w.text, w.start + start, w.end + start, w.confidence)
            for seg in self._asr.transcribe(self._buf, self._language)
            for w in seg.words
        ]

    def _trim(self) -> None:
        """Drop committed audio so the window stays short (committed audio drops out).

        Only audio older than ``left_context`` before the last committed word is
        dropped — never uncommitted audio. The window length is instead bounded by
        the overflow-flush in :meth:`feed`, which keeps the retained buffer within
        ``window_cap`` without ever discarding un-transcribed speech.
        """
        keep_from = max(self._committed_end() - self.left_context, self._buf_start or 0.0)
        drop = round((keep_from - (self._buf_start or 0.0)) * SAMPLE_RATE)
        if drop > 0:
            self._buf = self._buf[drop:]
            self._buf_start = (self._buf_start or 0.0) + drop / SAMPLE_RATE

    def _reset_buf(self) -> None:
        """Keep only a short silence pre-roll of the buffer."""
        keep = round(self.pre_roll * SAMPLE_RATE)
        if len(self._buf) > keep:
            self._buf_start = self._audio_end() - keep / SAMPLE_RATE
            self._buf = self._buf[-keep:]

    def _finalize_utterance(self, speech: list[SpeechSegment] | None) -> list[Word]:
        """Commit everything still pending (utterance done → no grey zone) and reset."""
        committed: list[Word] = []
        if len(self._buf) > 0 and (speech is None or speech):
            committed = self._extend_committed(self._filter_new(self._decode()))
        self._buffer = []
        self._reset_buf()
        return committed

    # -- LocalAgreement-2 --------------------------------------------------

    def _filter_new(self, words: list[Word]) -> list[Word]:
        """Drop words the window re-emitted from the already-committed region."""
        if not self._committed:
            return list(words)
        cutoff = self._committed[-1].end - self.match_tolerance
        new = [w for w in words if w.start > cutoff]
        # Timestamps drift between decodes, so also drop a leading run of words
        # that just repeats the committed tail by text (whisper_streaming's
        # n-gram cleanup — PLAN.md §1 SimulStreaming reference).
        max_n = min(len(self._committed), len(new), 5)
        for n in range(max_n, 0, -1):
            if [_key(w) for w in self._committed[-n:]] == [_key(w) for w in new[:n]]:
                return new[n:]
        return new

    def _commit(self, new: list[Word], audio_end: float) -> list[Word]:
        """Commit the LocalAgreement-2 stable prefix, holding back the grey zone."""
        agree = 0
        for nw, bw in zip(new, self._buffer, strict=False):
            if _key(nw) != _key(bw):
                break
            agree += 1
        horizon = audio_end - self.grey_zone
        count = 0
        while count < agree and new[count].end <= horizon:
            count += 1
        self._buffer = new[count:]
        return self._extend_committed(new[:count])

    def _extend_committed(self, words: list[Word]) -> list[Word]:
        """Append words, enforcing a non-decreasing start time.

        Re-decoding jitters word boundaries by a few ms, so a fresh window can
        place a boundary word a hair before the last committed word's start. Such
        a regressor is a re-emitted duplicate, never genuinely new text — dropping
        it keeps the committed stream strictly append-only (the monotonicity
        invariant; PLAN.md §5) with no visible loss.
        """
        kept: list[Word] = []
        last = self._committed[-1].start if self._committed else float("-inf")
        for word in words:
            if word.start + 1e-6 < last:
                continue
            self._committed.append(word)
            last = word.start
            kept.append(word)
        return kept

    def _interim_text(self) -> str:
        return " ".join(w.text for w in self._buffer)


def _key(word: Word) -> str:
    """Match key for LocalAgreement: case- and punctuation-insensitive."""
    stripped = _WORD_KEY.sub("", word.text.lower())
    return stripped or word.text.strip().lower()
