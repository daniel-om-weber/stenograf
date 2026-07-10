"""Voice activity detection and ASR windowing.

The finalize pass never feeds raw sliding windows to the ASR model: Silero
VAD finds speech, and ``pack_windows`` merges adjacent speech runs into
windows of at most ~30 s that always cut in silence (PLAN.md §2).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stenograf.audio import SAMPLE_RATE


@dataclass(frozen=True)
class SpeechSegment:
    start: float
    end: float


class SileroVAD:
    """Silero v5 via sherpa-onnx (ONNX/CPU on every platform)."""

    def __init__(
        self,
        model_path: Path,
        *,
        threshold: float = 0.5,
        min_silence: float = 0.5,
        min_speech: float = 0.25,
        max_speech: float = 28.0,
    ) -> None:
        import sherpa_onnx

        self._config = sherpa_onnx.VadModelConfig(
            silero_vad=sherpa_onnx.SileroVadModelConfig(
                model=str(model_path),
                threshold=threshold,
                min_silence_duration=min_silence,
                min_speech_duration=min_speech,
                # Bounds a single speech run so every ASR window fits the
                # pack_windows budget even in gap-free speech.
                max_speech_duration=max_speech,
            ),
            sample_rate=SAMPLE_RATE,
        )

    def speech_segments(self, samples: np.ndarray) -> list[SpeechSegment]:
        """Detect speech runs in mono 16 kHz float32 PCM."""
        import sherpa_onnx

        vad = sherpa_onnx.VoiceActivityDetector(self._config, buffer_size_in_seconds=120)
        segments: list[SpeechSegment] = []

        def drain() -> None:
            while not vad.empty():
                seg = vad.front
                start = seg.start / SAMPLE_RATE
                segments.append(SpeechSegment(start, start + len(seg.samples) / SAMPLE_RATE))
                vad.pop()

        chunk = self._config.silero_vad.window_size
        for offset in range(0, len(samples), chunk):
            vad.accept_waveform(samples[offset : offset + chunk])
            drain()
        vad.flush()
        drain()
        return segments

    def stream(self, origin: float) -> SileroVADStream:
        """A persistent incremental detector for the live pass.

        ``origin`` is the absolute session time of the first pushed sample;
        reported segments are on that clock.
        """
        return SileroVADStream(self._config, origin)


class SileroVADStream:
    """One long-lived Silero detector fed only new samples (the live pass's VAD).

    Constructing sherpa's detector costs ~25 ms and the live decoder asks for
    speech on every audio frame, so :meth:`SileroVAD.speech_segments` — a fresh
    detector re-scanning the whole retained window per call — dominated the
    session's CPU. This keeps one causal detector per channel: completed
    segments accumulate as they close, and the in-progress run comes from
    sherpa's ``current_segment``, so each call costs only the new audio.
    """

    def __init__(self, config, origin: float) -> None:
        import sherpa_onnx

        self._vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=120)
        self._window = config.silero_vad.window_size
        self._origin = origin
        self._fed = 0  # samples pushed through the detector (whole windows only)
        self._pending = np.zeros(0, dtype=np.float32)
        self._segments: list[SpeechSegment] = []

    def push(self, samples: np.ndarray) -> None:
        """Feed mono 16 kHz float32 PCM continuing the stream (any length)."""
        buf = np.concatenate([self._pending, samples]) if len(self._pending) else samples
        end = len(buf) - len(buf) % self._window
        for offset in range(0, end, self._window):
            self._vad.accept_waveform(buf[offset : offset + self._window])
        self._fed += end
        self._pending = buf[end:]
        self._drain()

    def segments(self, min_end: float) -> list[SpeechSegment]:
        """Speech runs (absolute time) ending after ``min_end``, open tail included.

        ``min_end`` is the decoder's retained-buffer start; it only moves
        forward, so segments are pruned as they fall out of the window.
        """
        self._segments = [s for s in self._segments if s.end > min_end]
        out = list(self._segments)
        if self._vad.is_speech_detected():
            start = self._origin + self._vad.current_segment.start / SAMPLE_RATE
            open_seg = SpeechSegment(start, self._origin + self._fed / SAMPLE_RATE)
            if open_seg.end > min_end:
                out.append(open_seg)
        return out

    def _drain(self) -> None:
        while not self._vad.empty():
            seg = self._vad.front
            start = self._origin + seg.start / SAMPLE_RATE
            self._segments.append(SpeechSegment(start, start + len(seg.samples) / SAMPLE_RATE))
            self._vad.pop()


def pack_windows(
    segments: list[SpeechSegment],
    total_duration: float,
    *,
    max_window: float = 30.0,
    pad: float = 0.15,
) -> list[tuple[float, float]]:
    """Merge speech segments into ASR windows of at most ``max_window`` s.

    Consecutive speech runs share a window while they fit; each window is
    padded slightly into the surrounding silence so VAD onset jitter never
    clips a word. Returned windows are disjoint and sorted.
    """
    windows: list[list[float]] = []
    for seg in segments:
        # Oversized run (VAD's max_speech_duration should prevent this):
        # hard-split rather than hand the model an unbounded window.
        if seg.end - seg.start > max_window:
            for cut in np.arange(seg.start, seg.end, max_window):
                windows.append([cut, min(cut + max_window, seg.end)])
            continue
        if windows and seg.end - windows[-1][0] <= max_window:
            windows[-1][1] = seg.end
        else:
            windows.append([seg.start, seg.end])

    padded: list[tuple[float, float]] = []
    for start, end in windows:
        start = max(0.0, start - pad)
        if padded:
            # Keep windows disjoint: hard-split neighbours touch, so the pad
            # must not reach back into the previous window.
            start = max(start, padded[-1][1])
        end = min(total_duration, end + pad)
        padded.append((start, end))
    return padded
