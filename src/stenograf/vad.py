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
