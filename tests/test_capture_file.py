import wave

import numpy as np

from stenograf.capture.base import SAMPLE_RATE, Channel
from stenograf.capture.file import FileCaptureProvider


def write_wav(path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(samples.tobytes())


def test_replays_two_channels_in_timestamp_order(tmp_path):
    mic = (np.arange(SAMPLE_RATE, dtype=np.int16))  # 1 s
    system = np.full(SAMPLE_RATE // 2, 7, dtype=np.int16)  # 0.5 s
    write_wav(tmp_path / "mic.wav", mic)
    write_wav(tmp_path / "sys.wav", system)

    provider = FileCaptureProvider(
        {Channel.MIC: tmp_path / "mic.wav", Channel.SYSTEM: tmp_path / "sys.wav"},
        frame_ms=200,
    )
    provider.start({Channel.MIC, Channel.SYSTEM})
    frames = list(provider.frames())

    # Non-decreasing timestamps across the merged stream.
    assert [f.timestamp for f in frames] == sorted(f.timestamp for f in frames)
    # Every sample of each channel is delivered exactly once.
    mic_total = sum(len(f.samples) for f in frames if f.channel is Channel.MIC)
    sys_total = sum(len(f.samples) for f in frames if f.channel is Channel.SYSTEM)
    assert mic_total == SAMPLE_RATE
    assert sys_total == SAMPLE_RATE // 2


def test_only_starts_requested_channels(tmp_path):
    write_wav(tmp_path / "mic.wav", np.zeros(SAMPLE_RATE, dtype=np.int16))
    write_wav(tmp_path / "sys.wav", np.zeros(SAMPLE_RATE, dtype=np.int16))
    provider = FileCaptureProvider(
        {Channel.MIC: tmp_path / "mic.wav", Channel.SYSTEM: tmp_path / "sys.wav"}
    )
    provider.start({Channel.MIC})  # in-room: no system channel
    channels = {f.channel for f in provider.frames()}
    assert channels == {Channel.MIC}
