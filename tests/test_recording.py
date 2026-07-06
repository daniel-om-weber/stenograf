import wave

import numpy as np
import pytest

from stenograf.capture.base import SAMPLE_RATE, AudioFrame, Channel
from stenograf.recording import WavTee


def frame(channel: Channel, timestamp: float, samples: np.ndarray) -> AudioFrame:
    return AudioFrame(channel=channel, timestamp=timestamp, samples=samples)


def read_wav(path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SAMPLE_RATE
        assert w.getsampwidth() == 2
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        return w.getnchannels(), data


def test_mono_records_a_single_channel(tmp_path):
    path = tmp_path / "rec.wav"
    tee = WavTee(path, {Channel.MIC})
    tee.add(frame(Channel.MIC, 0.0, np.array([1, 2, 3], dtype=np.int16)))
    tee.add(frame(Channel.MIC, 3 / SAMPLE_RATE, np.array([4, 5], dtype=np.int16)))
    tee.close()

    nchannels, data = read_wav(path)
    assert nchannels == 1
    assert data.tolist() == [1, 2, 3, 4, 5]


def test_stereo_interleaves_mic_left_system_right(tmp_path):
    path = tmp_path / "rec.wav"
    tee = WavTee(path, {Channel.MIC, Channel.SYSTEM})
    tee.add(frame(Channel.MIC, 0.0, np.array([10, 11], dtype=np.int16)))
    tee.add(frame(Channel.SYSTEM, 0.0, np.array([20, 21], dtype=np.int16)))
    tee.close()

    nchannels, data = read_wav(path)
    assert nchannels == 2
    assert data[0::2].tolist() == [10, 11]  # left = mic
    assert data[1::2].tolist() == [20, 21]  # right = system


def test_stereo_pads_the_shorter_channel_on_close(tmp_path):
    path = tmp_path / "rec.wav"
    tee = WavTee(path, {Channel.MIC, Channel.SYSTEM})
    tee.add(frame(Channel.MIC, 0.0, np.array([1, 2, 3, 4], dtype=np.int16)))
    tee.add(frame(Channel.SYSTEM, 0.0, np.array([9], dtype=np.int16)))
    tee.close()

    _, data = read_wav(path)
    assert data[0::2].tolist() == [1, 2, 3, 4]  # mic in full
    assert data[1::2].tolist() == [9, 0, 0, 0]  # system padded with silence


def test_gap_between_frames_pads_silence(tmp_path):
    path = tmp_path / "rec.wav"
    tee = WavTee(path, {Channel.MIC})
    tee.add(frame(Channel.MIC, 0.0, np.array([7], dtype=np.int16)))
    tee.add(frame(Channel.MIC, 1.0, np.array([8], dtype=np.int16)))  # 1 s later
    tee.close()

    _, data = read_wav(path)
    assert len(data) == SAMPLE_RATE + 1
    assert data[0] == 7
    assert np.all(data[1:SAMPLE_RATE] == 0)
    assert data[SAMPLE_RATE] == 8


def test_backward_frame_raises_instead_of_misaligning(tmp_path):
    tee = WavTee(tmp_path / "rec.wav", {Channel.MIC})
    tee.add(frame(Channel.MIC, 1.0, np.ones(SAMPLE_RATE, dtype=np.int16)))
    with pytest.raises(ValueError, match="backwards"):
        tee.add(frame(Channel.MIC, 0.0, np.ones(10, dtype=np.int16)))
    tee.close()


def test_file_is_playable_before_close(tmp_path):
    # Crash safety: a process killed mid-meeting (no close()) still leaves a
    # valid WAV of everything aligned and drained so far.
    path = tmp_path / "rec.wav"
    tee = WavTee(path, {Channel.MIC, Channel.SYSTEM})
    tee.add(frame(Channel.MIC, 0.0, np.array([1, 2], dtype=np.int16)))
    tee.add(frame(Channel.SYSTEM, 0.0, np.array([3, 4], dtype=np.int16)))
    # deliberately no close() — simulate a crash

    nchannels, data = read_wav(path)
    assert nchannels == 2
    assert data.tolist() == [1, 3, 2, 4]
