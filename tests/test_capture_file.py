import time
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
    mic = np.arange(SAMPLE_RATE, dtype=np.int16)  # 1 s
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


def test_unpaced_replay_emits_as_fast_as_read(tmp_path):
    write_wav(tmp_path / "mic.wav", np.zeros(SAMPLE_RATE, dtype=np.int16))  # 1 s
    provider = FileCaptureProvider({Channel.MIC: tmp_path / "mic.wav"}, frame_ms=200)
    provider.start({Channel.MIC})
    t0 = time.monotonic()
    frames = list(provider.frames())
    assert len(frames) == 5
    assert time.monotonic() - t0 < 0.2  # no real-time wait — the default dev path


def test_paced_replay_releases_frames_in_real_time(tmp_path):
    # 0.6 s of audio at 200 ms frames → frames at t=0.0, 0.2, 0.4; the last is
    # released at ~0.4 s wall clock, so paced replay is decisively slower than the
    # near-instant unpaced path — this is what drives the live pass at meeting cadence.
    write_wav(tmp_path / "mic.wav", np.zeros(int(SAMPLE_RATE * 0.6), dtype=np.int16))
    provider = FileCaptureProvider({Channel.MIC: tmp_path / "mic.wav"}, frame_ms=200, paced=True)
    provider.start({Channel.MIC})
    t0 = time.monotonic()
    frames = list(provider.frames())
    elapsed = time.monotonic() - t0
    assert len(frames) == 3
    assert 0.35 <= elapsed < 3.0
