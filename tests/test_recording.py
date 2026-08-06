import threading
import time
import wave

import numpy as np
import pytest

from stenograf import recording
from stenograf.audio import load_audio, load_audio_channels
from stenograf.capture.base import SAMPLE_RATE, AudioFrame, Channel
from stenograf.recording import AudioTee, read_channels


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
    tee = AudioTee(path, {Channel.MIC})
    tee.add(frame(Channel.MIC, 0.0, np.array([1, 2, 3], dtype=np.int16)))
    tee.add(frame(Channel.MIC, 3 / SAMPLE_RATE, np.array([4, 5], dtype=np.int16)))
    tee.close()

    nchannels, data = read_wav(path)
    assert nchannels == 1
    assert data.tolist() == [1, 2, 3, 4, 5]


def test_stereo_interleaves_mic_left_system_right(tmp_path):
    path = tmp_path / "rec.wav"
    tee = AudioTee(path, {Channel.MIC, Channel.SYSTEM})
    tee.add(frame(Channel.MIC, 0.0, np.array([10, 11], dtype=np.int16)))
    tee.add(frame(Channel.SYSTEM, 0.0, np.array([20, 21], dtype=np.int16)))
    tee.close()

    nchannels, data = read_wav(path)
    assert nchannels == 2
    assert data[0::2].tolist() == [10, 11]  # left = mic
    assert data[1::2].tolist() == [20, 21]  # right = system


def test_stereo_pads_the_shorter_channel_on_close(tmp_path):
    path = tmp_path / "rec.wav"
    tee = AudioTee(path, {Channel.MIC, Channel.SYSTEM})
    tee.add(frame(Channel.MIC, 0.0, np.array([1, 2, 3, 4], dtype=np.int16)))
    tee.add(frame(Channel.SYSTEM, 0.0, np.array([9], dtype=np.int16)))
    tee.close()

    _, data = read_wav(path)
    assert data[0::2].tolist() == [1, 2, 3, 4]  # mic in full
    assert data[1::2].tolist() == [9, 0, 0, 0]  # system padded with silence


def test_gap_between_frames_pads_silence(tmp_path):
    path = tmp_path / "rec.wav"
    tee = AudioTee(path, {Channel.MIC})
    tee.add(frame(Channel.MIC, 0.0, np.array([7], dtype=np.int16)))
    tee.add(frame(Channel.MIC, 1.0, np.array([8], dtype=np.int16)))  # 1 s later
    tee.close()

    _, data = read_wav(path)
    assert len(data) == SAMPLE_RATE + 1
    assert data[0] == 7
    assert np.all(data[1:SAMPLE_RATE] == 0)
    assert data[SAMPLE_RATE] == 8


def test_even_a_jitter_sized_gap_is_padded(tmp_path):
    # Unlike the AEC tracks (which absorb jitter-sized gaps), the recording
    # pads every gap exactly, so the WAV clock stays honest.
    path = tmp_path / "rec.wav"
    tee = AudioTee(path, {Channel.MIC})
    tee.add(frame(Channel.MIC, 0.0, np.array([7], dtype=np.int16)))
    tee.add(frame(Channel.MIC, 4 / SAMPLE_RATE, np.array([8], dtype=np.int16)))
    tee.close()

    _, data = read_wav(path)
    assert data.tolist() == [7, 0, 0, 0, 8]


def test_late_first_frame_pads_the_head_to_session_start(tmp_path):
    # Every recording is anchored at the capture clock's t=0, not at its first
    # frame — the AEC dump triple relies on this for sample alignment.
    path = tmp_path / "rec.wav"
    tee = AudioTee(path, {Channel.MIC})
    tee.add(frame(Channel.MIC, 2 / SAMPLE_RATE, np.array([5], dtype=np.int16)))
    tee.close()

    _, data = read_wav(path)
    assert data.tolist() == [0, 0, 5]


def test_backward_frame_raises_instead_of_misaligning(tmp_path):
    tee = AudioTee(tmp_path / "rec.wav", {Channel.MIC})
    tee.add(frame(Channel.MIC, 1.0, np.ones(SAMPLE_RATE, dtype=np.int16)))
    with pytest.raises(ValueError, match="backwards"):
        tee.add(frame(Channel.MIC, 0.0, np.ones(10, dtype=np.int16)))
    tee.close()


def test_file_is_playable_before_close(tmp_path):
    # Crash safety: a process killed mid-meeting (no close()) still leaves a
    # valid WAV of everything aligned and drained so far.
    path = tmp_path / "rec.wav"
    tee = AudioTee(path, {Channel.MIC, Channel.SYSTEM})
    tee.add(frame(Channel.MIC, 0.0, np.array([1, 2], dtype=np.int16)))
    tee.add(frame(Channel.SYSTEM, 0.0, np.array([3, 4], dtype=np.int16)))
    # deliberately no close() — simulate a crash

    nchannels, data = read_wav(path)
    assert nchannels == 2
    assert data.tolist() == [1, 3, 2, 4]


def tone(seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE))
    return (8000 * np.sin(2 * np.pi * 440 / SAMPLE_RATE * t)).astype(np.int16)


def test_opus_default_pipes_through_the_bundled_encoder(tmp_path):
    # A non-.wav target is Ogg Opus. The encode is lossy, so the round trip
    # asserts duration, not content.
    path = tmp_path / "audio.opus"
    tee = AudioTee(path, {Channel.MIC})
    tee.add(frame(Channel.MIC, 0.0, tone()))
    tee.close()

    assert tee.fallback_path is None
    decoded = load_audio(path)
    assert abs(len(decoded) - SAMPLE_RATE) < SAMPLE_RATE // 100


def test_opus_stereo_keeps_the_channels_separate(tmp_path):
    # Mic left / system right survives the encode: a tone on the mic channel
    # must not leak into a silent system channel.
    path = tmp_path / "audio.opus"
    tee = AudioTee(path, {Channel.MIC, Channel.SYSTEM})
    tee.add(frame(Channel.MIC, 0.0, tone()))
    tee.add(frame(Channel.SYSTEM, 0.0, np.zeros(SAMPLE_RATE, dtype=np.int16)))
    tee.close()

    left, right = load_audio_channels(path)
    rms = lambda x: float(np.sqrt(np.mean(np.square(x))))  # noqa: E731
    assert rms(right) < 0.1 * rms(left)


def test_opus_file_is_readable_after_encoder_kill(tmp_path):
    # The Opus counterpart of test_file_is_playable_before_close: a killed
    # encoder (crashed meeting) must leave a decodable file. -flush_packets 1
    # is what makes this hold — without it ffmpeg holds a 256 KB output block
    # and a sub-minute recording dies at zero bytes.
    path = tmp_path / "audio.opus"
    tee = AudioTee(path, {Channel.MIC})
    for i in range(10):
        tee.add(frame(Channel.MIC, float(i), tone()))
    sink = tee._sink
    assert isinstance(sink, recording._OpusSink)
    deadline = time.monotonic() + 10
    while sink._queued and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.5)  # let the encoder flush its pages
    assert sink._proc is not None
    sink._proc.kill()
    sink._proc.wait()
    tee.close()

    assert tee.fallback_path is None
    decoded = load_audio(path)
    assert len(decoded) >= 5 * SAMPLE_RATE  # of 10 s fed; the kill eats a tail


def test_opus_spawn_failure_falls_back_to_wav(tmp_path, monkeypatch):
    # A recording the user asked for must survive encoder trouble: no ffmpeg
    # binary → the tee degrades to a WAV beside the target and reports it.
    monkeypatch.setattr(recording, "ffmpeg_exe", lambda: str(tmp_path / "missing-ffmpeg"))
    path = tmp_path / "audio.opus"
    tee = AudioTee(path, {Channel.MIC})
    tee.add(frame(Channel.MIC, 0.0, np.array([1, 2, 3], dtype=np.int16)))
    tee.close()

    assert tee.fallback_path == tmp_path / "audio.opus.wav"
    assert not path.exists()
    nchannels, data = read_wav(tee.fallback_path)
    assert nchannels == 1
    assert data.tolist() == [1, 2, 3]


def test_missing_ffmpeg_runtimeerror_also_falls_back(tmp_path, monkeypatch):
    # imageio_ffmpeg raises RuntimeError (not OSError) when it has no binary
    # at all; that too must degrade, not abort the meeting.
    def boom() -> str:
        raise RuntimeError("No ffmpeg exe could be found")

    monkeypatch.setattr(recording, "ffmpeg_exe", boom)
    tee = AudioTee(tmp_path / "audio.opus", {Channel.MIC})
    tee.add(frame(Channel.MIC, 0.0, np.array([7], dtype=np.int16)))
    tee.close()

    assert tee.fallback_path == tmp_path / "audio.opus.wav"
    _, data = read_wav(tee.fallback_path)
    assert data.tolist() == [7]


def test_fallback_never_clobbers_a_sibling_wav(tmp_path, monkeypatch):
    # --record-audio talk.opus beside an unrelated talk.wav: the fallback
    # appends its suffix instead of substituting, so talk.wav survives.
    sibling = tmp_path / "talk.wav"
    sibling.write_bytes(b"precious" * 100)
    monkeypatch.setattr(recording, "ffmpeg_exe", lambda: str(tmp_path / "missing-ffmpeg"))
    tee = AudioTee(tmp_path / "talk.opus", {Channel.MIC})
    tee.add(frame(Channel.MIC, 0.0, np.array([1], dtype=np.int16)))
    tee.close()

    assert tee.fallback_path == tmp_path / "talk.opus.wav"
    assert sibling.read_bytes() == b"precious" * 100


class _DeadEncoder:
    """Popen stand-in whose pipe is broken from the start."""

    def __init__(self) -> None:
        self.stdin = self

    def write(self, block: bytes) -> None:
        raise BrokenPipeError

    def close(self) -> None:
        pass

    def wait(self, timeout: float | None = None) -> int:
        return 1

    def poll(self) -> int:
        return 1

    def kill(self) -> None:
        pass


def test_opus_encoder_death_mid_meeting_degrades_to_wav(tmp_path, monkeypatch):
    monkeypatch.setattr(recording.subprocess, "Popen", lambda *a, **k: _DeadEncoder())
    tee = AudioTee(tmp_path / "audio.opus", {Channel.MIC})
    tee.add(frame(Channel.MIC, 0.0, np.array([1, 2], dtype=np.int16)))
    deadline = time.monotonic() + 5
    while tee.fallback_path is None and time.monotonic() < deadline:
        time.sleep(0.01)  # the writer thread notices the broken pipe
    assert tee.fallback_path == tmp_path / "audio.opus.wav"

    # From the switch on, the WAV carries the meeting.
    tee.add(frame(Channel.MIC, 2 / SAMPLE_RATE, np.array([3, 4], dtype=np.int16)))
    tee.close()
    nchannels, data = read_wav(tee.fallback_path)
    assert nchannels == 1
    assert data.tolist() == [3, 4]


class _WedgedEncoder:
    """Popen stand-in that accepts writes but never drains — a full disk."""

    def __init__(self) -> None:
        self.stdin = self
        self._dead = threading.Event()

    def write(self, block: bytes) -> None:
        self._dead.wait()  # blocks like a full pipe until killed
        raise BrokenPipeError

    def close(self) -> None:
        pass

    def wait(self, timeout: float | None = None) -> int:
        return 1

    def poll(self) -> int | None:
        return 1 if self._dead.is_set() else None

    def kill(self) -> None:
        self._dead.set()


def test_wedged_encoder_never_blocks_the_capture_thread(tmp_path, monkeypatch):
    # CaptureLoop's contract: nothing on the capture thread may stall. A
    # wedged encoder must be killed and the tee must keep accepting frames.
    monkeypatch.setattr(recording.subprocess, "Popen", lambda *a, **k: _WedgedEncoder())
    tee = AudioTee(tmp_path / "audio.opus", {Channel.MIC})
    start = time.monotonic()
    seconds = recording._WEDGE_SECONDS + 3
    for i in range(seconds):
        tee.add(frame(Channel.MIC, float(i), tone()))
    tee.add(frame(Channel.MIC, float(seconds), np.array([42, 43], dtype=np.int16)))
    tee.close()
    elapsed = time.monotonic() - start

    assert elapsed < 5  # never sat on the blocked pipe
    assert tee.fallback_path == tmp_path / "audio.opus.wav"
    _, data = read_wav(tee.fallback_path)
    assert data[-2:].tolist() == [42, 43]  # post-switch audio survives


def test_read_channels_round_trips_stereo(tmp_path):
    # read_channels is the exact inverse of the stereo tee layout.
    path = tmp_path / "rec.wav"
    tee = AudioTee(path, {Channel.MIC, Channel.SYSTEM})
    tee.add(frame(Channel.MIC, 0.0, np.array([10, 11, 12], dtype=np.int16)))
    tee.add(frame(Channel.SYSTEM, 0.0, np.array([20, 21, 22], dtype=np.int16)))
    tee.close()

    channels = read_channels(path, [Channel.MIC, Channel.SYSTEM])
    assert channels[Channel.MIC].tolist() == [10, 11, 12]
    assert channels[Channel.SYSTEM].tolist() == [20, 21, 22]
    assert channels[Channel.MIC].dtype == np.int16


def test_read_channels_maps_mono_to_the_given_channel(tmp_path):
    # A mono file is ambiguous (mic-only or system-only) — the caller's channel
    # list disambiguates it. Here the single stream is the SYSTEM channel.
    path = tmp_path / "rec.wav"
    tee = AudioTee(path, {Channel.SYSTEM})
    tee.add(frame(Channel.SYSTEM, 0.0, np.array([5, 6], dtype=np.int16)))
    tee.close()

    channels = read_channels(path, [Channel.SYSTEM])
    assert list(channels) == [Channel.SYSTEM]
    assert channels[Channel.SYSTEM].tolist() == [5, 6]


def test_read_channels_rejects_a_channel_count_mismatch(tmp_path):
    path = tmp_path / "rec.wav"
    tee = AudioTee(path, {Channel.MIC})
    tee.add(frame(Channel.MIC, 0.0, np.array([1, 2], dtype=np.int16)))
    tee.close()

    with pytest.raises(ValueError, match="channel"):
        read_channels(path, [Channel.MIC, Channel.SYSTEM])  # expected 2, file has 1
