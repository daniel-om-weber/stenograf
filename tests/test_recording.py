import os
import sys
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


# Windows refuses to rename over or delete a file another handle holds open,
# so there the incident cannot happen and the write-back never runs.
posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="Windows cannot take an open file's name away"
)


def _sync_client_replaces(path, content: bytes = b"") -> None:
    """What a two-way sync client did on 2026-11-27: download its own copy of
    the file to a temporary name and rename it over the one being written."""
    stand_in = path.with_name(".download")
    stand_in.write_bytes(content)
    os.replace(stand_in, path)


def _leftovers(folder) -> list[str]:
    return sorted(p.name for p in folder.iterdir() if p.name.endswith(".restore"))


def _ordinary_mode(folder) -> int:
    """The mode a file created the ordinary way gets under the current umask."""
    probe = folder / "probe"
    probe.write_bytes(b"")
    mode = probe.stat().st_mode & 0o777
    probe.unlink()
    return mode


@posix_only
def test_opus_replaced_mid_meeting_is_written_back(tmp_path):
    path = tmp_path / "audio.opus"
    tee = AudioTee(path, {Channel.MIC})
    for i in range(3):
        tee.add(frame(Channel.MIC, float(i), tone()))
    _sync_client_replaces(path)
    for i in range(3, 6):
        tee.add(frame(Channel.MIC, float(i), tone()))
    tee.close()

    assert tee.error is None
    assert tee.restored
    assert tee.saved == [path]
    decoded = load_audio(path)
    assert abs(len(decoded) - 6 * SAMPLE_RATE) < SAMPLE_RATE // 10  # both halves
    assert _leftovers(tmp_path) == []
    assert path.stat().st_mode & 0o777 == _ordinary_mode(tmp_path)  # not mkstemp's 0600


@posix_only
def test_opus_removed_mid_meeting_is_written_back(tmp_path):
    path = tmp_path / "audio.opus"
    tee = AudioTee(path, {Channel.MIC})
    tee.add(frame(Channel.MIC, 0.0, tone()))
    path.unlink()
    tee.add(frame(Channel.MIC, 1.0, tone()))
    tee.close()

    assert tee.restored
    assert abs(len(load_audio(path)) - 2 * SAMPLE_RATE) < SAMPLE_RATE // 10


def test_an_untouched_recording_is_left_alone(tmp_path):
    path = tmp_path / "audio.opus"
    tee = AudioTee(path, {Channel.MIC})
    inode = path.stat().st_ino
    tee.add(frame(Channel.MIC, 0.0, tone()))
    tee.close()

    assert not tee.restored
    assert tee.error is None
    assert path.stat().st_ino == inode


@posix_only
def test_wav_replaced_mid_meeting_is_written_back(tmp_path):
    path = tmp_path / "rec.wav"
    tee = AudioTee(path, {Channel.MIC})
    tee.add(frame(Channel.MIC, 0.0, np.array([1, 2], dtype=np.int16)))
    _sync_client_replaces(path)
    tee.add(frame(Channel.MIC, 2 / SAMPLE_RATE, np.array([3, 4], dtype=np.int16)))
    tee.close()

    assert tee.error is None
    assert tee.restored
    assert read_wav(path)[1].tolist() == [1, 2, 3, 4]


@posix_only
def test_a_replaced_wav_continuation_is_written_back(tmp_path, monkeypatch):
    monkeypatch.setattr(recording.subprocess, "Popen", lambda *a, **k: _DeadEncoder())
    tee = AudioTee(tmp_path / "audio.opus", {Channel.MIC})
    tee.add(frame(Channel.MIC, 0.0, np.array([1, 2], dtype=np.int16)))
    deadline = time.monotonic() + 5
    while tee.fallback_path is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert tee.fallback_path is not None
    tee.add(frame(Channel.MIC, 2 / SAMPLE_RATE, np.array([3, 4], dtype=np.int16)))
    _sync_client_replaces(tee.fallback_path)
    tee.add(frame(Channel.MIC, 4 / SAMPLE_RATE, np.array([5, 6], dtype=np.int16)))
    tee.close()

    assert tee.error is None
    assert tee.restored
    assert read_wav(tee.fallback_path)[1].tolist() == [3, 4, 5, 6]


@posix_only
def test_a_failed_rename_names_where_the_recording_is(tmp_path, monkeypatch):
    path = tmp_path / "audio.opus"
    tee = AudioTee(path, {Channel.MIC})
    tee.add(frame(Channel.MIC, 0.0, tone()))
    _sync_client_replaces(path)

    def no_rename(src, dst):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(recording.os, "replace", no_rename)
    tee.close()

    assert not tee.restored
    assert tee.error is not None
    assert "could not be written back" in tee.error
    [kept] = tee.saved  # the stand-in at ``path`` is not the recording
    assert kept.name in tee.error
    assert abs(len(load_audio(kept)) - SAMPLE_RATE) < SAMPLE_RATE // 10


@posix_only
def test_a_failed_copy_leaves_no_junk(tmp_path, monkeypatch):
    path = tmp_path / "audio.opus"
    tee = AudioTee(path, {Channel.MIC})
    tee.add(frame(Channel.MIC, 0.0, tone()))
    _sync_client_replaces(path)

    def full_disk(src, dst):
        dst.write(b"partial")
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(recording.shutil, "copyfileobj", full_disk)
    tee.close()

    assert tee.error is not None
    assert "lost" in tee.error
    assert tee.saved == []  # the empty stand-in is not offered as the recording
    assert _leftovers(tmp_path) == []


@posix_only
def test_a_moved_meeting_folder_is_not_called_a_replacement(tmp_path):
    folder = tmp_path / "meeting"
    folder.mkdir()
    tee = AudioTee(folder / "audio.opus", {Channel.MIC})
    tee.add(frame(Channel.MIC, 0.0, tone()))
    folder.rename(tmp_path / "meeting-renamed")
    tee.add(frame(Channel.MIC, 1.0, tone()))
    tee.close()

    assert tee.error is not None
    assert "moved" in tee.error and "replaced" not in tee.error
    moved = tmp_path / "meeting-renamed" / "audio.opus"
    assert abs(len(load_audio(moved)) - 2 * SAMPLE_RATE) < SAMPLE_RATE // 10


@posix_only
def test_a_symlinked_target_keeps_its_link(tmp_path):
    real = tmp_path / "real.opus"
    link = tmp_path / "audio.opus"
    link.symlink_to(real)
    tee = AudioTee(link, {Channel.MIC})
    tee.add(frame(Channel.MIC, 0.0, tone()))
    _sync_client_replaces(real)
    tee.close()

    assert tee.restored
    assert link.is_symlink()
    assert abs(len(load_audio(real)) - SAMPLE_RATE) < SAMPLE_RATE // 10


def test_missing_ffmpeg_creates_no_target(tmp_path, monkeypatch):
    def boom() -> str:
        raise RuntimeError("No ffmpeg exe could be found")

    monkeypatch.setattr(recording, "ffmpeg_exe", boom)
    path = tmp_path / "audio.opus"
    tee = AudioTee(path, {Channel.MIC})
    assert not path.exists()  # not even while the meeting runs
    tee.close()


@posix_only
def test_an_abandoned_target_never_deletes_another_programs_file(tmp_path, monkeypatch):
    # The empty target of a failed encoder is deleted on close — but only
    # while its name is still the tee's.
    def no_spawn(*args, **kwargs):
        raise OSError(8, "Exec format error")

    monkeypatch.setattr(recording.subprocess, "Popen", no_spawn)
    path = tmp_path / "audio.opus"
    tee = AudioTee(path, {Channel.MIC})
    _sync_client_replaces(path, b"someone else's file")
    tee.add(frame(Channel.MIC, 0.0, np.array([1, 2], dtype=np.int16)))
    tee.close()

    assert path.read_bytes() == b"someone else's file"
    assert tee.saved == [tmp_path / "audio.opus.wav"]


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
    # Wedge = no pipe progress for the window; the test shrinks the window.
    monkeypatch.setattr(recording, "_WEDGE_SECONDS", 0.2)
    monkeypatch.setattr(recording.subprocess, "Popen", lambda *a, **k: _WedgedEncoder())
    tee = AudioTee(tmp_path / "audio.opus", {Channel.MIC})
    start = time.monotonic()
    tee.add(frame(Channel.MIC, 0.0, tone()))  # the writer blocks on this one
    time.sleep(0.3)  # exceed the progress window
    tee.add(frame(Channel.MIC, 1.0, tone()))  # detects the stall, kills
    deadline = time.monotonic() + 5
    while tee.fallback_path is None and time.monotonic() < deadline:
        time.sleep(0.01)  # the writer notices the kill and engages
    tee.add(frame(Channel.MIC, 2.0, np.array([42, 43], dtype=np.int16)))
    tee.close()
    elapsed = time.monotonic() - start

    assert elapsed < 5  # never sat on the blocked pipe
    assert tee.fallback_path == tmp_path / "audio.opus.wav"
    _, data = read_wav(tee.fallback_path)
    assert data[-2:].tolist() == [42, 43]  # post-switch audio survives


def test_disk_full_stops_the_recording_not_the_meeting(tmp_path, monkeypatch):
    # ENOSPC mid-recording (the WAV path, which fills a disk fastest) must
    # neither escape add() — that would end the capture loop — nor close().
    real_write = recording._WavSink.write
    state = {"writes": 0}

    def failing_write(self: recording._WavSink, block: bytes) -> None:
        state["writes"] += 1
        if state["writes"] > 1:
            raise OSError(28, "No space left on device")
        real_write(self, block)

    monkeypatch.setattr(recording._WavSink, "write", failing_write)
    tee = AudioTee(tmp_path / "rec.wav", {Channel.MIC})
    tee.add(frame(Channel.MIC, 0.0, np.array([1, 2], dtype=np.int16)))
    tee.add(frame(Channel.MIC, 2 / SAMPLE_RATE, np.array([3, 4], dtype=np.int16)))
    tee.add(frame(Channel.MIC, 4 / SAMPLE_RATE, np.array([5, 6], dtype=np.int16)))
    tee.close()  # must not raise

    assert tee.error is not None
    assert "space" in tee.error


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
