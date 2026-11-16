"""The voice-sample player: the command per platform, and the play/stop/fail
lifecycle — exercised with a stand-in process that swallows the audio, so no
test ever makes a sound."""

import sys
import threading

import numpy as np
import pytest

from stenograf import playback

CLIP = np.zeros(1600, np.int16)


def _done():
    finished = threading.Event()
    errors = []

    def on_done(error):
        errors.append(error)
        finished.set()

    return finished, errors, on_done


def test_macos_and_linux_play_through_the_bundled_ffmpeg_from_a_pipe(monkeypatch):
    monkeypatch.setattr(playback.sys, "platform", "darwin")
    mac = playback.player_argv()
    assert "pipe:0" in mac and mac[-3:] == ["-f", "audiotoolbox", "-"]
    monkeypatch.setattr(playback.sys, "platform", "linux")
    linux = playback.player_argv()
    assert linux[0] == mac[0], "one binary, one code path"
    assert linux[-3:-1] == ["-f", "pulse"]


def test_windows_has_no_subprocess_player(monkeypatch):
    monkeypatch.setattr(playback.sys, "platform", "win32")
    with pytest.raises(playback.PlaybackUnavailableError):
        playback.player_argv()


def test_the_bundled_ffmpeg_has_this_platforms_audio_output():
    """Guards the pin: an imageio-ffmpeg whose build drops the output device
    would silence the button on this platform. Lists devices, plays nothing."""
    import subprocess

    from stenograf.audio import ffmpeg_exe

    devices = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-devices"], capture_output=True, text=True
    ).stdout
    wanted = {"linux": "pulse", "darwin": "audiotoolbox"}.get(sys.platform)
    if wanted is None:
        pytest.skip("Windows plays through winsound")
    assert wanted in devices


@pytest.mark.skipif(sys.platform == "win32", reason="Windows plays in-process")
class TestSubprocessPlayer:
    def test_a_clip_is_written_to_the_player_and_finishes(self, monkeypatch):
        check = "import sys; assert len(sys.stdin.buffer.read()) == 3200"  # 1600 int16 samples
        monkeypatch.setattr(playback, "player_argv", lambda: [sys.executable, "-c", check])
        finished, errors, on_done = _done()
        playback.Playback(CLIP, on_done).start()
        assert finished.wait(10)
        assert errors == [None]

    def test_stop_ends_a_playing_clip_without_an_error(self, monkeypatch):
        monkeypatch.setattr(
            playback,
            "player_argv",
            lambda: [sys.executable, "-c", "import time; time.sleep(30)"],
        )
        finished, errors, on_done = _done()
        player = playback.Playback(CLIP, on_done).start()
        player.stop()
        assert finished.wait(10)
        assert errors == [None]

    def test_a_failing_player_says_so(self, monkeypatch):
        monkeypatch.setattr(
            playback, "player_argv", lambda: [sys.executable, "-c", "raise SystemExit(3)"]
        )
        finished, errors, on_done = _done()
        playback.Playback(CLIP, on_done).start()
        assert finished.wait(10)
        assert errors == ["the audio player exited with code 3"]

    def test_no_player_is_an_error_not_a_crash(self, monkeypatch):
        def missing():
            raise playback.PlaybackUnavailableError("no audio player found")

        monkeypatch.setattr(playback, "player_argv", missing)
        finished, errors, on_done = _done()
        playback.Playback(CLIP, on_done).start()
        assert finished.wait(10)
        assert errors == ["no audio player found"]
