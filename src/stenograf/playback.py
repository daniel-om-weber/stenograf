"""Play a few seconds of 16 kHz mono PCM through the speakers, from memory.

The one thing stenograf plays is a "Who spoke?" voice sample, so this is a
sample player, not a media player: one clip at a time, stoppable, and the clip
never touches the disk — it is the reason the samples exist only in memory.
That rules out the obvious players, all of which want a file. What remains:

- **macOS and Linux:** the ffmpeg already in the wheel (imageio-ffmpeg,
  pinned) reads raw PCM from a pipe and writes to AudioToolbox / PulseAudio —
  the Linux build has the PulseAudio client compiled in (0.8.1's binary needs
  nothing beyond libc, measured 2026-11-16), and a PipeWire desktop serves the
  same protocol through pipewire-pulse, exactly as capture relies on. ``afplay``
  cannot do this: it needs a seekable file (``AudioFileOpen failed`` on a pipe,
  measured 2026-11-16); ``pacat``/``pw-cat`` would, but stock Debian 12 GNOME
  ships no ``pacat`` and Debian 12 KDE no ``pw-cat``.
- **Windows:** ffmpeg has no audio output there at all; ``winsound`` (standard
  library) plays a WAV image from memory — synchronously only (``SND_MEMORY``
  refuses ``SND_ASYNC``), so on a thread.

No Qt Multimedia: PySide6-Essentials does not ship it, and the add-on package
is a 40–90 MB download for one button.
"""

from __future__ import annotations

import io
import subprocess
import sys
import threading
import wave
from collections.abc import Callable

import numpy as np

from stenograf.audio import SAMPLE_RATE, ffmpeg_exe


class PlaybackUnavailableError(RuntimeError):
    """No way to play audio here; the message names what is missing."""


def player_argv() -> list[str]:
    """The command that plays raw s16le 16 kHz mono from stdin on this machine.

    Raises :class:`PlaybackUnavailableError` on Windows, which plays
    in-process instead."""
    if sys.platform == "darwin":
        output = ["-f", "audiotoolbox", "-"]
    elif sys.platform.startswith("linux"):
        output = ["-f", "pulse", "stenograf"]  # the stream name the mixer shows
    else:
        raise PlaybackUnavailableError(f"no subprocess player on {sys.platform}")
    return [ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
            "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "pipe:0",
            *output]  # fmt: skip


class Playback:
    """One clip playing on its own thread; :meth:`stop` cuts it short.

    ``on_done`` fires once, on the playing thread, when the clip ends for any
    reason — finished, stopped, or failed (with the error's message)."""

    def __init__(self, clip: np.ndarray, on_done: Callable[[str | None], None]) -> None:
        self._clip = np.asarray(clip, dtype=np.int16)
        self._on_done = on_done
        self._process: subprocess.Popen[bytes] | None = None
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._play, name="voice-sample", daemon=True)

    def start(self) -> Playback:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stopped.set()
        if sys.platform == "win32":
            import winsound

            winsound.PlaySound(None, 0)  # ends the synchronous play on the other thread
        elif self._process is not None:
            self._process.terminate()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def _play(self) -> None:
        error = None
        try:
            if sys.platform == "win32":
                import winsound

                winsound.PlaySound(_wav_bytes(self._clip), winsound.SND_MEMORY)
            else:
                self._process = subprocess.Popen(
                    player_argv(),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                if self._stopped.is_set():  # stopped while the player was starting
                    self._process.terminate()
                stdin = self._process.stdin
                assert stdin is not None
                try:
                    stdin.write(self._clip.tobytes())
                    stdin.close()
                except BrokenPipeError:  # terminated mid-write: that is a stop
                    pass
                code = self._process.wait()
                if code != 0 and not self._stopped.is_set():
                    error = f"the audio player exited with code {code}"
        except (OSError, PlaybackUnavailableError) as exc:
            error = str(exc)
        self._on_done(error)


def _wav_bytes(clip: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(clip.tobytes())
    return buffer.getvalue()
