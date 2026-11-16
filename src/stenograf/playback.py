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
- **Windows:** ffmpeg has no audio output there at all. The waveOut API
  (winmm, unchanged since Windows 95) plays a memory buffer and ``waveOutReset``
  stops it at once, reached through ``ctypes``. Not ``winsound``: its in-memory
  play is synchronous only (``SND_MEMORY`` refuses ``SND_ASYNC``), and
  ``PlaySound(None, 0)`` from another thread does not interrupt it — a sample
  "stopped" after 1 s played all 4 s (Windows CI, 2026-11-16).

No Qt Multimedia: PySide6-Essentials does not ship it, and the add-on package
is a 40–90 MB download for one button.
"""

from __future__ import annotations

import subprocess
import sys
import threading
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
        # ffmpeg's PulseAudio muxer disconnects at the end without draining
        # (pulse_write_trailer, FFmpeg 9.0), discarding whatever is buffered —
        # and the server only starts a stream once its default ~2 s target is
        # full. A short buffer starts playback at once; a second of appended
        # silence is what the disconnect throws away instead of the voice.
        # "stenograf" is the stream name the system mixer shows.
        output = ["-af", "apad=pad_dur=1", "-f", "pulse", "-buffer_duration", "250",
                  "stenograf"]  # fmt: skip
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
        self._stopped.set()  # the waveOut loop polls it
        if self._process is not None:
            self._process.terminate()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def _play(self) -> None:
        error = None
        try:
            if sys.platform == "win32":
                _play_waveout(self._clip, self._stopped)
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


def _play_waveout(clip: np.ndarray, stopped: threading.Event) -> None:
    """Play ``clip`` through winmm's waveOut until it ends or ``stopped`` is set."""
    assert sys.platform == "win32"
    import ctypes
    from ctypes import wintypes

    class WaveFormatEx(ctypes.Structure):
        _fields_ = [
            ("wFormatTag", wintypes.WORD),
            ("nChannels", wintypes.WORD),
            ("nSamplesPerSec", wintypes.DWORD),
            ("nAvgBytesPerSec", wintypes.DWORD),
            ("nBlockAlign", wintypes.WORD),
            ("wBitsPerSample", wintypes.WORD),
            ("cbSize", wintypes.WORD),
        ]

    class WaveHdr(ctypes.Structure):
        _fields_ = [
            ("lpData", ctypes.c_void_p),
            ("dwBufferLength", wintypes.DWORD),
            ("dwBytesRecorded", wintypes.DWORD),
            ("dwUser", ctypes.c_size_t),
            ("dwFlags", wintypes.DWORD),
            ("dwLoops", wintypes.DWORD),
            ("lpNext", ctypes.c_void_p),
            ("reserved", ctypes.c_size_t),
        ]

    winmm = ctypes.WinDLL("winmm")
    handle_t, header_p = ctypes.c_void_p, ctypes.POINTER(WaveHdr)
    winmm.waveOutOpen.argtypes = [ctypes.POINTER(handle_t), wintypes.UINT,
                                  ctypes.POINTER(WaveFormatEx), ctypes.c_size_t,
                                  ctypes.c_size_t, wintypes.DWORD]  # fmt: skip
    for name in ("waveOutPrepareHeader", "waveOutWrite", "waveOutUnprepareHeader"):
        getattr(winmm, name).argtypes = [handle_t, header_p, wintypes.UINT]
    for name in ("waveOutReset", "waveOutClose"):
        getattr(winmm, name).argtypes = [handle_t]

    def check(result: int, call: str) -> None:
        if result != 0:  # MMSYSERR_NOERROR
            raise OSError(f"Windows audio output failed ({call}: MMRESULT {result})")

    wave_mapper, callback_null, whdr_done = 0xFFFFFFFF, 0, 0x1
    fmt = WaveFormatEx(1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16, 0)  # PCM, mono, s16
    data = clip.tobytes()
    buffer = ctypes.create_string_buffer(data, len(data))
    header = WaveHdr(lpData=ctypes.cast(buffer, ctypes.c_void_p), dwBufferLength=len(data))
    size = ctypes.sizeof(header)
    device = handle_t()
    check(winmm.waveOutOpen(ctypes.byref(device), wave_mapper, ctypes.byref(fmt), 0, 0,
                            callback_null), "waveOutOpen")  # fmt: skip
    try:
        check(winmm.waveOutPrepareHeader(device, ctypes.byref(header), size), "prepare")
        try:
            check(winmm.waveOutWrite(device, ctypes.byref(header), size), "write")
            # The driver sets WHDR_DONE once the buffer has played.
            while not header.dwFlags & whdr_done and not stopped.wait(0.02):
                pass
        finally:
            winmm.waveOutReset(device)  # the stop: drops whatever is still queued
            winmm.waveOutUnprepareHeader(device, ctypes.byref(header), size)
    finally:
        winmm.waveOutClose(device)
