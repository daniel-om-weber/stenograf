"""Linux capture provider — spawns the Rust helper and reads its frames.

The helper (``stenocap``, `native/stenocap`) opens one PulseAudio-protocol
record stream per channel — the mic from ``@DEFAULT_SOURCE@``, system audio
from ``@DEFAULT_MONITOR@`` (the default output's monitor) — and streams them
as framed PCM on its stdout, resampled server-side to mono 16 kHz. One client
protocol serves both sound servers: PipeWire ships ``pipewire-pulse``
precisely so pulse clients need no second path. The transport is shared with
macOS and Windows (:mod:`stenograf.capture.helper`).

**Why a helper at all, when ``parec`` worked.** Capture ran through one parec
subprocess per channel until 2026-08, and the reason it moved is timestamps:
parec's PCM stream is raw, so frames were stamped when they *arrived*
(``SessionClock``), each channel silently carrying its own transport latency
— the exact shape that measured a dead echo canceller on Windows (2.6 dB
ERLE), and was never measured here because losing echo cancellation reports
nothing. The helper stamps both taps on CLOCK_MONOTONIC with the server's own
latency accounting subtracted, so the question cannot arise. See
`native/stenocap/src/pulse.rs` for where the stamps come from, and
`PLAN-CAPTURE-HELPER.md` for the decision record.

The measured device-name behaviours carried over with the names themselves
(the helper passes the same server-resolved aliases parec did):
``@DEFAULT_MONITOR@`` follows a default-sink change mid-capture — the user
plugs in a headset, WirePlumber moves the meeting app's playback to it, and
the capture moves too; ``@DEFAULT_SOURCE@`` pins to the mic that was default
at start (an acceptable asymmetry — the remote channel is the one that must
survive an output-device switch). A muted sink's monitor still delivers
(sink volume does not attenuate it). No code path writes audio to disk.
"""

from __future__ import annotations

from stenograf.capture.base import Channel
from stenograf.capture.helper import HelperCaptureProvider, query_devices


class LinuxCaptureProvider(HelperCaptureProvider):
    """Streams frames from the ``stenocap`` subprocess.

    Stopping closes the helper's stdin rather than signalling it — the Rust
    helper's one stop gesture on both its platforms (it exists for Windows,
    where no signal can reach a single child; see
    :attr:`HelperCaptureProvider._stop_signal`). The helper ignores SIGINT for
    exactly this reason: a terminal's Ctrl+C hits the whole process group, and
    the parent must drain the pipe before the helper lets go.
    """

    _stop_signal = None


def default_devices(channels: set[Channel]) -> dict[Channel, str]:
    """What each channel would record from right now.

    The shared helper preflight, undecorated: a monitor's name already says
    what it is (``….monitor``), so nothing is appended the way Windows adds
    "(loopback)".
    """
    return query_devices(channels)
