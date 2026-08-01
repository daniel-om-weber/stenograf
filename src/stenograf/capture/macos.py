"""macOS capture provider — spawns the signed Swift helper and reads its frames.

The helper (``stenocap``, `native/stenocap-macos`) captures system audio via a
Core Audio process tap and the mic via AVAudioEngine, resamples both to mono
16 kHz int16, and streams them as framed PCM over its stdout. No Python package
exposes the process-tap API, which is why the native helper exists.

The transport — spawning, framing, the crash retry, the stderr routing — is
shared with the Windows provider in :mod:`stenograf.capture.helper`, which is
also where the wire protocol is documented. What is left here is the one thing
that is genuinely macOS's: the helper takes SIGINT to stop.
"""

from __future__ import annotations

from stenograf.capture.helper import HelperCaptureProvider


class MacOSCaptureProvider(HelperCaptureProvider):
    """Streams frames from the ``stenocap`` subprocess.

    Echo cancellation is *not* done here. The helper used to expose a ``--aec``
    flag backed by Voice Processing IO; measured on macOS 26 it emitted no mic
    frames at all and attenuated the system channel by ~36 dB, so it was removed
    (see native/README.md). Echo is cancelled downstream, with the system channel
    as the far-end reference.
    """
