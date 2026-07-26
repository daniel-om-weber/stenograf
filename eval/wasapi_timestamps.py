"""Do WASAPI capture packets carry usable device-side timestamps on both taps?

The measurement that decided the capture-helper design (2026-07-26). Read
PLAN-CAPTURE-HELPER.md for what was done with it; this file is the evidence,
kept so a second machine or a different driver can be checked without
re-deriving the method.

**Why it existed.** ``EchoCanceller`` pairs the mic and the system reference by
timestamp, and both are stamped when they *arrive*, so each channel carries its
own transport latency. Windows compensates with a declared constant
(``capture.windows.FAR_END_LAG_S``). PLAN-WINDOWS.md deferred the real fix --
device-side timestamps, which is how macOS avoids the problem entirely -- on the
belief that the transport does not carry them. It does: soundcard asks
``IAudioCaptureClient::GetBuffer`` for ``pu64DevicePosition`` /
``pu64QPCPosition`` and passes NULL for both (``mediafoundation.py:699``), with
the header already declaring them (``mediafoundation.py.h:249``).

**This script monkeypatches those two NULLs into real pointers.** That is a
measurement instrument, not a proposed design -- the shipped answer is a native
helper that never involves soundcard at all. The patch is confined here.

Usage (Windows, plain ``uv run`` -- ``--group eval`` cannot resolve here):

    uv run python eval/wasapi_timestamps.py [--seconds 30]

It needs something rendering so the loopback tap has packets, and synthesizes a
far-end clip if ``eval/audio/far-en.wav`` is missing. Output volume is left
alone: this measures timestamps, not acoustics, so no echo path is required.

ASCII output only -- see ``eval/aec_echo_present.py`` for why.
"""

from __future__ import annotations

import argparse
import ctypes
import statistics
import subprocess
import sys
from pathlib import Path

if sys.platform != "win32":
    raise SystemExit("wasapi_timestamps.py measures WASAPI; it is Windows-only")

import threading  # noqa: E402
import time  # noqa: E402

import aec_windows  # noqa: E402

SAMPLE_RATE = 16_000

AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY = 0x1
AUDCLNT_BUFFERFLAGS_SILENT = 0x2
AUDCLNT_BUFFERFLAGS_TIMESTAMP_ERROR = 0x4

_k32 = ctypes.windll.kernel32
_FREQ = ctypes.c_int64()
_k32.QueryPerformanceFrequency(ctypes.byref(_FREQ))


def qpc_100ns() -> float:
    """QueryPerformanceCounter in the 100 ns units WASAPI reports positions in.

    The same machine-wide counter both streams are stamped against, which is
    the whole point: a mic stream and a loopback stream are directly
    comparable without either one anchoring on its own first frame.
    """
    counter = ctypes.c_int64()
    _k32.QueryPerformanceCounter(ctypes.byref(counter))
    return counter.value * 10_000_000 / _FREQ.value


import soundcard  # noqa: E402
import soundcard.mediafoundation as mf  # noqa: E402

_ffi = mf._ffi
_com = mf._com


def _capture_buffer_stamped(self):
    """soundcard's ``_capture_buffer``, keeping the two discarded out-params."""
    data = _ffi.new("BYTE**")
    toread = _ffi.new("UINT32*")
    flags = _ffi.new("DWORD*")
    devpos = _ffi.new("UINT64*")
    qpcpos = _ffi.new("UINT64*")
    hr = self._ppCaptureClient[0][0].lpVtbl.GetBuffer(
        self._ppCaptureClient[0], data, toread, flags, devpos, qpcpos
    )
    _com.check_error(hr)
    stamps = getattr(self, "stamps", None)
    if stamps is not None:
        stamps.append((qpc_100ns(), devpos[0], qpcpos[0], toread[0], flags[0]))
    return data[0], toread[0], flags[0]


mf._Recorder._capture_buffer = _capture_buffer_stamped


def capture(label: str, loopback: bool, seconds: float, out: list) -> None:
    """One channel, device resolved and recorder opened in this thread.

    Mirrors ``capture/windows.py``'s pump: COM objects are apartment-bound, so
    the device must be resolved where it is used.
    """
    try:
        if loopback:
            speaker = soundcard.default_speaker()
            device = soundcard.get_microphone(speaker.id, include_loopback=True)
        else:
            device = soundcard.default_microphone()
        with device.recorder(samplerate=SAMPLE_RATE) as recorder:
            recorder.stamps = out
            end = time.perf_counter() + seconds
            while time.perf_counter() < end:
                recorder.record(numframes=None)
    except Exception as exc:  # noqa: BLE001 - report and let the other channel finish
        print(f"{label}: FAILED {type(exc).__name__}: {exc}")


def summarize(label: str, stamps: list) -> dict | None:
    if not stamps:
        print(f"{label}: no packets captured")
        return None
    recv, devpos, pktqpc, _nframes, flags = (list(c) for c in zip(*stamps, strict=True))

    zero_qpc = sum(1 for q in pktqpc if q == 0)
    ts_error = sum(1 for f in flags if f & AUDCLNT_BUFFERFLAGS_TIMESTAMP_ERROR)
    silent = sum(1 for f in flags if f & AUDCLNT_BUFFERFLAGS_SILENT)
    # The first packet always reports a discontinuity; soundcard clears it in
    # _record_chunk, downstream of the raw flags collected here.
    disc = sum(1 for f in flags if f & AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY)
    monotonic_qpc = all(b >= a for a, b in zip(pktqpc, pktqpc[1:], strict=False))
    monotonic_pos = all(b >= a for a, b in zip(devpos, devpos[1:], strict=False))

    skew_ms = [(r - q) / 10_000 for r, q in zip(recv, pktqpc, strict=True) if q]
    span_s = (pktqpc[-1] - pktqpc[0]) / 10_000_000 if len(pktqpc) > 1 else 0.0
    pos_rate = (devpos[-1] - devpos[0]) / span_s if span_s > 0 else float("nan")

    print(f"\n{label}")
    print(f"  packets            {len(stamps)} over {span_s:.1f} s")
    print(f"  qpc zero / error   {zero_qpc} zero, {ts_error} TIMESTAMP_ERROR")
    print(f"  flags              {silent} silent, {disc} discontinuity")
    print(f"  monotonic          qpc={monotonic_qpc}  devpos={monotonic_pos}")
    print(f"  devpos rate        {pos_rate:,.0f} frames/s (stream asked for {SAMPLE_RATE:,})")
    median = statistics.median(skew_ms) if skew_ms else None
    if skew_ms:
        fifth = max(1, len(skew_ms) // 5)
        parts = [statistics.median(skew_ms[i : i + fifth]) for i in range(0, len(skew_ms), fifth)]
        print(f"  recv - devstamp    median {median:7.1f} ms")
        print(f"                     per-fifth {' '.join(f'{p:.1f}' for p in parts[:5])} ms")
    return {
        "median_skew_ms": median,
        "usable": bool(skew_ms) and not zero_qpc and not ts_error and monotonic_qpc,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=30.0, help="capture length [30]")
    args = parser.parse_args()

    source = Path(__file__).parent / "audio" / "far-en.wav"
    if not source.exists():
        print(f"synthesizing {source.name} ({aec_windows.VOICE}) so the tap has packets")
        aec_windows.synthesize(source)

    print(f"playing {source.name} for {args.seconds:.0f} s; volume is left where it is")
    print("(this measures timestamps, not acoustics -- no echo path needed)")

    mic_stamps: list = []
    lpb_stamps: list = []
    threads = [
        threading.Thread(target=capture, args=("mic", False, args.seconds, mic_stamps)),
        threading.Thread(target=capture, args=("loopback", True, args.seconds, lpb_stamps)),
    ]
    player = subprocess.Popen(aec_windows.player_command(source))
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        if player.poll() is None:
            player.terminate()

    mic = summarize("mic", mic_stamps)
    lpb = summarize("loopback", lpb_stamps)

    print("\n--- verdict ---")
    if not (mic and lpb):
        print("INCONCLUSIVE: a channel produced nothing")
        return 2
    if not (mic["usable"] and lpb["usable"]):
        print("FAIL: device timestamps are not usable on at least one tap.")
        print("  A helper cannot place frames on one clock here; keep far_end_lag_s.")
        return 1

    # Sign convention: a positive skew means the packet reached us *after* the
    # device stamped it (the mic's capture path). Loopback is normally negative
    # because its stamp is render-side -- the moment the audio hits the endpoint,
    # which is slightly ahead of us reading it.
    relative_ms = mic["median_skew_ms"] - lpb["median_skew_ms"]
    print(
        f"mic skew {mic['median_skew_ms']:+.1f} ms, loopback skew {lpb['median_skew_ms']:+.1f} ms"
    )
    print(f"stamping on arrival therefore places the far end {relative_ms:.1f} ms early")
    print("relative to the mic, in steady state. Early is the SAFE direction for AEC3,")
    print("so this term is not the ~60 ms bug: that one is per-meeting anchor skew,")
    print("which a machine-wide clock removes and a declared constant cannot.")
    print("\nPASS: both taps carry populated, monotonic, machine-wide QPC stamps.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
