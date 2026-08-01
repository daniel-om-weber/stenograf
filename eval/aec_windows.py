"""Windows far-end source, playback and output volume for the AEC rig.

``aec_rig.py`` drives real speakers and a real microphone. On macOS that needs
nothing but ``afplay`` and a clip sitting in ``eval/audio/``; on Windows all
three pieces have to be built, and each one has already cost a run:

- **The clip.** ``eval/audio/`` is gitignored, so a fresh Windows checkout has
  no far-end speech at all. :func:`synthesize` builds one from SAPI, which is
  the only speech source guaranteed present on the machine.
- **The voice.** The default SAPI voice follows the *system* language, and on
  the notebook this file was written for that is ``Microsoft Hedda Desktop``
  (de-DE). Left alone she reads English text with German phonetics and the ASR
  output is mangled past recognition ("throughput" -> "Troput", "peak hours" ->
  "Pika Oaz") -- which reads as a model failure and is nothing of the kind.
  :data:`VOICE` is therefore always selected explicitly, and
  :func:`installed_voices` exists so a caller can fail loudly when it is
  missing rather than silently fall back to the system default.
- **The volume.** 40 % on a laptop chassis puts no echo into the microphone at
  all, and a 33-minute run was spent discovering that (2026-07-26). 90 % puts
  the mic 22 dB above its own noise floor. Since the level is a precondition
  for the measurement rather than a preference, :func:`set_output_volume` sets
  it and the rig restores it afterwards -- an unattended long run cannot ask
  anyone to drag a slider.

**Everything printed here is ASCII**, for the reason ``aec_echo_present.py``
documents at length: a piped Python stdout on this machine encodes as cp1252,
and a stray "greater-than-or-equal" raises ``UnicodeEncodeError`` at the end of
the run, after the audio is gone.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import wave
from itertools import cycle
from pathlib import Path

SAMPLE_RATE = 16_000

VOICE = "Microsoft David Desktop"
"""The en-US voice to read the far end. See the module docstring: never let
SAPI pick, because on a German system it picks German."""

SCRIPT = (
    "Let's start with the throughput numbers from last week.",
    "The queue depth peaked on Wednesday afternoon and recovered overnight.",
    "Operational cost is still the largest line item this quarter.",
    "We moved four of the six migrations onto the new scheduler.",
    "Latency at the ninety fifth percentile came down by twelve percent.",
    "The remaining risk is the storage tier during peak hours.",
    "I would like a decision on the rollout window before Friday.",
    "Nothing else from me, so we can hand over to the platform team.",
)
"""Far-end speech. Deliberately the same vocabulary as the 2026-07-26 runs --
"throughput numbers", "operational cost" -- so a leaked ``Local-N`` line can be
compared against the two that run produced. Distinctive enough that echo in the
transcript is recognisable as echo rather than plausible local speech."""

GAPS = (2.0, 3.0, 2.0, 6.0)
"""Silence between sentences, cycled. Two constraints meet here.
``aec_echo_present.py`` refuses to score a dump whose far end never rests
(``playing.all()`` is INCONCLUSIVE), so gaps are what make the probe
*possible*; and the 6 s member is the case the long run exists to test, because
WASAPI's loopback tap delivers nothing at all while nothing renders. That gap
used to be filled by ``soundcard`` from wall-clock time, which let the far end
drift out from under the alignment; the capture helper fills it from the same
device clock it stamps audio with, so the 6 s member is now checking that the
seam is *invisible* rather than merely survivable."""


def _powershell(script: str) -> str:
    """Run a PowerShell snippet, raising with its stderr on failure."""
    done = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
    )
    if done.returncode != 0:
        raise RuntimeError(f"powershell failed:\n{done.stdout}\n{done.stderr}")
    return done.stdout


def installed_voices() -> list[str]:
    out = _powershell(
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$s.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name }"
    )
    return [line.strip() for line in out.splitlines() if line.strip()]


_SYNTH_PS = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$lines = Get-Content -Raw -Encoding UTF8 '{lines}' | ConvertFrom-Json
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SelectVoice('{voice}')
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(
    {rate},
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
    [System.Speech.AudioFormat.AudioChannel]::Mono)
for ($i = 0; $i -lt $lines.Count; $i++) {{
    $part = Join-Path '{parts}' ("part-{{0:d3}}.wav" -f $i)
    $synth.SetOutputToWaveFile($part, $fmt)
    $synth.Speak($lines[$i])
    $synth.SetOutputToNull()
}}
$synth.Dispose()
"""


def synthesize(
    dest: Path,
    *,
    lines: tuple[str, ...] = SCRIPT,
    voice: str = VOICE,
    gaps: tuple[float, ...] = GAPS,
    lead_s: float = 1.0,
) -> float:
    """Write a mono 16 kHz far-end clip and return its duration in seconds.

    One WAV per sentence out of SAPI, concatenated here with exact silences
    rather than SSML ``<break>``s -- the gap lengths are the experiment (see
    :data:`GAPS`), so they are measured in samples and not left to a speech
    engine's prosody. ``lead_s`` of leading silence covers the moment between
    the rig seeing ``capturing:`` and the speakers actually producing sound.
    """
    if voice not in (available := installed_voices()):
        raise SystemExit(
            f"voice {voice!r} is not installed -- have: {', '.join(available)}\n"
            "Pick an en-US one explicitly; the system default here is German."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        parts_dir = Path(tmp)
        lines_json = parts_dir / "lines.json"
        lines_json.write_text(json.dumps(list(lines)), encoding="utf-8")
        _powershell(
            _SYNTH_PS.format(
                lines=lines_json,
                voice=voice,
                rate=SAMPLE_RATE,
                parts=parts_dir,
            )
        )
        parts = sorted(parts_dir.glob("part-*.wav"))
        if len(parts) != len(lines):
            raise RuntimeError(f"SAPI wrote {len(parts)} parts for {len(lines)} lines")

        gap_cycle = cycle(gaps)
        with wave.open(str(dest), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(SAMPLE_RATE)
            out.writeframes(b"\x00\x00" * int(lead_s * SAMPLE_RATE))
            for part in parts:
                with wave.open(str(part), "rb") as src:
                    if src.getframerate() != SAMPLE_RATE or src.getnchannels() != 1:
                        raise RuntimeError(f"{part} is not mono {SAMPLE_RATE} Hz")
                    out.writeframes(src.readframes(src.getnframes()))
                out.writeframes(b"\x00\x00" * int(next(gap_cycle) * SAMPLE_RATE))
            frames = out.getnframes()
    return frames / SAMPLE_RATE


def player_command(source: Path) -> list[str]:
    """Play a WAV out the default output device until killed.

    ``winsound`` is stdlib and blocks for the length of the clip, so a child
    process running it behaves exactly like ``afplay`` does on macOS: the rig
    terminates it to stop playback.
    """
    return [
        sys.executable,
        "-c",
        "import sys, winsound; winsound.PlaySound(sys.argv[1], winsound.SND_FILENAME)",
        str(source),
    ]


_VOLUME_PS = r"""
$ErrorActionPreference = 'Stop'
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;

[Guid("5CDF2C82-841E-4546-9722-0CF74078229A"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IAudioEndpointVolume {
  int RegisterControlChangeNotify(IntPtr p);
  int UnregisterControlChangeNotify(IntPtr p);
  int GetChannelCount(out uint c);
  int SetMasterVolumeLevel(float f, ref Guid g);
  int SetMasterVolumeLevelScalar(float f, ref Guid g);
  int GetMasterVolumeLevel(out float f);
  int GetMasterVolumeLevelScalar(out float f);
}

[Guid("D666063F-1587-4E43-81F1-B948E807363F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IMMDevice {
  int Activate(ref Guid iid, int ctx, IntPtr p, [MarshalAs(UnmanagedType.IUnknown)] out object o);
}

[Guid("A95664D2-9614-4F35-A746-DE8DB63617E6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IMMDeviceEnumerator {
  int EnumAudioEndpoints(int flow, int mask, IntPtr p);
  int GetDefaultAudioEndpoint(int flow, int role, out IMMDevice dev);
}

[ComImport, Guid("BCDE0395-E52F-467C-8E3D-C4579291692E")]
class MMDeviceEnumeratorComObject { }

public class StenoVol {
  static IAudioEndpointVolume Endpoint() {
    var e = (IMMDeviceEnumerator)(new MMDeviceEnumeratorComObject());
    IMMDevice dev;
    Marshal.ThrowExceptionForHR(e.GetDefaultAudioEndpoint(0, 1, out dev));
    Guid iid = typeof(IAudioEndpointVolume).GUID;
    object o;
    Marshal.ThrowExceptionForHR(dev.Activate(ref iid, 23, IntPtr.Zero, out o));
    return (IAudioEndpointVolume)o;
  }
  public static float Get() {
    float level;
    Marshal.ThrowExceptionForHR(Endpoint().GetMasterVolumeLevelScalar(out level));
    return level;
  }
  public static void Set(float level) {
    Guid empty = Guid.Empty;
    Marshal.ThrowExceptionForHR(Endpoint().SetMasterVolumeLevelScalar(level, ref empty));
  }
}
'@
"""
"""``IAudioEndpointVolume`` through ctypes-free C#, for the same reason
``winlink.py`` reaches ``IShellLink`` through ctypes COM: the scriptable
surface (``WScript.Shell``, the volume-up virtual key) cannot set an absolute
level, only nudge it in 2 % steps from wherever it happens to be."""


def output_volume() -> float:
    """Master volume of the default render endpoint, 0.0-1.0.

    ``ToString(InvariantCulture)`` is not decoration: PowerShell formats a
    float in the *system* locale, so on the German machine this file was
    written for a bare ``[StenoVol]::Get()`` prints ``0,4`` and ``float()``
    rejects it. (The reverse direction is safe -- PowerShell number *literals*
    are always dot-separated, whatever the locale.)
    """
    # Concatenated, never .format()ed: the C# above is full of braces.
    script = (
        _VOLUME_PS + "\n[StenoVol]::Get().ToString([Globalization.CultureInfo]::InvariantCulture)"
    )
    return float(_powershell(script).strip())


def set_output_volume(level: float) -> None:
    if not 0.0 <= level <= 1.0:
        raise ValueError(f"volume must be 0.0-1.0, got {level}")
    _powershell(_VOLUME_PS + f"\n[StenoVol]::Set({level})")


if __name__ == "__main__":  # quick manual check of the three primitives
    print(f"voices    : {len(installed_voices())} installed, using {VOICE}")
    print(f"volume    : {output_volume() * 100:.0f}%")
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("eval/audio/far-en.wav")
    print(f"clip      : {synthesize(target):.1f} s -> {target}")
