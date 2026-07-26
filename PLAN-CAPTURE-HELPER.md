# One capture helper, two backends — retiring arrival-stamped audio

**Status: designed and evidenced, not built.** Decided 2026-07-26 after the
Windows AEC work (`PLAN-WINDOWS.md`) exposed the root cause and one measurement
overturned the cost estimate that had deferred the fix.

Read the problem first, then the evidence, then what is decided and what is not.
The evidence section is the part not to re-derive: `eval/wasapi_timestamps.py`
re-runs it in twelve seconds on any Windows machine.

---

## The problem, stated once

`EchoCanceller` pairs the microphone against the system reference **by
timestamp** (`aec.py`). Off macOS, both channels are stamped when their frames
*arrive*, by `SessionClock` — so each channel silently carries its own transport
latency, and the two disagree by however much their paths differ.

That is not a small bug. On Windows it made echo cancellation dead: 2.6 dB ERLE
and far-end speech attributed to the local speaker, because WASAPI's loopback
tap is the longer path and AEC3 only searches its far-end history *backwards*.
A reference labelled after its own echo is unusable at any delay hint.

Today's answer is `CaptureProvider.far_end_lag_s`, a declared constant
(`capture/windows.py:73`, 0.15 s, 2.5× the largest measurement) chosen generously
because the error is one-sided. It works — 13.7 dB ERLE, zero leaked lines — and
it is still a workaround with three known holes:

1. **It is not a per-machine number.** The same Realtek endpoint measured 60 ms
   on one run and 10–25 ms minutes later, because each channel anchors on its own
   first frame and the two pump threads open independently. Nothing can be
   measured once at install.
2. **It fails silently past its margin.** A driver whose tap is slower than
   150 ms gets a dead canceller and no diagnostic.
3. **Linux has the same shape and no constant at all.** `parec` runs one
   subprocess per channel, both arrival-stamped. Unmeasured to this day, and it
   costs nothing but echo cancellation, so nothing would report it.

macOS has none of this, and not by luck: `stenocap` gets each buffer's Mach host
time from CoreAudio for **both** taps in one clock domain, so equal timestamps
mean simultaneous (`capture/macos.py:9-16`).

---

## The evidence that changed the decision

`PLAN-WINDOWS.md` deferred device-side timestamps as "a transport rewrite on
both — owning WASAPI capture through ctypes COM", priced against one number a
constant already got right. **That premise was wrong**, and it took reading the
dependency rather than the plan to see it:

- `soundcard/mediafoundation.py.h:249` **already declares** the full signature:
  `GetBuffer(..., UINT64 *pu64DevicePosition, UINT64 *pu64QPCPosition)`.
- `soundcard/mediafoundation.py:699` calls it and passes `_ffi.NULL` for both.
- `mediafoundation.py:773` says so outright: *"ignore
  AUDCLNT_BUFFERFLAGS_TIMESTAMP_ERROR, since we don't use time stamps."*

So the data was never unreachable. `eval/wasapi_timestamps.py` turns those two
NULLs into pointers and measures both taps at once. Measured 2026-07-26 on the
GPD notebook (Realtek HD Audio), reproduced across 30 s and 12 s runs:

| | mic | loopback |
|---|---|---|
| packets | 3000 / 30 s | 3001 / 30 s |
| QPC zero, TIMESTAMP_ERROR | 0, 0 | 0, 0 |
| monotonic (qpc, devpos) | yes, yes | yes, yes |
| `recv − devstamp` median | **+11.1 ms** | **−9.6 ms** |
| per-fifth spread | 11.1 11.1 11.0 11.1 11.1 | −9.6 −9.5 −9.6 −9.6 −9.6 |

**Stable to ±0.1 ms over 30 seconds**, against an arrival-derived estimate that
scattered by hundreds of milliseconds and needed a 2.5× safety factor.

Three findings that shape the implementation, none of which block it:

- **`pu64DevicePosition` is in device frames, not stream frames.** It advances at
  47,999 frames/s while the stream asked for 16,000 — the position is upstream of
  the `AUTOCONVERTPCM` resampler. **Use QPC as the clock; never use devpos as a
  sample index.** This would have been a silent 3× bug.
- **The loopback stamp is render-side and therefore *ahead* of receipt.** The mic's
  trails receipt (+11 ms), the tap's leads it (−9.6 ms) — it marks when the audio
  reaches the endpoint, i.e. when the echo is born, which is the ideal reference
  time. The signs differ between taps and getting that backwards reproduces the
  original bug mirrored.
- **The steady-state term is small and points the safe way.** Arrival-stamping
  places the far end ~20.5 ms *early* relative to the mic, and early is what AEC3
  searches. So the dangerous ~60 ms is **not** transport latency — it is the
  per-meeting anchor skew from two independently-opened streams. A machine-wide
  clock removes that; a declared constant cannot.

**Not covered by this evidence**, and still open: the run always had something
rendering, so soundcard's zero-fill path (`mediafoundation.py:736-756`) never
fired. Whether the loopback stream keeps delivering during far-end silence
depends on whether the call app holds its render stream open. That is where
`_REANCHOR_TOLERANCE_S` lives.

---

## The design

**A native capture helper per platform, speaking the wire protocol that already
exists.** From `capture/macos.py:9-16`, little-endian, stdout carries frames only:

    each frame = channel:u8  timestamp:f64  count:u32  samples:count×i16

`channel` 0 = mic, 1 = system; `timestamp` is seconds since capture start on **one
clock for both taps**. The Windows helper anchors that clock on QPC, the Linux one
on PipeWire/PulseAudio's, exactly as macOS anchors on Mach host time.

What this buys, in order of importance:

1. `far_end_lag_s` is **deleted**, not zeroed — the concept stops existing.
2. `_REANCHOR_TOLERANCE_S` and the forward re-anchor heuristic very likely go with
   it: a real stamp on the next packet turns the silence fill from a wall-clock
   guess into padding to a known time.
3. Linux's unmeasured echo-cancellation risk is closed by construction rather than
   by another measurement campaign.
4. `soundcard` and the external `parec` binary both leave the dependency set.
5. One capture architecture on three platforms instead of three.

**Language: Rust**, matching `stenodiar`. The `windows` crate binds WASAPI; the
`AUTOCONVERTPCM | SRC_DEFAULT_QUALITY` init flag keeps Windows' server-side 48 k→16 k
resampling, so no resampler dependency appears. Linux takes libpulse or PipeWire
directly — the dependency the `parec` decision existed to avoid, now paid for
deliberately because it is the only way to get `pa_timing_info` / `pw_time`.

The Python side is mostly deletion: `read_frame` and `_HEADER` already exist and are
tested — they move out of `capture/macos.py` into a shared module and the two
providers collapse onto it.

---

## The packaging consequence — decided: expand the matrix

Today a platform with no tagged wheel gets the pure `py3-none-any` wheel and
**capture still works**: Linux through the external `parec`, Windows through
`soundcard` (pure Python + cffi). Moving capture into a helper makes the wheel
matrix load-bearing, and these currently-working platforms would lose capture:

- **Linux x86_64 with glibc 2.28–2.38** — Ubuntu 22.04 LTS (2.35), Debian 12
  (2.36), RHEL 9 (2.34). The largest group and the reason this section exists.
- **Linux aarch64** — no tagged wheel today.
- **Windows ARM64** — no tagged wheel today.

(macOS x86_64 is already in this bucket and the loss is already accepted, which is
the precedent — but Intel Macs are a far smaller population than Debian 12.)

**Decision: expand the matrix rather than accept the narrowing.** A silent
capability loss on Ubuntu 22.04 surfaces as a bug report six months later, and the
floor is not intrinsic: `hatch_build.py:47-51` records that manylinux_2_39 is forced
by *onnxruntime* — "the onnxruntime static library ort-sys downloads needs glibc
2.38 symbols" — which is **stenodiar's** constraint. The capture helper links no
ONNX, so it can be built on a much older manylinux image. The shape is a low-floor
wheel carrying capture only, alongside the current 2_39 wheel carrying both; pip
prefers the most specific compatible tag, so it resolves without a chooser.

**This lands with the helper, not before**: a new wheel tag with nothing
platform-specific in it is exactly the lie `hatch_build.py:16-20` refuses to ship.

Two build-system facts to honour, both already written down there:
- **A capture helper is `stenocap`'s category, not `stenodiar`'s.** Without it
  `steno start` fails, so its build failure must fail the wheel — where a missing
  stenodiar only degrades speaker-count estimation.
- **Requiring Rust for development is accepted** (decided 2026-07-26). A checkout
  without cargo loses live capture on that platform, the way it already loses the
  diarization helper. Users installing from PyPI need no toolchain.

---

## Decided — do not re-litigate

- **Not vendoring soundcard, and not monkeypatching it.** Both were costed and
  rejected 2026-07-26: a fork to track forever, or a runtime dependency on a
  private method (`_capture_buffer`) that a release can rename silently. The
  monkeypatch survives only inside `eval/wasapi_timestamps.py`, which is a
  measurement instrument and says so.
- **Not PortAudio** (`pyaudiowpatch`, `sounddevice`). pyaudiowpatch was already
  spiked and rejected on real hardware (2026-07-11). Both abstract timing into a
  per-stream PortAudio clock rather than handing over the device's, which
  reintroduces at a higher level the exact question the spike answered at the API.
- **The far-end lag stays until the helper replaces it.** `FAR_END_LAG_S = 0.15` is
  working (13.7 dB ERLE, zero leaked lines) and must not be removed in advance of
  its replacement.
- **Linux keeps a declared constant in the interim**, in the same field, the way
  macOS returns 0.0 today — honestly marked rather than quietly broken. Windows
  does not wait for Linux.

## Deferred, with a trigger — livekit

`livekit` is **25.6 MB** imported in exactly one place (`aec.py:193`) for exactly
one class: `rtc.AudioProcessingModule`, i.e. WebRTC's AEC3. The code comment
already concedes the shape — *"the native lib is ~9 MB and no other path needs
it."* It is structurally the soundcard case: a large SDK used as a delivery
vehicle for one native component.

**It is not a candidate on its own**, because it fails the test that made soundcard
one: livekit discards nothing and costs no correctness. Replacing it buys install
size while risking a canceller that measures 37.6 dB on macOS.

**The trigger is the helper landing.** Once the helper holds both channels with
device timestamps, running AEC3 *inside* it (Rust bindings to the same WebRTC APM)
means channel pairing never crosses a language boundary — `aec.py`'s timestamp
machinery would stop existing rather than get a better input. Revisit then, on all
three platforms at once, and not before.

## Not candidates — recorded so the question stops recurring

The test a dependency must pass: it wraps an **OS API rather than an algorithm**;
we use a **sliver** of it; the abstraction is **lossy in a way that costs
correctness**; correctness is **verifiable by comparison, not judgment**; and the
native ship pipeline **already exists**. soundcard hits all five.

| dependency | installed | why it stays |
|---|---|---|
| onnxruntime | 67.1 MB | the inference engine |
| numpy | 44.0 MB | foundational numerics |
| sherpa-onnx | 25.7 MB | real work — Silero VAD, pyannote segmentation, speaker embeddings |
| textual | 4.9 MB | the Python surface *is* the value; on the retirement path anyway |
| click | 0.8 MB | replacing it means writing an argument parser |
| imageio-ffmpeg | — | already just a native binary + a `get_ffmpeg_exe()` shim |

These are algorithms. Reimplementing them trades a verifiable property for an
accuracy risk detectable only through the eval harness — the opposite of the
soundcard trade.

---

## Sequencing

Each step ships working; the constant stays until its replacement is proven.

1. **Windows helper**, mic + loopback, QPC-anchored, emitting the existing frame
   format. Validate against `eval/aec_alignment.py` (must report no correction
   needed with `far_end_lag_s = 0`) and `eval/aec_rig.py far-only` (ERLE at or
   above today's, zero leaked lines).
2. **Shared frame reader** — lift `read_frame`/`_HEADER` out of `capture/macos.py`;
   `capture/windows.py` becomes a helper client and `soundcard` leaves `pyproject`.
3. **Wheel matrix** — the low-floor capture wheel, `smoke-windows` asserting the
   helper the way it asserts `stenodiar.exe` today.
4. **Delete `far_end_lag_s`** and, if step 1 shows the silence case is handled,
   `_REANCHOR_TOLERANCE_S` with it.
5. **Linux backend**, same helper, PipeWire/PulseAudio. This is the step that closes
   the never-measured Linux echo-cancellation risk.
6. **Re-ask the livekit question** with the helper in place.

The ≥30-minute Windows AEC run (`PLAN-WINDOWS.md`, W4) is **not** a prerequisite and
its framing changes: its alignment half is superseded by step 1, and what remains —
double-talk, the ERLE gap to macOS, whether cancellation holds over half an hour —
is better spent validating the helper than adjudicating the constant it replaces.
