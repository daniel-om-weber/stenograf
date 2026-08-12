# PLAN-MIC-DEVICE — choosing which microphone a meeting records

Status: **steps 1–3 BUILT 2026-08-12; G5 and G5r are green (§7), the macOS and
Windows gates and G6 remain.** G5 found and cost a real defect: the Linux pin
held only until the device went away, and the fix is in `pulse.rs` (§7).
Scope: the **mic** channel's input device becomes selectable — "the OS default"
or one named device — on macOS, Windows and Linux, chosen in the Qt setup form
(which also names the current default) and settable standing in
`settings.toml`.

What shipped: the three helpers (`--list-inputs`, `--mic-device`, strict argv),
the Python layer (`[capture] mic_device`, `steno devices`, `--mic-device`, the
preflight and the doctor checks), and the setup form's picker. The Windows and
Linux helper code compiles only in CI — no toolchain for either target exists
on the development Mac.

**One design decision changed under measurement, and §4a below is now wrong
about macOS.** Retargeting `AVAudioEngine`'s input node (§4a.2) cannot be used
while the system-audio tap runs: with the tap's aggregate live,
`kAudioOutputUnitProperty_CurrentDevice` wedges `engine.start()` in the HAL
permanently, and so does opening the device with its own IO proc; retargeting
also posts a configuration change 9–13 ms after every build, which drove the
supervisor into a rebuild loop (nine rebuilds in ten seconds). A pinned
microphone therefore takes an `AudioDeviceIOProc` on the device when it is the
only channel, and **joins the tap's aggregate as a sub-device** when system
audio is on — where a silent keep-alive IO proc on the output device is
required, because that aggregate delivers nothing at all until something
renders (25 s of silence produced zero buffers). All measured 2026-08-12 on
macOS 26.5.1; `native/README.md` carries the record.

Decisions Daniel took 2026-08-12, each the plan's recommendation: **no GUI
persistence** (per-run picker + hand-edited standing key; O1 stays option (a),
`ui-state.json` remains the cheap upgrade if the friction turns out real);
**a missing pinned device is fatal**, with the remedy named in the message
(O3 closed — no advisory/warn-and-fall-back mode, not even for a standing file
pin); **names are accepted alongside IDs** (D3 as written); **mic only**, the
system channel keeps following the default (D1); **measure before building**
(§0 — M1 is now done, M2 is half done).

## 1. What exists today (measured, by reading the code)

Every platform records **whatever the OS calls the default input**, and nothing
in the tool can change that:

- macOS: `AVAudioEngine().inputNode` (`native/stenocap-macos/main.swift:646`).
  `MicSupervisor` (`:689`) *watches* `kAudioHardwarePropertyDefaultInputDevice`
  and rebuilds the engine on a change — it follows the default by design.
- Windows: `GetDefaultAudioEndpoint(eCapture, eConsole)`
  (`native/stenocap/src/wasapi.rs:130`), pinned at start.
- Linux: `@DEFAULT_SOURCE@` (`native/stenocap/src/pulse.rs:114`). Not pinned at
  start, as this once said: measured 2026-08-12, an unpinned mic follows a
  default-source change mid-capture, the server moving the stream it already
  resolved.
- Helper argv is `[--mic] [--system] | --devices` (`main.rs:48-66`); the Swift
  helper accepts `--mic`/`--system` only (`main.swift:764-771`). `--devices` is
  a read-only preflight consumed by `query_devices`
  (`src/stenograf/capture/helper.py:137`) — it answers *what would this run
  record*, keyed by channel.
- No `device` key in `settings.py`; no flag in `cli/start.py`; nothing in the
  GUI.

So the feature is genuinely absent at every layer, and every layer needs a
change. The wire protocol (frame format) does **not**.

## 0. Step 0 — measure the two assumptions the design rests on, BEFORE any code

Both are minutes of work with no code, and both can invalidate §2–§6. Doing
them last would be the mistake the repo's experiment-cost rule exists to stop.

### M1 — is device *enumeration* TCC-clean on macOS 26? **MEASURED 2026-08-12: yes.**

Gated §4a.4 (`--list-inputs` on the Swift helper), §5.3's macOS preflight,
§5.7's macOS doctor check, and O2. All four are unblocked.

Method, with the control that makes it mean anything. A probe walking
`kAudioHardwarePropertyDevices` → input `StreamConfiguration` → UID + name,
touching no AVFoundation capture API, listed all three inputs with full names
from the shell — **but that proved nothing**, because
`AVCaptureDevice.authorizationStatus(for: .audio)` reported that this shell
context already *has* a microphone grant. The same probe was therefore rebuilt
inside a throwaway app bundle (`dev.stenograf.micprobe-throwaway`, ad-hoc
signed, its own TCC identity), which reported `notDetermined` — never asked, no
grant — and still returned the complete list, names included, with no prompt.
Enumeration is not TCC-gated. (`log show --predicate 'subsystem ==
"com.apple.TCC"'` produced zero lines for *any* client in the window, so it is
useless as evidence either way; the authorization-status control is what
carries the result.)

Machine: macOS 26.5.1 (25F80), arm64. Not established: whether reading a
device's name is gated on *other* macOS versions, and whether an app that has
been explicitly **denied** (rather than never asked) can still enumerate — the
probe measured `notDetermined`, not `denied`.

### M2 — are the IDs stable? **Baseline recorded 2026-08-12; needs Daniel.**

The macOS inventory on this Mac right now:

| ID | Name | Note |
|---|---|---|
| `BuiltInMicrophoneDevice` | MacBook Pro-Mikrofon | default; a static string, stable by construction |
| `20CBF5DD-622F-4F4A-8DBE-09F300000003` | Mikrofon von „Daniels iPhone" | Continuity mic — a UUID, comes and goes with the phone |
| `MSLoopbackDriverDevice_UID` | Microsoft Teams Audio | **a virtual loopback driver** |

Two things this already settles:

- **§8.7 is not hypothetical.** A virtual device is present on this machine
  today, so the picker *will* offer one, and the invalid-host-time warning
  (§4a.5) is load-bearing rather than defensive.
- **The Continuity mic is a real use case** (pin the iPhone as the room mic)
  and simultaneously the worst case for D4: it is absent whenever the phone is.

Still owed, and it needs hands on hardware: re-run the probe **after a reboot**
and **after re-plugging a USB mic into a different port**, and diff against the
baseline; the same on the CachyOS notebook (pulse source names) and the Windows
box (endpoint IDs). D2 stays ASSUMED for the UUID-shaped and pulse/WASAPI IDs
until then — the two static-string macOS IDs above need no further proof.

**Linux, 2026-08-12: the baseline is taken and the reboot diff is set up, not
yet run.** `~/.local/share/steno-m2/` (outside the repo, and outside `/tmp`
because the reboot destroys that) holds `snapshot.sh`, the pre-reboot
`snap-boot0.txt` and the procedure. This machine's two ids are
`rnnoise_source` (a filter-chain source, and the default) and
`alsa_input.pci-0000_c4_00.6.analog-stereo` — both derived from a PCI address
or a module name rather than from an enumeration order, so the reboot half is
expected to be clean; the re-plug half is the one that can bite and **cannot be
run here at all: this machine has no USB audio device**, so that half stays
not-run rather than passed.

The probe (source + binary + baseline) is in this session's scratchpad; it is
deliberately *not* committed, because `steno devices` (§5.6) replaces it the
moment the helper work lands.

## 2. Decisions, with reasons

**D1 — mic only; the system channel keeps following the default.** The system
channel's device is not a user choice on any platform: on Linux
`@DEFAULT_MONITOR@`'s measured follow-the-default-sink behaviour is what keeps
capture with the meeting when a headset is plugged in (`pulse.rs:1-35`), and on
macOS the tap is a *process* tap whose aggregate is pinned to the default
output. Selecting an output to monitor is a separate feature with a separate
justification; not in this plan.

**D2 — identity is a platform-stable ID, not a name.** Store what survives a
reboot and a re-plug (pending M2):

| Platform | ID | Human label | Default from |
|---|---|---|---|
| macOS | `kAudioDevicePropertyDeviceUID` | `kAudioObjectPropertyName` | `kAudioHardwarePropertyDefaultInputDevice` → its UID |
| Windows | `IMMDevice::GetId()` endpoint string | `PKEY_Device_FriendlyName` | `GetDefaultAudioEndpoint(eCapture, eConsole)` → its ID |
| Linux | pulse source `name` | source `description` | server info `default_source_name` |

`AudioDeviceID`/`IMMDevice` handles are *not* stable and are never stored — the
UID/endpoint-ID is re-resolved at every engine rebuild.

**D3 — a stored value may also be an exact device *name*, under one normative
matching rule.** A hand-edited `settings.toml` saying
`mic_device = "Yeti Stereo Microphone"` must work; a Windows endpoint GUID is
not something a human types. The rule, stated once here and cited by all four
implementations (§3): **trim surrounding whitespace, normalize both sides to
NFC, then compare case-sensitively.** NFC is not optional — macOS
`kAudioObjectPropertyName` returns NFD, so `"Bürgel-Mikrofon"` typed into a
TOML file will not match an unnormalized comparison. Resolution order is
ID → name → error; two present devices sharing a name is an error naming both
IDs, never a silent pick. Windows caveat to document, not fix:
`PKEY_Device_FriendlyName` carries a mutable enumeration index
(`Microphone (2- Yeti Stereo Microphone)`), so a name pin there is *less*
stable than an ID pin — which is why `steno devices` prints copy-pasteable IDs.

**D4 — a missing selected device FAILS, loudly, before models load — with a
bounded grace and a named remedy.** No silent fallback to the default: silently
recording the built-in mic when the user asked for the desk mic is exactly the
failure this feature exists to prevent, and it is invisible until the
transcript is bad. Two refinements the first draft got wrong:

- **A ~3 s re-resolve grace at startup**, matching the supervisor's own retry
  cadence (`main.swift:747`), so a USB interface that enumerates two seconds
  after wake-from-sleep is not a hard failure — today's asymmetry would be
  "missing at t=0 → exit 1, vanishes at t=1 s → retry forever".
- **The message names the one-word fix** (`--mic-device default`, or the key to
  delete), because `settings.toml` is a text file users copy between machines
  and a stale pin would otherwise brick every meeting on the second machine.
  The template gains a "this key is machine-specific" line for the same reason.
  (The alternative — treat a *standing file* pin as advisory and
  warn-and-fall-back, while a *per-run* choice still fails hard — was
  considered and **declined 2026-08-12**: a warning on a line the user is not
  reading is how the wrong-mic transcript happens. One behaviour, both
  provenances.)

**D5 — the helpers must reject unknown arguments, which means a real
parser.** Today both scan argv for known flags and ignore the rest
(`main.rs:50-52` uses `.any()`, `main.swift:765-766` uses `.contains`). A new
Python passing `--mic-device X` to a stale dev-checkout helper would therefore
record the *default* mic while the UI says otherwise — D4's failure mode
through the back door. `.any()` also cannot implement a value-taking flag at
all: `stenocap --mic-device --mic` would swallow `--mic` as the device ID. So
both helpers get an index-walking loop that consumes `--mic-device`'s value,
rejects a value beginning with `--`, and exits 2 with usage on anything
unrecognized. The Swift helper gains `--help` at exit 0 in the same change
(`main.rs:57-59` has it as the wheel smoke test; `main.swift:768` currently
exits 2 on `--help`).

**D6 — the GUI picker is per-run, and the *selection lives in the list order*,
not in a QML binding.** The setup form gets a combo under the Microphone
toggle whose first entry is `System default (<name of the current default>)` —
that is how the form "shows which is the default". The mechanism matters and
the first draft hand-waved it: `Combo.qml:13-14` exposes `currentOption`/
`value` as **readonly** derivations of `currentIndex`, and no combo in
`Setup.qml` (`:38`, `:91`, `:104`, `:125`) ever writes `currentIndex` — every
one of them relies on index 0 being the right default. A model that arrives
asynchronously *resets* `currentIndex` to 0, and binding `currentIndex` is a
QML trap (ComboBox writes it on selection, killing the binding). So: **Python
returns the list already ordered with the entry that should be selected
first**, and the combo is not rendered until the list lands. Zero seeding code,
one convention, and it is the same "index 0 is the default" rule the other four
combos already follow. Persisting the choice is O1.

**D7 — `--devices` keeps its shape; listing gets a new flag.** `--devices`
answers "what will this run record" (a channel→name map, already consumed by
`query_devices`); `--list-inputs` answers "what could the mic record from" (an
array). Different questions, different shapes, different consumers. `--devices`
*does* start honouring `--mic-device`, so the preflight names and validates the
chosen device.

## 3. Wire protocol (all three helpers)

New flags, identical on the Swift and Rust helpers:

```
stenocap --list-inputs                        # JSON array on stdout, exit 0
stenocap [--mic] [--system] --mic-device ID   # pin the mic channel
stenocap --devices [--mic-device ID]          # preflight, honouring the pin
```

`--list-inputs` output (stdout, one line, UTF-8):

```json
[{"id":"BuiltInMicrophoneDevice","name":"MacBook Pro Microphone","default":true},
 {"id":"AppleUSBAudioEngine:Blue:Yeti:...","name":"Yeti Stereo Microphone","default":false}]
```

At most one entry carries `"default": true`; none does when the machine has no
default input (a legitimate state — the GUI's first entry then reads "System
default (none configured)").

`--mic-device` resolution is **D3's rule verbatim** in every implementation:
trim, NFC-normalize, ID first, then exact name, ambiguity is an error.

Failure modes, all `FATAL:` on stderr + non-zero exit, so `_helper_complaint`
(`helper.py:177`) already carries the reason to the user:

- names nothing present (after D4's grace) → `the selected microphone "<id>" is
  not available — run \`steno devices\` to list what is connected, or pass
  --mic-device default`
- matches two present devices by name → names both IDs
- unknown argument → usage + exit 2 (D5)

The frame format, the shared clock, the stop gestures and the silence filler
are untouched.

## 4. Step 1 — the helpers

### 4a. macOS (`native/stenocap-macos/main.swift`)

**Rebuilding this helper is safe.** `native/appbundle/README.md:53` records
(measured 2026-07-25) that TCC keys the grant to the *bundle stub's* cdhash and
that "children inherit the responsible process, so `stenocap`'s request is
attributed to `dev.stenograf.app` and rebuilding `stenocap` never re-prompts".
The frozen tree is `src/stenograf/assets/Stenograf.app`, untouched here.

1. **Enumerate**: `kAudioHardwarePropertyDevices` → keep devices whose input
   `kAudioDevicePropertyStreamConfiguration` reports ≥1 channel → read UID and
   name. The default input's UID marks the `default` flag.
2. **Pin**: before reading `inputFormat(forBus:)` (`:648`) and before
   `engine.start()`, set `kAudioOutputUnitProperty_CurrentDevice` (global
   scope, element 0) on `engine.inputNode.audioUnit` with the `AudioDeviceID`
   resolved *now* from the stored UID. Order is load-bearing: the input node
   caches its format from whatever device is current when the format is first
   read (§8.2).
3. **Supervisor** (`MicSupervisor`, `:689`) — three changes, each protecting a
   specific regression:
   - When pinned, a `kAudioHardwarePropertyDefaultInputDevice` change must
     **not** move capture (that listener's whole point today, `:702-708`).
   - `AVAudioEngineConfigurationChange` must still rebuild — the measured
     2026-07-20 AirPods bug (a default-*output* switch permanently stops a
     running engine, `:683-688`) is orthogonal to pinning and must not regress.
   - Every rebuild re-resolves UID → `AudioDeviceID` and re-pins; a rebuild
     that cannot find the UID keeps the existing "mic capture lost … retrying
     until an input device returns" behaviour (`:745`) and must never fall
     through to the default (D4).
4. **`--devices`** does not exist on this helper today; add it (mic = the
   selected-or-default input's name, system = the default output's name), plus
   `--list-inputs`. Neither may call `requestMicrophoneAccess()` (`:631`) —
   gated on M1.
5. **Host-time validity warning** (§8.7): `:658` falls back to `Clock.now()`
   when `when.isHostTimeValid` is false — arrival stamping, the exact thing the
   helper transport exists to eliminate (`capture/helper.py:20-24`). Pinning
   makes virtual/aggregate devices (BlackHole, Loopback, aggregates)
   selectable, and those are where invalid host times live. Log once, loudly,
   when the first N buffers of a pinned device report an invalid host time.

### 4b. Windows (`native/stenocap/src/wasapi.rs`)

1. **Enumerate**: `EnumAudioEndpoints(eCapture, DEVICE_STATE_ACTIVE)` → per
   item `GetId()` (allocates; `CoTaskMemFree` it) + `friendly_name()` (`:146`,
   already written). Default via the existing `default_endpoint` (`:130`).
2. **Pin**: `open()` (`:196`) and `device_names()` (`:156`) take an optional ID
   and call `enumerator.GetDevice(PCWSTR)` instead of
   `GetDefaultAudioEndpoint`. Which HRESULT an absent endpoint ID returns is
   ASSUMED to be `E_NOTFOUND`; map whatever it is to D4's message rather than
   matching on the code.
3. Everything else — the int16/float32 format ladder, `AUTOCONVERTPCM`, the QPC
   stamping — is device-independent and unchanged.
4. The consent-store check (`capture/windows.py:67`) is unchanged: a per-user
   privacy gate, not per-device.

### 4c. Linux (`native/stenocap/src/pulse.rs`)

1. **Enumerate**: `introspect().get_source_info_list()`, dropping every source
   with `monitor_of_sink.is_some()` — those are output monitors, not
   microphones. `name` → ID, `description` → label, `default_source_name` from
   `get_server_info` (already fetched in `device_names()`, `:189`) marks the
   default.
2. **Pin**: `stream.connect_record(Some(id), …)` (`:258`) instead of
   `Tap::device()`'s `@DEFAULT_SOURCE@`; validate against the source list first
   so failure is a clean D4 message rather than a stream that goes `Failed` ten
   seconds later (`:274-287`).
3. Known wart to document, not fix: a Bluetooth headset's source name changes
   with the card profile (A2DP↔HSP), so a pinned BT mic can legitimately
   "vanish". D4's message covers it.

### 4d. Both helpers: the parser (D5) and the fake

`tests/fake_stenocap.py` gains `--list-inputs`, `--mic-device` (resolved
against a fixed fake inventory, with a miss that FATALs), and `--devices`
honouring the pin. Its unknown-arg rejection needs an **explicit whitelist that
includes its value-taking test flags** — `--frames N` (`:98`), `--die-once
MARKER` (`:105`), `--die-after N` (`:113`) — or every test in
`tests/test_capture_transport.py` and `tests/test_capture_windows.py` that
passes a positional value starts failing.

## 5. Step 2 — the Python layer

1. **`settings.py`**: a `CaptureSettings` dataclass (`mic_device: str | None`),
   a `[capture]` table in `SETTINGS_TEMPLATE` (commented — the pristine
   template must still load as exactly `Settings()`, pinned by
   `tests/test_settings.py`), `_capture_from_table`, and the field on
   `Settings`. `reject_unknown` gives the typo guard for free, and
   `_preset_from_table`'s own `reject_unknown` (`:540-571`) already blocks
   `[meetings.*] mic_device` without extra work.
2. **`capture/helper.py`**:
   - `list_input_devices() -> list[InputDevice]` (frozen dataclass: `id`,
     `name`, `is_default`) wrapping `--list-inputs` with the same
     timeout/complaint handling `query_devices` already has (`:149-174`). A
     helper that rejects the flag (old binary, D5) raises
     `CaptureUnavailableError`.
   - `query_devices(channels, *, mic_device=None)` passes the pin through.
   - `HelperCaptureProvider(..., mic_device: str | None = None)` appends
     `--mic-device <id>` in `_spawn` (`:262`); the retry-once respawn path
     inherits it automatically.
3. **`loaders.py`**: `make_provider(..., mic_device=None)` → `_base_provider`
   → the platform branches (`:274-292`). Three signature changes the first
   draft missed: `capture/windows.py:43 default_devices` takes the pin,
   `_native_provider`'s `Callable[[set[Channel]], dict[Channel, str]]`
   (`:302`) widens to match, and **the darwin branch grows a second shape** —
   it returns `HelperCaptureProvider(on_log=on_log)` directly today (`:274-279`)
   and must route through a preflight *when pinned* (default path unchanged:
   no new subprocess, no new TCC surface). `_native_provider` already announces
   `capture: mic ← <name>` (`:319-320`), which becomes the user-visible
   confirmation that the right mic was chosen.
4. **`flow.py`**: `MeetingRequest` gains `mic_device: str | None = None` — not
   `RunOptions`, which is explicitly "the per-run knobs *beyond* a setup form's
   controls" (`:106`). One resolution point,
   `resolve_mic_device(explicit, settings)`: explicit wins, the literal
   `"default"` clears a standing pin for one run, else
   `settings.capture.mic_device`. `resolve_meeting_request` (`:172`) takes the
   form's value; `cli/start.py:284` fills the same field from its flag;
   `MeetingRun.run` (`:338`) passes it to `make_provider`.
5. **`cli/start.py`**: `--mic-device TEXT` (ID or name; `default` = the OS
   default).
6. **New `steno devices`**: prints ID, name and a `(default)` marker, one per
   line — the only way to learn the ID a `settings.toml` needs. (Named for the
   user, not the helper flag; it lists *microphones*, while the helper's
   `--devices` answers a different question. If the collision reads badly in
   use, `steno inputs` is the fallback name.)
7. **`doctor.py`**: `_linux_capture_check` (`:135`) and `_windows_capture_check`
   (`:157`) report the mic device they would use, and a configured-but-absent
   pin is a FAIL naming it. Both call sites pass the pin. **The macOS check is
   gated on M1** — `_capture_helper_check` (`:107`) spawns nothing today, and
   making it spawn `--devices` would grow a microphone prompt inside
   `steno doctor` if enumeration turns out to be TCC-gated.
8. **`settings_rows`** (`settings.py:647`): the `[capture] mic_device` row, so
   `steno settings show` reveals a standing pin.
9. **`.claude/skills/verify/SKILL.md:28-29`** names
   `capture.windows.default_devices` as *the* documented hardware-faking seam;
   its signature changes here, so the skill gets a line (and `list_input_devices`
   joins it as a seam).

## 6. Step 3 — the Qt setup form

- `SetupScreen.opened()` (`gui/screens.py:70`) kicks off
  `self.work(list_input_devices, done=…, failed=…)` — **off the GUI thread**,
  because it spawns a subprocess and `opened()` runs on the event loop.
- **Python orders the list; QML never writes `currentIndex`** (D6). The `done`
  handler builds `[{label, value}]` with the entry that should be selected
  **first**: the standing pin if it is present, otherwise
  `System default (<default name>)`. The default device's own row is suffixed
  `(default)`. A standing pin that is *not* present is emitted first as
  `"<id> — not connected"`, so the form never silently resets to the default
  behind the user's back (D4 at the UI layer).
- **The combo is not rendered until the list lands** (`state.micDevicesReady`).
  It is the app's first combo whose model mutates after first paint, and a
  model swap resets `currentIndex` — hiding it until then means the reset
  always lands on the intended entry and a user pick can never be clobbered.
- Placement: directly under the `Microphone` toggle (`Setup.qml:55`), inside a
  `ColumnLayout` whose `visible: mic.checked` — so the failure hint below has
  somewhere to render. The combo itself carries
  `visible: page.screen.state.micDevices.length > 1`; the hint carries
  `visible: page.screen.state.micError.length > 0`. (The first draft hid the
  hint together with the combo, which meant a listing failure showed nothing at
  all.)
- `start()` passes `"micDevice": micDevice.value || ""` — the `|| ""` guard is
  not optional, `Combo.qml:14` yields `undefined` on an empty model, which is
  why `Setup.qml:181` already writes `preset.value || ""`.
- **Test seam.** `SetupScreen.__init__` calls `opened()` (`screens.py:67`) and
  `Setup.qml:19` calls it again, and the `gui` fixture (`tests/test_gui.py:57`)
  builds the shell for nearly every GUI test — so without a stub, every GUI
  test spawns a real helper two or three times on a daemon thread. The fixture
  must `monkeypatch` `screens.list_input_devices` to a fixed fake inventory,
  with tests for the `done` branch (list renders, ordering correct, pinned
  entry first) and the `failed` branch (hint shows, Start still works).

## 7. Step 4 — gates (each needs hardware; none is a unit test) — THE REMAINING WORK

| # | Platform | What it proves |
|---|---|---|
| G1 | macOS | Two inputs connected; pick the non-default; speak into the chosen one only → transcript has the speech. |
| G2 | macOS | Unplug the pinned device mid-meeting → "mic capture lost … retrying", **no** switch to the built-in mic; re-plug resumes. |
| G3 | macOS | Connect AirPods (a default-*output* switch) while pinned → capture stays on the pinned input. Guards the 2026-07-20 regression. Partly covered already: switching the default output between the built-in speakers and an external monitor mid-run left a pinned mic delivering without a gap (2026-08-12), but AirPods are a Bluetooth *input* too and only real hardware settles it. |
| G4 | Windows | Pin works; preflight names it; an absent ID fails before model load. |
| G5 | Linux | **MEASURED 2026-08-12 — green, after fixing what it found (below).** Pin works on pipewire-pulse; the **system** channel still follows a default-sink change (D1's reason survives). |
| G5r | Linux | **MEASURED 2026-08-12 — the Linux twin of G2, red then green.** A pinned device removed mid-capture must not become a different microphone. |
| G6 | any | **Dual-clock AEC run**: mic on a USB interface, system audio on the built-in output — two independent crystals, which pinning is what makes possible. Run ≥15 min, score ERLE *over time*, and count the framer's gap-correction complaints in the helper's stderr as the primary signal. `frame.rs:180-184` is explicit that slow stamp-vs-device-clock drift accumulates until it crosses `GAP_TOLERANCE_UNITS` and is then corrected **in one step** — a timeline discontinuity AEC3 cannot follow. A one-shot ERLE spot-check cannot see this; the first draft's version of this gate was wrong. |
| G7 | macOS + a virtual device | A pinned aggregate/BlackHole device logs the invalid-host-time warning (§4a.5) rather than silently arrival-stamping. |

### What the Linux gates measured (2026-08-12, CachyOS, PipeWire 1.6.8)

The rig is a `module-null-sink` + `module-remap-source` pair, which yields a
source whose content is exactly known — so *which device was recorded* is
answered by the audio rather than by a log line — plus a second pair carrying a
different tone, so both sides of every switch are positively identified. The
first attempt read silence→tone and was thrown away: a suspended sink's monitor
delivers nothing, so silence→tone is equally the signature of a stream that was
dead all along.

Green: the pin is honoured, shown three ways at once — content (the pinned tone
source at full amplitude, the pinned physical mic and the unpinned default at
the noise floor), the server's own routing table (source-output → pid), and the
device name the server reports back after connect. By id and by description
alike; a name two devices share is refused naming both ids; a bogus, mis-cased
or monitor-named pin is fatal, never a fallback. The system channel still
follows a default-sink change mid-capture — its stream's source moves while the
pinned mic's stays put, with real audio on *both* sides of the switch. An absent
pin fails before models load, proven by call order (the preflight in `flow.py`
precedes `load_backends`) rather than by wall clock, exit 1, remedy named. And
end to end: speech played only into the pinned device is the speech in the
transcript, both mic-only and in the full pinned-mic + system + AEC shape.

**The defect it found.** The pin was connect-time only — `connect_record`
omitted `PA_STREAM_DONT_MOVE`, so the server could re-route the stream, and
removing a pinned device mid-capture silently moved the recording to the
*fallback* source (measured: the fallback's tone in the mic channel one second
later, no diagnostic). That is exactly the failure D4 and §8.1 exist to prevent,
reached with no user action at all. `pulse.rs` now sets `DONT_MOVE` for a pinned
mic **only** — the system tap and an unpinned mic follow the default *by* being
moved, so the flag must never reach them, which is now a unit test rather than a
comment. That alone converts wrong-audio into silent-audio, because losing the
device leaves no trace in the stream API (index, state, `suspended` and device
name all unchanged across the removal — which also rules out
`pa_stream_get_device_index` as the cheaper signal), so a pinned mic delivering
nothing for 5 s is the failure signal; the same removal then yields no wrong
audio and a FATAL naming the device and the remedy.

**The watchdog's own trap, found in review and fixed before it shipped.** A
first version judged silence at the top of the pump loop, which made the pump's
*own* stall indistinguishable from the device's silence: stopping the process
for 6 s declared a live, delivering microphone disconnected. The helper shares
the CLI's process group, so `Ctrl+Z` on `steno start` for five seconds would
have ended the meeting with a false reason. Silence is therefore only counted
while the pump is actually running, a hole counts as liveness (the server
reporting it dropped *this* stream's audio proves the device is there), and the
judgement happens after the drain. Guarded by 30 s of continuous virtual and
25 s of continuous physical pinned capture with no trip, by the stall repro, and
by both follow-the-default behaviours re-measured unchanged.

Bounds worth knowing before trusting it: a trip ends the *helper*, so losing a
pinned mic also ends the system channel and the session finalizes what arrived —
a false trip truncates a meeting rather than losing one, which is what makes 5 s
survivable. The stream never reattaches, so re-plugging does not resume a
meeting (unlike G2's macOS expectation). And the guard is Linux-only by need:
`wasapi.rs` already turns device invalidation into a stream-died error.

Not established, and not to be written up as if it were: two *physical* mics
discriminated by content (the physical arm's audio evidence is the absence of a
tone, so its positive identity rests on the routing table); **NFC matching,
which is untested and untestable here** — every id and description on this
machine is ASCII, and NFD is a macOS behaviour, though trim and case-sensitivity
were checked against real devices; the 5 s window, chosen against a module
unload rather than an unplugged USB interface; and **classic PulseAudio, which
was never exercised** — all of this is pipewire-pulse, while the low-floor
manylinux_2_28 wheel exists for exactly the distros that may still run the real
thing, whose move and suspend semantics are its own. (`pactl suspend-source` is
a no-op here: delivery continued unbroken through a nominal suspension.) The
framer's 30 ms complaint also fired within ~2 s on *every* run, virtual and
physical alike, which is trap 7 arriving on schedule — no timestamp or AEC
conclusion may ride on this rig, and G6 still needs real crystals.

Two smaller defects turned up alongside, recorded rather than fixed because
neither is a gate: a `mic_device` value beginning with `--` reaches the helper's
parser, which exits 2, and `_ask_helper` reports *every* exit 2 as "the capture
helper … is older than this version of stenograf; rebuild it" — wrong diagnosis
and wrong remedy for a settings typo (`steno devices` handles the same value
correctly); and `--devices` answers with the source *id* for an unpinned mic but
the *description* for a pinned one, so `capture: mic ← …` and `steno doctor`
mix vocabularies depending on whether a pin exists — as does the new
disconnected-mid-meeting FATAL, which names the id. One pass should settle which
of the two a user is shown.

One consequence of the fix worth stating, because the picker makes it one
click: pinning the device that is *already* the default is not a no-op. It
changes both the follow behaviour (it stops following) and the liveness policy
(it becomes fatal for that device to go away).

Docs when green: `README.md` (the setting + `steno devices`),
`native/README.md` (the new flags), `.claude/skills/verify/SKILL.md` (§5.9).

## 8. Traps and risks

1. **Silent fallback is the whole bug** (D4/D5). Review anything that turns
   "not found" into "use the default".
2. **macOS format caching** — reading `inputFormat` before pinning yields the
   *old* device's format and the resampler is then wrong; the symptom is
   garbled or silent audio, not an error.
3. **Re-pinning on every rebuild** — `AVAudioEngine` reconstructs its input
   node on a configuration change; a pin set once at startup is lost.
4. **`GetId()` leaks** without `CoTaskMemFree`.
5. **JSON escaping**: `escape()` (`main.rs:228`) covers `"` and `\` only.
   Device names are driver-supplied; a control character would produce invalid
   JSON. Harden it while adding the array output.
6. **Name matching is ambiguity-prone** (D3) — two identical USB mics, and
   Windows' mutable `2-` index prefix. Erroring is correct; the error names
   both IDs.
7. **Virtual devices break the clock story** (§4a.5): pinning is what makes
   BlackHole/Loopback/VB-Cable selectable, and their host times are often
   invalid. This is the one way this feature can silently reintroduce arrival
   stamping.
8. **The GUI listing spawns a helper on every setup-screen visit** — cheap, but
   it is a subprocess on a UI path; keep it on `Screen.work`, and stub it in
   tests (§6).

## 9. Explicitly out of scope — and what needs no change

Out of scope: output/system-audio device selection (D1); writing
`settings.toml` from the GUI (O1); mid-meeting device switching from the UI;
per-preset pinning (`[meetings.*] mic_device`).

Checked and needing **no** change: `hatch_build.py` and the wheel matrix (no
new binaries, no new deps), CI, `aec.py`, the session/recorder layer, the
replay/file providers, `gui/wintray.py`, and the frame protocol itself.

## 10. Questions, as decided 2026-08-12

- **O1 — should the GUI persist the choice? DECIDED: no, option (a).** Ship
  the per-run picker plus the hand-edited standing key; revisit only if the
  friction is real, in which case (b) is the cheap upgrade. Three options were
  on the table:
  (a) hand-edited `settings.toml` only (this plan);
  (b) a GUI-owned `ui-state.json` in `data_dir()`, written with the
  `atomic_write_text` the repo already uses (`voiceprints.py:180-187`,
  `settings.py:643`) — **no new dependency, no settings.toml write, no comment
  round-trip, and `settings.toml` stays user-owned**;
  (c) writing `settings.toml` itself (a `tomlkit` dependency or a line
  patcher). The first draft posed this as (a)-vs-(c) and recommended (a) on a
  cost that isn't real. (b) is the cheap path, and what it actually costs is a
  **precedence rule** — where a GUI-remembered pick sits relative to a
  `settings.toml` pin and a CLI flag — which is the question to answer before
  building it, not now.
- **O2 — should the macOS default path get the `--devices` preflight too?**
  M1 removed the objection (enumeration is TCC-clean, §0), so this is now a
  pure cost question: one extra subprocess per meeting start on macOS, in
  exchange for all three platforms announcing `capture: mic ← <name>` and
  failing identically on a broken audio stack. **Still deferred** — §5.3 ships
  the pinned-only preflight, and the always-on version is a one-line follow-up
  once the startup cost is measured against the meeting-start budget.
- **O3 — should a *standing file* pin be advisory rather than fatal? DECIDED:
  no.** Fatal in both cases; the message names the remedy (D4). Folded into D4;
  no open question remains.
