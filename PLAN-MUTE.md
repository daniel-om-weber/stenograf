# PLAN-MUTE — muting the microphone or system audio during a meeting

Status: **steps 0–3 BUILT 2026-11-16 (M1 measured, library, meeting screen,
menu bar); the hardware gates in §8 remain.** M1 killed the cheap clock, so
the capture layer now reports the time it is capturing (`CaptureProvider.now`).
Scope: while a meeting records,
either channel — the microphone or the system audio — can be muted and unmuted
at any time, from the meeting screen and from the menu-bar icon.

Decisions Daniel took 2026-11-16: **the transcript marks each muted span** (Q1,
as recommended); **no keyboard shortcut for now** — the UI only, the shortcut
research is kept in §7 for later; **no retroactive cut**, a mute acts from the
moment it is pressed. Muted audio never reaches memory, the transcript,
the notes or `audio.opus`; the transcript says where the gap is. The motivating
case (Daniel, 2026-11-16): a phone call comes in mid-meeting and must not end up
in the transcript.

"Pause" is not a separate feature: it is both channels muted.

## 1. What exists today (read from the code, 2026-11-16)

- Nothing mutes anything. No mute, pause or hold exists in `session.py`,
  `flow.py`, the GUI or the CLI.
- **Every captured frame passes one function**, `_consume_frame`
  (`src/stenograf/session.py`), shared by the live and batch loops. It appends
  the frame to the in-RAM `SessionStore`, hands it to `on_frame` (the
  `--record-audio` tee, `recording.AudioTee`) and advances the live pass's
  `AudioBus`. The live decoder, the crash checkpoints and the on-stop finalize
  all read the store, never the provider.
- **The echo canceller sits upstream of that seam**: `loaders.make_provider`
  wraps the capture provider in `aec.py`'s provider, which pairs mic and system
  frames by timestamp and needs the *real* system audio as its far-end
  reference. Everything after `_consume_frame` sees cancelled audio.
- **Frame timestamps are seconds since capture start on one clock shared by
  both channels** (`capture/helper.py`'s frame format). The store places frames
  by timestamp and pads gaps with zeros, so the session clock is the
  transcript's clock.
- **Capture starts before the models load** (`MeetingRun.run`: `provider.start`
  then `load_backends`), and the provider queue buffers the first ~4–8 s
  meanwhile. So at the start of a meeting, frames are *consumed* several
  seconds after they were *captured*. This is why D3 below exists.
- The GUI keeps the `MeetingRun` (`MeetingScreen._run`) and already calls
  methods on it from the GUI thread (`name_speakers`, `settle_names`); the
  menu-bar icon (`gui/tray.py`) reads the meeting screen's state and has
  `idle`/`rec`/`busy` icon tints.
- The notes prompt renders entries itself (`notes/prompt.py:_render_entries`),
  not through `Transcript.to_markdown`.

## 2. Decisions, with reasons

**D1 — Muting writes silence; it never stops capture.** Stopping and restarting
capture is the fragile operation in this codebase (CoreAudio wedges under a
concurrent capture, device-switch rebuilds; see the capture-helper history).
Zeros keep the session clock continuous, so the two channels still interleave,
the AEC keeps its pairing, and nothing downstream needs to know about gaps.

**D2 — The mute applies at the `_consume_frame` seam, after the echo
canceller.** Muting the system channel *before* the canceller would take away
its reference, and the remote voices leaving the speakers would pass straight
into the mic channel. At the seam, the canceller works as it does today and the
store, the tee and the live pass all receive the same silence. The native
helpers are untouched.

**D3 — A mute is a span on the session clock, not a state applied to whatever
frame is consumed next.** Applying "muted now" to the next consumed frame breaks
in the start-up backlog: mute 2 s into the meeting, take a short call, unmute at
6 s, and the models finish loading at 8 s — every buffered frame, call
included, is consumed with the switch already back on *unmuted* and is kept.
So a press is stamped with the current session time and recorded as a span;
`_consume_frame` zeroes the samples of each frame that fall inside a span,
sample-precise, whenever the frame happens to be consumed.

**D4 — Spans live beside the entries, not as fake entries.** A
`TranscriptEntry` with an empty speaker would leak into diarization counts,
speaker lists, "Who spoke?", re-ID, echo dedup and the notes chunker. The
transcript gets a separate `muted: [{channel, start, end}]` field, and the
renderers that show time (markdown, the notes prompt) insert one line per span
in time order: `*(microphone muted 10:32–10:41)*`. Plain text, SRT and VTT stay
as they are. No schema-version bump: `from_json` reads the field with a default,
and older readers ignore an unknown key (to be confirmed by a round-trip test,
step 1).

**D5 — The library owns the switch; the front-ends only call it.**
`MeetingRun.mute(channel, on)` and `MeetingRun.muted` (the current state per
channel), usable from any thread and before capture is up (a press during model
loading must count — D3). The GUI header and the tray both call the
same method, so the two can never disagree. The CLI gets no control in this
plan (§9), but the API is there for it.

**D6 — Every meeting starts unmuted, and nothing is persisted.** A mute that
carried over to the next meeting would silently lose a whole meeting of the
user's own voice.

**D7 — In the app, a source switch means "live at the start", not "recorded"
(Daniel, 2026-11-16).** The start form always records both sources; one
switched off starts muted from the first sample (`MeetingRequest.start_muted`),
so the meeting can switch it on later. Consequence, chosen over an automatic
fallback: both captures must work. A denied system-audio permission or a
missing output device fails the meeting at start, with a message that says the
switched-off source is still recorded (`MeetingRun._explain_capture_failure`);
the app no longer offers a mic-only capture. The CLI is unchanged —
`--local 0` / `--remote 0` still leave a channel out, since it has no control
to switch one on.

## 3. Step 0 — measure the clock anchor, BEFORE any code

D3 needs "the session time right now" at the moment of a press. Two candidates:

- **Wall anchor**: `time.monotonic() − started`, where `started` is what
  `MeetingRun.run` already records right after `provider.start`. Free, but wrong
  by however long the helper takes from spawn to its own t=0 (the macOS tap
  setup is the suspect). An error in that direction leaves audio *after* the
  press unmuted, which is the privacy-wrong direction.
- **Arrival anchor**: the provider's frame reader (the helper's drainer thread)
  records the arrival time of the newest frame it read; "now" is that frame's
  timestamp plus the time since it arrived. Exact to a frame, but it touches the
  capture layer (and the AEC wrapper must pass it through). Read
  `git log --follow -p PLAN-CAPTURE-HELPER.md` first.

**M1**: on this Mac, log the first frame's arrival time against `started`
across ~10 meeting starts, with and without system audio. The test that would
prove the cheap option wrong: an offset above 150 ms in any start. Then the
arrival anchor is built; otherwise the wall anchor ships, with its measured
offset written next to it. Windows and Linux are checked in the step-4 gates.

**M1 result (measured 2026-11-16, macOS 27.2, this Mac, `eval`-free scratch
script over ten starts):** per frame, *(arrival − `started`) − (timestamp +
duration)*. Warm starts: 26–41 ms (min), 35–50 ms (median), mic-only and with
system audio alike. **One cold start (the first of the ten): 1.85 s for the
whole run** — the helper's t=0 came 1.8 s after `provider.start` returned.
The falsifier fired, so the wall anchor is out: it would have left ~1.8 s after
a press unmuted. Built instead: the helper's drainer stamps each frame's
arrival, `HelperCaptureProvider.now()` = newest frame end + time since it
arrived (short of the truth by the ~35 ms pipe latency, which mutes slightly
early), `EchoCancellingProvider.now()` passes the inner clock through, paced
replay reports its own wall clock, and providers without a clock return `None`
— `MuteSwitch` then falls back to how far consumption has got. The capture
transport itself is unchanged.
Note: file replay (`FileCaptureProvider`, paced) starts its clock when frames are
first *iterated*, i.e. after model load, so replay-based tests must use the same
anchor the real provider uses, or they test a different clock.

## 4. Step 1 — the library

- `session.py`: a small `MuteSwitch` — per-channel state plus the list of spans
  (open span = muted until now), guarded by a lock, written from any thread,
  read by the capture thread. `apply(frame) -> frame` returns the frame with
  every sample inside a span replaced by zeros (a copy; frames are shared with
  the tee). Called at the top of `_consume_frame`, before `store.append` and
  `on_frame`.
- `MeetingRecorder.run(..., mute=None)` threads it to both capture loops; the
  finalized `Transcript` carries the closed spans (an open span closes at the
  stop's session time).
- `MeetingRun` constructs the switch in `__init__` (so it exists before the run
  starts) and exposes `mute()` / `muted`.
- `Transcript.muted`, JSON round-trip, markdown rendering; the notes prompt adds
  the same line in time order.
- Live captions: not touched. The live decoder sees the silence, so the open
  caption line closes on its own at the next idle flush (the 1 Hz tick), the
  same way it does after anyone stops talking.

Tests (offline, file replay):
- A span zeroes exactly its samples in the store and in the tee's output; a
  frame straddling a span boundary is partly zeroed.
- **The backlog case from D3**: mute and unmute while frames are still queued
  (before consumption starts); the span is silent anyway. This is the test the
  "next-frame" design fails.
- Muting the system channel leaves the AEC's reference intact (a replay with
  speaker echo in the mic channel: the mic's cancelled output is the same with
  and without a system-channel mute).
- `Transcript.from_json(t.to_json()) == t` with spans; an old transcript without
  the field loads; the markdown and the notes prompt show the line where it
  belongs.
- A meeting that ends while muted closes its span at the stop.

Built as listed, in `tests/test_mute.py` (plus the helper clock in
`tests/test_capture_transport.py`). The backlog test was checked against the
"next-frame" design swapped in: it fails there, as it must.

## 5. Step 2 — the meeting screen

- Two toggles in the header, one per channel the meeting actually records
  (a mic-only meeting shows one): "Mute mic" / "Mute system audio" (then
  "Unmute …"), enabled from the moment capture is up — model loading included.
- A muted channel is unmistakable: a grey `MIC MUTED` / `SYSTEM AUDIO MUTED` /
  `MIC + SYSTEM MUTED` badge (`Theme.off`) in
  the header next to `REC`. Forgetting to unmute is the main way this feature
  loses data (§8), so the state must be readable at a glance, not only from the
  toggle's position.
- The toggles are event-driven; nothing animates (the redraw budget in
  `gui/app.py`'s docstring).
- Tests in `tests/test_gui.py` (offscreen harness): a toggle reaches
  `MeetingRun.mute`, the badge follows, the finished transcript has the span.
  Screenshot via `grabWindow()` at the minimum window width.

## 6. Step 3 — the menu-bar icon

- While a meeting records, the menu gets two checkable entries, "Mute
  microphone" and "Mute system audio", calling the same `MeetingRun.mute`.
- The summary line reads `Recording · 12:34 · mic muted`, and the icon gets a
  muted variant (a fourth tint beside `idle`/`rec`/`busy`), so a hidden window
  still shows the state. On macOS the icon stays a template image.
- Labels are set when the menu opens (`_relabel`), as today; the icon follows the
  meeting's *state* through `_refresh`, never its clock.

## 7. Deferred — a system-wide shortcut (not built; Daniel 2026-11-16)

Kept for when it is wanted. One shortcut toggles the **microphone** (the phone-call case); system audio has
none by default. Desk research plus a ctypes prototype on this Mac, 2026-11-16.
Labels: *measured* = run here, *docs* = official documentation or source,
*secondary* = credible third party, *inferred* = reasoning only.

**No platform needs a new dependency or a change to the frozen bundle.** Qt 6
has no global-shortcut API (*secondary* — QHotkey exists because it lacks one),
so each platform gets its own small module, the way `gui/wintray.py` already
does for the status icon.

- **macOS — Carbon `RegisterEventHotKey` through ctypes.** Registered and fired
  inside a running PySide6 event loop on macOS 27.2.1 (*measured*); Qt's Cocoa
  loop delivers the event, so registration must happen on the GUI thread, and
  nothing fires without that loop (*measured*). No permission prompt and no
  Info.plist key (*secondary*: sindresorhus/KeyboardShortcuts, built on the same
  call, runs sandboxed in the App Store and says so). The prototype ran under a
  process already trusted for Accessibility, so it **cannot** prove the
  no-prompt claim — a gate must launch from `Stenograf.app` on a fresh grant.
  Rejected: `NSEvent` global monitors (need Accessibility, *docs*), `CGEventTap`
  and pynput (need Input Monitoring, *docs*). macOS 15.0 refused hotkeys whose
  only modifiers are Option/Shift and 15.2 relaxed it (*docs*); a default with
  Ctrl avoids the question.
- **Windows — `RegisterHotKey` with `MOD_NOREPEAT`** (user32 via ctypes, no
  prompt, *docs*). It needs a window created on the GUI thread: a separate hidden
  window built like `wintray._make_window`, not the tray icon's own, so the
  shortcut does not depend on the icon existing (*inferred*). A combination held
  by another app fails with `ERROR_HOTKEY_ALREADY_REGISTERED` (1409) and must be
  reported, not swallowed.
- **Linux Wayland — the `org.freedesktop.portal.GlobalShortcuts` portal over
  QtDBus** (QtDBus ships in PySide6-Essentials: *measured* on the macOS wheel,
  *inferred* for Linux). KDE Plasma ≥ 5.27 and GNOME ≥ 48 implement it (*docs*);
  Sway and older GNOME do not. The desktop shows its own confirmation dialog the
  first time; that cannot be avoided (*docs*). Trap: for an app that is not a
  Flatpak, the portal derives the app identity from the systemd unit name and a
  terminal launch gets none; `org.freedesktop.host.portal.Registry.Register`
  before the first portal call is the documented fix (*secondary*, not verified
  on either desktop — check on Daniel's CachyOS notebook first).
- **Linux X11 — `XGrabKey` through ctypes on libX11** (*inferred*, not
  prototyped): one grab per NumLock/CapsLock combination, and an X error handler,
  because a combination taken by another client fails asynchronously. Useless
  under XWayland. Built only if the portal turns out to be missing on an X11
  session that matters (KDE's X11 session may serve the portal too — open).

**Default: Ctrl+Option+Shift+M on macOS, Ctrl+Alt+Shift+M elsewhere**
(*inferred*). It avoids every meeting app's own mute key — Teams
Ctrl/Cmd+Shift+M, Discord Ctrl+Shift+M, Webex Ctrl+M, Zoom Alt+A / Cmd+Shift+A,
Meet Ctrl/Cmd+D — which a system-wide grab would steal from the app being muted;
it contains Ctrl, so even macOS 15.0's rule allows it; and the Shift keeps it off
AltGr+M on a German keyboard (Windows reads AltGr as Ctrl+Alt, so a plain
Ctrl+Alt+M would swallow every typed µ). No KDE or GNOME default is known to use
it; not checked on real desktops.

Shape of the step:
- Settable in `settings.toml` (`[shortcuts] mute_microphone = "ctrl+alt+shift+m"`,
  empty = off) and shown on the Settings screen; flow reads it, the library
  never does (settings charter).
- Registered when a meeting starts recording and released when it stops, on
  macOS, Windows and X11 — a combination held all day for a feature used only in
  meetings is a collision waiting to happen. On Wayland the portal session is
  created at the first meeting and kept for the app's lifetime, because a session
  binds only once and each new one may ask again; presses outside a meeting are
  ignored.
- A press while the window is hidden must be acknowledged: a short
  notification through the status icon ("Microphone muted" / "unmuted") on top
  of the icon tint from step 3.
- A registration that fails (combination taken, no portal) is said once on the
  meeting screen's status line, naming the setting that changes it.

## 8. Step 4 — gates (need hardware; none is a unit test)

- **G1, macOS, real meeting**: during a call, mute the mic, speak a sentinel
  phrase, unmute. The phrase is in neither the transcript, the notes nor
  `audio.opus`; the transcript shows the span; speech right after unmuting is
  transcribed.
- **G2, mute during model loading** on a cold start: the D3 case on the real
  helper.
- **G3, system audio muted while the speakers play**: no remote line leaks into
  the mic channel (the canceller still has its reference).
- **G4, Windows and Linux**: G1 on each, plus the M1 anchor offset.

## 9. Traps and risks

- **Forgetting to unmute** loses the user's own voice for the rest of the
  meeting. Mitigation in this plan: the header badge and the tray icon variant.
  A "you are speaking while muted" hint (VAD on the pre-mute audio) is possible
  but costs power during every muted second; deferred until a forgotten mute is
  actually reported.
- **Echo dedup with system audio muted**: the cross-channel text dedup has no
  system text to match during the span. It only arms when the canceller reports
  a lost reference, and the canceller keeps its reference under D2, so the
  exposure should be nil (inferred, checked by G3).
- **The crash checkpoint** (`transcript.partial.*`) is built from the live
  decoders, which only ever see the zeroed store; no extra handling needed.
- **Voice samples for "Who spoke?"** are cut from speaker turns, which never lie
  in a zeroed span; no change.

## 10. Out of scope

- **Retroactive cut ("drop the last 30 seconds")**, for the case where muting
  came too late. It needs to rewrite audio the store already holds, whose chunks
  are deliberately immutable for lock-free readers, and to retract words the live
  pass already committed. That is its own design; this plan's span field and
  markdown line are what it would reuse.
- **CLI control.** `steno start` streams plain captions and reads no keys. The
  library API makes it a small addition; reopen when someone asks for it.
- **A remembered mute state** between meetings (D6).

## 11. Questions — decided 2026-11-16

- **Q1 — the transcript line:** yes, `*(microphone muted 10:32–10:41)*` in the
  markdown transcript and the notes prompt.
- **Q2 — the shortcut:** not now (§7).
- **Q3 — order:** M1, then steps 1–3, then the gates.
