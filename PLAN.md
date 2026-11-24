# stenograf — open work, gates, and what was declined

This file holds three things only: **open work** (needs code or a measurement),
**gates** (needs real hardware or Daniel's eyes, no code), and **deferred and
declined** ideas, kept so they are not re-proposed. Everything shipped is in
git history: this file's own (`git log --follow -p PLAN.md` — the GUI build,
notes and presets, the Windows and Linux platform work, the diarization
program, all pruned 2026-11-24) and the deleted plan files, each retrievable
with `git log --follow -p <file>`: `PLAN-AEC.md`, `PLAN-CLEANUP.md`,
`PLAN-GUI-DEFAULT.md`, `PLAN-LINUX.md`, `PLAN-WINDOWS.md`,
`PLAN-ASR-CHALLENGER.md`, `PLAN-NOTES-MARKDOWN.md`, `PLAN-MEETING-PRESETS.md`,
`PLAN-CAPTURE-HELPER.md`, `PLAN-DIARIZATION.md`, `PLAN-DIARIZATION-SPEED.md`,
`PLAN-NEMOTRON.md`, `PLAN-FINISH.md`, `PLAN-MUTE.md`, `PLAN-MIC-DEVICE.md` and
`PLAN-UPDATE.md`. Measured evidence for every shipped default is in
`eval/README.md`; design reasons live in the code's docstrings.

---

## Open work

1. **One stop gesture for the capture helpers (macOS-gated).** The last
   per-platform difference in `capture/helper.py` is `_stop_signal`: SIGINT
   for the Swift helper, stdin EOF for the Rust one. Teach
   `native/stenocap-macos/main.swift` to exit on stdin EOF (mirror the stdin
   watcher in `native/stenocap/src/main.rs`), then delete `_stop_signal` and
   the signal arm of `_request_exit`. The helper is not inside the frozen app
   bundle, so this is allowed, but it recompiles and re-signs a signed binary:
   do it in a real macOS session and finish with a live-capture meeting.

2. **Stop holding the whole meeting audio in RAM (deferred 2026-11-16,
   Daniel's call).** `SessionStore` (`session.py`) keeps every channel's
   audio until the stop (~115 MB/hour/channel, never on disk). The motive is
   privacy — the voice should exist as briefly as possible. Already done:
   diarization and voiceprint collection run during the meeting. Still to
   answer: Nemotron's streaming-mode accuracy on held-out AMI against the
   offline 97.8 % (`eval/README.md`), and what the fallbacks (live pass off
   or failed) do without audio — they would lose speaker separation. The
   finish card's voice samples survive if cut during the meeting.

3. **Memory on 8–16 GB machines — unmeasured, and now the common case.**
   Diarization is on by default since 0.8.1. On the M4 Max a 1 h 45 min
   meeting's finalize peaks at 7.9 GB with diarization (6.9 GB without;
   `eval/README.md`, "A real finalize, and its peak memory"), and the voice
   model's session sits at 860 MB for the whole meeting, measured alone, not
   beside the live pipeline. Measure both on a small machine; the way out
   today is `[speakers] diarization = false`.

4. **What the ReDimNet2 voice model left** (`eval/README.md`, "Speaker-embedding
   models in the naming seat" and after). (a) **Energy** per minute of speech:
   `eval/embedders/energy.sh` with `shipped.py`, background QoS and
   foreground, on a quiet M4 Max; it decides whether the 120 s speech budget
   stays (60 s names as well, narrower margin) and whether the smaller B3 is
   worth its saving. (b) **A solo channel is still embedded at the stop**
   (`session.py` `_name_solo_channel`): with a stated count of one no
   diarizer runs, so nothing collects during the meeting. (c) **A
   fixed-shape path** (GPU, CoreML) would pad or cut spans to a few lengths,
   changing what a voiceprint is for short clusters; score it only if (a)
   says the CPU path costs too much.

5. **WebGPU as an optional extra for Nemotron — its re-open trigger has
   fired** ("once the CPU path has shipped in a release": 0.8.0). CPU runs
   15 s per audio-hour on the M4 Max; on a GPU the fp16 export is faster (x86
   iGPU: 8.3 vs 14 s per 37.6-min channel). Trap: the `onnxruntime-ep-webgpu`
   plugin loads a second onnxruntime beside the wheel's on macOS and
   segfaulted on a second session in one process — re-test that first.

6. **livekit re-ask — its trigger has fired** (both channels device-stamped
   by one helper on every platform, since 2026-08-02). `livekit` (25.6 MB)
   serves one class, AEC3, in `aec.py`. The idea: run AEC3 inside the helpers
   so channel pairing never crosses a language boundary and `aec.py`'s
   timestamp machinery disappears. Judge on all three platforms at once
   against the shipped ERLE; reasoning in
   `git log --follow -p PLAN-CAPTURE-HELPER.md`.

7. **System-wide mute shortcut (deferred by Daniel 2026-11-16).** One
   shortcut toggles the microphone. Qt has no global-shortcut API, so one
   small ctypes module per platform, like `gui/wintray.py`; no dependency or
   bundle change. macOS: Carbon `RegisterEventHotKey`, fires inside the
   PySide6 loop on the GUI thread (measured); "no permission prompt" is
   unproven — check from `Stenograf.app` on a fresh grant (`NSEvent`
   monitors, `CGEventTap` and pynput all need a privacy grant). Windows:
   `RegisterHotKey` + `MOD_NOREPEAT` on its own hidden window; report error
   1409 (combination taken). Wayland: the `GlobalShortcuts` portal over QtDBus
   (KDE ≥ 5.27, GNOME ≥ 48; a non-Flatpak app must call
   `org.freedesktop.host.portal.Registry.Register` first — unverified).
   Default Ctrl+Option+Shift+M / Ctrl+Alt+Shift+M: no meeting app's own mute
   key, and Shift keeps it off AltGr+M (µ) on German keyboards. Shape:
   `[shortcuts] mute_microphone` (empty = off), registered only while a
   meeting records, a notification when the window is hidden.

8. **Repair-only overhang — the one open accuracy idea.** Decode stays the
   exact current slice; overhang fires only on a detected ≥1.5 s speech hole.
   It is the salvage from the reverted cut-overlap work, immune to the
   language-flip sites that killed the always-on version. The VAD threshold
   0.4 re-attempt is blocked on it (its known cost is more forced mid-speech
   cuts); a ~0.5 s always-on left overhang is a separate candidate. Any of
   these re-runs the full battery against a control worktree.

9. **Ollama's response budget on the longest answer.** The reduce step of a
   long meeting is the longest output the notes produce and was never
   measured against the `num_predict` limit `notes/ollama.py` now sends.

---

## Gates

Real hardware or Daniel's observation, no code. "Open" means never run.

| Feature | Platform | What to observe | Status |
|---|---|---|---|
| GUI power, watt half | macOS, quiet machine | Absolute watt delta, app open vs closed, against the pipeline's ~0.6 W; also wakeups/s with the window hidden in tray mode | Deferred since 2026-07-25 (needs an idle machine) |
| Bare `steno` over SSH | macOS | Prints help, opens no window (the display heuristic's bet) | Optional; needs Remote Login enabled |
| Notes equivalence (Gate A) | any | Daniel reads the before/after note pairs in `gate-a/` (gitignored, real meetings) | Captured; read pending |
| Meeting presets in use | this Mac | Are any `[meetings.*]` sections written after weeks of use? If not, reconsider the preset layer, not the picker | Open since 2026-07-31 |
| Windows echo cancellation | Windows (GPD) | Helper's ERLE ≥ 13.7 dB with 0 leaked lines (the deleted lag constant's score); double-talk; why 13.7 vs macOS 37.6 dB | Owed since 2026-08-02 |
| DirectML fails loudly | Windows (GPD) | A healthy `[asr] provider = "dml"` run transcribes; `STENOGRAF_ASR_PROVIDER=cuda` on the DirectML build exits with a clean error naming `[asr] provider = "cpu"` | Unrecorded, overdue (due "before the next release" 2026-08-02) |
| `.lnk`-only launcher | Windows (GPD) | `steno setup` lands both `Stenograf.lnk` links (Start Menu, Desktop) and both start the app | Unrecorded, overdue (same) |
| Shared room mic | any | First real hybrid meetings: people merged into one speaker, crosstalk phantoms, a profile naming the wrong person | Open since 2026-11-15 |
| Finish card | macOS | A real meeting with a wrong name match corrected on the card | Open since 2026-11-16 |
| Mute G1 | macOS | During a call: mute mic, say a sentinel phrase, unmute. Phrase absent from transcript, notes and `audio.opus`; span shown; speech right after unmuting kept | Open |
| Mute G2 | macOS | Mute during model loading on a cold start: the backlog case on the real helper | Open |
| Mute G3 | macOS | System audio muted while speakers play: no remote line leaks into the mic channel | Open |
| Mute G4 | Windows, Linux | G1 on each, plus the mute clock's offset (pipe latency ~30–40 ms on macOS) | Open |
| Mic picker G1 | macOS | Two inputs; pick the non-default; speech into it alone is in the transcript | Open |
| Mic picker G2 | macOS | Unplug the pinned device mid-meeting: "retrying", no switch to the built-in mic; re-plug resumes | Open |
| Mic picker G3 | macOS | Connect AirPods while pinned: capture stays on the pinned input (switching speakers↔monitor already passed, 2026-08-12) | Partly measured |
| Mic picker G4 | Windows | Pin works; preflight names it; an absent ID fails before model load | Open |
| Mic picker G5, G5r | Linux | Pin honoured on pipewire-pulse; system channel still follows the default sink; a removed pinned device never becomes another mic | Measured green 2026-08-12 (CachyOS) |
| Mic picker G6 | any | Dual-clock echo run: mic on USB, system audio built-in, ≥ 15 min, ERLE over time | Open |
| Mic picker G7 | macOS | A pinned virtual device (BlackHole, Teams Audio) logs the invalid-host-time warning | Open |
| Device-ID stability (M2) | all | IDs unchanged after a reboot and after re-plugging a USB mic into another port | macOS static IDs: proven by construction. Linux reboot: measured green 2026-08-12; re-plug, macOS UUID IDs, Windows: open |
| Mic pin on classic PulseAudio | Linux | `DONT_MOVE`, device removal and the 5 s watchdog on real PulseAudio (CI only live-captures there, unpinned) | Open |
| Update, macOS | macOS | Update from the app in `~/Applications`: first meeting needs no new mic prompt; no alert when the old app exits; same with the bundle moved | Open |
| Update, Windows | Windows | No console window; tray keeps its place | Measured green 2026-11-20 |
| Update, Windows extras | Windows | A pinned taskbar icon still starts the app; installer's `--force` retry and `install.yml`'s step | Open (no pinned item existed) |
| Update, Linux | Linux desktop | Menu entry still starts the app; restarted window carries its icon | Open |
| Update, broken releases | all, real machines | A release failing its self-check, one crashing at start, one whose install is killed: each ends in the old version with the failure line; after the third with the updater killed too, the install command repairs | Crash-at-start covered in CI; the rest open |
| Update lock on network home | Linux | Shared/exclusive lock behaves as locally | Open (no such machine) |

Recipes and traps:

- **GUI watt half:** on battery, several minutes per arm, app-open and
  app-closed interleaved; Spotlight/SeaDrive indexing put the noise floor at
  446–795 mW, the size of the quantity measured. Use `powermetrics --samplers
  tasks` for per-process wakeups (`sudo -n` works). An empty
  `ApplicationWindow` is not a valid control (no scene graph, so it never
  exercises the render loop), and `sample` at 1 ms cannot see ~17 µs wakes.
- **Windows echo run:** `uv run python eval/aec_rig.py far-only --seconds 1800
  --volume 90`, then `eval/aec_alignment.py` on the dump. Plain `uv run`,
  never `--group eval` (mlx has no win_amd64 wheel). Half an hour of speech
  aloud, so an empty room; the rig pins an en-US SAPI voice because the system
  default is German; 40 % volume puts no echo into the mic. Read the Windows
  observation recipe in `git log --follow -p PLAN-WINDOWS.md` (last section)
  first. A Linux ERLE run is a nice-to-have alongside, not its own gate.
- **Mic picker G6:** the primary signal is the framer's gap-correction
  complaints in the helper's stderr — drift is corrected in one step
  (`native/stenocap/src/frame.rs`), which AEC3 cannot follow and a one-shot
  ERLE check cannot see. The Linux null-sink rig carries no timestamp claim.
- **M2:** `steno devices` lists IDs; the Linux harness is
  `~/.local/share/steno-m2/` on the CachyOS notebook (`snapshot.sh LABEL`,
  then `diff`). NFC/NFD name matching is untested anywhere.
- **Update, macOS:** also look at a release that dies at start through the
  launcher: the stub shows its own alert and stays alive, so the rollback's
  second `open` may only bring that alert forward (inferred, not observed).
  Never try an update against the machine's own install.

---

## Deferred and declined

Kept so future sessions do not re-open them. Each says why, and what would
re-open it.

**Speech recognition and accuracy**

- **Challenger ASR models — declined 2026-07-27.** Word timestamps are
  mandatory (`asr/base.py`; speaker assignment needs them), and every
  leaderboard leader so far lost to Parakeet on our meeting audio (Voxtral
  Small 24B 32:38, Canary-1B-v2 ~1:2, blind, de+en). Re-open: a leader that
  emits word timestamps — a `BackendSpec` in `asr/registry.py`
  (`git log --follow -p PLAN-ASR-CHALLENGER.md`).
- **Never move VAD window bounds.** Both bound-moving fixes were tried and
  reverted; only the decode slice may change. A detector swap moves every
  bound and is judged on what it buys against that churn (Silero v6 passed
  2026-11-19, `eval/README.md`, "Speech detector").
- **Lower priority, independent:** greedy re-ID → Hungarian assignment;
  SRT/VTT dropping text not covered by `words` (latent: Parakeet emits
  full-or-none); meeting-mode auto-detect; hybrid cross-channel dedup;
  acoustic first-segment language ID for the live pass; re-running the
  winning biasing config over the full test set.

**Diarization and speaker naming** (read `eval/diarization-sota-2026.md` and
`eval/README.md`'s corpus-harness and Nemotron sections first)

- **Hand-labelled references of our own audio — declined.** The AMI/ICSI
  harness (`eval/ami.py`) builds them from public corpora in our two-channel
  topology; the 0.62 naming threshold on our own audio stays unmeasured.
- **The owned diarization loop — closed 2026-08-09, replaced by Nemotron 3
  2026-11-15;** its PLDA + VBx fallback is closed with it
  (`git log --follow -p PLAN-DIARIZATION.md`).
- **DiariZen meeting-base (2026-08-09):** lost on the fair frame (+1.3 mean
  paired ΔDER, wins 4/20, sign p≈0.012). Re-open: the meeting mix shifts to
  many-speaker rooms (it won all four ICSI k=4–8 loops) — and it must now
  beat Nemotron 3.
- **Other embedding models (2026-08-07, naming seat 2026-11-18):**
  ERes2NetV2-zh and ResNet34-LM regressed; ERes2Net-large, TitaNet-L and
  w2v-BERT 2.0 SV trail ReDimNet2 (`eval/README.md`). No trigger.
- **Automatic profile updates:** ungateable wrong updates while the oracle
  ties no-update; growth stays user-confirmed. Re-run `eval/auto_update.py`
  and `eval/threshold_pick.py` in any future diarization change's gate.
- **Sortformer** (superseded by Nemotron 3); **joint/LLM SA-ASR** (re-open: a
  permissive, CPU-practical joint model beating the cascade); **TS-VAD**
  (re-open: an Apache/MIT ONNX implementation — nomo-pvad examined
  2026-11-15, not pursued); **AS-Norm** (degraded the strongest models);
  **EEND-TA** (re-open: a release); **pyannote.audio as a dependency**
  (opt-out telemetry, paid-SDK pull; weights may be extracted).
- **Custom GPU kernels and second inference engines — declined 2026-11-13**
  (WGSL/wgpu, IREE, Burn, ncnn measured; ORT's WebGPU plugin reaches a GPU on
  the engine already shipped; Triton has no Vulkan/D3D12 backend).
- **Speaker labels in live captions:** out of scope; captions are
  channel-coarse by design.

**Notes and presets**

- **A directory of preset *files* — rejected.** `[notes] command` runs
  unattended after every meeting, so a shared checkout is arbitrary code
  execution, and `git pull` must never change which commands run. If sharing
  becomes real: a directory loaded *without* executable keys unless
  settings.toml allowlists them.
- **A preset-name stamp in `MeetingProfile` — rejected.** Profiles serialize
  into every transcript and must record resolved values, not machine-local
  references; regenerating old notes takes `--preset` explicitly.
- **An override channel so a preset's backend beats `STENOGRAF_NOTES_BACKEND`
  in the app — not worth it.** The variable is a developer escape hatch and
  the Settings report names it (`resolve_meeting_request`'s docstring).
- **Owner-grouped action items and `.notes.json` — removed deliberately** with
  the markdown notes (the template is the schema); history in
  `git log --follow -p PLAN-NOTES-MARKDOWN.md`.
- **A cancel button for an in-flight notes call — deferred:** interrupting a
  blocking `subprocess.run` / mlx generate needs a Popen restructure.
- **In-process notes on Windows/Linux:** Ollama stays the default; the
  fallback, if ever needed, is llama-cpp-python (off-PyPI wheels).
  onnxruntime-genai-directml rejected (DX12-only, EOL DirectML EP).

**Capture, mute and microphone choice**

- **Muting by stopping capture, or before the echo canceller — rejected.**
  Restarting capture is the fragile operation here; muting the system channel
  upstream would take away the canceller's reference (`session.MuteSwitch`).
- **Retroactive cut ("drop the last 30 s")** — its own design (immutable
  store chunks, committed live words). **CLI mute control** — when asked.
  **Mute remembered between meetings** — never (silently loses a meeting).
  **"Speaking while muted" hint** — VAD power every muted second; deferred
  until a forgotten mute is reported.
- **System-audio device selection, per-preset `mic_device`, mid-meeting
  device switching** — out of scope. **A standing pin that falls back to the
  default** — rejected: silent fallback is the bug the picker exists to
  prevent; a missing pinned device is fatal with the remedy named.
- **The GUI remembering the picked microphone — deferred.** If wanted, a
  GUI-owned `ui-state.json` in `data_dir()` is the cheap path (no
  settings.toml writes); decide its precedence against the settings pin and
  the CLI flag first.
- **The device preflight on macOS's unpinned path — deferred.** Enumeration is
  TCC-clean (measured 2026-08-12), so it is purely one extra subprocess per
  start; measure that against the start budget, then it is a one-liner.

**Updates and distribution**

- **Silent updates, updating `uv`, a user-facing "go back a version", a
  signature scheme beyond PyPI's HTTPS + uv's checksums** — declined.
  **`steno update`** — only if wanted.
- **Accepted holes in the updater:** a kill or power cut mid-swap leaves the
  install broken until the install command is re-run (it repairs with
  `--force`); closing that needs a self-repairing launcher, and the macOS one
  is frozen. An app slower than 45 s to reach Python is rolled back while
  starting (raise `NEVER_STARTED_S` if seen). An install with recorded uv
  options (a package mirror writes `find-links` into the receipt) gets the
  notice, never the button. uvx/pipx installs get a true but vague message.
- **Packaged signed installers / any downloadable artifact:** needs the $99
  Developer ID and notarization (locally written bundles need neither), and a
  PyInstaller build would owe PySide6's LGPL relinking. Re-open only if
  non-terminal users ask for a double-click installer.
- **GUI toolkits other than Qt Quick (2026-07-25):** `textual-serve` +
  pywebview (a browser engine — the wrong side of the no-web-UI lock); our own
  terminal renderer (owning a terminal emulator's bug surface); Tk (uv's
  python-build-standalone cannot open a Tk window: "Cannot find a usable
  init.tcl"); Slint (GPLv3, no stable Python release); Toga (sdist-only GTK
  deps); Dear PyGui (redraws continuously); PyQt6 (GPL would relicense the
  project). **`LSUIElement` in the bundle** stays rejected: it stops `open`
  from bringing the window forward.
