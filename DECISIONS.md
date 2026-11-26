# stenograf — decisions, and what was never verified

**No features are planned (2026-11-25).** Work starts from a reported problem
or a new idea of Daniel's, not from a backlog. This file keeps two things so
nobody has to work them out again: ideas that were decided against or set
aside, each with its reason and what would re-open it, and shipped features
that were never observed on real hardware.

Everything built is in git history: this file's own (it was `PLAN.md` until
2026-11-25; `git log --follow -p DECISIONS.md` holds the full designs, traps and
measurements behind every entry below) and the deleted side-plans, each
retrievable with `git log --follow -p <file>`: `PLAN-AEC.md`,
`PLAN-CLEANUP.md`, `PLAN-GUI-DEFAULT.md`, `PLAN-LINUX.md`, `PLAN-WINDOWS.md`,
`PLAN-ASR-CHALLENGER.md`, `PLAN-NOTES-MARKDOWN.md`, `PLAN-MEETING-PRESETS.md`,
`PLAN-CAPTURE-HELPER.md`, `PLAN-DIARIZATION.md`, `PLAN-DIARIZATION-SPEED.md`,
`PLAN-NEMOTRON.md`, `PLAN-FINISH.md`, `PLAN-MUTE.md`, `PLAN-MIC-DEVICE.md` and
`PLAN-UPDATE.md`. Measured evidence for every shipped default is in
`eval/README.md`; design reasons live in the code's docstrings.

---

## Set aside 2026-11-25 (Daniel)

Open work until that day. The working notes for each are in this file's
history.

- **Memory on 8–16 GB machines — until someone reports trouble.** On the M4
  Max a 1 h 45 min meeting's finalize peaks at 7.9 GB with diarization (6.9 GB
  without), and the voice model's session holds 860 MB for the whole meeting
  (`eval/README.md`, "A real finalize, and its peak memory"). The way out
  today is `[speakers] diarization = false`.
- **Ollama's output limit on the longest notes answer — until someone reports
  trouble.** A long meeting's reduce step was never measured against the
  `num_predict` limit `notes/ollama.py` sends. Hitting it fails loudly
  (`done_reason == "length"`), never as quietly cut notes.
- **Nemotron on the GPU (ORT's WebGPU plugin) — only if speed becomes a
  problem.** The CPU path runs 15 s per audio-hour on the M4 Max. First thing
  to re-test: on macOS the plugin loads a second onnxruntime beside the
  wheel's and segfaulted on a second session in one process.
- **Not holding the whole meeting's audio in RAM — perhaps later.** The motive
  is privacy (`SessionStore` keeps ~115 MB/hour/channel until the stop). Still
  unanswered: Nemotron's streaming-mode accuracy on held-out AMI against the
  offline 97.8 %, and what the fallbacks would do without audio.
- **A system-wide shortcut to start a meeting — perhaps later** (when a call
  comes in). The per-platform research for a mute shortcut carries over:
  Carbon `RegisterEventHotKey` on macOS, `RegisterHotKey` on Windows, the
  `GlobalShortcuts` portal on Wayland, each a small ctypes module.
- **A system-wide mute shortcut — declined.**
- **Echo cancellation inside the capture helpers, dropping `livekit` —
  declined:** the current path works.
- **Repair-only overhang (decode overhang only across a detected speech hole)
  — declined.** The VAD threshold 0.4 re-attempt, which waited on it, goes with
  it.
- **One stop gesture for both capture helpers — dropped.** Internal tidying
  only (`_stop_signal` in `capture/helper.py`: SIGINT for the Swift helper,
  stdin EOF for the Rust one) that costs a re-sign of the macOS helper; worth
  doing only if that helper is rebuilt for another reason.
- **The voice model's leftovers — dropped unless battery drain is reported:**
  energy per minute of speech (would decide the 120 s speech budget), a solo
  channel still embedded at the stop, a fixed-shape GPU/CoreML path.

---

## Never verified on real hardware

Shipped features nobody has yet observed on the machine they are for, and two
observations only Daniel can make. Not a to-do list: where to look first when a
problem is reported, and how to check one if it ever comes to that. "Open"
means never run.

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
| Two-voice split | any | First real meetings where a speaker is split: the halves right by ear on the finish card, no single voice split, the stop's extra seconds (7.5 s measured on a stored meeting) | Open since 2026-11-26 |
| Parakeet Ultra by ear | this Mac | First real German meetings on Ultra: the names and jargon the glossary used to fix, numbers and acronyms now written as "12" and "DVD", no other language appearing on short interjections | Open since 2026-11-26 |
| Parakeet Ultra off macOS | Windows, Linux | The ONNX export on real hardware: a first meeting transcribes; DirectML still word for word the CPU's (measured on v3 only) | Open since 2026-11-26 |
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

## Declined and deferred earlier

Each says why, and what would re-open it.

**Speech recognition and accuracy**

- **Challenger ASR models — declined 2026-07-27.** Word timestamps are
  mandatory (`asr/base.py`; speaker assignment needs them), and every
  leaderboard leader so far lost to Parakeet on our meeting audio (Voxtral
  Small 24B 32:38, Canary-1B-v2 ~1:2, blind, de+en). Re-open: a leader that
  emits word timestamps — a `BackendSpec` in `asr/registry.py`
  (`git log --follow -p PLAN-ASR-CHALLENGER.md`). Parakeet Ultra was no
  challenger in this sense — Parakeet-v3's architecture, tokenizer and
  timestamps with other weights — and replaced v3 on measured word error
  rates (2026-11-26, `eval/README.md`, "Parakeet Ultra").
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
