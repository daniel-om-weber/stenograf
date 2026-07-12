# Code cleanup & maintainability plan

> **Status: not started (authored 2026-07-12).** Produced by a six-subsystem
> deep review (CLI, live orchestration, audio/capture, ML pipeline,
> notes/config, eval/native/tests). No pass has begun. Work top to bottom:
> §2 bugs first, then passes in order (§3 → §4 → §5 → §6). §4 (capture) may
> be pulled forward to precede the Phase 5 CachyOS capture work — see the
> note there. Tick checkboxes and update this blockquote as tasks land.

This document is the backlog for making the codebase clean and maintainable.
It is **behavior-preserving by charter**: no task here changes what stenograf
does, only how the code is organized. Product scope stays LOCKED per
CLAUDE.md. Line numbers below are as of commit `653ecaf` (2026-07-12) and
will drift as passes land — treat them as anchors, not gospel; re-locate by
the named symbol.

## 0. Review verdict (context for future sessions)

The codebase is engineering-strong and hygiene-clean (ruff with E/F/I/UP/B/SIM
passes in CI; docstrings record *measured* decisions; dataclass-first data
model; crash-safety via atomic writes and checkpoints; the concurrency model
in `session.py` is careful and well-tested). It does **not** need a rewrite.
It has three structural diseases:

1. **`cli.py` is a god-module** (2,151 lines): arg parsing + run
   orchestration + backend factory + capture-provider construction +
   transcript I/O + settings rendering in one file, including a full
   re-implementation of the mixed-file finalize pipeline that duplicates
   what the library already does for split channels.
2. **Duplication that has begun to drift**: the same primitive implemented
   2–5× across layers (atomic write ×4, timestamp formatter ×4, gap-padding
   buffer ×2, float→int16 ×2 *with different scale factors*, schema
   instruction ×2, ffmpeg invocation ×5, …). Several copies have already
   diverged; the subtle timing/gap logic is exactly what must not be
   duplicated.
3. **Missing seams**: no `tests/conftest.py` (fakes copy-pasted across 4+
   files; 409 `monkeypatch` calls, mostly patching private module symbols
   because collaborators are constructed internally instead of injected);
   god-object state in `MeetingRecorder`; inheritance abuse in
   `WindowedLiveDecoder`; no type checker despite ~86% annotation coverage.

Counter-model within the repo: `doctor.py` (one function per check, frozen
`Check` dataclass, uniform detail strings) is what the rest of the CLI should
be factored toward. Leave it essentially alone.

## 1. Guiding principles for every task

- **Behavior-preserving.** The existing behavioral test suite is the safety
  net; it stays green after every commit. If a refactor needs a behavior
  change, that's a separate, explicitly-argued commit.
- **One home per invariant.** Anything correctness-critical (timestamp
  anchoring, gap padding, atomic replace, sample-rate constant, schema
  instruction wording) exists exactly once and is imported everywhere else.
- **Seams over patching.** When a test needs to monkeypatch a private
  symbol, the production code is missing a constructor argument or a
  factory function. Fix the seam, then simplify the test.
- **CLI is glue, library is truth.** The CLI resolves settings/flags and
  calls one library function per command; domain logic (glossary, LID,
  provenance, transcript assembly) lives in the library. The settings
  charter stands: the library never reads `settings.toml`.
- **Interfaces are declared, not probed.** No `getattr(obj, "attr", None)`
  duck-typing across module boundaries; extend the base class/Protocol
  instead.
- **Each task = one focused commit to `main`** (repo convention), message
  describing the change itself. Run `uv run ruff check . && uv run pytest -q`
  before every commit; use the `verify` skill for end-to-end confidence
  after each pass.
- **Delete, don't deprecate.** Internal APIs have no external consumers;
  when a helper moves, the old copy is removed in the same commit.

## 2. Bug fixes first (small, independent — do before/alongside Pass 1)

Real defects found during review. Each is a standalone commit with a test.

- [x] **B1 — macOS `stop()` races `frames()`** — `capture/macos.py:135-136`
  closes `proc.stdout` from another thread while `frames()`
  (`macos.py:112-117`) may be mid-`read()` → `ValueError: read of closed
  file` instead of clean `StopIteration` (fires when e.g. `max_seconds`
  elapses mid-decode). Fix: `stop()` stops the process only (SIGINT/kill +
  `wait()` already ends the stream); the reading thread owns and closes
  stdout — matching the Linux/Windows ownership model.
- [x] **B2 — leaked checkpointer thread** — `session.py:845-846`
  (`_run_batch`): the `_TailCheckpointer` is `.start()`ed before
  `provider.start(...)`, which sits outside the `try`. If `provider.start`
  raises, the bus never closes and the checkpointer blocks on `bus.wait`
  forever. Fix: start the provider first, or move it inside the
  `try/finally` that closes the bus.
- [x] **B3 — model downloads: no timeout, no integrity check** —
  `models.py:77-100`. `urllib.request.urlretrieve` (line 93) can hang
  indefinitely; any complete-looking file (e.g. a CDN error page saved as
  `.onnx`) passes `target.exists()` forever. Fix: add `sha256` (and/or
  `size`) to `ModelAsset`, verify the `.part` before `replace(target)`;
  download with an explicit timeout.
- [x] **B4 — `notes_command` flattens all exceptions** — `cli.py:2029-2030`:
  `except Exception as exc: raise click.ClickException(str(exc))` makes a
  programming bug (AttributeError/KeyError) indistinguishable from "Ollama
  down". Fix: catch the documented typed set
  `(NotesBackendError, SettingsError, ValueError, OSError)` — the same set
  `doctor._notes_check` already uses — and let the rest propagate.
  (`_notes_after_run`'s bare except at `cli.py:2147` is contractually
  non-fatal and stays.)
- [x] **B5 — `ollama_url` is an undocumented setting** — defined
  `settings.py:171`, read `settings.py:386`, consumed `notes/ollama.py:50`,
  but absent from the schema docstring (`settings.py:34-46`) and
  `SETTINGS_TEMPLATE` (`settings.py:110-119`), which promise to be the
  schema of record. Fix: document in both places.

## 3. Pass 1 — foundations (cheap, stops the drift immediately)

### 3.1 Tooling

- [x] **T1 — add a type checker to CI.** *(pyright basic over src/, macOS CI job)* Functions are ~86% annotated but
  nothing verifies it. Add pyright (preferred: faster, better inference) or
  mypy; wire into `ci.yml` next to ruff. Start permissive (basic mode /
  ignore missing stubs for mlx/sherpa/onnx), tighten later. Annotate the
  worst offenders found in review while at it: `settings` threading through
  `cli.py` helpers, the `_load_*` return tuples, `plans`/`counts`/`channel`.

### 3.2 Test infrastructure

- [x] **T2 — create `tests/conftest.py`.** Zero shared fixtures exist across
  42 files today. Consolidate the copy-pasted doubles: base
  `FakeASR(ASRBackend)` (currently reimplemented near-verbatim in
  `test_pipeline.py:126`, `test_session.py:235`, `test_cli.py:31`,
  `test_live_orchestration.py:60`), `FakeDiarizer` (`test_pipeline.py:181`,
  `test_session.py:319`, `test_cli.py:54`), and `write_wav`/PCM helpers
  (`test_capture_file.py:10`, `test_audio.py:16`, `test_cli.py:79`). Keep
  genuinely specialized doubles (`WordlessASR`, `SilentASR`, `GermanASR`,
  `TwoSpeakerASR`, `AmplitudeASR`, `RecordingASR`) as thin subclasses of the
  shared base. Pure test-code move; no production change.

### 3.3 Single-home the duplicated primitives

Each bullet: create/choose the one canonical home, port all call sites,
delete the copies, keep behavior identical.

- [x] **T3 — `atomic_write_text(path, text)`** *(home: `output.py`; models.py keeps its streaming `.part` dance — binary downloads, not text)* in a small shared util (or
  `output.py`). Currently 4 implementations in 2 idioms:
  `notes/export.py:121-124` and `cli.py:1490-1499` (byte-identical `.tmp` +
  `os.replace`, no parent mkdir) vs `profiles.py:134-139` and
  `models.py:90/107` (`NamedTemporaryFile(".part")` + `.replace`, with
  mkdir). One implementation, parent-mkdir included.
- [x] **T4 — `format_timestamp(seconds)`** — byte-identical 4-line divmod
  formatter exists in `transcript.py:305`, `notes/prompt.py:173`,
  `notes/export.py:127`, `notes/model.py:143`. Canonical home:
  `transcript.py`; others import.
- [x] **T5 — one `SAMPLE_RATE`.** `capture/base.py:25` and `audio.py:18`
  both define `16_000` independently. Owner: `capture/base.py`; `audio.py`
  re-exports for its callers.
- [x] **T6 — one `CaptureUnavailableError`** in `capture/base.py`.
  `linux.py:59` and `windows.py:71` currently define two distinct classes —
  a latent trap for any shared CLI error handler. Consider making macOS's
  `HelperNotFoundError` a subclass.
- [x] **T7 — one float→int16 conversion.** `windows.py:304` uses `*32767.0`
  while the canonical `audio.to_int16` (`audio.py:36`) uses `*32768.0` (the
  exact inverse of `to_float32`). Windows calls
  `to_int16(block.mean(axis=1))`; the divergent copy is deleted. (Tiny
  numeric change on Windows samples — acoustically negligible, restores the
  documented single-conversion invariant; note it in the commit message.)
- [ ] **T8 — `schema_instruction(schema)` in `notes/prompt.py`.** The
  "respond with exactly one JSON object matching this JSON Schema" string +
  render step is copy-pasted, already subtly divergent, in
  `notes/mlx.py:166-170` and `notes/command.py:98-104`. Both grammarless
  backends call the shared helper; Ollama correctly keeps not using it
  (server-side `format=`).
- [ ] **T9 — eval I/O helpers in `eval/common.py`.**
  `to_wav16k(src, dst, start=None, end=None)` replaces the open-coded
  ffmpeg `-ac 1 -ar 16000 -c:a pcm_s16le` invocation in `extract.py:23`,
  `extract.py:54`, `scan_languages.py:46`, `backends.py:150` (the
  `adjudicate.py:155` libmp3lame variant stays separate);
  `read_pcm16(path)` replaces the WAV→int16 readers in `parity.py:45`,
  `diarize.py:66`, `aec_score.py:45`. Also: document the two harness tiers
  (package-free vs package-importing scripts) in `common.py`'s docstring —
  its "deliberately does not import stenograf" claim is contradicted by
  `parity.py`/`live.py`/`diarize.py`.
- [ ] **T10 — notes markdown/dedup cleanups.** Consolidate the drifted
  action-item formatters (`notes/export.py` `_item_line` vs
  `notes/model.py` `_action_item_line`) and the shared `## Decisions` /
  `## Action items` / `## Open questions` section skeleton
  (`model.py:111-129` vs `export.py:48-71`) into one section-emitter — the
  two artifacts stay intentionally different, only the skeleton is shared.
  Extract `_system_prompt(...)` from the duplicated assembly in
  `prompt.py:84-95` and `prompt.py:110-116` so the anti-hallucination rules
  cannot diverge between map and reduce.

**Definition of done for Pass 1:** type checker green in CI; conftest
exists and the four duplicated doubles are gone; `grep` finds exactly one
definition each of the primitives above; full suite green.

## 4. Pass 2 — dismantle `cli.py` (the big one)

Order matters; each step is committable alone.

- [ ] **C1 — `finalize_file(...)` in `pipeline.py` (keystone).** The mixed-
  file transcribe branch (`cli.py:1201-1277`) hand-assembles the pipeline —
  `finalize_channel` → `relabel_speakers` → `apply_glossary` →
  `detect_language` → hand-built `MeetingProfile`/`ResolvedParameters`/
  `Transcript` — while the split-channel path gets a finished transcript
  from the library. Add
  `finalize_file(samples, *, profile, asr, vad, diarizer, reid,
  num_speakers, glossary_threshold) -> Transcript` (or
  `MeetingRecorder.finalize_samples`) so **both** branches collapse to
  "load backends → one library call → write". Deletes the CLI's shadow
  pipeline; glossary/LID/provenance return to the library layer.
- [ ] **C2 — `ResolvedRunConfig` + one resolution helper.** The identical
  settings preamble — `_cli_settings()`, `_resolve_formats`,
  `_collect_terms(vocab=...)`, `glossary_threshold` default,
  `reid_threshold` default, `reid_store` fallback — is copy-pasted at
  `cli.py:415-428`, `1118-1131`, and partially in
  `_transcribe_split_channels`. One dataclass, one
  `resolve_run_config(settings, ...)` builder. Also dedupe the
  `_notes_after_run(...)` tail (`cli.py:576-583` / `1299-1306`).
- [ ] **C3 — extract the backend factory to `stenograf/loaders.py`.**
  Move `_load_backends`, `_load_diarizer`, `_load_reid`,
  `_prefetch_models`. While moving: promote `doctor._installed` to a public
  util (CLI currently imports a sibling's private helper at `cli.py:1330`
  and `1609`); move the sherpa-vs-speakrs selection into
  `stenograf.diarization.build_diarizer()` (it's a diarization-domain
  decision and a shared seam with `profiles enroll` — `cli.py:1370-1397`).
- [ ] **C4 — declared ASR diagnostics instead of `getattr` probing.**
  `cli.py:1353-1364` probes `hasattr(asr, "provider")` /
  `getattr(asr, "provider_fallback"/"active_provider"/"model_id")` — an
  informal side-channel only the ONNX backend implements. Give `ASRBackend`
  (`asr/base.py:43-64`) an optional `status() -> dict[str, str]`
  (default `{}`); the CLI renders whatever is present. The bare-string
  attribute docstrings at `asr/parakeet_onnx.py:55-60` become real members.
- [ ] **C5 — extract a capture-provider factory.** Move `_make_provider` /
  `_base_provider` out of the CLI; collapse the near-duplicate Linux/Windows
  blocks (`cli.py:833-869` — ~18 lines each differing only in module +
  label) into one helper taking the provider class, `default_devices`, and
  error type.
- [ ] **C6 — data-driven settings table.** `_settings_rows`
  (`cli.py:1843-1931`, 88 lines) hand-builds rows with per-backend defaults
  resolved via `if notes_backend == ...` chains with inline imports. Replace
  with row descriptors iterated once; per-backend defaults become a
  `settings_defaults()` classmethod on each notes backend.
- [ ] **C7 — split into a `cli/` package.** Only after C1–C6 (the bodies are
  then small): `cli/__init__.py` (thin `main` + group), `cli/start.py`,
  `cli/transcribe.py`, `cli/notes.py`, `cli/profiles.py`,
  `cli/settings_cmd.py`, `cli/doctor_cmd.py`, `cli/format.py`
  (`_report_speaker_counts`, `_lock_hint`, `_describe_channel`,
  `_fmt_setting`). Transcript-I/O helpers (`_write_transcript`,
  `_atomic_write_text` → gone via T3, `_cleanup_checkpoints`,
  `_checkpoint_writer`, `_prepare_output`) move to `output.py`. Keep entry
  point `stenograf.cli:main` working (`pyproject.toml`).
- [ ] **C8 — re-seam the CLI tests.** With C1–C7 done, replace the ~15 tests
  patching `cli._load_backends` / `cli._load_diarizer` / `cli._prefetch_models`
  with fakes injected via the new loaders/factory seams. Tests should get
  simpler, not just move.

Low/cosmetic (fold into whichever commit touches the line): map
`transcript.FORMATS` to callables instead of method-name strings dispatched
via `getattr` (`transcript.py:29-35`, `cli.py:1484`); fix stale "archiving"
docstring in `_make_tee` (`cli.py:764-766`); `_prepare_output`'s discarded
3rd tuple element (`cli.py:1138`); `doctor`'s lone `raise SystemExit(1)`
(`cli.py:1532`) — leave if the silent-table contract requires it, but note
why in a comment.

**Definition of done for Pass 2:** no file in `cli/` exceeds ~300 lines; the
mixed vs split transcribe paths call the same library function; no CLI
import of any `_private` symbol from another module; command bodies ≤ ~60
lines; test suite green with fewer monkeypatches than before.

## 5. Pass 3 — capture-layer consolidation

> **Sequencing note:** strongly consider doing this pass **before** the
> Phase 5 CachyOS capture work (PLAN.md §5). That work touches exactly
> these files on real PipeWire; building on the consolidated base avoids
> making today's Linux/Windows duplication permanent, and the injectable
> clock makes the new Linux code testable from day one.

- [ ] **A1 — hoist `_QueueStreamingProvider` base + `SessionClock`.**
  Linux (`capture/linux.py`) and Windows (`capture/windows.py`) share ~70%
  of their machinery with no base class: byte-identical `frames()`
  (`linux.py:162-171` / `windows.py:219-228`), the
  `SimpleQueue[AudioFrame | Channel]` sentinel protocol, `_t0`/`_started`
  setup, per-channel daemon pump threads, the first-frame anchor formula
  (`anchor = max(0.0, elapsed - len(samples)/SAMPLE_RATE)`;
  `linux.py:211-214` / `windows.py:279-287`), and "stream death tears down
  siblings" (`linux.py:217-221` / `windows.py:295-298`). Base class owns
  queue + `frames()` + sentinel + teardown; a
  `SessionClock(clock=time.monotonic, reanchor_tolerance=...)` owns
  `stamp(nsamples, arrival) -> float` — tolerance ∞ for Linux (parec
  delivers gap-free PCM), `_REANCHOR_TOLERANCE_S` for Windows (WASAPI
  loopback wall-clock-estimates silence). That tolerance is the *only* real
  behavioral difference; each backend keeps only its transport. macOS stays
  separate (single synchronous helper process). Preserves the load-bearing
  invariant: timestamps derive from cumulative sample count, never arrival
  jitter.
- [ ] **A2 — injectable clock for Linux (falls out of A1).** Windows already
  takes `clock=` (`windows.py:197`) and has a deterministic re-anchor test
  (`test_capture_windows.py:172-189`); Linux hardcodes `time.monotonic`
  (`linux.py:212`) so its test can only assert `timestamp < 0.5` against a
  wall clock. After A1, port the Windows-quality anchor test to Linux.
- [ ] **A3 — extract `GapPaddedBuffer`.** `_Track` (`aec.py:71-149`) and
  `_PendingChannel` (`recording.py:160-209`) independently implement the
  same primitive: timestamp-anchored int16 buffer, silence-padded forward
  gaps, raise past `-ORDER_TOLERANCE_SAMPLES` backward jumps, front-pop.
  The gap arithmetic (`aec.py:100-111` vs `recording.py:168-185`) is the
  subtle part written twice. One buffer class; `_Track` layers
  `window()`/`trim_before()`, `_PendingChannel` layers its deque. A
  divergence here silently misaligns the recording or the AEC reference —
  this is the highest-value dedup in the codebase.
- [ ] **A4 — smaller consolidations.** One `DEFAULT_FRAME_MS` + frame-size
  computation (copied at `linux.py:49/127`, `windows.py:53/201`,
  `file.py:25/40`); merge the near-duplicate pipe readers `_read_exact`
  (`macos.py:139`) / `_read_up_to` (`linux.py:224`) once a shared home
  exists (differ only in EOF policy); unify the `command=` injection-seam
  type across backends (`str | Path | list[str]`); close the `AecDump` in
  `EchoCancellingProvider.stop()` so `start()`→`stop()` without iteration
  doesn't leak three WAV handles (`aec.py:331-355`); consider extracting the
  Windows-only silent-mic watchdog (`windows.py:265-278`) into a shared
  consumer (a dead mic is possible on every platform) — optional, judgment
  call at implementation time. Decide whether `default_devices` joins the
  `CaptureProvider` ABC (with a documented not-supported story for
  macOS/file) or the divergence gets documented instead.

**Definition of done for Pass 3:** `linux.py`/`windows.py` contain transport
code only (roughly half their current size); the anchor formula and gap
arithmetic each exist once; Linux timestamping has a deterministic unit
test; `test_capture_*` suites green on CI (all three OSes).

## 6. Pass 4 — `session.py` and `live.py` extractions

- [ ] **S1 — `MeetingRecorder.run()` returns a result object.** The
  constructor currently sets immutable config (`asr`, `vad`, `diarizer`,
  `reid`) *and* per-run mutable outputs: `speaker_counts`
  (`session.py:1047`), `dropped_echo_lines` (`:1057`), `reference_gap_s`
  (`:733/1010`), and `language` locked mid-run (`:1212`) — so the instance
  is not reentrant and results travel through two channels (attributes and
  `transcript.parameters`). Fold outputs into the returned
  `Transcript`/result dataclass; make language-locking local to one
  `run()`/`finalize()` call. This is the core god-object fix.
- [ ] **S2 — shared finalize epilogue.** The stop→finalize tail —
  `view.finalizing()`, `_note_reference_gap`, `with _shield_interrupt():
  finalize(...)`, `view.finalized`, plus identical status literals — is
  duplicated between `_run_batch` (`session.py:848-882`) and `_run_live`
  (`:958-999`). Extract `_finalize_and_publish(...)`. The *capture loop*
  stays inline in batch deliberately (KeyboardInterrupt must land on the
  main thread) — share the frame-body helper, keep the loop structure.
- [ ] **S3 — `LiveDecoder` becomes a Protocol; Windowed composes.**
  `WindowedLiveDecoder` (`live.py:359-559`) inherits `LiveDecoder` but
  overrides the entire core (`feed`/`flush`/`drop_window`/`_reset_buf`/
  `_audio_end`), reinterprets inherited `_buf_start`, and carries the base's
  unused LocalAgreement-2 machinery — a base method touching `_buf_start`
  directly would corrupt the byte-identical-slice guarantee. Define a
  Protocol with exactly what `LiveWorker` uses (`feed`, `flush`,
  `drop_window`, `window_cap`, `committed_words` — see
  `session.py:551-564, 991`; already used polymorphically as
  `dict[Channel, LiveDecoder]` at `:919-932`); share a small buffer/commit
  helper by composition. Gives the private methods tests currently reach
  for (`dec._buf` in test_live.py) a public surface.
- [ ] **S4 — decouple `_TailCheckpointer` from recorder privates.** It holds
  `self._recorder` and calls `recorder._tail_entries(...)` /
  `recorder._checkpoint_transcript(...)` (`session.py:666, 675`). Pass two
  bound callables (`finalize_tail`, `wrap_checkpoint`) or extract a
  `TailFinalizer` collaborator — also gives `test_session.py`'s direct
  private-method tests a public surface.
- [ ] **S5 — collapse `run()`'s two callback dialects.** `run()` takes 11
  params including both legacy `on_update`/`on_status` and `view`, wrapping
  the former in `_CallbackView` internally (`session.py:750-815, 794`).
  Standardize on the `LiveView` sink (tests already build `LiveView`
  subclasses); group the checkpoint knobs into a small config object.
- [ ] **S6 — smaller items.** Split `finalize` (80 lines, five jobs;
  `session.py:1012-1090`) into `_apply_echo_backstop` +
  `_assemble_transcript`; fix the close-path lock inconsistency (`:551-564`
  — capture `flush()` under `inference_lock`, run `_emit` after releasing,
  matching the main loop's "never hold the accelerator lock across a
  callback" rule); replace the stringly `_phase` state machine driving three
  parallel dicts in `tui.py:143/303/312` with an enum; lift `pipeline.py`'s
  magic progress-stage strings (`"asr"`/`"diarization"`) to constants.

**Pipeline-layer companions (same pass, `pipeline.py`/`glossary.py`):**

- [ ] **S7 — split `finalize_channel`** (`pipeline.py:43-133`): it
  interleaves precomputed-words vs VAD+ASR, single-speaker vs diarized,
  re-ID vs plain, and a wordless fallback, with `words`/`segments` shared
  across branches. Extract `_decode(...) -> (words, segments)` and
  `_attribute(...) -> entries`; `finalize_channel` becomes a ~15-line
  dispatcher.
- [ ] **S8 — merge the duplicated run-grouping machinery**:
  `merge_words_turns` (`pipeline.py:148-189`) and `group_words`
  (`:192-226`) both implement close-run-on-gap with a nested `close_run()`;
  one `_group(words, key, max_gap)` helper, constant key for the plain case.
- [ ] **S9 — micro-cleanups**: `_shift` uses `dataclasses.replace` instead
  of field-by-field reconstruction (`pipeline.py:136-145`, so a new `Word`
  field can't be silently dropped); simplify `_best_term`'s redundant gate
  (`glossary.py:185-192`); drop the per-call MLX re-imports in
  `parakeet.py:94-98`.

**Definition of done for Pass 4:** a `MeetingRecorder` can run twice without
state bleed (add that test); `LiveWorker` depends on a Protocol; no thread
class calls another object's `_private` methods; test_session/test_live no
longer unit-test private methods directly.

## 7. Deferred / awareness-only (no action planned)

Recorded so future sessions don't re-discover them:

- `tests/fake_parec.py:38`'s 0.5 s linger and
  `test_notes_backends.py:380`'s sleep-30-vs-timeout-0.2 are bounded and
  documented, but any sleep-to-dodge-a-race is a latent flake — revisit only
  if CI flakes.
- test_tui.py necessarily reaches into TUI privates (`app._tick()`,
  `app._phase`, …); partially unavoidable for a Textual app. S6's enum
  helps; don't chase further.
- `hatch_build.py`'s cargo probe checks PATH-or-`~/.cargo/bin` but the build
  runs via `/bin/sh` — probe and build can disagree. Advisory only (release
  workflow re-verifies); add a comment if touched.
- `native/stenodiar/src/main.rs:127-131`: `turns_json` uses `assert!` where
  the rest returns `Err(String)` — swap to `return Err(...)` if the file is
  touched; not worth a standalone commit.
- eval `aec_rig.py:46-47` shadows `common.OUT_DIR`; `transcribe.py:83`
  double-calls `mlx_peak_mb()`; fix opportunistically.
- `notes/*.from_settings` coupling each backend module to `NotesSettings`
  is consistent with the ASR pattern and stays; `_Table.number()`'s
  lo/hi-together invariant gets tightened if `settings.py` is touched.
