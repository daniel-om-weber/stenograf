# stenograf — meeting presets (deferred) + the per-run baseline (ship now)

Designed 2026-07-29; second adversarial review + Daniel's decisions 2026-07-30.
**Not built.** Two decisions reshaped this plan:

1. **Baseline first, presets later.** The plan's own admission stood up under
   review: the real justification for a preset *layer* is the GUI picker, which
   sits behind the Phase 8 step-7 / Textual-retire decision. So the ~30-line
   per-run baseline ships now, and the preset layer waits until that decision
   settles — its UI half must not be written for a front-end that may retire.
2. **History (old Decision 4) is deleted, not deferred.** "Past protocols as
   context" never meant meetings captured by stenograf — it means arbitrary
   files anywhere on Daniel's machine, including protocols a colleague wrote by
   hand. Stenograf cannot enumerate that set and should not try. Retrieval is
   the notes agent's job via Decision 3's positioning (cwd + env + per-preset
   instructions telling the agent where protocols live). With it die: the
   profile/transcript stamp, the output-home membership scan, the folder-name
   pre-filter, the oldest-first budget arithmetic, the contamination gate as a
   stenograf feature, and the entire "does this breach the no-index rule"
   philosophy section — stenograf never reads the output home as a collection.
   The prompt rule ("past protocols are background only; never report their
   decisions or action items as this meeting's") moves into the instructions
   file, where it belongs now.

Early-beta note (Daniel, 2026-07-30): backward compatibility is explicitly not
a constraint right now. Review findings about older installs hard-failing on a
new `[meetings.*]` table are release-notes material, not design constraints.

---

## Step 0 — the baseline, ship now (~30 lines)

Per-run flags on `steno start`: `--instructions FILE` and
`--notes-backend`/`--notes-model`. **Measured:** `_notes_options`
(`run.py:259-278`) gives `steno start` only `--notes/--no-notes` and `--print`
today — there is no way to choose a notes backend for a live meeting at all;
only `steno notes --backend` exists after the fact. `[notes] instructions`
already appends to the system prompt (`prompt.py:113`), so the flag is plumbing,
not features. This is the thing to actually use in real meetings while the
step-7 gate runs; what it cannot do (persistence, per-kind vocab, a GUI picker)
is the evidence base for whether the preset layer below is worth building.

## Independent fixes surfaced by review — land any time, preset layer or not

- **Closing the GUI during a long notes run blocks quit for the full
  `timeout_s`.** `MeetingScreen.shutdown()` (`gui/meeting.py:191-204`) stops
  capture only in phase `rec`, then calls `self.join()` unconditionally — with
  an agentic backend (`timeout_s = 1800`) that is a hang on exit, not merely
  "reads as done-and-hung". And `stop()` in phase `done` calls `app.back()`
  (`gui/meeting.py:222-223`) instead of cancelling. The fix lives in the
  library, not the screen: a cancel token on the notes step in
  `flow.MeetingRun.run` (`flow.py:285-292`), surfaced through `LiveView`
  (`flow.py:22-25`) so both front-ends get it — the drift rule applied.
  (The "emit a progress line before the notes call" idea from the first draft
  is already implemented: `view.status("generating notes…")`,
  `cli/notes.py:257`, both paths.)
- **`steno settings show` already fails to display a shipped key:** `[asr]
  boost` is absent from `_settings_rows` (`settings_cmd.py:107-158`). Any
  future preset-aware renderer inherits the gap; fix it now.

## The preset layer — deferred design, corrected by review

Everything below waits for the step-7 / Textual decision. Recorded now so it is
not re-derived; the review corrections are baked in.

### Naming — not "type"

`MeetingMode` (`config.py:73`) already means "kind of meeting"; `export.py:55`
already writes `type: meeting`; `profiles.py` holds "profile". So: **preset**,
`--preset`, `[meetings.<name>]`. German UI label: *Besprechungsart*.

### A preset is a named section in settings.toml, not a file

```toml
[meetings.controlling]
title    = "Controlling-Runde"
language = "de"
template = "~/steno/controlling-protokoll.md"   # needs PLAN-NOTES-MARKDOWN
instructions = "~/steno/controlling-stil.md"

[meetings.controlling.notes]     # sparse: unset keys fall through
backend = "command"
command = ["claude", "-p"]
timeout_s = 1800
```

The directory-of-files design stays rejected: a shared checkout is arbitrary
code execution (`[notes] command` argv runs unattended after every meeting,
`notes/command.py:83`; `git pull` must never change which commands run,
triggered by *recording a meeting*). Presets are personal. If sharing becomes
real, it returns as a directory loaded *without* executable keys unless
settings.toml allowlists them.

**Loader facts (measured in review):** `[meetings.*]` does not pass the loader
today — `load_settings` reads six table keys then `top.reject_unknown()`
(`settings.py:296`, `:305`) and a `[meetings.controlling]` section raises
`SettingsError` — so the namespace needs an explicit `top.table("meetings")`
read plus a per-name sub-loader. Every `_*_from_table` reader hardcodes its
table name in error messages (`unknown setting(s) in [notes]: …`), so reuse for
`[meetings.controlling.notes]` misreports the section — all six readers need a
name-prefix parameter, or the charter's "fail loudly, naming the file and key"
(`settings.py:80-83`) is violated for exactly the hand-edited file presets
live in. `reject_unknown` cannot tell a typo'd preset *name* from a new
preset; `steno presets` listing + echo-on-use is the mitigation.

### Overlay rules

- Sparse: a non-`None` scalar wins, a non-empty tuple wins (several fields use
  `()` as the unset marker — `NotesSettings.command`, `VocabSettings.attendees`,
  `TranscriptSettings.formats`, `MeetingProfile.glossary`/`.attendee_names`).
  Booleans and numeric zero overlay fine (`False`/`0` are non-`None`).
- **The overlay needs an explicit OFF-marker, because the canonical per-preset
  need is an unset** (review): a confidential-meeting preset must be able to
  turn *off* `notes.export_dir` — not writing into a synced Obsidian vault —
  and "non-empty wins" makes that inexpressible. Same shape:
  `notes.instructions` (revert to built-in prompt), `vocab.glossary_file`,
  `notes.command`. Chosen marker: `= ""` on a path key means *off/built-in*,
  stated in the settings docs.
- `[vocab]` **merges** with the standing vocabulary; every other table
  replaces. Cite honestly: `settings.py:73-78`'s exception is file-vs-flags —
  extending it to preset-vs-file is *our* extension, not "the charter's own".
  And it is per-key: `glossary_threshold` is a scalar in `[vocab]`
  (`settings.py:190`) and must replace, not merge.
- **Per-preset `[vocab]` changes the transcript, not just the notes** — terms
  flow into TurboBias decode-time boosting via
  `load_backends(glossary=…, boost=…)` (`start.py:326-329`). Consequences: it
  must be applied *before* `_collect_terms` (`run.py:298-326`, which takes one
  `vocab` object — signature change); and a transcript recorded under a preset
  is not reproducible without it. Acceptable in beta; noted.
- **`backend` and `model` overlay as a pair** — a preset setting one without
  the other nulls the other. `backend.py:151-152` clears a stale model only
  when `spec.name != settings.backend`; overwriting `settings.backend` with the
  preset's choice makes them agree and **suppresses** the guard — settings.toml
  `model = "claude-opus-4-8"` would ride into an mlx preset as an HF repo id.
  Add the ollama-file/mlx-preset pairing test beside
  `tests/test_notes_backends.py:65-79`.
- **Precedence: CLI flag > preset > env > settings.toml > built-in.** The stack
  below is already inconsistent — `ollama.py:46` / `mlx.py:78` do `model or
  env`, so settings.toml beats the env var while `settings_cmd.py:161`
  *displays* the env var as winning. Fix or document in the same pass.
- **Only `title`, `language` and `[vocab]` may reach the `MeetingProfile`**,
  allowlisted with a named error — profiles serialize into every transcript
  and `settings.py:3` forbids machine-specific config there. **No preset-name
  stamp, anywhere**: with history deleted the stamp lost its only real
  consumer, and review showed it violated this very rule (a preset name is a
  late-bound machine-local reference in a profile that otherwise records
  resolved values). Regenerating notes for an old meeting takes `--preset`
  explicitly; unknown name → hard `UsageError` before any work.

### One entry point — through the three seams that already exist

The first draft counted five `MeetingProfile(` construction sites and
prescribed `apply_meeting_preset()` called at each. Review: three of the five
(`start.py:258`, `transcribe.py:217`, `:275`) already share
`_resolve_run_config`. **The real seams are three** — `_resolve_run_config`
(`start.py:235` / `transcribe.py:154`), `resolve_meeting_request`
(`flow.py:88`), `transcribe_recording` (`flow.py:322`) — and the drift-proof
shape is a `preset: str | None` parameter on those three, resolved inside,
not a helper invoked at N call sites. This is also *required*, not just
cleaner: per-preset `[vocab]` only reaches TurboBias if the preset enters
inside `_resolve_run_config`. Test: CLI and `resolve_meeting_request` produce
identical results for the same preset.

`[notes]` resolution sites, recounted: three pass-throughs (`run.py:120`,
`start.py:384` — missed by the first draft — and `flow.py:293`), one loader
(`cli/notes.py:178`; `flow.py:508` delegates to it), one in doctor
(`doctor.py:308-309`). The loader branch (`notes_settings is None`) is what
discriminates "regeneration" from "in-run, already overlaid" — name that guard
in code; re-applying a preset on the in-run paths would double-merge `[vocab]`.

### Context is the notes command's job (kept, sharpened)

Stenograf supplies the *position*, not the payload: `notes/command.py:83`
gains `cwd` and an environment carrying `STENOGRAF_MEETING_DIR` and
`STENOGRAF_OUTPUT_HOME`; the preset's `instructions` tell the agent where the
board and past protocols live and which CLI reaches them. This now carries
past-protocol context too (see header) — the agent reads whatever files the
instructions point at, including colleague-written protocols stenograf has
never seen.

- **Instructions are backend-blind** (`prompt.py:113-114` appends them to the
  shared system prompt): a preset whose instructions say "run `glab issue
  list`" handed to mlx/ollama reaches a model with no tools, and the only
  thing between that and invented board state is `_SYSTEM`'s
  anti-hallucination text. Validate at preset load: agent-style instructions
  (mentioning commands/tools) on a non-`command` backend → warning.
- **From the `.app`, the user's exported env is absent and PATH is only
  widened, not sourced.** `native/appbundle/main.c:199-208` appends
  `~/.local/bin`, `/opt/homebrew/bin`, `/usr/local/bin`; no shell rc runs, so
  `GITLAB_TOKEN`/`ANTHROPIC_API_KEY` from `~/.zshrc` do not exist, while
  file-based credentials (`~/.config/glab-cli/`, `~/.claude/`) carry over.
  Review correction: an npm-installed `claude` is *not* the failing example —
  on this machine it resolves via `~/.local/bin`, which is on the widened
  PATH; the invisible case is nvm/fnm/volta-managed node
  (`~/.nvm/versions/node/*/bin`). Same preset, two outcomes, and the failing
  one is the one step 7 makes default. `steno doctor` must resolve each
  preset's `argv[0]` under the effective PATH and say whether this is an app
  launch (`STENOGRAF_APP_BUNDLE` is set for this purpose).
- The first draft's context-runner price list (stdin/DEVNULL, locale
  encoding, PATHEXT, byte ceilings, Popen+poll, 30 s default, ANSI/stderr
  hygiene) stays recorded in git history (`git log --follow -p` this file,
  2026-07-29 version) for the day a *local-model* preset needs stenograf-side
  fetching. Do not rebuild it naively.

### Injection — honest scope

After the decisions above, **no fetched text and no past protocol ever enters
stenograf's prompt**. The agent reads the board and old protocols inside its
own context, where it is prompt-injectable through that channel no matter what
stenograf does — out of stenograf's scope, worth one line in the docs. What
remains stenograf's: its own prompt contains only transcript + template +
instructions (all local, user-authored), and `notes/command.py:107-111`
flattens roles — so if stenograf ever *does* inject third-party text again,
the one mitigation that survives flattening is a per-run randomized fence
nonce; role labels and "data, not instructions" placement are theater against
an attacker who can write `SYSTEM:`. Recorded so the lesson outlives the
deleted feature.

### Inspection

- `steno settings show --preset NAME` rendering the effective layers with a
  `[meetings.NAME]` source label — **via `flow.settings_report(preset=…)`**
  (`flow.py:438`), not a CLI-only flag, so both Settings screens can render
  it (drift rule; `pick()` at `settings_cmd.py:165-170` grows a fourth
  source).
- `doctor.py:308-309` runs `create_backend(None, load_settings().notes)` — a
  preset selecting `command` gets a green doctor for mlx today. `doctor`
  validates every preset: `_notes_check` per preset + the `argv[0]`/PATH
  check above.

### Sequencing (when unblocked)

1. Loader + overlay + `preset` param on the three seams, `--preset` on
   `start`/`notes`, `steno presets`. No schema changes anywhere.
2. Per-preset `template` + `instructions` (needs `PLAN-NOTES-MARKDOWN.md`).
3. `cwd` + env on the command backend, the doctor checks, the instruction/
   backend mismatch warning.
4. UI picker + form prefill — after the Textual-retire decision, prefill logic
   in `flow.py` regardless.

## The baseline this must beat (unchanged, now sharpened by Daniel's call)

Step 0 *is* the baseline, shipped. The preset layer must earn its way in
against it on real use: persistence (typing flags every Tuesday gets old),
per-kind `[vocab]`, and the GUI picker are the three things the baseline
structurally cannot do. If real use shows the flags suffice, the layer above
never gets built — that is an acceptable outcome of this plan, not a failure
of it.
