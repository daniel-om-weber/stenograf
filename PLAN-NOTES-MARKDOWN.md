# stenograf — notes become markdown

Designed 2026-07-29, second adversarial review 2026-07-30 (three independent
reviewers; findings baked in). **BUILT 2026-07-30** — the code below is
implemented and the CI triple passes; what remains is the gates:

- **Gate 0 (Ollama probe): not run** — needs the CachyOS notebook.
- **Gate A: both cases captured, Daniel's read pending.** Outputs live in
  `gate-a/` (gitignored, real meeting content), before/after pairs on the
  same transcripts: `small-claude` and `mid-claude` succeeded on both paths
  (after-case: all headings matched, zero warnings). **mlx has no before-note
  to read: the JSON path failed 2 of 2 attempts** on the small meeting
  ("missing a usable 'title'", temp 0.6, both from the pinned `4fe5c76`
  worktree; mechanism not investigated — the path is deleted), while the
  markdown path succeeded on the same transcript+model first try with zero
  warnings (`gate-a/after/small-mlx`). n=2 vs n=1, but the direction is the
  point: the shipped macOS default could not produce notes for this real
  meeting on the old path and can on the new one.
- **Gate B: not run** — needs CachyOS/Windows. `format=` is now deleted, so
  the with-grammar baseline run must check out `4fe5c76` (the last commit
  with the JSON path) on that box first.

Two small deviations from the design, both stated in code: warnings travel on
`NotesProvenance.warnings` (no return-tuple arity change was needed), and
*unfenced* suffix chatter after the last section is accepted as body — there
is no delimiter to stop at (`notes/markdown.py` documents it); the fenced
case, which is the one the old tests pinned, is fully handled.

Independent of `PLAN-MEETING-PRESETS.md` — that plan needs this one, this one
needs nothing. **Landed in parallel with Phase 8 step-7 real-use gating**
(Daniel, 2026-07-30): PLAN.md scopes the step-7 gate to the live caption
screen, a notes failure is non-fatal and self-labelling, so per-subsystem
observation loses no attribution. The budget fix below landed first,
standalone.

---

## Why

Notes are produced as schema-constrained JSON, parsed into a fixed
`MeetingNotes`, and rendered to markdown by two renderers. The structure cannot
vary, so a protocol layout cannot vary per meeting. **Measured:** nothing in
`src/`, `eval/`, `scripts/` or `native/` reads `.notes.json` back —
`MeetingNotes.from_json` has exactly three callers, all in
`tests/test_notes_model.py` (the `from_json` hits in `flow.py:501`,
`cli/notes.py:85`, `eval/aec_rig.py:164` are `Transcript.from_json`). The
round-trip serves one grouping operation (`export.py:68`) and a validation
gate. The structured fields have no consumer beyond `export.py` — verified by
full-tree grep.

## The four jobs (unchanged diagnosis)

`NOTES_SCHEMA` + `_parse_notes_object` (`notes/generate.py:97`) do **four**
jobs: 1. structure, 2. **sanitizing** — the `{`-scan discards preamble, fence,
trailing chatter (`tests/test_notes_generate.py:143`), 3. **truncation
detection** — mlx hard-caps output (`notes/mlx.py:48`); a cut-off response has
no closing `}` and fails loudly, 4. **refusal detection** — a refusal contains
no JSON object. Deleting the schema deletes all four; the replacement must
cover each explicitly, and round two showed the first replacement design
covered none of them fully.

## Decision — the template is the schema, with a real gate in front of it

### The template has a home

- **`notes/template.py`** (new) owns the template: `DEFAULT_TEMPLATE` (the
  built-in layout — today's sections as markdown headings) and
  `headings(template) -> list[str]`, the single heading extractor. `template:
  str`, **never `None`** — review killed `str | None`: with no template there
  is no heading set and the hard-fail gate below is undefined, so the shipped
  macOS default would lose all four jobs and get back only "empty → fail".
- `headings()` parses ATX `##`+ headings only, **skipping fenced blocks** (a
  template with `#` lines inside a fence must not yield phantom headings). The
  H1 slot is the title, not a matched heading.
- Degenerate case, documented: a user template with zero `##` headings degrades
  validation to "non-empty after unwrap + body minimum" — stated as weaker,
  warned at load, not an error.
- **Headings are matched verbatim against the template actually used.** This
  makes per-preset and German templates language-safe by construction.
- **Language decoupling** (round-two finding: the word "language" appeared
  nowhere in the first draft). `prompt.py:95` says "write the notes in
  {language}"; template headings are literal text, possibly in another
  language. The prompt must say: *headings exactly as in the template; body
  prose in {language}*. Without this, a German meeting against the English
  default template emits `## Beschlüsse`, matches zero headings, and hard-fails
  a good note. (The JSON schema's English keys were language-neutral —
  `tests/test_notes_generate.py:26-37` pins German values under English keys —
  and that decoupling is exactly what a literal template destroys unless
  restated.)

### The format instruction stays LAST

Both grammarless backends deliberately append the format spec to the *end* of
the prompt: `mlx.py:182-183` appends `schema_instruction` to the final user
message; `command.py:107-111`'s docstring literally reads "schema instruction
last". Moving the template into `_system_prompt` puts it *before* up to 400 kB
of transcript (`command.py:26`) in a flat-concatenation prompt — a recency
inversion that would degrade adherence most in exactly the backend Gate B
never tests. So: `template_instruction(template)` replaces `schema_instruction`
**in the same position** — appended to the last user message per backend. The
system prompt carries role and rules; the template rides last.

### `unwrap_markdown()` — prefix AND suffix

The first draft's two-step rule ("strip one outer code fence, drop everything
before the first H1") fails its own cited test case
(`test_notes_generate.py:143`): with a preamble the fence is not outermost, so
the stray closing ``` and the trailing "Anything else?" survive into the note.
Spec:

- strip reasoning first — hoist mlx's `_THINK_BLOCK` (`mlx.py:60`) into a
  shared `strip_reasoning()` and drop the `\A` anchor (an unanchored think
  block whose `# ` line would otherwise become "the first H1");
- then drop everything before the first H1 **and** the suffix side: a trailing
  orphan fence and trailing chatter after the last matched heading's section
  (the counterpart of the `{`-scan's `raw_decode` stop);
- fence handling: if the *entire remaining output* is one fenced block, unwrap
  it; never unwrap a fence that starts after content (a legitimate fenced
  snippet inside a section stays);
- **the first H1 line is extracted as the model's title candidate and removed
  from the body** — `to_markdown()` renders exactly one H1 from the resolved
  title. (The first draft had the H1 both "being the title" and "in the
  body" — two H1s or a discarded model title, depending on reading.)
- No H1 at all (or a placeholder H1 echoing the template's literal slot text):
  title falls back per precedence below **with a warning** — today
  `generate.py:127-128` hard-fails on a missing title, so silently degrading
  to the meeting date would send `export.py:38` slugs like
  `2026-07-10 – 2026-07-10.md` into the vault; the warning is the trace.

### Validation

- Empty output after unwrap → hard fail (unchanged contract).
- **Zero template headings matched → hard fail** — what keeps a refusal from
  becoming the note body.
- Some headings missing → warning, per heading.
- **Per-section emptiness → warning** (round two: a model echoing the template
  verbatim matches every heading — the "silently half-empty note" the gate
  exists to catch — so heading presence alone is not a gate). A minimum of
  non-blank, non-heading body characters backs it.
- **Truncation via the backend's own completion signal, not a text heuristic.**
  mlx: `stream_generate`'s `GenerationResponse.finish_reason` ("length" vs
  "stop"); Ollama: `done_reason` in the `/api/chat` response body, which
  `ollama.py:94` currently discards. `finish_reason == "length"` → hard fail,
  matching today's loud missing-`}` failure. The command backend has no such
  signal — heading validation is its only truncation net; stated, not hidden.
- **Warnings are returned, not printed — through *both* channels.**
  `_generate_and_write_notes` grows a warnings slot (the tuple change touches
  `cli/notes.py:96`, `:228`, `:259`, `flow.py:508`, and the test double
  `tests/test_cli.py:848`, which hand-writes a 2-tuple), rendered where "notes
  failed" renders — but **not via `view.error()`** for partial success
  (`flow.py:380` prefixes "warning:", `cli/notes.py:268` uses it for hard
  failure; a warn-styled status line keeps success and failure
  distinguishable). **And `flow.generate_notes_for` (`flow.py:483`) returns
  `(paths, warnings)`** — round two: its two consumers, the Qt notes screen
  (`gui/screens.py:216-218`) and the Textual notes screen
  (`ui/notes.py:157-165`), render a bare status string and would silently drop
  warnings; both lambdas update. This is the two-front-end drift rule applied
  to this plan's own feature.
- **Warnings also land in the provenance block** (below), so the evidence
  survives the screen.

### Provenance becomes a named-line block

Today provenance is one italic line in `to_markdown()` (`model.py:121-124`)
and **absent from the vault note** (`export.py:48-65` emits only
`source: stenograf` frontmatter). Change: a named-line block in the sibling
`.notes.md` (backend, model, strategy, heading misses / warnings), and
backend + model as **YAML frontmatter keys** in the vault note — frontmatter is
the vault-idiomatic place and stays out of the reading view.

### `MeetingNotes` and the renderers

- `MeetingNotes` → `title` + `body` + `provenance`. `to_markdown()` = H1 + body
  + provenance block. `export_note()` = frontmatter + H1 + body + transcript
  callout; `_action_items_by_owner` dies.
- **Owner grouping is dropped, not regex-restored** (Daniel, 2026-07-30). The
  vault note shows action items as the model wrote them. `README.md:241-242`
  promises "action items per owner" — update it and note the removal in the
  release notes alongside `.notes.json`.
- **Title precedence unchanged**: `profile.title or <model H1> or <meeting
  date>`. `notes/generate.py:82` and `tests/test_notes_generate.py:112` pin
  it — a title typed into the setup form must keep winning.
- `.notes.json` stops being written; the write site (`cli/notes.py:204`) also
  `unlink(missing_ok=True)`s a stale sibling.

### Map-reduce

- **Content in map; shape in reduce** — map emits neutral bullets, reduce
  fills the template once. `_system_prompt` (`prompt.py:98`) stays one
  function; thread `template: str | None` through it (None for map calls — the
  one place None is correct, because map has no template by design).
- **Map partial gate, stated honestly:** non-empty + at least one `- ` bullet
  line. Round two: the JSON parse was the map path's refusal detector, and "a
  non-empty check" passes "I can't summarize this" — the bullet check catches
  empty and prose-shaped refusals, and is admittedly weaker than the parse. A
  refusal that emits bullets passes; accepted.
- Budget the reduce call: Σ partials + template + system checked against
  `max_input_chars` before the call; over → hard fail naming the overflow (no
  silent head-truncation on Ollama / output-truncation on mlx).

## Budget — fix it here, it is already wrong

`max_input_chars` is a *transcript-chunk* budget, not a prompt budget: its only
use is `chunk_entries(transcript.entries, max_chars=backend.max_input_chars)`
(`generate.py:61`); the system prompt escapes the accounting entirely.

Fix: render the system prompt once, pass
`max_chars = max_input_chars - len(system)` into `chunk_entries` — and **hard
error, not a floor, when the remainder is below a sane minimum** (~4 000
chars), naming the instructions file / template as the thing to cut. Round
two: `chunk_entries` (`prompt.py:170`) with `max_chars <= 0` splits on *every*
entry — an oversized instructions file would turn a 2-hour meeting into
hundreds of model calls, each carrying the full oversized system prompt. The
first draft's "(floored)" traded a silent overrun for a runaway.

Note which prompt is measured: `_system_prompt(partial=True)` differs from the
reduce form and the template rides only on reduce — measure the larger.

Standalone bugfix; can land before everything else.

## Regressions, named as accepted rather than discovered later

One verbatim body cannot reproduce both renderers' current output:

- the vault note gains **Highlights** (`export.py:59` never emitted them);
- the vault note gains inline `[h:mm:ss]` timestamps (`export.py:90` strips
  them on purpose — no audio in a vault to jump to);
- a template makes the model emit every heading, so a decision-free meeting
  gains an empty `## Decisions` (today `model.py:128` omits empty sections;
  the per-section emptiness warning is the trace);
- **owner-grouped action items are gone** — decided, not open (above).

Gate A is therefore stated per output.

## Gates

- **Gate 0 — the Ollama probe, before Gate B is built.** One `/api/chat` call
  to qwen3:8b, no `format`, oversized prompt. Answers three questions at once:
  (a) `prompt_eval_count` vs the declared ceiling — `ollama.py:86-91` sends
  **no `options.num_ctx`**, so the `128_000`-char ceiling (`ollama.py:23`) may
  be a client-side fiction and the server may silently truncate the prompt
  head, which would make Gate B measure truncation, not template adherence; if
  so, send `num_ctx` derived from `max_input_chars`. (b) `done_reason`
  present → the truncation check above is confirmed implementable. (c) Gate C:
  does reasoning arrive in `message.thinking` (separate; content always clean —
  likely for qwen3, in which case removing `format=` changes nothing and no
  stripping is needed) or leak into `message.content` (→ shared
  `strip_reasoning()`, and today's `format=` path was already at risk). Do not
  write the conclusion as "format= was doing this" — the repo reads neither
  `thinking` nor `done_reason` today, so the mechanism is genuinely unknown.
- **Gate A — the `.notes.md` is equivalent.** Same transcript, same model, JSON
  path vs markdown path. **The JSON-path outputs (sibling and vault note) must
  be generated and kept *before* landing** — after the change the old path no
  longer exists to compare against. Vault note judged separately per the
  regression list.
- **Gate B — template adherence on Ollama, on Linux or Windows.** Not mlx (mlx
  never had grammar constraint — `mlx.py:174-183` inlines the schema as plain
  text — so testing it measures nothing about what is removed). Run it **twice
  on the same box: first with `format=` still in place, then without** —
  PLAN.md's declined list records "a real Ollama notes e2e" as never run, so a
  single unconstrained run could not attribute a failure between "removing
  format= broke it" and "this path never worked". Needs the CachyOS notebook
  or a Windows session; machine access, not Phase 8, is this plan's real
  scheduling constraint.

## Mechanical work the CI triple will not catch

- `mlx.py:29` and `command.py:17` import `schema_instruction` — ImportError at
  import time, not a type error. Both `_render` helpers change signature
  (`mlx.py:174-183`, `command.py:107-111`), not just the import.
- `notes/__init__.py` re-exports `ActionItem`/`SpeakerHighlight` in `__all__` →
  ruff F822; dead imports in `model.py`/`export.py` → F401.
- **Pyright cannot see the protocol change**: `create_backend` reaches the
  class via `getattr(module, spec.cls)` (`backend.py:154`) → `Any`, pyright is
  `basic` and scoped to `src`. `NotesBackend` *is* `@runtime_checkable`
  (`backend.py:34`), but `runtime_checkable` checks member presence, never
  signatures — an `isinstance` would not help. Add
  `test_every_registered_backend_matches_the_protocol` inspecting
  `signature(cls.complete)` — via the registry *classes*
  (`importlib.import_module(spec.module)` + `getattr`), not instances:
  `create_backend("command", …)` raises `NotesBackendUnavailableError` on
  empty argv (`command.py:43-47`).
- **`tests/test_notes_backends.py` — the largest single body of affected test
  code, absent from the first draft:** the `SCHEMA` constant (`:27`) and ~20
  `complete(MESSAGES, SCHEMA)` call sites; `payload["format"] == SCHEMA`
  (`:134-142`); the schema-on-stdin assertion (`:346-353`); the
  schema-in-last-message assertion (`:246-249`); the opt-in real-`claude` e2e
  asserting `notes.summary` (`:442`).
- **`tests/test_notes_model.py`**: five of six tests die (round-trips,
  missing-optional-keys, `test_markdown_renders_all_sections` incl. the
  provenance assertion, `test_markdown_omits_empty_sections` — which pins the
  empty-section behavior the regression list gives up).
- `tests/test_notes_generate.py:8` imports `NOTES_SCHEMA`; `:109` asserts
  `backend.calls[0][1] is NOTES_SCHEMA`. Test doubles define
  `complete(self, messages, schema)` in `tests/test_notes_generate.py:53` and
  `tests/test_cli_notes.py:49` (the latter imported by `tests/test_ui.py` and
  `tests/test_gui.py`).
- `tests/test_cli_notes.py:163` is **not** a malformed-output test — it is the
  backend-unavailable path; keep it, delete only its stale `.notes.json`
  assertion (`:174`). There is no malformed-output test in that file. Keep
  `tests/test_notes_export.py:72-99` (live slug logic, constructor update
  only). Do not simply delete `tests/test_cli_notes.py:90` (the only e2e
  provenance check — move it to the `.notes.md` block) or
  `test_notes_generate.py:149` — replace with tests pinning what
  refusal-shaped, heading-less, and template-echo responses actually produce.
- Docstrings describing the deleted contract: `notes/__init__.py:5`,
  `notes/generate.py:1`, `notes/command.py:5-7`, `notes/backend.py:31`,
  `:37-38`, `notes/mlx.py:176-179`, `model.py:4`.
- Docs: `README.md:226`, `:241-242` (the "per owner" promise), `:251`,
  `output.py:13`, `cli/notes.py:72`, `:160`. `.notes.json` removal and the
  owner-grouping removal are user-visible contract changes → release notes.

## Sequencing

Budget fix first (standalone). Gate 0 probe at the next CachyOS/Windows
session, before Gate B is designed in stone. The main change lands **in
parallel with step-7 real-use gating** — record notes observations
per-subsystem so a bad session attributes cleanly. Gate B runs when a
non-macOS box is next available; it does not block landing on macOS, it blocks
*declaring the Ollama path healthy*.
