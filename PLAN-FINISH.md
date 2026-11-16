# PLAN-FINISH — the finish card after a meeting

Status: **steps 1–3 BUILT 2026-11-16; the gate (step 4) is Daniel's.** Daniel picked design A of three
mockups: one tidy card, icons instead of headings, text only where it earns its
place. It replaces the "Who spoke?" panel on the meeting screen.

## What the card holds

- **Title**, editable, pre-filled from the start form or meeting type. Under it,
  the folder name it produces: `meeting-20261116-091500-weekly-sync`. Done
  renames the folder (Hendrik's issue 4: a suffix after the date, so folders
  still sort by date).
- **Speakers**: a circle in the channel colour (initials once named), a play
  button for the voice samples, one quote, the speaking time. An unknown voice
  shows a "Who is this?" field. **A recognised voice shows its name as a
  button with a pencil** — clicking it opens the field pre-filled, because a
  wrong match must be obviously fixable (Daniel 2026-11-16: "Dennis Bank" came
  back as "Simon Ehlers").
- **Notes**: a switch (sparkles icon), the meeting type (which notes setup to
  use — chosen after the fact is often when it is known), and an optional
  one-line context the notes prompt receives.
- **Actions** as icons with tooltips: open folder, open transcript, copy
  transcript, discard (moves the folder to the trash, asks once), and **Done**
  as the one primary button.

## Decisions

- **F1 — The card opens after every meeting in the app**, not only when a voice
  is nameable, and the notes wait for it (Done, the existing timeout, or
  leaving the screen), whatever the start form said — the card can switch notes
  on. `names_panel_skip_if_recognised` keeps its meaning: when every voice was
  recognised the card does not open and the run finishes with the form's
  choices; a pencil in the footer opens it later.
- **F2 — Leaving or the timeout settles with the choices as they stand**: the
  untouched card means the form's notes choice, the start title, no context.
  Names typed but not saved are not applied (as today).
- **F3 — The library owns every effect** (`MeetingRun`): naming (exists),
  `retitle()` (rename the folder, write the title into `transcript.json`), and
  `settle(notes=, preset=, context=)` (the notes step uses those choices).
  Moving a folder to the trash is the one platform call the GUI makes itself
  (`QFile.moveToTrash`), like opening a folder; the library only abandons the
  notes.
- **F4 — The context lives on the profile** (`MeetingProfile.context`), so it
  is persisted in `transcript.json` and a later `steno notes` sees it too.
- **F5 — A changed meeting type changes only the notes setup**; vocabulary and
  language were applied to a recording that is already transcribed. A notes
  backend warmed for the old type is not used.
- **F6 — Retitling is refused while the notes are being written** (they write
  into the folder); after them it is allowed, and the notes files move along.

## Steps

1. Library: `output.titled_dir_name` / `retitle_meeting_dir`, `MeetingProfile.context`
   (+ JSON + prompt line), `MeetingRequest.preset`, `MeetingRun.retitle` /
   `settle(...)` / wait for the card. Tests.
2. QML: `FinishCard.qml` (variant A), editable recognised names, icons added to
   `Icon.qml`. Tests through the offscreen harness.
3. Meeting screen wiring, screenshots at full and minimum width.
4. Gate (Daniel): a real meeting with a wrong match corrected on the card.
