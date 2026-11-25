# Competitor landscape — November 2026

Research record: which other tools do what stenograf does, what is still
unique, and which competitor features might be worth building. Compiled
2026-11-24 from three parallel web research passes (open-source/local tools,
commercial notetakers, and a feature-by-feature sweep). The load-bearing claims
about the closest competitors were then re-checked by hand on the vendors' own
pages and GitHub (marked *re-checked*). No competitor app was installed or
tested hands-on; every competitor fact is what the vendor or a third party
*says*.

This field moves monthly — OpenWhispr went from no diarization to voiceprints
plus echo cancellation between April and November 2026. Treat anything here
older than ~6 months as stale and re-check before relying on it.

**How to read the source labels**

- **V** — stated on the vendor's own website, docs, README or code.
- **S** — secondary: press, a review, or a *competitor's* comparison page.
- **?** — not stated anywhere found. This means "not advertised", not "proven
  absent".

---

## 1. Verdict

Feature by feature, stenograf is **no longer unique**. Since mid-2026 other
tools ship echo cancellation, cross-meeting voiceprints, the same Nemotron 3
diarizer, separate mic/system streams and live-then-final transcripts. What no
product found combines is:

1. **audio never written to disk by default**, with nothing leaving the device;
2. the **full pipeline on macOS, Windows and Linux**;
3. **several people on one room microphone named across meetings**, locally;
4. **German as a first-class language**, including decode-time vocabulary
   boosting that handles German compound words;
5. **free and MIT, nothing paywalled**.

Points 1 + 3 together are the core selling point; 2, 4 and 5 make it hard to
copy quickly.

**Caveat on point 3:** it is built (every channel is diarized, and re-ID
relabels diarized speakers on any channel — `pipeline.py`), but it is the one
case no reference recording covers. Watching the first real hybrid meetings on
a shared room mic has not happened yet (`DECISIONS.md`, never verified on
real hardware). Until it has, market it as a design, not as a measured result.

**Closest competitors, in order:**

| # | Tool | Why it is close | Where stenograf differs |
|---|---|---|---|
| 1 | **OpenWhispr** (MIT, ~12.1k stars, mac/win/linux) | Closest overall: three OSes, PipeWire capture on Linux, WebRTC echo cancellation + duplicate-line removal (stenograf's design), local voiceprints across meetings, raw audio deleted after diarization | Mic is "definitionally you" — never diarized, so no room-mic separation; audio does hit disk (temp dir); SQLite storage; vocabulary is prompt hints; dictation-first with paid cloud plans; no German focus |
| 2 | **pasrom/meeting-transcriber** (MIT, 256 stars, macOS) | Closest on internals: mic and app tracks diarized *separately* (so a room mic is split), Nemotron 3, voiceprints across meetings, Parakeet with CTC vocabulary boosting, plain `.md`/`.txt` in a folder, menu-bar app | macOS only; echo cancellation **off by default**; no real live captions (proof-of-concept overlay); records audio to disk; auto-records Teams/Zoom/Webex |
| 3 | **MimicScribe** (closed, $50 one-time, macOS 15+) | Most polished commercial match: AEC on both channels, live transcript then final one seconds after the call, voice profiles that grow with each meeting, markdown to Obsidian/any folder, local LLM option | Writes WAV to disk during recording; German not named among its 25 languages; Mac only; closed source |
| 4 | **Anarlog** (ex-Hyprnote, MIT, ~10k stars) and **Meetily** (MIT open-core, ~33k stars) | The mindshare leaders among open-source tools | Meetily: speaker ID only in paid Pro, saves audio by default, no AEC found. Anarlog: local ASR Apple Silicon only, custom vocabulary in paid plan, voiceprints opt-in, SQLite storage |
| 5 | **Jamie** (German company, cloud) and **Granola** (cloud, market default) | Jamie owns the "audio streamed, not stored, GDPR" message and has cross-meeting Speaker Memory. Granola is what buyers compare against | Both process audio in the cloud. Granola has no voice recognition and is US-hosted |

---

## 2. Feature by feature: who else has it

| stenograf feature | Verdict | Who else has it |
|---|---|---|
| **No audio on disk by default** (RAM only, text out; recording opt-in) | **Unique locally** | Nobody local keeps audio in RAM only. Closest: OpenWhispr and Humla write to a temp file and delete it after processing (V, *re-checked*); Anarlog has an opt-in "zero-retention" mode (V, PR #7856). Granola and Jamie don't store audio but upload it first (V). Saving by default: Meetily (`auto_save: true` in `recording_preferences.rs`, V *re-checked*), MimicScribe (WAV, V *re-checked*), MacWhisper, ownscribe, Biscotti, CLNMR/MeetingAssistant, Wispr Flow Notetaker |
| **German legal framing** (§201 StGB, works council, GDPR) | **Unclaimed** | No product markets "no audio, ever" on §201 or works-council grounds. Caveat: dr-datenschutz.de argues even *buffering* can count as recording ([source](https://www.dr-datenschutz.de/protokollierung-von-teams-besprechungen/)); the GDD published a short paper on transcription ([GDD Kurzpapier 4](https://www.gdd.de/wp-content/uploads/2026/01/GDD-Kurzpapier-4-Gespraechstranskription.pdf)). The legal advantage is argued, not settled |
| **Cross-meeting voiceprints** | **Common** (cloud), **rare-ish** (local) | Cloud: Jamie Speaker Memory (V), Otter (S; subject of a biometric-privacy lawsuit), Fellow (V), Microsoft Teams voice profiles (V), Limitless (V, now Meta). Local: OpenWhispr (CAM++, auto-confirm ≥0.70, "Is this Alice?" prompt 0.55–0.70, V *re-checked*), Mac Note Taker (CAM++, cosine 0.65, renaming relabels past meetings, V *re-checked*), MimicScribe (V *re-checked*), pasrom (V *re-checked*), Anarlog (opt-in, V), CLNMR/MeetingAssistant, heed, millet |
| **Several people on one room mic, named across meetings** | **Rare** | Cloud with enterprise licences: Microsoft Teams Rooms (voice/face enrollment, Teams Premium/Copilot, V) and Fellow (workspace members only, V). Local: pasrom only (macOS). MimicScribe and Mac Note Taker don't say whether they split a shared mic. Anarlog 1.4.18 and Natively claim room-mic separation within a meeting (S) |
| **Echo cancellation, so laptop speakers work** | **Rare, no longer unique** | OpenWhispr (WebRTC AEC + text dedup, V *re-checked*), MimicScribe (AEC on both channels, V *re-checked*), Anarlog (skipped on headphones, V), Wisp (WebRTC AEC3, V), pasrom and Muesli (neural LocalVQE; pasrom's off by default), Wispr Flow (cloud, V). Plaud and Granola tell users to wear headphones |
| **Separate mic and system channels** | **Common** | Granola ("Me"/"Them"), Superwhisper, Whisper Notes, MimicScribe, pasrom, Humla, Wisp, tapeback, heed, millet. Remaining difference: diarizing *both* channels (stenograf, pasrom, CLNMR) vs only the system side (OpenWhispr, Granola) |
| **Nemotron 3 diarizer, up to 8 speakers** | **Common** | MacWhisper 15.2 (S), Whisper Notes (V), pasrom (V), Humla (default, V), CLNMR (V), MimicScribe (combined with pyannote, V) |
| **Live captions, then final transcript** | **Common** | MimicScribe, CLNMR, heed, Muesli, MacParakeet, tapeback, Wispr Flow. *Note on stenograf itself: by default the stop step reuses the live decodes (`reuse_live_finalize`, `session.py`); a from-scratch pass runs only with `--full-finalize` or after a live-worker error — see §6* |
| **Decode-time vocabulary boosting** | **Rare; German-compound handling unique** | pasrom (CTC boosting via FluidAudio — a second keyword-spotting model that corrects text after decoding, close to but not inside the decoder), Superwhisper (Argmax keyword boosting, up to 1,000 terms), cloud APIs (Speechmatics, AssemblyAI keyterms, Deepgram). Prompt hints only: OpenWhispr, ownscribe. Paid feature: Anarlog. Nobody mentions German compounds |
| **German first-class, on device** | **Common as "supported"** | Every Whisper- or Parakeet-v3 tool runs German. No independent German meeting benchmark found. stenograf's measured German work (biasing, glossary, int8 ruining German) is depth nobody else shows |
| **Local LLM notes on all three OSes, with templates** | **Common in open source** | Meetily (Ollama; templates are Pro), Anarlog, OpenWhispr. In-process MLX like stenograf's macOS path: Muesli (MLX Gemma) |
| **Plain files as the only storage** | **Rare cross-platform** | pasrom (macOS), tapeback (Obsidian markdown with YAML front matter, Linux), Ghost Pepper (macOS), ownscribe. Database-backed: Anarlog, Meetily, OpenWhispr, Humla, MacWhisper |
| **Linux with system-audio capture** | **Rare commercially, fairly common in open source** | No commercial notetaker supports Linux. Open source: OpenWhispr (PipeWire), Meetily (native PipeWire since v0.5.0), Anarlog (beta), tapeback, HushNote, millet (Linux-only capture). stenograf's real edge: device-clock-stamped capture through one helper on all three OSes (competitors most likely stamp on arrival — inferred) |
| **Measured low power during meetings** | **Unique as a published number** | Nobody publishes watts. MimicScribe states "about 30% of one CPU core" and 145 MB |
| **"Who spoke?" panel playing a voice sample** | **Not found anywhere** (?) | Absence from READMEs only; apps not tested. Jamie has "Speaker Clips" (a few seconds per speaker kept, V) — related |
| **Free and open source, nothing paywalled** | **Rare among the strong ones** | The strongest local rivals are closed and paid (MimicScribe, Mac Note Taker $149, MacWhisper, Whisper Notes, Superwhisper) or open-core with key features paid (Meetily Pro, Anarlog Pro). OpenWhispr and pasrom are fully free |

---

## 3. The landscape in detail

### 3a. Open-source and local tools

Star counts and release dates are live GitHub data from 2026-11-24.

| Tool | Licence · platforms | Mic/system | Live | Diarization | Cross-meeting voices | AEC | Vocab | Notes | Audio kept by default | Output | Stars / last release |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [OpenWhispr](https://github.com/OpenWhispr/openwhispr) | MIT · mac/win/linux | Separate capture, then merged with dedup (V) | Yes | pyannote-seg-3.0 + CAM++, system side only (V) | Yes, 3-tier matching, retroactive relabel (V) | WebRTC AEC + dedup (V) | Hints | llama.cpp | Temp file, deleted after diarization (V) | SQLite | 12.1k / v1.12.0, 2026-11-17 |
| [pasrom/meeting-transcriber](https://github.com/pasrom/meeting-transcriber) | MIT · macOS 14.2+ | Dual track, each diarized (V) | PoC only | FluidAudio incl. Nemotron 3 (V) | Yes (V) | LocalVQE, off by default; bleed detection on (V) | CTC boosting + normalization (V) | Ollama/LM Studio/Claude CLI | Yes, own retention policy (V) | `.md`/`.txt` files | 256 / v0.9.0, 2026-11-08 |
| [Anarlog](https://github.com/fastrepl/anarlog) (ex-Hyprnote) | MIT + commercial enterprise parts · mac; win/linux beta | Yes | Yes | On-device (PR #7244) | Opt-in WeSpeaker (PR #6647) | Yes, skipped on headphones (PRs #5218, #7273) | Pro only | Ollama/LM Studio | Zero-retention option (PR #7856); default ? | SQLite, markdown export | ~10k / 1.4.35, 2026-11-15 |
| [Meetily](https://github.com/Zackriya-Solutions/meetily) | MIT open-core · mac/win/linux | Mixed with ducking (S) | Yes | **Pro only** (V) | No | None found | No | Ollama | **Yes** (V) | App storage; markdown export Pro | ~33k / v0.5.0, 2026-11-14 |
| [Humla](https://github.com/michaelwilhelmsen/humla) | MIT · macOS 13+ | Two streams | Near-live | Nemotron 3 default + pyannote | No (per transcript) | ? | Biasing | Ollama | **No** — temp dir deleted; "Keep recorded audio" opt-in (V) | SQLite | 355 / v0.81.0, 2026-11-19 |
| [CLNMR/MeetingAssistant](https://github.com/CLNMR/MeetingAssistant) | No licence · macOS 15 | Two tracks | Yes, then full second pass | Nemotron 3 | CAM++, 20 s enrolment | Yes | German-oriented | Claude via Claude Code | Yes | JSON/MD/SRT | 0 stars — a near-mirror of stenograf's design, no adoption |
| [Muesli](https://github.com/Muesli-HQ/muesli) | MIT · macOS 14.2+ | Process tap | Opt-in | pyannote via FluidAudio | ? | LocalVQE | Yes | Ollama, MLX Gemma | Yes (for re-transcription) | DB + MD/PDF | 1.7k / v0.8.6, 2026-11-08 |
| [MacParakeet](https://github.com/moona3k/macparakeet) | GPL-3.0 · macOS | Mic/system/both | Preview then finalize | Yes | Experimental, off in releases | In development | Replacement-based | Ollama/LM Studio | Keep / N days / delete; default ? | Library + 8 formats | 797 / 0.8.10, 2026-10-14 |
| [Wisp](https://github.com/ppXD/Wisp) | MIT · mac/win | Fused "You vs Them" | Sub-second | sherpa-onnx | Within call only | WebRTC AEC3 | Term biasing | Local live copilot | Library | Library + MD | 170 / v0.5.3, 2026-07-28 |
| [heed](https://github.com/isjunrod/heed) | MIT · mac, linux partial | Stereo, one each | Mac only | Sortformer + WeSpeaker | Yes | ? | ? | Ollama | Yes | Browser UI | 20 |
| [millet](https://github.com/pretyflaco/millet) | GPL-3.0 · capture Linux only | L/R channels | No | WhisperX + pyannote | Voiceprint enrol | ? | ? | Local / private enclave | Yes (ffmpeg) | PDF/files | 375 / v0.23.2, 2026-11-05 |
| [ownscribe](https://github.com/paberr/ownscribe) | MIT · mac (linux via sounddevice) | Both | No | pyannote | ? | ? | Hotwords/prompt | Phi-4-mini built in | Yes | MD/JSON | 111 / v0.15.0, 2026-08-14 |
| [Biscotti](https://github.com/scosman/Biscotti) | Non-standard · macOS | Separate channels | ? | pyannote via SpeakerKit | Calendar names | Via channel isolation | Whisper only | Gemma 4 / llama.cpp | Yes, sold as a feature | ? | 541 / v1.0.0, 2026-10-08 |
| [Ghost Pepper](https://github.com/matthartman/ghost-pepper) | MIT · macOS | ScreenCaptureKit | Chunked | ? | No | No | ? | Local | Yes | Markdown files | 3.2k / v2.4.4, 2026-07-27 |
| [Natively](https://github.com/Natively-AI-assistant/natively-cluely-ai-assistant) | Source-available, personal use · mac/win | Separate | Yes | Pro only | ? | Leak fix | ? | Ollama | ? | DB | 3.4k / v3.2.1, 2026-11-19 |
| tapeback | Linux | Stereo mic L / others R | Optional, every 60 s, then full pass | Yes | Roadmap | Mic gating only | ? | llama.cpp | WAV linked from notes (S) | Obsidian MD + YAML | small |

File-only tools (no live meeting capture): noScribe (GPL, 2.3k), aTrain (AGPL,
1.2k, Uni Graz, GDPR-framed, offline faster-whisper + pyannote), Buzz (MIT,
23k), Vibe (MIT, 8.2k), Scriberr (paused), Whishper→anysub, Speech Note (Linux).
Not comparable: Screenpipe (whole-screen recorder, commercial licence), Pluely
and Glass (interview copilots), Amical and FluidVoice (dictation; Amical's
meetings "coming soon"), VoxMinutes (Chinese-focused).

### 3b. Commercial tools that run on the device

| Tool | Platforms | Processing | Mic/system | Live | Diarization | Cross-meeting voices | German | Audio kept | Output | Price |
|---|---|---|---|---|---|---|---|---|---|---|
| [MimicScribe](https://mimicscribe.app/) | macOS 15+, Apple Silicon | On device; notes via local Gemma, own endpoint, or Gemini proxy | Separate | Live, final seconds after call | Nemotron 3 + pyannote community-1 + grammar cues; vendor: 99% speech credited correctly | Profiles from ~30 s, auto-grow, multiple centroids per person; vendor: 96% returning voices named, 0 wrong in 779 trials | 25 languages, German not named | **WAV during recording** (crash recovery) | MD to Obsidian/Logseq/folder; MCP; Spotlight/Shortcuts/Siri | Free tier; Pro $50 launch / $75; cloud $10/month |
| [Mac Note Taker](https://macnotetaker.com/speaker-recognition) | macOS 14.2+ | On device; Ollama or own OpenAI key | Both captured | ? | pyannote-seg-3.0 on Neural Engine | CAM++ 192-dim, cosine 0.65; renaming relabels every past meeting | ? | ? | MD/JSON/SRT/txt + CLI | $149 lifetime |
| MacWhisper 15.2 | macOS | On device (Whisper, Parakeet, Qwen3-ASR); summaries cloud by default (S) | ? | Since 15.2 (S) | Nemotron 3, 8 speakers (S) | No — name autofill only (V) | Yes | Yes | App; Notion/Obsidian/Zapier | ~$69 Pro (S) |
| [Superwhisper Meetings](https://superwhisper.com/docs/modes/meeting) | macOS | Cloud default, local models available; speaker separation always local | Separate | Yes | Local | Not stated | Model-dependent | ? | App | Pro subscription |
| [Whisper Notes](https://whispernotes.app/blog/whisper-speaker-diarization) | Mac, iOS | On device | "Me"/"Others" | Me/Others only | Nemotron 3 after meeting | No | Yes | Yes | App | $14 one-time |
| [Hedy](https://hedy.ai/post/local-ai-meetings-hedy-3-2/) | Mac, Win (GPU), iOS | On-device ASR default, optional local AI | ? | Yes | "sometimes" (S) | ? | ? | On device | Device, optional sync | Free tier |
| Alter, Shadow | Mac | Local transcription (S) | ? | ? | Local (Alter) | ? | ? | ? | ? | Free + paid |

### 3c. Cloud notetakers (the mainstream)

All upload audio. Bot-free desktop capture is now the norm, not a
differentiator — even Otter, Fireflies, Fathom and Krisp offer it.

- **[Granola](https://granola.ai/security)** — Mac/Win/iOS/Android/Watch.
  Cloud ASR (Deepgram/AssemblyAI) and LLMs (OpenAI/Anthropic), AWS US, no EU
  residency. "Me"/"Them" channels, live transcript, audio not stored after
  transcription. Speaker names come from Zoom/Meet/Teams, not voices; can't
  separate people on a shared device (V). Workspace dictionary on Business
  plan. Free (30-day history), $14/$35 per user per month.
- **[Jamie](https://www.meetjamie.ai/en)** (Germany) — Mac/Win/iOS/web, cloud
  inside EEA/CH/UK, "audio streamed, not stored", Speaker Memory across
  meetings (described by a competitor as biometric profiles needing consent,
  S), optional "Speaker Clips", custom vocabulary 50–250 terms, webhooks/API/MCP.
  Free (10 notes); €21–39/month.
- **Krisp** — cloud; on-device transcription only on Enterprise (V), recordings
  stored with quotas. $8–15/month.
- **[Fellow](https://help.fellow.ai/en/articles/10657637-speaker-identification-for-remote-hybrid-meetings)**
  — cloud; separates several voices on one room mic, workspace members only.
- **[Microsoft Teams voice profiles](https://learn.microsoft.com/microsoftteams/rooms/voice-and-face-recognition)**
  — the reference for the hybrid-room case: enrolled voiceprints (express and
  passive enrollment), in-room people named on Teams Rooms devices, ≤10 people
  in the room advised, profiles deleted after a year unused, 25 enrollment
  languages incl. German. Needs Teams Rooms Pro or Teams Premium/Copilot.
- **Otter, Fireflies, Fathom, Read.ai, tl;dv, Sembly** — cloud, US. Otter has
  voiceprint tagging (S) and a biometric-privacy (BIPA) lawsuit; Fireflies is
  under a similar investigation (S).
- **Plaud Desktop** — cloud (Whisper large-v3 + Azure, S), recommends
  headphones against echo (V).
- **Notion AI Meeting Notes** — cloud, Business/Enterprise; shows a consent
  announcement.
- **Tactiq** — browser extension reading the platform's own captions; no audio.
- **Apple call recording (macOS Tahoe)** — Phone/FaceTime/Notes; recording
  kept; on-device status undocumented.
- **Limitless** — bought by Meta in December 2025; pendant discontinued.

### 3d. German and EU providers

Jamie (above) is the only one close in positioning. Sally (cloud, hosted in
Germany, desktop app since August 2026), tucan.ai (own ASR, on-premise option,
public-sector focus), Bluedot (EU hosting, sources conflict on default),
Leexi, MeetGeek, noota, Amberscript and Happy Scribe are all cloud tools,
mostly bot-based, none close to stenograf.

---

## 4. What competitors have that stenograf lacks

Grouped by fit with stenograf's locked product philosophy (`CLAUDE.md`: the
tool ends at transcript + notes in a visible folder; no meeting management,
browsing, playback, index or web UI).

**Excluded by the philosophy** — listed so future sessions know buyers expect
them, not as candidates:

- Search and AI chat across past meetings, semantic search (Granola, Jamie,
  MimicScribe, Humla, Muesli, Wisp, Anarlog). stenograf's answer: the
  folder is plain markdown, so Obsidian, Spotlight or any AI agent can already
  read it.
- An MCP server or app library exposing meetings (Granola, Jamie, MimicScribe,
  Meetily Pro, Krisp, Fireflies).
- Meeting playback with word highlighting (Humla), undo-history ledgers and
  access logs per transcript (MimicScribe).
- Team workspaces, shared folders, SSO/SCIM, admin retention policies,
  follow-up emails, CRM sync.
- Cloud sync to phone.

**Different products** — out of scope rather than excluded:

- Dictation / voice typing in the same app (OpenWhispr, MimicScribe, Muesli,
  MacParakeet, Ghost Pepper).
- Mobile apps for in-person recording, dedicated recorder hardware (Plaud).
- Live in-call AI copilots (Wisp, Natively, Hedy, Cluely) — also at odds with
  stenograf's power budget.
- Live translation (VoxMinutes).

---

## 5. Feature ideas worth considering

Ideas taken from competitors that fit the philosophy. **None is planned** —
this is a menu, not a backlog. Effort is a rough guess, not a price. Items
already on `DECISIONS.md`'s deferred list are cross-referenced there rather than
re-argued here.

| Idea | Seen in | Fit | Rough effort | Notes |
|---|---|---|---|---|
| **YAML front matter in `transcript.md`** (date, title, duration, language, speakers, meeting type) | tapeback, MimicScribe/pasrom Obsidian export | Good — makes every meeting folder Obsidian/Dataview-queryable without any index of our own | Small | Half exists: `[notes.export]` already writes a self-contained Obsidian note *with* front matter, but only for meetings that got notes. `transcript.md` itself (`Transcript.to_markdown`) starts straight with the first line |
| **Rename a voice → relabel earlier meetings too** | Mac Note Taker, OpenWhispr | Good — files stay the index; a CLI/finish-card action rewrites the label in older transcripts that carry that voice | Medium | Today `steno profiles assign` corrects one meeting; later meetings pick the name up, earlier ones keep `Remote-1`. Needs per-meeting embeddings kept with the transcript (check what `transcript.json` already holds) |
| **Mark a moment during the meeting** (hotkey or menu-bar item that drops a marker; the notes prompt is told to cover marked spots) | MimicScribe bookmarks | Strong — cheap, local, improves notes | Small–medium | Depends on the system-wide shortcut deferred in `PLAN-MUTE.md`; the menu-bar item works without it |
| **Meeting auto-detect / prompt to start when Zoom/Teams/Meet begins, auto-stop when it ends** | MimicScribe, pasrom, MacWhisper, OpenWhispr, Superwhisper, Meetily Pro | Good | Medium per platform | Already on `DECISIONS.md`'s deferred list ("meeting-mode auto-detect") |
| **Calendar: meeting title and attendee names from the local calendar** (EventKit on macOS), fed to the title field and the finish card's name suggestions | OpenWhispr, Mac Note Taker, Jamie, Biscotti | Good if read-only and local | Medium on Windows/Linux; **blocked on macOS as things stand** | Calendar access on macOS needs a usage-description key in `Info.plist`, and the frozen bundle's Info.plist is sealed into the cdhash that holds every user's microphone grant (`native/appbundle/README.md`). Adding the key means rebuilding the bundle and every macOS user granting the microphone again. Only worth it if bundled with other reasons to rebuild |
| **"Is this X?" confirm band** for voices just below the match threshold | OpenWhispr (0.55–0.70) | Good | Small | The finish card already shows an unknown voice as "Who is this?" and offers known names as you type (0.8.1); this would pre-fill the closest near-miss name as a suggestion. Check how often near-misses occur before building |
| **Consent helper**: a ready-to-paste chat line ("This meeting is transcribed locally; no audio is recorded") or a start-form reminder | Notion (consent announcement) | Strong — turns the no-audio design into something the user can say out loud | Small | Pure UX; directly supports the main selling point |
| **Network transparency statement / log**: say exactly which requests stenograf ever makes (daily PyPI check, model downloads) | MimicScribe "Network Log" | Good | Small as documentation; medium as a UI log | A README section may be enough |
| **Post-meeting hook**: run a user command with the meeting folder path when it is written (move to a vault, sync, notify) | Meetily Pro (webhooks/CLI) | Good — Unix-style, no integrations to maintain | Small | Replaces most "integration" requests |
| **More languages** — Parakeet v3 covers 25 European languages | MimicScribe (25), cloud tools (30–100+) | Possible | Medium–large: per-language notes prompts, glossary handling, eval | The German depth is a selling point; breadth risks diluting it. Only if users ask |
| **Signed one-click installer** | All paid Mac apps | Good for non-terminal users | Large ($99/yr Developer ID + notarization) | Already on `DECISIONS.md`'s deferred list ("Packaged signed installers") |
| **Multiple voiceprint centroids per person** (different rooms/mics) | MimicScribe | Maybe | Medium | stenograf already matches a profile by the *median* of its entries' scores, which covers some of this. Evaluate on the harness before building |
| **Screen-share snapshots in the transcript** (repeats skipped, faces blurred) | MimicScribe | Questionable — needs screen-recording permission and puts images in the folder | Large | Probably not; listed for completeness |

Previously declined and **not** to be re-opened from this research without
the trigger named in `DECISIONS.md`: automatic profile updates (MimicScribe's
"profiles grow with each meeting" is exactly that), joint speaker-attributed
ASR, ASR challenger models.

---

## 6. Positioning notes for stenograf's own texts

- **Lead with the combination**, not single features: "local, no audio on
  disk, names everyone — including several people in one room — on Mac,
  Windows and Linux, German-first, free."
- **Don't claim echo cancellation, voiceprints, separate channels, Nemotron or
  live+final as unique** — each has at least three other implementations.
- **README wording on the stop step**: it says stenograf runs "a high-accuracy
  re-transcription of the full in-memory buffer" when the meeting ends. By
  default the stop step reuses the live pass's decodes (`reuse_live_finalize`
  in `session.py`); a from-scratch pass runs only with `--full-finalize` or
  when the live worker dropped audio or failed. The live pass decodes the same
  VAD-packed windows the finalize would, and `eval/README.md` measures 0.0 %
  WER between them (the "reuse guarantee"), so accuracy is not at stake —
  only the wording.
- **The legal argument** (no recording → lighter §201/GDPR obligations) is a
  real differentiator but contested by at least one German commentary; phrase
  it as "no audio is ever written unless you ask for it" (a fact) rather than as
  legal advice. Do not say "nothing but text is written": with speaker
  separation on (the default since 0.8.1) every meeting folder also gets
  `voiceprints.json`, each speaker's voice embedding, which is what makes
  naming a voice after the meeting possible. Keeping it beside the transcript
  is deliberate (Daniel, 2026-11-24); a user who wants no person-specific data
  turns speaker separation off. The README says so.

---

## 7. Re-check triggers

Re-run this research when any of these happens:

- **OpenWhispr** starts diarizing the microphone channel or writes plain files
  — it would then match nearly the whole combination.
- **pasrom/meeting-transcriber** ships Windows or Linux, or turns echo
  cancellation on by default.
- **MimicScribe** ships Windows, names German, or stops writing WAV files.
- **Anarlog** makes local ASR cross-platform or moves vocabulary out of Pro.
- **Meetily** moves speaker identification into the free edition.
- Any tool starts marketing "no audio on disk" for German legal reasons.
- Six months pass (next: ~2027-05).
