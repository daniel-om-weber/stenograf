# stenograf on Linux — the desktop app, measured and still to fix

Phase 8's app was designed, built and tuned on macOS; `PLAN.md` recorded it as
*never run on a real desktop* off that platform. On **2026-07-25** it was run
for the first time on a real Linux session, end to end including a live
meeting. This file holds what that found: the parts that already work (so no
future session re-measures them), and the work that is left, worst first.

**The machine every number below comes from:** CachyOS on the GPD notebook
(`cachyos-gpdmini`, x86_64, kernel 7.1.4-1-cachyos-deckify), **KDE Plasma
6.7.3 on Wayland**, single panel `eDP-1` 1920×1080@120 at **scale 1.5** — i.e.
a 1280×720 *logical* desktop, which is small enough to catch layout
assumptions. PySide6 6.11.1 from the `gui` extra. Where a claim was measured
somewhere else (XWayland, DBus introspection) it says so.

---

## What already works — do not re-verify

Every item here was observed on that session, not reasoned about:

- **The window.** Renders correctly and crisply at 150 % fractional scaling
  (the grab comes back at 1.5×, so Qt gets the fractional scale rather than
  rounding to 2). KDE draws a server-side titlebar carrying our icon; the task
  manager shows **one** entry with the app tile.
- **Launcher identity on Wayland.** KWin reports the window as
  `resourceClass=stenograf`, `desktopFileName=stenograf` — `setDesktopFileName`
  does exactly what `shortcut.py`'s `DESKTOP_FILE_NAME` comment claims. The X11
  half is *not* fine; see item 5.
- **The tray.** `isSystemTrayAvailable()` and `supportsMessages()` are both
  true; the status item registers as an SNI (`Id`/`Title` = `Stenograf`,
  `Category` = `ApplicationStatus`), shows the two commas in the brand inks on
  the dark panel, and **re-inks red while recording**.
- **The menu, over DBusMenu.** The `aboutToShow` relabelling design — the one
  thing headless tests could not reach — works: before `AboutToShow` the status
  entry is empty; after it reads `No meeting running`, with *Open Stenograf*
  and *Stop* correctly disabled. Qt exports `Stop && finalize` as
  `Stop _& finalize`, which is the DBusMenu mnemonic convention (`_` marks the
  accelerator), so Plasma should render `Stop & finalize` — *the rendered menu
  was never looked at, only the exported strings.* Worth one eyeball.
- **Tray → window on Wayland.** `Activate` on the SNI both shows **and focuses**
  the window; KWin's focus-stealing prevention does not eat
  `requestActivate()`. (`set_dock_icon` is a no-op here, as intended.)
- **Notifications.** `showMessage` on finish renders with app name *Stenograf*,
  our icon and the folder path as the body.
- **Closing the window during a live meeting.** Close → hides, meeting keeps
  recording, tray goes red, and *Stop & finalize* **from the menu bar** ran the
  finalize and wrote the transcript with no window on screen at any point.
- **A real meeting through the GUI.** The setup form → capture via
  `parec`/PipeWire → live pass → Stop → finalize → `transcript.{md,json,txt}`
  (+ `audio.wav` when asked) → the DONE screen with diarized entries. Three
  runs, one with real speech in the room. `steno doctor` is green apart from
  optional Ollama.
- **Startup**: 0.61 s from exec to first frame, warm (0.23 s imports, 0.29 s
  building the QML tree). No launch-feedback problem worth solving.
- **The event loop stays live after the finalize** (120 s of 1 Hz heartbeat,
  `phase=done`, `running=False`) — an earlier apparent freeze was an artefact of
  the observing shell, not the app.

---

## Work items

### 1. Nothing stops a second instance — the one real bug

**Symptom.** Launching the desktop entry twice starts two independent apps
(measured: PIDs 13701 and 13827, **two** SNI items registered with the
watcher). Both have a tray icon, both can start a meeting, both would open the
microphone and write their own meeting folder.

**Why it is now load-bearing.** Step 6 made closing the window *hide* it, so
the natural gesture for "bring stenograf back" — click the launcher again — is
exactly the gesture that produces the second copy. macOS is covered by
LaunchServices (measured in `PLAN.md` step 6: a second `open` starts no second
instance, and the app answers the reopen as `ApplicationActivate`); Linux and
Windows have no equivalent. `grep` confirms there is no lock, `QLocalServer` or
`QSharedMemory` anywhere in the tree.

**Fix.** In `gui/app.py: run()`, before building anything: claim a per-user
name (`stenograf-$UID`, `$XDG_RUNTIME_DIR` is the natural home) with
`QLocalServer`. If the connect succeeds, the app is already running — send it
`show`, print nothing, exit 0. If it fails, `removeServer()` (a crashed
instance leaves a stale socket) and listen; incoming connections call
`gui.show_window()`, which is the Linux/Windows counterpart of `Tray._reopened`.
Do it unconditionally rather than Linux-only: a terminal `steno --gui` can
start a second copy on macOS too, beside the bundle.

**Test.** Factor the claim into its own function so a headless test can call it
twice in one process and assert the second call reports "already running";
`QLocalServer` needs no display.

**Then** add `SingleMainWindow=true` to the desktop entry (item 3) — KDE honours
it, everything else ignores it.

### 2. `~/Documents/Meetings` is the wrong folder on a localised desktop

**Symptom.** `output.default_output_home()` hardcodes `Path.home() /
"Documents" / "Meetings"`. On this machine `xdg-user-dir DOCUMENTS` is
`/home/deck/Dokumente`, so the tool silently creates a *second* documents tree
that the file manager's sidebar does not list.

**Why it outranks every GUI detail.** The product lock is "the filesystem is
the index" — the folder has to be the one the user's file manager shows.
macOS is unaffected (`~/Documents` is the real path with a localised display
name); Windows has the same class of problem through OneDrive redirection,
which `shortcut.py` already handles for the Desktop but `output.py` does not
for Documents.

**Fix.** Resolve the documents directory in one place: on Linux read
`XDG_DOCUMENTS_DIR` out of `$XDG_CONFIG_HOME/user-dirs.dirs` (expanding
`$HOME`), everywhere else keep `~/Documents`. Read the file rather than
shelling out to `xdg-user-dir`, which is not installed everywhere.

**Decision to take first:** what happens to an existing `~/Documents/Meetings`
on a machine that also has a localised documents dir. Recommendation: prefer
the XDG dir, and have `doctor` name the output home it resolved (it already
prints a settings line) so a moved folder is discoverable rather than
mysterious. A permanent "use the legacy path if it exists" branch is the thing
to avoid — one always-used resolver, and a release note.

**Also update the places that name the default in prose:** `output.py` (module
docstring + `default_output_home`), `settings.py` (three), `cli/notes.py`,
`cli/transcribe.py` (two), `README.md` (four).

**Test.** Monkeypatch `HOME`/`XDG_CONFIG_HOME` with and without a
`user-dirs.dirs`, assert both resolutions.

### 3. The desktop entry needs four small fixes

All in `shortcut.py` (`_LINUX_DESKTOP`, `_install_desktop_entry`), all cheap:

- **`Categories=AudioVideo;Audio;Utility;` names two main categories**, and
  `desktop-file-validate` warns the app may appear **twice** in the menu.
  `AudioVideo;Audio;Recorder;` is the idiomatic set (`Recorder` is an
  additional category that requires `AudioVideo`).
- **`Icon=` is an absolute path into site-packages.** It works today, but it
  points inside a venv that a reinstall can move, and it gives the notification
  daemon a path instead of a themed name. Install
  `assets/icon.png` (it is 512×512) to
  `$XDG_DATA_HOME/icons/hicolor/512x512/apps/stenograf.png` and write
  `Icon=stenograf`. A scalable SVG would be nicer still; the geometry exists in
  `native/appbundle/icon.svg`, and a *new* sibling file under `assets/` is
  allowed by the bundle freeze — but one PNG is the single path.
- **No `StartupNotify=true`.** Startup is 0.6 s, so this is polish, not rescue.
- **`SingleMainWindow=true`**, once item 1 lands.

**Test.** `tests/test_shortcut.py` already has a `_linux` helper — extend it:
the icon file is installed, `Icon=stenograf`, exactly one main category,
`StartupNotify`.

### 4. The title bar reads "stenograf — Stenograf"

Qt appends `applicationDisplayName` to the window title, so the KDE titlebar
and task manager show the name twice (measured: KWin caption
`stenograf — Stenograf`; `Main.qml` sets `title: "stenograf"` and
`gui/app.py` sets both `setApplicationName` and `setApplicationDisplayName`).
Drop one — the display name is the one to lose, since the tray, the SNI title
and the notification app name all come from `applicationName`. **Check the
macOS window title after the change**, since it is the same code path there.

### 5. On X11 the window class is not what the entry declares

**Measured under XWayland** (`QT_QPA_PLATFORM=xcb`, `xprop`):
`WM_CLASS = "steno", "Stenograf"` — the instance name is `argv[0]`'s basename,
the class is `applicationName`. The entry says `StartupWMClass=stenograf`, so
matching an X11 window back to its launcher relies on the shells'
case-insensitive fallbacks (KWin lowercases `resourceClass`; GNOME retries a
lowercased lookup) rather than on the declared string.

**Fix.** Write `StartupWMClass=Stenograf` — the class Qt actually sets — and
correct the `DESKTOP_FILE_NAME` docstring, which currently reasons the other
way round. `StartupWMClass` is consulted only for X11 windows, so this cannot
disturb the Wayland matching that already works.

**Not fixed by this:** the claim is still unverified on a *real* X11 session
(see below).

### 6. Small screens: the default window is bigger than this desktop

`Main.qml` opens 1000×680; the logical desktop here is 1280×720 minus a 46 px
panel, so KWin shrank the frame to 1000×690 and the content to ~662. Nothing
was cut off and the layout held, but clamping the initial size to
`Screen.desktopAvailableWidth/Height` is a two-line change worth making with
the other cosmetics.

---

## Decisions taken here — do not re-litigate

- **File dialogs stay Qt's own.** The `FileDialog` that opens belongs to our
  process (measured: the dialog window's class is `stenograf`, modal, 598×491),
  i.e. Qt Quick's fallback, not the KDE dialog and not the portal — the
  PySide6 wheel ships `libqxdgdesktopportal.so` but no KDE platform theme, and
  nothing selects the portal theme. It renders dark (Qt's built-in KDE theme
  reads `kdeglobals`), is perfectly usable, and is identical on every desktop.
  `QT_QPA_PLATFORMTHEME=xdgdesktopportal` would buy native dialogs with
  Places/Recent/KIO at the cost of a second process, a portal dependency and a
  different palette/font path per desktop. Not worth it; revisit only if a
  user asks for their file manager's places.
- **The portal warning before `steno setup` is accepted.** Until the entry
  exists, every launch prints `qt.qpa.services: Failed to register with host
  portal … App info not found for 'stenograf'`; installing the entry makes it
  disappear (both states measured). It is Qt reporting a missing desktop entry,
  and setup is what installs one.
- **The app stays dark on a light desktop.** Same as macOS: the palette is the
  product's, not the system's. The only system-themed surface is the file
  dialog, which will follow a light desktop — acceptable.
- **Fixed `pixelSize`s stay.** Qt reports the desktop font as `Sans Serif 9`
  (no KDE platform-theme plugin in the wheel), and the app sets explicit sizes
  anyway. Scaling is honoured; a user's *font size* preference is not — the
  same trade the macOS build makes.

---

## Still unmeasured — needs a different desktop

Nothing below is worth coding against blind:

- **Stock GNOME (Wayland) without the AppIndicator extension** — the
  degrade-to-window path (`install()` returns `None`, closing the window quits).
  This is the branch the whole tray design leans on and it has never run.
- **A real X11 session** — whether the taskbar groups the window with its
  launcher after item 5's change.
- **Any other desktop** (XFCE, Cinnamon) and a **light** system theme.
- **The rendered tray menu** — only the DBus-exported labels were read.

---

## How to observe the app on this machine

The `verify` skill covers the CLI/TUI headlessly; none of it reaches a real
desktop. What worked here, and the traps paid for:

- **Drive the real app from a harness**, not by hand: build with
  `stenograf.gui.app.build()`, `window.show()`, then a chain of
  `QTimer.singleShot` calls that navigate (`gui.open("Doctor")`), submit the
  setup form with the same map QML sends (`setup.start({...})`), and save
  `window.grabWindow()` per screen. `grabWindow` needs no compositor
  permission and captures at the real scale factor.
- **Keep test meetings out of `~/Documents`**: `STENOGRAF_DATA=<scratch>` with a
  `settings.toml` holding `[output] dir = <scratch>/meetings`.
- **Window facts come from KWin scripting**, not from guesswork: write a `.js`
  that walks `workspace.windowList()` and `print()`s
  `resourceClass`/`resourceName`/`desktopFileName`/`caption`, load it with
  `gdbus … org.kde.kwin.Scripting.loadScript`, run
  `/Scripting/Script<N>.run`, read the output from `journalctl --user`. The
  same script can `workspace.activeWindow = w` to raise a window, and
  `w.closeWindow()` to press the titlebar X. **A loaded script runs once** —
  re-running needs a fresh `loadScript`.
- **Tray facts come from DBus**: the item's properties from
  `org.kde.StatusNotifierItem`, the menu from
  `busctl --user call <name> /MenuBar com.canonical.dbusmenu GetLayout iias 0 -- -1 …`,
  and `AboutToShow` can be called directly to prove the relabelling path.
  `Activate ii x y` simulates the left click.
- **Screenshots**: `spectacle -b -n -f -o out.png` for the whole screen (`-a`
  grabs the *active* window, which is usually not ours), then `magick … -crop`
  to inspect the panel and the tray at 8×.
- **Two traps**: `pkill -f "…stenograf…"` matches the agent's own shell command
  and kills the session — resolve PIDs with `pgrep` and kill by number; and a
  backgrounded Python harness needs `python -u`, or its output dies with the
  process and looks like a hang in the app.
