# stenograf on Linux — what the desktop app was measured to do

Phase 8's app was designed, built and tuned on macOS; on **2026-07-25** it was
run for the first time on a real Linux session, end to end including a live
meeting, and the six problems that found were fixed the same day (single
instance, the localised documents dir, the desktop entry, the doubled window
title, the X11 `WM_CLASS`, the window-size clamp — `git log` has them). Later
the same day rung 0 of the ladder below was climbed and the two gestures only a
hand can make were made, which took the tray-less degrade path, the X11 taskbar
question, a light desktop, the rendered menu and the launcher gestures off the
unmeasured list without another machine — and found one more bug on the way
(*Open Stenograf*, greyed for a window that was merely buried). **No work item
is open here.** What is left is
this file's real job: the evidence, so
no future session re-measures it; the decisions, so none re-litigates them; the
desktops that have never run it and how to reach them without buying a machine;
and the recipe for observing a running GUI.

**The machine every number below comes from:** CachyOS on the GPD notebook
(`cachyos-gpdmini`, x86_64, kernel 7.1.4-1-cachyos-deckify), **KDE Plasma
6.7.3 on Wayland**, single panel `eDP-1` 1920×1080@120 at **scale 1.5** — i.e.
a 1280×720 *logical* desktop, which is small enough to catch layout
assumptions. PySide6 6.11.1 from the `gui` extra. Where a claim was measured
somewhere else (XWayland, DBus introspection) it says so.

---

## What works — do not re-verify

Every item here was observed on that session, not reasoned about:

- **The window.** Renders correctly and crisply at 150 % fractional scaling
  (the grab comes back at 1.5×, so Qt gets the fractional scale rather than
  rounding to 2). KDE draws a server-side titlebar carrying our icon; the task
  manager shows **one** entry with the app tile. The caption reads `stenograf`,
  once — `applicationDisplayName` is deliberately unset, or Qt appends it.
- **Launcher identity on Wayland.** KWin reports the window as
  `resourceClass=stenograf`, `desktopFileName=stenograf` — `setDesktopFileName`
  does exactly what `shortcut.py`'s `DESKTOP_FILE_NAME` comment claims.
- **The desktop entry.** Both variants (app and TUI) pass
  `desktop-file-validate` with no output, and `Icon=stenograf` resolves out of
  the user's own hicolor theme (`QIcon.fromTheme` finds it at 512×512 on the
  live session; under `QT_QPA_PLATFORM=offscreen` it does not, because that
  platform has no icon theme at all — not a lookup failure).
- **X11 `WM_CLASS`** (measured under XWayland, `QT_QPA_PLATFORM=xcb`): the class
  is `applicationName`, the instance is `argv[0]`'s basename — `"steno",
  "Stenograf"` from the console script, `"__main__.py", "Stenograf"` from the
  entry's `python -m stenograf`. Only the class is stable, and it is what
  `StartupWMClass=Stenograf` now declares; KWin agrees
  (`resourceClass=Stenograf` for the X11 window).
- **One app per user.** `gui.app.claim_single_instance` claims
  `/tmp/stenograf-$USER`; a second launch exits 0, prints nothing, and the first
  app's **hidden** window comes back on screen. One SNI item stays registered
  with the watcher, not two. A `SIGTERM`-killed instance leaves its socket
  behind and the next launch still starts (`removeServer` clears it).
- **The tray.** `isSystemTrayAvailable()` and `supportsMessages()` are both
  true; the status item registers as an SNI (`Id`/`Title` = `Stenograf`,
  `Category` = `ApplicationStatus`), shows the two commas in the brand inks on
  the dark panel, and **re-inks red while recording**.
- **The menu, over DBusMenu.** The `aboutToShow` relabelling design — the one
  thing headless tests could not reach — works: before `AboutToShow` the status
  entry is empty; after it reads `No meeting running`, and while a meeting runs
  `Recording · 0:45`, with *Start meeting…* disabled and *Stop* enabled.
- **The menu as Plasma draws it** (opened by hand, caught by a screenshot
  recorder rather than a keypress that would dismiss it). `Stop _& finalize` —
  Qt's DBusMenu mnemonic encoding of `Stop && finalize` — arrives on screen as
  **`Stop & finalize`**, so the convention holds end to end and nothing needs
  escaping differently. Labels, both separators and the greying all render as
  the exported properties say.
- **What the rendered menu found, and it was a real one.** *Open Stenograf* came
  out greyed while the window was merely **buried** behind another window:
  `isVisible()` means *mapped*, not looked at, and an occluded window is the
  shape the whole menu-bar mode is designed around (`app.py`'s redraw budget
  leans on exactly that). The same predicate suppressed the "Meeting finished"
  notification in the same common case. Both were fixed the same day — the entry
  is never disabled now, and the notification keys off `isActive()` — and both
  re-measured here afterwards: the exported menu reports *Open Stenograf*
  enabled with the window in front, and stopping a meeting from the menu bar
  with the window buried rendered **Meeting finished** with the folder as its
  body. `steno start`'s Stop and Quit were driven the same way, over
  `com.canonical.dbusmenu`'s `Event`, and both did the whole job (finalize,
  transcript on disk, clean exit).
- **The launcher gestures** (a hand on the trackpad, the one thing no harness
  reaches). Launching the K-menu entry shows a busy cursor before the window
  appears — `StartupNotify=true` doing its job. Clicking the same entry a second
  time while the app is running starts **no** second instance and brings the
  existing window forward; a left-click on the tray icon does the same. Which of
  the two mechanisms refuses the second launch — KDE's `SingleMainWindow=true`
  or our own `QLocalServer` claim — is not observable from outside, and both are
  in force.
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
  optional Ollama, and now names the output home it resolved
  (`/home/deck/Dokumente/Meetings` here, read out of `user-dirs.dirs`).
- **Startup**: 0.61 s from exec to first frame, warm (0.23 s imports, 0.29 s
  building the QML tree); the single-instance claim added `PySide6.QtNetwork`,
  which is 9 ms on top of QtCore. No launch-feedback problem worth solving.
- **The no-tray-host branch, with a window actually on screen** — rung 0 of the
  ladder below, climbed the same day. Under `dbus-run-session` the app reports
  `isSystemTrayAvailable()` false, `install()` returns `None`, and
  `setQuitOnLastWindowClosed` is left **true**: the stock-GNOME shape, driven
  against the real compositor rather than offscreen. `--tray` prints its warning
  and opens a window anyway, and closing that window exits 0.
- **Closing that window during a live meeting** — the case that decides whether
  the degrade path is *safe*, since there the close is the app's only exit. A
  meeting was started from the GUI's own setup form (a real recording replayed
  through the capture provider) and KWin's `closeWindow()` — the titlebar X —
  was pressed while it was recording. The app printed the stderr explanation,
  stopped capture, finalized, and wrote `transcript.{md,json,txt}` with the
  checkpoints cleaned up; `run()` returned 0 about a second after the click.
  No orphaned meeting, nothing for a user to rescue by hand.
- **X11 launcher identity is not only `WM_CLASS`** (measured under XWayland).
  Qt also exports `_KDE_NET_WM_DESKTOP_FILE=stenograf` **and**
  `_GTK_APPLICATION_ID=stenograf`, KWin reports `desktopFileName=stenograf` for
  the X11 window, and Plasma's task manager shows one entry carrying the app's
  own icon. Every desktop-file matcher — Plasma, GNOME Shell — therefore groups
  an X11 window with its launcher without consulting `StartupWMClass` at all;
  the key is the fallback for panels that know only `WM_CLASS`.
- **A light system theme**, without touching the session: Qt's built-in KDE
  theme reads `kdeglobals` out of `$XDG_CONFIG_HOME`, so one process can be
  handed Breeze Light while the desktop stays dark. The app's own screens render
  in the product palette unchanged; the file dialog follows the light desktop
  and is perfectly readable. The trade named in the decisions below, seen rather
  than assumed. (The titlebar in that grab is still the real session's — a
  client cannot fake the decoration.)
- **The event loop stays live after the finalize** (120 s of 1 Hz heartbeat,
  `phase=done`, `running=False`) — an earlier apparent freeze was an artefact of
  the observing shell, not the app.

**One local leftover:** this machine's pre-fix meeting sits in
`~/Documents/Meetings/meeting-20260725-215405`, off the resolved output home
now that it is `~/Dokumente/Meetings`. Nothing moves it — the resolver has no
legacy branch by design — so it is a `mv` whenever anyone cares.

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
  dialog, which follows a light desktop and stays readable there — seen, not
  assumed (above).
- **"Is the user looking at it?" is `isActive()`, never `isVisible()`.** The
  distinction cost a real bug (above) and it will cost another one: on both
  Wayland and X11 a window is *visible* the whole time it sits behind the video
  call, which is where this app expects to spend a meeting. Qt exposes no
  occlusion query on any platform, so focus is the only proxy there is.
- **Fixed `pixelSize`s stay.** Qt reports the desktop font as `Sans Serif 9`
  (no KDE platform-theme plugin in the wheel), and the app sets explicit sizes
  anyway. Scaling is honoured; a user's *font size* preference is not — the
  same trade the macOS build makes.
- **The window-size clamp is a guard, not a fit.** `Main.qml` clamps its initial
  1000×680 to `Screen.desktopAvailable*`, which on **Wayland is the whole panel**
  — a client cannot learn panel geometry there, so 720 clamps nothing — and on
  X11 is the work area (690). Both are *content* sizes while the compositor fits
  the decorated frame, so KWin's own shrink to 1000×690 is what keeps the window
  on screen either way, and the layout holds at that size. Chasing the
  decoration height from QML is not worth it.

---

## Still unmeasured

Nothing below is worth coding against blind. Rung 0 and two minutes of
trackpad time shortened this list to one line the same day; the section after
it says how to reach what is left without buying a machine:

- **Stock GNOME (Wayland) without the AppIndicator extension** — *narrowed*.
  The degrade-to-window path itself is now measured with a window on screen
  (above), so what is left is GNOME's own rendering, and the nastier case rung 0
  cannot stage: a watcher registered on the bus with nothing actually drawing
  the item.
- **A real X11 session** — *narrowed*. The matching mechanism is measured
  (above), so what is left is a panel that matches on `WM_CLASS` alone — XFCE —
  and a launcher pinned to a taskbar rather than a window merely appearing in
  one.
- **Any other desktop** (XFCE, Cinnamon) — which is rung 1 below, and all that
  is left.

---

## Reaching those desktops — containers, not virtual machines

The list above reads like it needs more machines. It does not. Every item on it
is a fact about a **session** — which shell owns the panel, what is registered
on the session bus, which display protocol the window speaks, which distro
packaged the userland around it — and not one is a fact about a kernel.
Containers share the host kernel and bring their own userland and their own
session, which is precisely the axis these items vary along; a VM adds a
kernel, a display manager and ten gigabytes, and buys nothing the list asks
for. Nothing needed here is installed on this machine — `podman`, `distrobox`,
`xorg-server-xephyr` and `weston` are all in `extra`, all absent (checked
2026-07-25).

Climb in order and stop when it stops paying; the curve flattens hard after
rung 1.

**Rung 0 — an isolated session bus. CLIMBED 2026-07-25; its findings are in
"What works" above.** `QSystemTrayIcon.isSystemTrayAvailable()` keys off
`org.kde.StatusNotifierWatcher` on the **session bus**, not off the compositor,
so a fresh bus *is* a desktop with no tray host:

```
$ uv run --extra gui python probe.py                    # the real Plasma session
platform: wayland | bus: /run/user/1000/bus   | trayAvailable: True
$ dbus-run-session -- uv run --extra gui python probe.py
platform: wayland | bus: /tmp/dbus-VvATr5yLPn | trayAvailable: False
```

The window still goes to the real compositor, which is the half
`test_no_tray_host_means_no_status_item` cannot reach: offscreen proves the
branch is taken, not that closing the window then really quits the app and
leaves no meeting orphaned. Both were driven this way and both hold — see the
two bullets above. Two things learned climbing it:

- **A meeting has to be running for the interesting half.** An empty app exits
  on the close trivially; the claim worth testing is that the *finalize* still
  happens, and it does, off `run()`'s own `join_meetings` after the event loop
  has already stopped. The transcript is written by the worker thread at the
  `finalized` event, so the dead event loop costs nothing.
- **A fresh bus activates its own `xdg-desktop-portal`**, and Qt's registration
  against it fails differently there — `Could not register app ID: Connection
  already associated with an application ID`, even with the desktop entry
  installed. An artefact of the staged session, not a regression of the
  pre-`steno setup` warning above it.

**What it is not:** GNOME's own rendering, nor the nastier case where a watcher
is registered but nothing draws the item. It also strips the notification daemon
out of that session, so it is a probe for this one branch and not a general
GNOME stand-in.

**Rung 1 — one container, and most of the remaining value.** `podman` +
`distrobox`, an Ubuntu LTS box, `Xephyr` and an XFCE session inside it. One
environment collects **a `WM_CLASS`-only taskbar** (the one X11 matcher rung 0's
XWayland grab does not stand in for, since Plasma and GNOME both read the
desktop-file properties Qt sets), **the SNI menu drawn by a panel that is not
Plasma's** (Plasma's own rendering is measured, mnemonic and all), and **another
desktop entirely**. It is also the only rung
that exercises *userland*: a stable-distro glibc against the PySide6 and onnxruntime
wheels, `parec` out of `pulseaudio-utils` rather than `pipewire-pulse`, and
`desktop-file-validate` plus the icon cache as a distro actually ships them.
XFCE is the right second desktop precisely because it has SNI **and** X11 where
GNOME has neither: between them they span both sides of every conditional in
`tray.py`. The launcher-gesture item splits across this line — `StartupNotify`
is observable there, but `SingleMainWindow` is a KDE key, so watching it do
something is a hand on *this* trackpad, not another desktop.

**Rung 2 — nested GNOME, once, as a check on rung 0.**
`gnome-shell --wayland --nested` in a Fedora box, to confirm the real shell
behaves the way an empty bus predicts. Finickier than XFCE; worth one sitting,
not a standing environment.

**Rung 3 — a virtual machine.** Only if something session-*level* — display
manager, autostart, portal handoff — misbehaves in a way a nested session
cannot show. Expect never to climb here. Windows is not an argument for one
either: its three open items are hardware items, and the notebook exists.

**What no amount of this reaches:** AEC over real speakers, GPU vendor tiers,
real PipeWire devices, HiDPI panels, and the deferred watt number. A VM's audio
device and GPU are fictions. Those stay on real machines, which is where they
already are.

**The by-product worth having.** The recipe below is KWin scripting, i.e.
Plasma-only. A container has no KWin and forces protocol-generic probes instead
— `xprop` and `wmctrl` for X11, `busctl`/`gdbus` for SNI and DBusMenu — which
then work on Plasma too, and are the form these checks have to be in before any
of them can live in CI. `capture-linux` in `.github/workflows/ci.yml` already
proves that pattern end to end: stand a session up inside the job, drive it,
assert on what comes out.

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
- **Keep test meetings out of the documents folder**: `STENOGRAF_DATA=<scratch>`
  with a `settings.toml` holding `[output] dir = <scratch>/meetings`. Safe for a
  throwaway run — `data_dir()` holds settings and speaker profiles, *not* the
  model cache (that is the HF cache, 2.6 GB here, and moving it would re-download
  everything).
- **Give the GUI a meeting without a microphone.** `flow.py` deliberately has no
  replay (a developer flag), so patch the seam the run uses:
  `loaders.make_provider` is called with `replay=None`, and a wrapper that
  substitutes a wav path replays it paced to wall clock. That is the hardware
  boundary the `verify` skill says to fake at. A previous meeting's `audio.wav`
  is the best source — real speech, real ASR — repeated a few times with the
  stdlib `wave` module when one is too short to leave room to act.
- **Drive the whole of `run()`, not just `build()`, when identity matters.**
  Construct the `QApplication` first (`run()` reuses `QApplication.instance()`),
  arm a `QTimer` state machine, then call `run()`: the application name, the
  desktop file name, the window icon and the single-instance claim are all set
  *inside* it, so a harness that calls `build()` directly gets a nameless,
  generic-icon app and will mislead anyone reading its screenshots.
- **A light desktop for one process**: `XDG_CONFIG_HOME=<scratch>` with
  `/usr/share/color-schemes/BreezeLight.colors` copied in as `kdeglobals`. Qt's
  built-in KDE theme reads it; the real session is untouched. QML's `FileDialog`
  can be opened from the harness — walk the root object's children for a
  class name containing `FileDialog` and `QMetaObject.invokeMethod(dialog,
  "open")` — since a dialog nobody clicks is the only system-themed surface
  there is to look at.
- **Window facts come from KWin scripting**, not from guesswork: write a `.js`
  that walks `workspace.windowList()` and `print()`s
  `resourceClass`/`resourceName`/`desktopFileName`/`caption`, load it with
  `gdbus … org.kde.kwin.Scripting.loadScript`, run
  `/Scripting/Script<N>.run`, read the output from `journalctl --user`. The
  same script can `workspace.activeWindow = w` to raise a window, and
  `w.closeWindow()` to press the titlebar X — the honest way to test a close,
  and it works on an app running against an isolated session bus, because KWin
  is reached over the *real* bus while the window is reached over Wayland.
  **A loaded script runs once** — re-running needs a fresh `loadScript`.
- **Tray facts come from DBus**: the item's properties from
  `org.kde.StatusNotifierItem`, the menu from
  `busctl --user call <name> /MenuBar com.canonical.dbusmenu GetLayout iias 0 -- -1 …`,
  and `AboutToShow` can be called directly to prove the relabelling path.
  `Activate ii x y` simulates the left click. `RegisteredStatusNotifierItems` on
  `org.kde.StatusNotifierWatcher` lists the items; map a bus name to a process
  with `org.freedesktop.DBus.GetConnectionUnixProcessID` — and read that reply
  with care, since `tr -dc '0-9'` glues the `uint32` in the D-Bus signature onto
  the front of the pid.
- **Screenshots**: `spectacle -b -n -f -o out.png` for the whole screen (`-a`
  grabs the *active* window, which is usually not ours), then `magick … -crop`
  to inspect the panel and the tray at 8×.
- **What is not installed here**: `xwininfo` (use `xprop -name <title>` and KWin
  scripting instead), `bc`, `/usr/bin/time`.
- **Two traps**: `pkill -f "…stenograf…"` matches the agent's own shell command
  and kills the session — resolve PIDs with `pgrep` and kill by number, one
  number per `kill`; and a backgrounded Python harness needs `python -u`, or its
  output dies with the process and looks like a hang in the app.
