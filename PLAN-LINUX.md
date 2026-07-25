# stenograf on Linux — what the desktop app was measured to do

Phase 8's app was designed, built and tuned on macOS; on **2026-07-25** it was
run for the first time on a real Linux session, end to end including a live
meeting, and the six problems that found were fixed the same day (single
instance, the localised documents dir, the desktop entry, the doubled window
title, the X11 `WM_CLASS`, the window-size clamp — `git log` has them). **No
work item is open here.** What is left is this file's real job: the evidence, so
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
  entry is empty; after it reads `No meeting running`, with *Open Stenograf*
  and *Stop* correctly disabled. Qt exports `Stop && finalize` as
  `Stop _& finalize`, which is the DBusMenu mnemonic convention (`_` marks the
  accelerator), so Plasma should render `Stop & finalize` — *the rendered menu
  was never looked at, only the exported strings*; see below.
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
  dialog, which will follow a light desktop — acceptable.
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

Nothing below is worth coding against blind. The section after it says how to
reach each one without buying a machine:

- **Stock GNOME (Wayland) without the AppIndicator extension** — the
  degrade-to-window path (`install()` returns `None`, closing the window quits).
  This is the branch the whole tray design leans on and it has never run *on a
  real desktop*; `tests/test_gui.py` takes the branch offscreen, and rung 0
  below takes it with a window on screen, without GNOME.
- **A real X11 session** — whether the taskbar groups the window with its
  launcher now that `StartupWMClass` names the class Qt actually sets.
- **The launcher gesture itself.** The single-instance path was driven with two
  `python -m stenograf --gui` launches, not with two clicks on the K-menu entry,
  so `StartupNotify=true` and `SingleMainWindow=true` are validated as keys but
  not observed doing anything.
- **The rendered tray menu** — only the DBus-exported labels were read. There is
  no way to open the systray applet's menu over DBus (`ContextMenu` asks the
  *app* to draw one, which Qt leaves to the host) and no pointer-injection tool
  on this machine, so this needs a human hand on the trackpad.
- **Any other desktop** (XFCE, Cinnamon) and a **light** system theme.

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

**Rung 0 — an isolated session bus. Nothing to install, and it lands today.**
`QSystemTrayIcon.isSystemTrayAvailable()` keys off `org.kde.StatusNotifierWatcher`
on the **session bus**, not off the compositor, so a fresh bus *is* a desktop
with no tray host. Measured here 2026-07-25:

```
$ uv run --extra gui python probe.py                    # the real Plasma session
platform: wayland | bus: /run/user/1000/bus   | trayAvailable: True
$ dbus-run-session -- uv run --extra gui python probe.py
platform: wayland | bus: /tmp/dbus-VvATr5yLPn | trayAvailable: False
```

The window still goes to the real compositor, so this drives the GNOME-shaped
branch — `install()` returns `None`, `setQuitOnLastWindowClosed` is left true,
`--tray` prints its warning and opens a window anyway — **with a window
actually on screen**, which is the half `test_no_tray_host_means_no_status_item`
cannot reach: offscreen proves the branch is taken, not that closing the window
then really quits the app and leaves no meeting orphaned. Run the observation
harness below under `dbus-run-session`, close the window, assert the process
exits. **What it is not:** GNOME's own rendering, nor the nastier case where a
watcher is registered but nothing draws the item. It also strips the
notification daemon and the portal out of that session, so it is a probe for
this one branch and not a general GNOME stand-in.

**Rung 1 — one container, and most of the remaining value.** `podman` +
`distrobox`, an Ubuntu LTS box, `Xephyr` and an XFCE session inside it. One
environment collects **a real X11 session** (does the taskbar group the window
with its launcher now that `StartupWMClass` names the class Qt actually sets),
**the rendered SNI menu** on a panel that is not Plasma's — including whether
`Stop _& finalize` really arrives as `Stop & finalize` — **a light system
theme**, and **another desktop entirely**. It is also the only rung that
exercises *userland*: a stable-distro glibc against the PySide6 and onnxruntime
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
  with a `settings.toml` holding `[output] dir = <scratch>/meetings`.
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
