# Stenograf.app — the frozen launcher bundle

The macOS desktop launcher `steno setup` installs into `~/Applications`
(Phase 8 step 5). Sources live here; the built product is committed at
`src/stenograf/assets/Stenograf.app` and ships inside the wheel, which is the
opposite of how `helper/` and `stenodiar/` work — and the reason is the whole
design.

## Why it is frozen

macOS records the app's microphone and system-audio permissions against the
**cdhash of `Contents/MacOS/Stenograf`, and nothing else**: no bundle
identifier, no signing anchor, no path (measured on 2026-07-25 by reading
`TCC.db` and decoding the requirement blob with `csreq -r <blob> -t`). The
Info.plist and the icon are sealed into that hash too, through the code
directory's special slots.

Three consequences, all load-bearing:

- **Nothing may be generated per machine or per release.** A bundle assembled or
  re-signed at install time is a different app to TCC on every install, and a
  rebuild in a later release re-prompts everyone who upgrades. So the binary is
  compiled once, signed once, and committed with its signature; `shortcut.py`
  only ever copies it.
- **Whatever might change lives outside.** The stub reads the command to launch
  from `~/Library/Application Support/stenograf/launch-target` at every launch
  (one argv element per line; blank lines and `#` comments ignored; `~/`
  expanded). `steno setup` rewrites that file, never the bundle. With no file
  the stub falls back to `~/.local/bin/steno --gui`, which is why the CLI has to
  keep accepting `--gui` forever.
- **Adding the icon later would have re-prompted everyone**, so the layout was
  frozen complete — icon included, and with the Info.plist keys the menu-bar
  mode needs (see below) — before any user was asked for permission.

## Why it spawns instead of exec'ing

The main executable **may not be a script and may not `exec`**. A `#!/bin/sh`
wrapper that execs Python makes the process launchd started *become* the
interpreter — a Mach-O outside the bundle — and TCC then drops the bundle
identity and path-keys the grant to the shared uv `python3.13` instead
(`client_type=1`, prompt titled "python3.13"). That is worse than granting
Terminal: every uv-managed tool shares that interpreter, and a `uv python`
upgrade moves the path and re-prompts.

`main.c` therefore `posix_spawn`s the real program and stays alive as its
parent. Children inherit the responsible process, so `stenocap`'s request is
attributed to `dev.stenograf.app` and rebuilding `stenocap` never re-prompts.

## What else the stub does, and why it is worth the bytes

Everything here exists because a GUI launch has no terminal to complain into,
and this is the only chance to build diagnostics in:

- The child's stdout and stderr go to `~/Library/Logs/Stenograf.log`
  (truncated past 1 MiB), and a non-zero exit raises an alert naming it.
  Without that, a failed launch from the Dock is completely silent.
- A missing launch target raises its own alert rather than doing nothing.
- SIGTERM/SIGINT/SIGHUP are forwarded to the child so logout doesn't orphan the
  window — and the handler restores the default disposition once the child is
  gone, or the stub would be unkillable while an alert is on screen.
- `PATH` gains `~/.local/bin`, `/opt/homebrew/bin` and `/usr/local/bin`
  (appended, so system copies still win): launchd hands a GUI app a bare PATH.
- `STENOGRAF_APP_BUNDLE` marks the launch as coming from the app.

## The grant, as actually stored (2026-07-25)

Verified against this exact bundle: starting capture from the app prompted
once, and `TCC.db` then held

    kTCCServiceMicrophone    dev.stenograf.app  client_type=0  auth_value=2
    kTCCServiceAudioCapture  dev.stenograf.app  client_type=0  auth_value=2

with the requirement `cdhash H"1651c78f…" or cdhash H"b0516786…"` — **both
slices of the universal binary**, so the same grant follows the app across an
Intel/Apple-Silicon migration. No `python3.13` row appeared, which is the
failure this design exists to avoid. The step-2 test bundle's older grant, still
in the database and pinned to *its* cdhash, was simply replaced: proof of the
freeze rule from the other direction, since nothing migrates an old grant.

## Measured behaviour of the running app (2026-07-25)

- **LaunchServices moves the bundle's registration onto the spawned child** when
  Qt checks in: one Dock tile, our icon and name, with `originalPid` still
  pointing at the stub. There is no second, nameless tile to suppress.
- **The child owns its activation policy at runtime.**
  `setActivationPolicy(accessory)` flips the record from `Foreground` to
  `UIElement` live, so **step 6's menu-bar mode needs no Info.plist key** — the
  coupling the plan feared does not exist. `LSUIElement` was tried and rejected:
  it also stops `open` from bringing the window forward, so the app opened
  behind whatever was in front.
- A second `open` re-activates the running instance instead of starting a
  second one, which is the single-instance behaviour we want for free.

## Rebuilding (don't, unless you mean it)

    sh build.sh

Renders `icon.svg` (Qt, SVG Tiny 1.2 — no filters), compiles `main.c` universal
(arm64 + x86_64, each slice with its own cdhash), copies `Info.plist`, ad-hoc
signs the bundle and prints both cdhashes. `tests/test_shortcut.py` pins a
sha256 over the committed tree and will fail until the constant is updated —
that failure is the point. Updating it means accepting that every existing user
answers the microphone prompt again.
