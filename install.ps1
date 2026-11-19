# stenograf installer — one command sets up everything:
#
#   powershell -NoProfile -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/daniel-om-weber/stenograf/main/install.ps1 | iex"
#
# Installs uv if missing, installs stenograf as a uv tool, then runs
# `steno setup` (permission prompts, desktop launcher, model downloads).
# Safe to re-run: every step is idempotent and re-running upgrades stenograf.
#
# -ExecutionPolicy Bypass is load-bearing, not decoration: uv's own installer
# refuses to run under Restricted, the default on every Windows client.
#
# Failures throw rather than exit: pasted as a bare `irm … | iex` into the
# user's own window, `exit` would close it and take the message with it.
# No global $ErrorActionPreference = "Stop": in Windows PowerShell 5.1 it
# never covers a native program's exit code (only 7.4+ can), so it looked
# like protection while a failed upgrade still ended in "stenograf is
# installed". Every program below is checked by hand instead.

function Assert-Ran([string]$What) {
    if ($LASTEXITCODE -ne 0) {
        throw "$What failed (exit code $LASTEXITCODE). The messages above say why; running the install command again is safe."
    }
}

function Assert-Exists([string]$Path, [string]$What) {
    if (-not (Test-Path $Path)) {
        throw "$What was not found at $Path. Open a new PowerShell window and run the install command again."
    }
}

$uvHome = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
if (-not (Get-Command uv -ErrorAction SilentlyContinue) -and -not (Test-Path $uvHome)) {
    Write-Host "installing uv (https://docs.astral.sh/uv/) ..."
    powershell -NoProfile -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
    Assert-Ran "Installing uv"
}
$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) { $uv = $uvHome }
Assert-Exists $uv "uv"

# --python: uv picks the interpreter before it reads requires-python, so on a
# PC with no Python 3.12/3.13 it fetches the newest one, and stenograf has no
# install there (see pyproject.toml).
#
# The retry repairs an install whose folder is damaged or gone while its
# steno.exe is still there: an interrupted update leaves exactly that, and uv
# then refuses with "Executable already exists" until it is told to overwrite
# (measured 2026-11-19, eval/update_mechanics.sh).
& $uv tool install --upgrade --python 3.13 stenograf
if ($LASTEXITCODE -ne 0) {
    Write-Host "that did not work - reinstalling from scratch ..."
    & $uv tool install --force --python 3.13 stenograf
}
Assert-Ran "Installing stenograf"

# A freshly created uv bin dir isn't on this shell's PATH yet — ask uv where it is.
$steno = (Get-Command steno -ErrorAction SilentlyContinue).Source
if (-not $steno) {
    $bin = & $uv tool dir --bin
    Assert-Ran "Locating stenograf"
    $steno = Join-Path $bin "steno.exe"
}
Assert-Exists $steno "steno"

& $steno setup
if ($LASTEXITCODE -ne 0) {
    # Installed, but setup stopped (mic toggle off, a model download cut off):
    # setup's own message above names the fix.
    throw "stenograf is installed, but 'steno setup' did not finish. Fix what it reports above, then run the install command again."
}

Write-Host ""
Write-Host "stenograf is installed. Start it from the desktop launcher above,"
Write-Host "or run: steno"
