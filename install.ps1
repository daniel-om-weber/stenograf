# stenograf installer — one command sets up everything:
#
#   powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/daniel-om-weber/stenograf/main/install.ps1 | iex"
#
# Installs uv if missing, installs stenograf as a uv tool, then runs
# `steno setup` (permission prompts, desktop launcher, model downloads).
# Safe to re-run: every step is idempotent and re-running upgrades stenograf.
$ErrorActionPreference = "Stop"

$uvHome = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
if (-not (Get-Command uv -ErrorAction SilentlyContinue) -and -not (Test-Path $uvHome)) {
    Write-Host "installing uv (https://docs.astral.sh/uv/) ..."
    powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
}
$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) { $uv = $uvHome }

& $uv tool install --upgrade stenograf

# A freshly created uv bin dir isn't on this shell's PATH yet — ask uv where it is.
$steno = (Get-Command steno -ErrorAction SilentlyContinue).Source
if (-not $steno) { $steno = Join-Path (& $uv tool dir --bin) "steno.exe" }

& $steno setup

Write-Host ""
Write-Host "stenograf is installed. Start it from the desktop launcher above,"
Write-Host "or run: steno"
