# Build the stenocap capture helper and drop the binary next to this script,
# where stenograf's dev fallback looks for it (mirrors ../stenodiar/build.ps1).
# Needs a Rust toolchain (rustup) and the VS Build Tools linker. No signing:
# Windows gates the microphone through the privacy consent store, which is
# read per user rather than per binary, so an unsigned helper prompts nothing
# and loses nothing — unlike macOS, where the signature *is* the grant.
#
# The wheel build hook runs this script too (hatch_build.py), and unlike
# stenodiar a failure here fails the wheel: without this binary there is no
# live capture at all on Windows.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

cargo build --release --locked
Copy-Item target\release\stenocap.exe stenocap.exe -Force

Write-Output "built: $(Join-Path (Get-Location) 'stenocap.exe')"
