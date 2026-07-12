# Build the stenodiar diarization helper and drop the binary next to this
# script, where stenograf's dev fallback looks for it (twin of build.sh for
# Windows). Needs a Rust toolchain (rustup) and the VS Build Tools linker. No
# signing: stenodiar touches no guarded resource, so an unsigned binary is fine.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Default features = ORT CPU (matches the shipped Linux/Windows path); CUDA is a
# manual opt-in — cargo build --release --features cuda (nvidia-*-cu12 DLLs +
# onnxruntime_providers_cuda.dll on PATH). CoreML is macOS-only (build.sh).
cargo build --release --locked
Copy-Item target\release\stenodiar.exe stenodiar.exe -Force

Write-Output "built: $(Join-Path (Get-Location) 'stenodiar.exe')"
