# Give a hosted Windows runner a sound card: VB-CABLE, a virtual cable whose
# playback side ("CABLE Input") is recordable from its capture side ("CABLE
# Output"). Hosted runners have no audio endpoint, and the runner-images
# maintainers decline to add one (actions/runner-images#6983).
#
# The driver pack is fetched from this repository's own release, unchanged
# and pinned by SHA-256 — never from the vendor at CI time (VB-Audio's
# licence permits redistributing the package AS IS with credit; the credit is
# in the release notes). The steps follow Mozilla cubeb's CI script
# (.github/scripts/setup_windows_audio.ps1), with one change: the device is
# created with tools already on the runner — Microsoft's DevCon replacement
# (devgen + the inbox pnputil) where the WDK provides devgen, else the WDK's
# devcon — so no Microsoft binary is redistributed either.

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0

$url = "https://github.com/daniel-om-weber/stenograf/releases/download/ci-vbcable-pack43/VBCABLE_Driver_Pack43.zip"
$sha256 = "66fd0a4d9f4896ff41632b7e3d53892c085c4561f53e8ae8d0f0bc10eedd1cdd"
$hardwareId = "VBAudioVACWDM"

$work = Join-Path $env:RUNNER_TEMP "vbcable"
New-Item -ItemType Directory -Force -Path $work | Out-Null
$zip = Join-Path $work "VBCABLE_Driver_Pack43.zip"
Invoke-WebRequest -Uri $url -OutFile $zip -MaximumRetryCount 3 -RetryIntervalSec 2
$actual = (Get-FileHash -Algorithm SHA256 -Path $zip).Hash.ToLowerInvariant()
if ($actual -ne $sha256) { throw "VB-CABLE package hash $actual, expected $sha256" }
$driver = Join-Path $work "driver"
Expand-Archive -LiteralPath $zip -DestinationPath $driver -Force

# Trust the driver's signer, or the install stops at a GUI prompt nobody answers.
$signature = Get-AuthenticodeSignature -FilePath (Join-Path $driver "vbaudio_cable64_win7.cat")
if ($null -eq $signature.SignerCertificate) { throw "no signer certificate on the VB-CABLE catalog" }
$cert = Join-Path $work "vbcable.cer"
[System.IO.File]::WriteAllBytes($cert, $signature.SignerCertificate.Export(
    [System.Security.Cryptography.X509Certificates.X509ContentType]::Cert))
& certutil.exe -addstore -f "TrustedPublisher" $cert | Out-Null
if ($LASTEXITCODE -ne 0) { throw "certutil failed ($LASTEXITCODE)" }

$inf = Join-Path $driver "vbMmeCable64_win7.inf"
$kits = Join-Path ${env:ProgramFiles(x86)} "Windows Kits\10\Tools"
$devgen = Get-ChildItem -Path $kits -Recurse -Filter devgen.exe -ErrorAction SilentlyContinue |
    Where-Object { $_.DirectoryName -like "*\x64" } | Select-Object -Last 1
$devcon = Get-ChildItem -Path $kits -Recurse -Filter devcon.exe -ErrorAction SilentlyContinue |
    Where-Object { $_.DirectoryName -like "*\x64" } | Select-Object -Last 1
if ($devgen) {
    Write-Host "creating the device with $($devgen.FullName) + pnputil"
    & $devgen.FullName /add /bus ROOT /hardwareid $hardwareId
    if ($LASTEXITCODE -ne 0) { throw "devgen failed ($LASTEXITCODE)" }
    & pnputil.exe /add-driver $inf /install
    if ($LASTEXITCODE -ne 0) { throw "pnputil failed ($LASTEXITCODE)" }
} elseif ($devcon) {
    Write-Host "creating the device with $($devcon.FullName)"
    & $devcon.FullName install $inf $hardwareId
    if ($LASTEXITCODE -ne 0) { throw "devcon failed ($LASTEXITCODE)" }
} else {
    Get-ChildItem -Path $kits -ErrorAction SilentlyContinue | Format-Table Name
    throw "neither devgen.exe nor devcon.exe under $kits"
}

# Off by default since the windows-2022 image (actions/runner-images#2528).
Start-Service AudioEndpointBuilder, Audiosrv
# Recording from "CABLE Output" counts as microphone use for the privacy gate.
$policy = "HKLM:\SOFTWARE\Policies\Microsoft\Windows\AppPrivacy"
New-Item -Path $policy -Force | Out-Null
New-ItemProperty -Path $policy -Name LetAppsAccessMicrophone -PropertyType DWord -Value 1 -Force | Out-Null

Get-CimInstance Win32_SoundDevice | Format-Table Name, Status
