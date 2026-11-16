# Give a hosted Windows runner a sound card: VB-CABLE, a virtual cable whose
# playback side ("CABLE Input") is recordable from its capture side ("CABLE
# Output"). Hosted runners have no audio endpoint, and the runner-images
# maintainers decline to add one (actions/runner-images#6983).
#
# The driver pack is fetched from this repository's own release, unchanged
# and pinned by SHA-256 — never from the vendor at CI time (VB-Audio's
# licence permits redistributing the package AS IS with credit; the credit is
# in the release notes). The steps follow Mozilla cubeb's CI script
# (.github/scripts/setup_windows_audio.ps1), with one change: no devcon. The
# root device is created through SetupAPI directly (what devcon's "install"
# does inside) and the inbox pnputil installs the driver on it, so nothing is
# needed from the WDK — windows-2025 ships neither devcon nor devgen
# (measured 2026-11-16) — and no Microsoft binary is redistributed.

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
Add-Type -TypeDefinition @"
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Text;

public static class RootDevice {
    [StructLayout(LayoutKind.Sequential)]
    struct SP_DEVINFO_DATA { public uint cbSize; public Guid ClassGuid; public uint DevInst; public IntPtr Reserved; }

    [DllImport("setupapi.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    static extern bool SetupDiGetINFClassW(string inf, out Guid classGuid, StringBuilder className, uint size, out uint required);
    [DllImport("setupapi.dll", SetLastError = true)]
    static extern IntPtr SetupDiCreateDeviceInfoList(ref Guid classGuid, IntPtr parent);
    [DllImport("setupapi.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    static extern bool SetupDiCreateDeviceInfoW(IntPtr set, string name, ref Guid classGuid, string description, IntPtr parent, uint flags, ref SP_DEVINFO_DATA data);
    [DllImport("setupapi.dll", SetLastError = true)]
    static extern bool SetupDiSetDeviceRegistryPropertyW(IntPtr set, ref SP_DEVINFO_DATA data, uint property, byte[] buffer, uint size);
    [DllImport("setupapi.dll", SetLastError = true)]
    static extern bool SetupDiCallClassInstaller(uint function, IntPtr set, ref SP_DEVINFO_DATA data);
    [DllImport("setupapi.dll")]
    static extern bool SetupDiDestroyDeviceInfoList(IntPtr set);

    const uint DICD_GENERATE_ID = 1, SPDRP_HARDWAREID = 1, DIF_REGISTERDEVICE = 0x19;

    static Exception Failed(string call) { return new Win32Exception(Marshal.GetLastWin32Error(), call + " failed"); }

    // A root-enumerated device node carrying the INF's class and the given
    // hardware ID, registered with PnP; the driver is installed onto it after.
    public static void Create(string inf, string hardwareId) {
        Guid cls; uint required; var className = new StringBuilder(256);
        if (!SetupDiGetINFClassW(inf, out cls, className, 256, out required)) throw Failed("SetupDiGetINFClass");
        IntPtr set = SetupDiCreateDeviceInfoList(ref cls, IntPtr.Zero);
        if (set == new IntPtr(-1)) throw Failed("SetupDiCreateDeviceInfoList");
        try {
            var data = new SP_DEVINFO_DATA();
            data.cbSize = (uint)Marshal.SizeOf(typeof(SP_DEVINFO_DATA));
            if (!SetupDiCreateDeviceInfoW(set, className.ToString(), ref cls, null, IntPtr.Zero, DICD_GENERATE_ID, ref data)) throw Failed("SetupDiCreateDeviceInfo");
            byte[] ids = Encoding.Unicode.GetBytes(hardwareId + "\0\0");
            if (!SetupDiSetDeviceRegistryPropertyW(set, ref data, SPDRP_HARDWAREID, ids, (uint)ids.Length)) throw Failed("SetupDiSetDeviceRegistryProperty");
            if (!SetupDiCallClassInstaller(DIF_REGISTERDEVICE, set, ref data)) throw Failed("SetupDiCallClassInstaller");
        } finally {
            SetupDiDestroyDeviceInfoList(set);
        }
    }
}
"@
Write-Host "creating the $hardwareId root device through SetupAPI"
[RootDevice]::Create($inf, $hardwareId)
& pnputil.exe /add-driver $inf /install
if ($LASTEXITCODE -ne 0) { throw "pnputil failed ($LASTEXITCODE)" }

# Off by default since the windows-2022 image (actions/runner-images#2528).
Start-Service AudioEndpointBuilder, Audiosrv
# Recording from "CABLE Output" counts as microphone use for the privacy gate.
$policy = "HKLM:\SOFTWARE\Policies\Microsoft\Windows\AppPrivacy"
New-Item -Path $policy -Force | Out-Null
New-ItemProperty -Path $policy -Name LetAppsAccessMicrophone -PropertyType DWord -Value 1 -Force | Out-Null

Get-CimInstance Win32_SoundDevice | Format-Table Name, Status
