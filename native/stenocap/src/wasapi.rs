//! The Windows backend: WASAPI shared-mode capture, stamped on the device clock.
//!
//! Both channels come from WASAPI — the mic from the default capture endpoint,
//! system audio from *loopback* on the default render endpoint — and the only
//! thing that makes them comparable is where their timestamps come from.
//! `IAudioCaptureClient::GetBuffer` hands back `pu64QPCPosition` for every
//! packet: the machine-wide performance counter, in 100-ns units, for the first
//! sample of that packet. Measured on Realtek HD Audio (2026-07-26, and
//! reproducible in twelve seconds with `eval/wasapi_timestamps.py`): populated
//! and monotonic on both taps, zero `TIMESTAMP_ERROR` flags, and stable to
//! ±0.1 ms over 30 s — against arrival stamping, which scattered by hundreds of
//! milliseconds because each channel anchored on its own first frame.
//!
//! Three details are load-bearing, and getting any of them wrong is silent:
//!
//! - **`pu64DevicePosition` is not a sample index.** It counts *device* frames,
//!   upstream of the resampler `AUTOCONVERTPCM` inserts, so it advances at
//!   47,999/s while the stream asked for 16,000. Only the QPC stamp is used.
//! - **The two taps' stamps mean different things, and that is correct.** The
//!   mic's marks capture, and trails receipt (+11.1 ms measured). The loopback
//!   tap's is render-side and *leads* receipt (−9.6 ms) — it marks when the
//!   audio reached the endpoint, i.e. when the echo was born, which is exactly
//!   the instant the canceller wants. Both are on one counter, so neither needs
//!   correcting; assuming they should agree in sign reintroduces the bug this
//!   file exists to remove.
//! - **Loopback delivers nothing while nothing renders.** There is no silent
//!   packet to read; the tap simply goes quiet. The shared silence filler
//!   (`frame::fill_silence`, ticking on [`now_units`]) is what covers it.
//!
//! Everything is timer-driven. Event-driven buffering is not available for a
//! loopback stream, and one polling loop for both channels is less machinery
//! than two lifecycles. Each channel owns its device end to end on its own
//! thread because COM objects are apartment-bound.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use windows::core::Result as WinResult;
use windows::core::PCWSTR;
use windows::Win32::Devices::FunctionDiscovery::PKEY_Device_FriendlyName;
use windows::Win32::Media::Audio::{
    eCapture, eConsole, eRender, IAudioCaptureClient, IAudioClient, IMMDevice,
    IMMDeviceEnumerator, MMDeviceEnumerator, AUDCLNT_BUFFERFLAGS_SILENT, AUDCLNT_SHAREMODE_SHARED,
    AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM, AUDCLNT_STREAMFLAGS_LOOPBACK,
    AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY, DEVICE_STATE_ACTIVE, WAVEFORMATEX,
};
use windows::Win32::System::Com::{
    CoCreateInstance, CoInitializeEx, CoTaskMemFree, CoUninitialize, CLSCTX_ALL,
    COINIT_MULTITHREADED, STGM_READ,
};
use windows::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};

use crate::device::{resolve_with_grace, InputDevice};
use crate::frame::{FrameSink, Framer, SAMPLE_RATE};

const WAVE_FORMAT_PCM: u16 = 1;
const WAVE_FORMAT_IEEE_FLOAT: u16 = 3;

/// Endpoint buffer asked of the audio engine (300 ms).
///
/// Only a ceiling: packets still arrive every ~10 ms. It is the slack a pump
/// thread has if the machine stalls it, and 300 ms is well past anything the
/// poll interval below can produce.
const BUFFER_DURATION_HNS: i64 = 3_000_000;

/// How often each pump asks for packets. WASAPI's own period is ~10 ms.
const POLL: Duration = Duration::from_millis(5);

/// Stamp/timeline disagreement absorbed rather than acted on (5 ms).
///
/// Packets arrive every ~10 ms and their QPC stamps jitter by a fraction of a
/// millisecond (±0.1 ms measured, `eval/wasapi_timestamps.py`), so this sits
/// well above the noise while staying far under a real dropout.
pub const GAP_TOLERANCE_UNITS: i64 = 50_000;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Tap {
    Mic,
    System,
}

impl Tap {
    pub fn label(self) -> &'static str {
        match self {
            Tap::Mic => "mic",
            Tap::System => "system",
        }
    }
}

/// The performance counter in the 100-ns units WASAPI reports positions in.
///
/// The same machine-wide counter both taps are stamped against, which is what
/// makes a mic frame and a system frame with equal timestamps simultaneous.
pub fn now_units() -> i64 {
    // Both calls are documented never to fail on Windows XP or later.
    let mut frequency = 0i64;
    let mut counter = 0i64;
    unsafe {
        let _ = QueryPerformanceFrequency(&mut frequency);
        let _ = QueryPerformanceCounter(&mut counter);
    }
    if frequency == 0 {
        return 0;
    }
    // 128-bit intermediate: counter * 10^7 overflows i64 after ~15 minutes of
    // uptime on a 10 MHz counter, which is every machine.
    ((counter as i128 * 10_000_000) / frequency as i128) as i64
}

struct Com;

impl Com {
    /// COM for the calling thread. Every pump needs its own — the devices and
    /// clients below are apartment-bound and may not cross threads.
    fn enter() -> Self {
        unsafe {
            // S_FALSE ("already initialized on this thread") is success; the
            // matching CoUninitialize below still balances it.
            let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
        }
        Com
    }
}

impl Drop for Com {
    fn drop(&mut self) {
        unsafe { CoUninitialize() };
    }
}

fn enumerator() -> WinResult<IMMDeviceEnumerator> {
    unsafe { CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL) }
}

fn default_endpoint(tap: Tap) -> WinResult<IMMDevice> {
    unsafe {
        // eConsole, not eCommunications: this is a meeting *recorder*, and the
        // endpoints a user hears the call on are the console ones.
        enumerator()?.GetDefaultAudioEndpoint(
            match tap {
                Tap::Mic => eCapture,
                Tap::System => eRender,
            },
            eConsole,
        )
    }
}

fn friendly_name(device: &IMMDevice) -> WinResult<String> {
    unsafe {
        let store = device.OpenPropertyStore(STGM_READ)?;
        let value = store.GetValue(&PKEY_Device_FriendlyName)?;
        let name = value.to_string();
        Ok(name)
    }
}

/// The endpoint ID string — what is stored, and what survives a reboot.
///
/// `GetId` hands back COM-allocated memory the caller owns, so the copy is
/// taken and the original freed here; leaking it once per enumeration would
/// leak once per device per meeting start.
fn endpoint_id(device: &IMMDevice) -> Result<String, String> {
    unsafe {
        let raw = device
            .GetId()
            .map_err(|e| format!("could not read an audio device's id ({e})"))?;
        let id = raw
            .to_string()
            .map_err(|e| format!("an audio device's id is not valid text ({e})"));
        CoTaskMemFree(Some(raw.0 as *const std::ffi::c_void));
        id
    }
}

/// Every active capture endpoint, with the default one marked.
pub fn list_inputs() -> Result<Vec<InputDevice>, String> {
    let _com = Com::enter();
    inputs()
}

/// [`list_inputs`] without entering COM — for callers already inside it.
fn inputs() -> Result<Vec<InputDevice>, String> {
    let enumerator =
        enumerator().map_err(|e| format!("could not open the audio device list ({e})"))?;
    // A machine with no default input is a legitimate state (nothing is
    // plugged in), so this marks the list rather than failing it.
    let default_id = unsafe { enumerator.GetDefaultAudioEndpoint(eCapture, eConsole) }
        .ok()
        .and_then(|device| endpoint_id(&device).ok());
    let collection = unsafe { enumerator.EnumAudioEndpoints(eCapture, DEVICE_STATE_ACTIVE) }
        .map_err(|e| format!("could not list the capture devices ({e})"))?;
    let count = unsafe { collection.GetCount() }
        .map_err(|e| format!("could not count the capture devices ({e})"))?;
    let mut devices = Vec::new();
    for index in 0..count {
        // One unreadable endpoint must not hide the rest: a driver that fails
        // its own property store is exactly when the user needs the picker.
        let Ok(device) = (unsafe { collection.Item(index) }) else { continue };
        let Ok(id) = endpoint_id(&device) else { continue };
        let name = friendly_name(&device).unwrap_or_else(|_| id.clone());
        let is_default = default_id.as_deref() == Some(id.as_str());
        devices.push(InputDevice { id, name, is_default });
    }
    Ok(devices)
}

/// The mic endpoint this run records from: the pinned one, or the default.
///
/// A pin is resolved through the shared rule (`device.rs`) with its grace
/// period, and a pin that names nothing present is an error — never the
/// default, which would silently record the wrong microphone.
fn capture_endpoint(pin: Option<&str>) -> Result<IMMDevice, String> {
    let Some(pin) = pin else {
        return default_endpoint(Tap::Mic).map_err(|e| {
            format!("no default mic device ({e}) — check Windows sound settings")
        });
    };
    let chosen = resolve_with_grace(pin, inputs)?;
    let enumerator =
        enumerator().map_err(|e| format!("could not open the audio device list ({e})"))?;
    let wide: Vec<u16> = chosen.id.encode_utf16().chain(std::iter::once(0)).collect();
    unsafe { enumerator.GetDevice(PCWSTR(wide.as_ptr())) }.map_err(|e| {
        format!("the selected microphone \"{}\" could not be opened ({e})", chosen.name)
    })
}

/// What each channel would record from right now, for the CLI's preflight.
pub fn device_names(taps: &[Tap], mic_device: Option<&str>) -> Result<Vec<(Tap, String)>, String> {
    let _com = Com::enter();
    let mut names = Vec::new();
    for &tap in taps {
        let device = match tap {
            Tap::Mic => capture_endpoint(mic_device)?,
            Tap::System => default_endpoint(tap).map_err(|e| {
                format!("no default {} device ({e}) — check Windows sound settings", tap.label())
            })?,
        };
        let name = friendly_name(&device)
            .map_err(|e| format!("could not read the {} device's name ({e})", tap.label()))?;
        names.push((tap, name));
    }
    Ok(names)
}

/// A mono 16 kHz stream description in one of the two formats we accept.
fn wave_format(float: bool) -> WAVEFORMATEX {
    let bits = if float { 32u16 } else { 16u16 };
    let block_align = bits / 8; // mono
    WAVEFORMATEX {
        wFormatTag: if float { WAVE_FORMAT_IEEE_FLOAT } else { WAVE_FORMAT_PCM },
        nChannels: 1,
        nSamplesPerSec: SAMPLE_RATE,
        nAvgBytesPerSec: SAMPLE_RATE * block_align as u32,
        nBlockAlign: block_align,
        wBitsPerSample: bits,
        cbSize: 0,
    }
}

struct Stream {
    client: IAudioClient,
    capture: IAudioCaptureClient,
    float: bool,
}

/// Open one tap at mono 16 kHz, letting the audio engine do the conversion.
///
/// `AUTOCONVERTPCM | SRC_DEFAULT_QUALITY` is what keeps a resampler out of this
/// crate: Windows converts rate and channel count server-side, the way parec
/// does on Linux. A client whose `Initialize` was rejected cannot be reused, so
/// the int16 attempt and the float32 fallback each get a fresh one.
fn open(tap: Tap, mic_device: Option<&str>) -> Result<Stream, String> {
    let device = match tap {
        Tap::Mic => capture_endpoint(mic_device)?,
        Tap::System => default_endpoint(tap).map_err(|e| {
            format!("no default {} device ({e}) — check Windows sound settings", tap.label())
        })?,
    };
    let mut flags = AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY;
    if tap == Tap::System {
        flags |= AUDCLNT_STREAMFLAGS_LOOPBACK;
    }

    let mut last = String::new();
    for float in [false, true] {
        let format = wave_format(float);
        let opened = unsafe {
            let client: IAudioClient = device.Activate(CLSCTX_ALL, None).map_err(|e| {
                format!("could not open the {} device ({e})", tap.label())
            })?;
            client
                .Initialize(
                    AUDCLNT_SHAREMODE_SHARED,
                    flags,
                    BUFFER_DURATION_HNS,
                    0,
                    &format,
                    None,
                )
                .and_then(|()| client.GetService::<IAudioCaptureClient>())
                .map(|capture| Stream { client, capture, float })
        };
        match opened {
            Ok(stream) => return Ok(stream),
            Err(err) => last = format!("{err}"),
        }
    }
    Err(format!(
        "the {} device accepted neither 16-bit nor float mono 16 kHz ({last})",
        tap.label()
    ))
}

/// Own one channel end to end until `stop`: device, stream, framing, stamps.
///
/// Returns the reason it ended abnormally, or `None` for a requested stop.
pub fn pump(
    tap: Tap,
    mic_device: Option<&str>,
    framer: Arc<Mutex<Framer>>,
    sink: Arc<FrameSink>,
    stop: Arc<AtomicBool>,
    started: std::sync::mpsc::Sender<()>,
    log: impl Fn(&str),
) -> Option<String> {
    let _com = Com::enter();
    let stream = match open(tap, mic_device) {
        Ok(stream) => stream,
        Err(why) => return Some(why),
    };
    if let Err(err) = unsafe { stream.client.Start() } {
        return Some(format!("could not start {} capture ({err})", tap.label()));
    }
    // The negotiated format is worth a line: the fallback path is invisible
    // otherwise, and "which format did this endpoint actually take" is the
    // first question a garbled channel raises.
    log(&format!(
        "{} capturing ({})",
        tap.label(),
        if stream.float { "float32" } else { "16-bit" }
    ));
    let _ = started.send(());

    let mut samples: Vec<i16> = Vec::new();
    let outcome = loop {
        if stop.load(Ordering::Relaxed) {
            break None;
        }
        let available = match unsafe { stream.capture.GetNextPacketSize() } {
            Ok(frames) => frames,
            Err(err) => break Some(format!("{} capture stream died ({err})", tap.label())),
        };
        if available == 0 {
            std::thread::sleep(POLL);
            continue;
        }
        let mut data = std::ptr::null_mut();
        let mut frames = 0u32;
        let mut flags = 0u32;
        let mut stamp = 0u64;
        let got = unsafe {
            stream.capture.GetBuffer(
                &mut data,
                &mut frames,
                &mut flags,
                None, // the device position counts device frames, not ours
                Some(&mut stamp),
            )
        };
        if let Err(err) = got {
            break Some(format!("{} capture stream died ({err})", tap.label()));
        }
        samples.clear();
        if flags & AUDCLNT_BUFFERFLAGS_SILENT.0 as u32 != 0 {
            // The buffer's contents are undefined when this is set; the audio
            // it stands for is digital silence.
            samples.resize(frames as usize, 0);
        } else if stream.float {
            let raw = unsafe { std::slice::from_raw_parts(data as *const f32, frames as usize) };
            samples.extend(raw.iter().map(|&s| (s.clamp(-1.0, 1.0) * 32767.0) as i16));
        } else {
            let raw = unsafe { std::slice::from_raw_parts(data as *const i16, frames as usize) };
            samples.extend_from_slice(raw);
        }
        let released = unsafe { stream.capture.ReleaseBuffer(frames) };
        if let Err(err) = released {
            break Some(format!("{} capture stream died ({err})", tap.label()));
        }

        let pushed = framer
            .lock()
            .expect("framer poisoned")
            .push(stamp as i64, &samples, &sink);
        match pushed {
            Ok(Some(complaint)) => log(&complaint),
            Ok(None) => {}
            Err(err) => break Some(format!("could not write frames ({err})")),
        }
    };

    let _ = unsafe { stream.client.Stop() };
    outcome
}

