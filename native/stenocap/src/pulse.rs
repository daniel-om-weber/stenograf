//! The Linux backend: PulseAudio-protocol capture, stamped on CLOCK_MONOTONIC.
//!
//! Both channels are pulse record streams — the mic from `@DEFAULT_SOURCE@`,
//! system audio from `@DEFAULT_MONITOR@`, the same server-resolved names the
//! retired `parec` transport used. Either alias follows a mid-capture default
//! change, because the server moves the stream it already resolved; a pinned
//! mic is the one stream that refuses ([`DONT_MOVE`][StreamFlagSet], and
//! [`DEVICE_LOST`]). One client library serves both sound
//! servers: PipeWire ships `pipewire-pulse` precisely so pulse clients need no
//! second path, and its timing answers come from `pw_time`, the native clock.
//!
//! **Where the timestamps come from.** The server accounts, per stream, how old
//! the sample at the client's read position is: device latency, transport, and
//! whatever sits unread in the buffer (`pa_stream_get_latency`, kept fresh by
//! `AUTO_TIMING_UPDATE` and interpolated between updates). So the first sample
//! of each peeked chunk was captured at `now − latency`, with `now` read from
//! CLOCK_MONOTONIC — the machine-wide clock both channels stamp against, which
//! is what makes a mic frame and a system frame with equal timestamps
//! simultaneous. For the monitor the server folds in the *sink's* latency, so
//! its stamps mark when the audio becomes audible — when the echo is born —
//! which is exactly the reference instant the canceller wants (the same
//! render-side semantics as WASAPI loopback's QPC stamps, see `wasapi.rs`).
//!
//! Stamp jitter between timing updates is absorbed by the framer's tolerance;
//! the sample count stays the authority in between (`frame.rs`).
//!
//! **A monitor goes quiet when its sink does.** Sinks suspend when idle
//! (module-suspend-on-idle, and PipeWire's node suspension), and a suspended
//! sink's monitor delivers nothing at all — the same shape as WASAPI loopback
//! during render silence, covered by the same silence filler.
//!
//! The server resamples to mono 16 kHz s16 per stream, the way it did for
//! `parec`, so no resampler dependency appears. Each channel owns its own
//! mainloop, context and stream on its own thread, mirroring the COM
//! apartments on Windows; the clock is machine-wide, so separate connections
//! cost the pairing nothing.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use libpulse_binding::callbacks::ListResult;
use libpulse_binding::context::{Context, FlagSet as ContextFlagSet, State as ContextState};
use libpulse_binding::def::BufferAttr;
use libpulse_binding::mainloop::standard::{IterateResult, Mainloop};
use libpulse_binding::operation::State as OperationState;
use libpulse_binding::sample::{Format, Spec};
use libpulse_binding::stream::{
    FlagSet as StreamFlagSet, Latency, PeekResult, State as StreamState, Stream,
};

use crate::device::{resolve_with_grace, InputDevice};
use crate::frame::{FrameSink, Framer, SAMPLE_RATE};

/// How often each pump dispatches server events and drains its stream.
/// Fragments arrive every ~[`FRAGSIZE_BYTES`] worth of audio; 5 ms keeps the
/// stop gesture prompt without waking the CPU for nothing.
const POLL: Duration = Duration::from_millis(5);

/// Fragment size asked of the server (40 ms of mono 16 kHz s16).
///
/// With `ADJUST_LATENCY` this is the delivery cadence. It only needs to sit
/// well under the silence filler's lead (100 ms, `frame.rs`): a fragment
/// delivered one cadence after its capture instant must still land ahead of
/// the fill line, or real audio would be filled over and placed late.
const FRAGSIZE_BYTES: u32 = (SAMPLE_RATE / 25) * 2;

/// How long a stream gets to reach Ready with timing info before the pump
/// reports it dead. Under the 15 s watchdog in `main.rs`, so a hang here still
/// names the channel rather than timing out anonymously.
const SETUP_TIMEOUT: Duration = Duration::from_secs(10);

/// Stamp/timeline disagreement absorbed rather than acted on (30 ms).
///
/// Stamps here are `now − latency` with the latency interpolated between the
/// server's timing updates, and that accounting is an order of magnitude
/// noisier than WASAPI's ±0.1 ms QPC stamps. Measured over 35 s on PipeWire
/// 1.6 (2026-08-02): the stamp error against a pure sample-count line holds a
/// bounded ±7 ms band — no unbounded drift — but left to the automatic update
/// cadence (which backs off to seconds) the interpolation wanders by ~±25 ms
/// before a correction lands, which is why the pump also refreshes the timing
/// info itself ([`TIMING_REFRESH`]). 30 ms sits above the refreshed noise
/// while staying far under a real dropout — a suspended sink's monitor goes
/// quiet for hundreds of milliseconds at the least.
pub const GAP_TOLERANCE_UNITS: i64 = 300_000;

/// How often each pump asks the server for fresh timing info.
///
/// `AUTO_TIMING_UPDATE` alone backs its cadence off to whole seconds, and the
/// interpolation drifts a few ms per second between corrections — that drift,
/// not delivery, is the noise floor of our stamps. One explicit refresh per
/// second (a single tiny round trip) keeps the error near the snapshot's own
/// accuracy instead of the drift's accumulation.
const TIMING_REFRESH: Duration = Duration::from_secs(1);

/// How long a pinned mic may deliver nothing before the run is called dead.
///
/// Losing the device mid-capture leaves no trace in the stream API — measured
/// 2026-08-12 on PipeWire 1.6.8, where unloading a pinned source left index,
/// state, `suspended` and device name all unchanged and merely stopped the
/// audio, so silence is the only symptom there is. Only a pinned mic is judged
/// this way: it cannot legitimately move, while a monitor whose sink is
/// suspended delivers nothing for entirely ordinary reasons.
///
/// Five seconds is chosen, not measured: long enough to dwarf any scheduling
/// hiccup the forgiveness below does not already cover, short enough that a
/// meeting recorded from a dead microphone is caught while it is still worth
/// telling someone. A trip ends the helper, so it truncates a meeting rather
/// than losing one.
const DEVICE_LOST: Duration = Duration::from_secs(5);

/// A pump gap this long is treated as our own stall, not the device's silence.
///
/// The watchdog can only judge what the pump observed, and a pump that was not
/// scheduled observed nothing: stopping the process for six seconds declared a
/// live, delivering microphone disconnected (measured 2026-08-12). The helper
/// shares the CLI's process group, so a terminal's SIGTSTP reaches exactly this
/// path — as does a starved thread on a loaded machine.
const STALL_FORGIVEN: Duration = Duration::from_secs(1);

/// Whether this stream is the one that may never be re-routed.
///
/// The system tap follows a default-sink change *by* being moved, and an
/// unpinned mic likewise follows the default source (both measured 2026-08-12),
/// so only a pinned mic may refuse a move — and only it may be judged dead for
/// going quiet.
fn pinned(tap: Tap, mic_device: Option<&str>) -> bool {
    matches!(tap, Tap::Mic) && mic_device.is_some()
}

/// Has a pinned mic stopped delivering, or did we simply stop looking?
///
/// Its own rules, kept out of the pump so both are testable without a sound
/// server: only data proves the device is alive, and silence is only counted
/// while the pump is actually running.
struct Liveness {
    last_data: Instant,
    last_tick: Instant,
}

impl Liveness {
    fn new(now: Instant) -> Self {
        Self { last_data: now, last_tick: now }
    }

    /// Once per pump iteration, before judging.
    fn tick(&mut self, now: Instant) {
        if now.duration_since(self.last_tick) >= STALL_FORGIVEN {
            self.last_data = now;
        }
        self.last_tick = now;
    }

    /// Audio arrived — including a hole, which is the server reporting that it
    /// dropped *this stream's* audio and so is proof the device is still there.
    fn observed(&mut self, now: Instant) {
        self.last_data = now;
    }

    fn lost(&self, now: Instant) -> bool {
        now.duration_since(self.last_data) >= DEVICE_LOST
    }
}

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

    /// The server-side default aliases, resolved (and re-resolved) by the
    /// server itself — passing anything else would lose the measured
    /// follow-the-default behaviour the monitor channel relies on.
    fn device(self) -> &'static str {
        match self {
            Tap::Mic => "@DEFAULT_SOURCE@",
            Tap::System => "@DEFAULT_MONITOR@",
        }
    }
}

/// CLOCK_MONOTONIC in the 100-ns units the framers keep their timelines in.
///
/// The same machine-wide clock the server's own rtclock interpolation runs on,
/// read directly rather than through a pulse connection so the silence filler
/// can tick without one.
pub fn now_units() -> i64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // Documented never to fail for CLOCK_MONOTONIC on Linux.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    ts.tv_sec as i64 * 10_000_000 + ts.tv_nsec as i64 / 100
}

/// A connected context plus the mainloop that drives it, on this thread.
///
/// Field order is load-bearing: struct fields drop in declaration order, and
/// the context must release its io events while the mainloop still exists —
/// the other way round trips libpulse's `!e->dead` teardown assertion.
struct Connection {
    context: Context,
    mainloop: Mainloop,
}

impl Drop for Connection {
    fn drop(&mut self) {
        self.context.disconnect();
    }
}

fn connect(client_name: &str) -> Result<Connection, String> {
    let mut mainloop =
        Mainloop::new().ok_or("could not allocate a PulseAudio mainloop".to_string())?;
    let mut context = Context::new(&mainloop, client_name)
        .ok_or("could not allocate a PulseAudio context".to_string())?;
    context
        .connect(None, ContextFlagSet::NOFLAGS, None)
        .map_err(|e| server_missing(&e.to_string().unwrap_or_default()))?;
    loop {
        iterate(&mut mainloop)?;
        match context.get_state() {
            ContextState::Ready => break,
            ContextState::Failed | ContextState::Terminated => {
                let why = context.errno().to_string().unwrap_or_default();
                return Err(server_missing(&why));
            }
            _ => {}
        }
    }
    Ok(Connection { mainloop, context })
}

fn server_missing(detail: &str) -> String {
    format!(
        "could not connect to the sound server ({detail}) — is PipeWire (pipewire-pulse) \
         or PulseAudio running?"
    )
}

/// One blocking mainloop dispatch; an error here means the server connection
/// is gone, which every caller treats as fatal for its channel.
fn iterate(mainloop: &mut Mainloop) -> Result<(), String> {
    match mainloop.iterate(true) {
        IterateResult::Success(_) => Ok(()),
        IterateResult::Quit(_) | IterateResult::Err(_) => {
            Err("the sound server connection died".to_string())
        }
    }
}

/// The server's default source and sink names, as it reports them right now.
fn server_defaults(conn: &mut Connection) -> Result<(Option<String>, Option<String>), String> {
    let fetched: Rc<RefCell<Option<(Option<String>, Option<String>)>>> =
        Rc::new(RefCell::new(None));
    let sink_slot = Rc::clone(&fetched);
    let op = conn.context.introspect().get_server_info(move |info| {
        *sink_slot.borrow_mut() = Some((
            info.default_source_name.as_ref().map(|n| n.to_string()),
            info.default_sink_name.as_ref().map(|n| n.to_string()),
        ));
    });
    while op.get_state() == OperationState::Running {
        iterate(&mut conn.mainloop)?;
    }
    // Bound, not returned as the tail expression: a `RefMut` in tail position
    // outlives the `Rc` it borrows from and the borrow checker refuses it.
    let answer = fetched.borrow_mut().take();
    answer.ok_or_else(|| "the sound server did not answer the device query".to_string())
}

/// Every microphone the server knows, with the default one marked.
///
/// Monitors are dropped: a sink's monitor is a *source* to the server, but it
/// is the far end of a loudspeaker, not a microphone, and offering one in the
/// mic picker would silently record the system channel twice.
fn sources(conn: &mut Connection) -> Result<Vec<InputDevice>, String> {
    let collected: Rc<RefCell<Vec<(String, String)>>> = Rc::new(RefCell::new(Vec::new()));
    let slot = Rc::clone(&collected);
    let op = conn.context.introspect().get_source_info_list(move |result| {
        let ListResult::Item(info) = result else { return };
        if info.monitor_of_sink.is_some() {
            return;
        }
        let Some(name) = info.name.as_ref().map(|n| n.to_string()) else { return };
        let description =
            info.description.as_ref().map(|d| d.to_string()).unwrap_or_else(|| name.clone());
        slot.borrow_mut().push((name, description));
    });
    while op.get_state() == OperationState::Running {
        iterate(&mut conn.mainloop)?;
    }
    let (default_source, _) = server_defaults(conn)?;
    let devices = collected
        .borrow()
        .iter()
        .map(|(name, description)| InputDevice {
            is_default: default_source.as_deref() == Some(name.as_str()),
            id: name.clone(),
            name: description.clone(),
        })
        .collect();
    Ok(devices)
}

/// Every microphone this machine could record from, for the picker.
pub fn list_inputs() -> Result<Vec<InputDevice>, String> {
    let mut conn = connect("stenograf")?;
    sources(&mut conn)
}

/// The source the mic channel records from: the pinned one, or the server's
/// default alias (which it re-resolves itself).
fn mic_source(conn: &mut Connection, pin: Option<&str>) -> Result<(String, String), String> {
    let Some(pin) = pin else {
        return Ok((Tap::Mic.device().to_string(), Tap::Mic.device().to_string()));
    };
    let chosen = resolve_with_grace(pin, || sources(conn))?;
    Ok((chosen.id, chosen.name))
}

/// What each channel would record from right now, for the CLI's preflight.
pub fn device_names(taps: &[Tap], mic_device: Option<&str>) -> Result<Vec<(Tap, String)>, String> {
    let mut conn = connect("stenograf")?;
    // A pinned mic is validated against the source list (and named by its
    // human description, the way the picker named it); an unpinned one is
    // whatever the server currently calls its default.
    let pinned = match mic_device {
        Some(pin) if taps.contains(&Tap::Mic) => Some(mic_source(&mut conn, Some(pin))?.1),
        _ => None,
    };
    let (source, sink) = server_defaults(&mut conn)?;

    let mut names = Vec::new();
    for &tap in taps {
        let name = match tap {
            Tap::Mic => match &pinned {
                Some(name) => name.clone(),
                None => source
                    .clone()
                    .ok_or("no default microphone is configured — check sound settings")?,
            },
            Tap::System => {
                let sink = sink
                    .clone()
                    .ok_or("no default output device is configured — check sound settings")?;
                format!("{sink}.monitor")
            }
        };
        names.push((tap, name));
    }
    Ok(names)
}

/// Own one channel end to end until `stop`: connection, stream, framing, stamps.
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
    let mut conn = match connect("stenograf") {
        Ok(conn) => conn,
        Err(why) => return Some(why),
    };
    // Resolved before the stream exists: a pin that names nothing present must
    // fail with its own reason here, not as a stream that goes Failed ten
    // seconds later with the server's generic complaint.
    let device = match tap {
        Tap::Mic => match mic_source(&mut conn, mic_device) {
            Ok((id, _)) => id,
            Err(why) => return Some(why),
        },
        Tap::System => Tap::System.device().to_string(),
    };
    let spec = Spec { format: Format::S16le, channels: 1, rate: SAMPLE_RATE };
    debug_assert!(spec.is_valid());
    let mut stream = match Stream::new(&mut conn.context, tap.label(), &spec, None) {
        Some(stream) => stream,
        None => return Some(format!("could not allocate the {} stream", tap.label())),
    };
    // tlength/prebuf/minreq are playback-only; MAX means server defaults.
    let attr = BufferAttr {
        maxlength: u32::MAX,
        tlength: u32::MAX,
        prebuf: u32::MAX,
        minreq: u32::MAX,
        fragsize: FRAGSIZE_BYTES,
    };
    let mut flags = StreamFlagSet::INTERPOLATE_TIMING
        | StreamFlagSet::AUTO_TIMING_UPDATE
        | StreamFlagSet::ADJUST_LATENCY;
    // A chosen mic must fail rather than drift: the server re-routes record
    // streams whose device disappears, so without this the pin lasts only until
    // the device is unplugged and then silently records the fallback source
    // instead — measured 2026-08-12, the audio changed device mid-capture with
    // no diagnostic, which is the wrong-microphone transcript the pin exists to
    // prevent.
    let pinned = pinned(tap, mic_device);
    if pinned {
        flags |= StreamFlagSet::DONT_MOVE;
    }
    if let Err(err) = stream.connect_record(Some(&device), Some(&attr), flags) {
        return Some(open_failure(tap, &device, &err.to_string().unwrap_or_default()));
    }

    // Ready, then timing: stamps derive from the server's latency accounting,
    // so nothing is peeked until the first timing update has landed
    // (`get_latency` returns `Latency::None` until then). Audio delivered in
    // the meantime waits in the stream buffer with the read position parked,
    // and the backlog is part of the latency once it is known — the wait loses
    // nothing and misplaces nothing.
    let setup_deadline = std::time::Instant::now() + SETUP_TIMEOUT;
    let mut latency_units: i64 = loop {
        if stop.load(Ordering::Relaxed) {
            let _ = stream.disconnect();
            return None;
        }
        if setup_deadline < std::time::Instant::now() {
            return Some(open_failure(tap, &device, "timed out waiting for the stream to start"));
        }
        if let Err(why) = iterate(&mut conn.mainloop) {
            return Some(why);
        }
        match stream.get_state() {
            StreamState::Ready => {}
            StreamState::Failed | StreamState::Terminated => {
                let why = conn.context.errno().to_string().unwrap_or_default();
                return Some(open_failure(tap, &device, &why));
            }
            _ => continue,
        }
        match latency(&stream) {
            Some(units) => break units,
            None => continue,
        }
    };

    // Worth a line: which device the server actually resolved the default
    // alias to — "which endpoint is this channel recording" is the first
    // question a silent channel raises.
    let device = stream.get_device_name().unwrap_or_else(|| tap.device().into());
    log(&format!("{} capturing from {}", tap.label(), device));
    let _ = started.send(());

    let mut samples: Vec<i16> = Vec::new();
    let mut carry: Option<u8> = None;
    let mut timing_refreshed = std::time::Instant::now();
    let mut liveness = Liveness::new(Instant::now());
    let outcome = 'pump: loop {
        if stop.load(Ordering::Relaxed) {
            break None;
        }
        liveness.tick(Instant::now());
        if timing_refreshed.elapsed() >= TIMING_REFRESH {
            timing_refreshed = std::time::Instant::now();
            // Fire and forget: the reply lands during a later dispatch, and
            // dropping the Operation only releases our handle on it.
            let _ = stream.update_timing_info(None);
        }
        // Dispatch whatever the socket holds, then drain the stream dry.
        match conn.mainloop.iterate(false) {
            IterateResult::Success(_) => {}
            IterateResult::Quit(_) | IterateResult::Err(_) => {
                break Some(format!("{} capture stream died (connection lost)", tap.label()));
            }
        }
        loop {
            // The latency belongs to the sample at the read position — the
            // first sample of whatever the peek below returns — so it is read
            // before the peek, and kept when the server has no fresh answer
            // (it drifts slowly; the framer's tolerance absorbs the error).
            latency_units = latency(&stream).unwrap_or(latency_units);
            match stream.peek() {
                Err(err) => {
                    break 'pump Some(format!(
                        "{} capture stream died ({err})",
                        tap.label()
                    ));
                }
                Ok(PeekResult::Empty) => break,
                Ok(PeekResult::Hole(_)) => {
                    liveness.observed(Instant::now());
                    // Overrun: audio the server dropped. Discarding advances
                    // the read position; the next chunk's stamp jumps forward
                    // and the framer fills the gap with silence, keeping later
                    // audio at the instant it happened.
                    let _ = stream.discard();
                }
                Ok(PeekResult::Data(bytes)) => {
                    liveness.observed(Instant::now());
                    let stamp = now_units() - latency_units;
                    decode(&mut carry, bytes, &mut samples);
                    let _ = stream.discard();
                    let pushed = framer
                        .lock()
                        .expect("framer poisoned")
                        .push(stamp, &samples, &sink);
                    match pushed {
                        Ok(Some(complaint)) => log(&complaint),
                        Ok(None) => {}
                        Err(err) => break 'pump Some(format!("could not write frames ({err})")),
                    }
                }
            }
        }
        // After the drain, so what the server had waiting for us has been taken
        // into account before the silence is held against the device.
        if pinned && liveness.lost(Instant::now()) {
            break Some(format!(
                "the selected microphone ({device}) stopped delivering audio for \
                 {}s — reconnect it and start again, or pass `--mic-device default` \
                 to follow the system default",
                DEVICE_LOST.as_secs()
            ));
        }
        std::thread::sleep(POLL);
    };

    let _ = stream.disconnect();
    outcome
}

fn open_failure(tap: Tap, device: &str, detail: &str) -> String {
    format!(
        "could not open {} capture from {device} ({detail}) — check sound settings",
        tap.label(),
    )
}

/// The stream's current latency in 100-ns units, or `None` before the first
/// timing update (and on any transient server error).
fn latency(stream: &Stream) -> Option<i64> {
    match stream.get_latency() {
        Ok(Latency::Positive(us)) => Some(us.0 as i64 * 10),
        // The monitor can momentarily run ahead of its sink's reported
        // latency; a negative latency means the sample is younger than now.
        Ok(Latency::Negative(us)) => Some(-(us.0 as i64 * 10)),
        Ok(Latency::None) | Err(_) => None,
    }
}

/// s16le bytes to samples, carrying a split byte across chunk boundaries.
///
/// The server delivers whole frames in practice, but the protocol speaks
/// bytes; silently dropping a stray odd byte would shift every later sample
/// by eight bits, which is a desync worth five lines to rule out.
fn decode(carry: &mut Option<u8>, bytes: &[u8], out: &mut Vec<i16>) {
    out.clear();
    let mut rest = bytes;
    if let Some(low) = carry.take() {
        match rest.split_first() {
            Some((&high, tail)) => {
                out.push(i16::from_le_bytes([low, high]));
                rest = tail;
            }
            None => {
                *carry = Some(low);
                return;
            }
        }
    }
    let chunks = rest.chunks_exact(2);
    *carry = chunks.remainder().first().copied();
    out.extend(chunks.map(|pair| i16::from_le_bytes([pair[0], pair[1]])));
}

#[cfg(test)]
mod tests {
    use super::{decode, pinned, Duration, Instant, Liveness, Tap, DEVICE_LOST};

    #[test]
    fn decode_reassembles_a_sample_split_across_chunks() {
        let mut carry = None;
        let mut out = Vec::new();
        decode(&mut carry, &[0x34, 0x12, 0x78], &mut out);
        assert_eq!(out, [0x1234]);
        assert_eq!(carry, Some(0x78));
        decode(&mut carry, &[0x56], &mut out);
        assert_eq!(out, [0x5678]);
        assert_eq!(carry, None);
    }

    #[test]
    fn decode_holds_a_lone_byte_until_its_partner_arrives() {
        let mut carry = None;
        let mut out = Vec::new();
        decode(&mut carry, &[0x0d], &mut out);
        assert!(out.is_empty());
        decode(&mut carry, &[], &mut out);
        assert!(out.is_empty());
        assert_eq!(carry, Some(0x0d));
    }

    #[test]
    fn only_a_pinned_mic_refuses_to_be_moved() {
        assert!(pinned(Tap::Mic, Some("usb-1")));
        // Both of these follow the default by being moved, so neither may take
        // DONT_MOVE or be judged dead for going quiet.
        assert!(!pinned(Tap::Mic, None));
        assert!(!pinned(Tap::System, Some("usb-1")));
    }

    #[test]
    fn a_pinned_mic_is_lost_only_after_the_full_silence() {
        let t0 = Instant::now();
        let mut live = Liveness::new(t0);
        assert!(!live.lost(t0 + DEVICE_LOST - Duration::from_millis(1)));
        assert!(live.lost(t0 + DEVICE_LOST));
        live.observed(t0 + DEVICE_LOST);
        assert!(!live.lost(t0 + DEVICE_LOST + Duration::from_secs(1)));
    }

    #[test]
    fn a_stalled_pump_is_not_a_lost_device() {
        // The pump stops being scheduled for longer than the whole budget and
        // then resumes: it observed nothing, which is not the same as the
        // device delivering nothing.
        let t0 = Instant::now();
        let mut live = Liveness::new(t0);
        let resumed = t0 + DEVICE_LOST + Duration::from_secs(5);
        live.tick(resumed);
        assert!(!live.lost(resumed));
    }

    #[test]
    fn steady_ticks_do_not_excuse_a_silent_device() {
        // Forgiveness must not become amnesty: a pump running normally holds
        // the device to the full DEVICE_LOST.
        let t0 = Instant::now();
        let mut live = Liveness::new(t0);
        let mut now = t0;
        for _ in 0..100 {
            now += Duration::from_millis(100);
            live.tick(now);
        }
        assert!(live.lost(now));
    }
}
