//! stenocap — the capture helper for stenograf on Windows and Linux.
//!
//! Captures the microphone and system audio, resamples both to mono 16 kHz
//! int16, and streams them as framed PCM on stdout. Audio is never written to
//! disk; this process only streams it.
//!
//! Usage:
//!   stenocap [--mic] [--system]   # at least one channel
//!   stenocap --devices            # what each channel would record from, as JSON
//!
//! stdout carries frames only and stderr carries status — see `frame.rs` for
//! the record layout and the timeline rules, and `wasapi.rs` / `pulse.rs` for
//! where each platform's timestamps come from. The macOS twin
//! (`../stenocap-macos`, Swift) speaks the identical protocol; the consumer of
//! all three is `stenograf.capture`.
//!
//! **Stopping is stdin reaching EOF**, not a signal. POSIX SIGINT is what the
//! macOS helper uses, and Windows has no equivalent that a parent can aim at
//! one child: `CTRL_C_EVENT` goes to a whole process group, and needs the child
//! in one of its own to be safe. Closing the pipe needs no console, no process
//! group and no handler, and it cannot arrive before the helper is ready to see
//! it. The parent closing stdin, or simply exiting, both stop capture cleanly.
//! Linux keeps the same gesture — one stop path for the one crate — and
//! therefore *ignores* SIGINT: a terminal's Ctrl+C hits the whole process
//! group, and a helper that died on it mid-meeting would drop its buffered
//! tail while the parent was still draining the pipe.

mod frame;
#[cfg(target_os = "linux")]
mod pulse;
#[cfg(windows)]
mod wasapi;

#[cfg(target_os = "linux")]
use pulse as backend;
#[cfg(windows)]
use wasapi as backend;

use std::io::Read;
use std::process::ExitCode;

use frame::{FrameSink, Framer, CHANNEL_MIC, CHANNEL_SYSTEM};

fn log(message: &str) {
    eprintln!("stenocap: {message}");
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let want_mic = args.iter().any(|a| a == "--mic");
    let want_system = args.iter().any(|a| a == "--system");
    let want_devices = args.iter().any(|a| a == "--devices");

    // --help succeeds and a bare invocation does not: the second is a caller
    // that forgot to name a channel, and `stenocap --help` is what a smoke test
    // runs to prove the binary in a wheel actually starts on the machine.
    if args.iter().any(|a| a == "-h" || a == "--help") {
        log("usage: stenocap [--mic] [--system] | --devices");
        return ExitCode::SUCCESS;
    }
    if !want_mic && !want_system && !want_devices {
        log("usage: stenocap [--mic] [--system] | --devices  (at least one channel)");
        return ExitCode::from(2);
    }
    run(want_mic, want_system, want_devices)
}

#[cfg(not(any(windows, target_os = "linux")))]
fn run(_mic: bool, _system: bool, _devices: bool) -> ExitCode {
    log("FATAL: this helper runs on Windows and Linux only (macOS uses the Swift helper)");
    ExitCode::FAILURE
}

#[cfg(any(windows, target_os = "linux"))]
fn run(want_mic: bool, want_system: bool, want_devices: bool) -> ExitCode {
    use backend::Tap;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    // Stopping is stdin EOF (module docstring); Ctrl+C belongs to the parent.
    #[cfg(target_os = "linux")]
    unsafe {
        libc::signal(libc::SIGINT, libc::SIG_IGN);
    }

    let mut taps = Vec::new();
    if want_mic {
        taps.push(Tap::Mic);
    }
    if want_system {
        taps.push(Tap::System);
    }

    if want_devices {
        if taps.is_empty() {
            taps = vec![Tap::Mic, Tap::System];
        }
        return match backend::device_names(&taps) {
            Ok(names) => {
                let fields: Vec<String> = names
                    .iter()
                    .map(|(tap, name)| format!("\"{}\": \"{}\"", tap.label(), escape(name)))
                    .collect();
                println!("{{{}}}", fields.join(", "));
                ExitCode::SUCCESS
            }
            Err(why) => {
                log(&format!("FATAL: {why}"));
                ExitCode::FAILURE
            }
        };
    }

    // The shared origin, fixed before either channel can stamp a frame.
    let t0 = backend::now_units();
    let (out, writer) = frame::queued_stdout();
    let sink = Arc::new(FrameSink::new(t0, out));
    let stop = Arc::new(AtomicBool::new(false));

    let mut pumps = Vec::new();
    let mut framers = Vec::new();
    let mut system_framer = None;
    let (started_tx, started_rx) = std::sync::mpsc::channel();
    for &tap in &taps {
        // The mic anchors on its first packet; the system reference exists from
        // t=0 whether or not anything ever renders (see Framer::anchored).
        let framer = Arc::new(Mutex::new(match tap {
            Tap::Mic => Framer::new(CHANNEL_MIC, backend::GAP_TOLERANCE_UNITS),
            Tap::System => Framer::anchored(CHANNEL_SYSTEM, t0, backend::GAP_TOLERANCE_UNITS),
        }));
        if tap == Tap::System {
            system_framer = Some(Arc::clone(&framer));
        }
        framers.push(Arc::clone(&framer));
        let (sink, stop) = (Arc::clone(&sink), Arc::clone(&stop));
        let started = started_tx.clone();
        pumps.push(std::thread::spawn(move || {
            let outcome = backend::pump(tap, framer, sink, Arc::clone(&stop), started, log);
            die_if_unrequested(outcome, &stop);
        }));
    }
    drop(started_tx);

    // The silence filler is the system channel's alone: the loopback tap stops
    // delivering entirely while nothing renders, and a reference that stops is
    // one the canceller waits for.
    let filler = system_framer.map(|framer| {
        let (sink, stop) = (Arc::clone(&sink), Arc::clone(&stop));
        std::thread::spawn(move || {
            let outcome = frame::fill_silence(&framer, &sink, &stop, backend::now_units);
            die_if_unrequested(outcome, &stop);
        })
    });

    // "ready" means every requested channel is running, not that the threads
    // exist. A device that hangs instead of failing — a concurrent capture app
    // wedging the audio service is the way this happens — otherwise looks
    // exactly like a meeting where nobody spoke. A pump that *fails* is
    // already exiting the process with its own FATAL; the disconnect arm
    // catches the race where we notice the dead channel first, and must not
    // log a second FATAL — the consumer reports the last one, and a generic
    // line here would bury the pump's actual reason.
    for _ in &taps {
        use std::sync::mpsc::RecvTimeoutError;
        match started_rx.recv_timeout(std::time::Duration::from_secs(15)) {
            Ok(()) => {}
            Err(RecvTimeoutError::Timeout) => {
                log("FATAL: capture did not start within 15 s — is another app using the device?");
                return ExitCode::FAILURE;
            }
            Err(RecvTimeoutError::Disconnected) => {
                std::thread::sleep(std::time::Duration::from_secs(1)); // let its exit land
                return ExitCode::FAILURE;
            }
        }
    }
    log("ready");

    // Stopping is stdin reaching EOF (module docstring). Nothing is ever sent
    // on it, so this blocks for the whole meeting.
    let mut ignored = Vec::new();
    let _ = std::io::stdin().read_to_end(&mut ignored);
    stop.store(true, Ordering::Relaxed);

    for pump in pumps {
        let _ = pump.join();
    }
    if let Some(filler) = filler {
        let _ = filler.join();
    }
    // Flush after every pump has stopped touching its framer: a partial frame
    // is a whole frame to the consumer, and dropping it would lose the tail.
    for framer in &framers {
        let _ = framer.lock().expect("framer poisoned").flush(&sink);
    }
    // Then let the writer drain what is queued. Every other reference to the
    // sink died with the threads above, so dropping this one closes the channel
    // and the writer finishes on its own.
    drop(sink);
    let _ = writer.join();
    log("stopped");
    ExitCode::SUCCESS
}

/// A channel that ends on its own has taken the meeting with it.
///
/// One dead stream means half-captured audio, and the consumer cannot tell that
/// from a quiet room — so the whole helper exits loudly instead, which is what
/// makes the provider raise `CaptureHelperError` (or retry once, if this
/// happened before any audio flowed) rather than finalize a transcript that
/// looks like a successful meeting. The `FATAL` prefix is what carries the
/// reason into that error's message.
#[cfg(any(windows, target_os = "linux"))]
fn die_if_unrequested(outcome: Option<String>, stop: &std::sync::atomic::AtomicBool) {
    use std::sync::atomic::Ordering;

    let Some(why) = outcome else { return };
    if stop.load(Ordering::Relaxed) {
        return; // it ended because we asked it to
    }
    log(&format!("FATAL: {why}"));
    std::process::exit(1);
}

/// The two characters JSON string values may not carry raw. Device names come
/// from the driver, so they are not ours to trust.
#[cfg(any(windows, target_os = "linux"))]
fn escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}
