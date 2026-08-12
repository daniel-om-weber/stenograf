//! Turning a stored microphone selection into one device that is present now.
//!
//! The mic channel may be pinned to a device instead of following the OS
//! default (`--mic-device`), and what is stored is a platform-stable ID — a
//! WASAPI endpoint ID, a PulseAudio source name. A hand-edited settings file
//! may name the device instead, because a Windows endpoint GUID is not
//! something a human types.
//!
//! **The matching rule is normative and shared** with the Swift helper
//! (`../../stenocap-macos/main.swift`), the Python layer
//! (`stenograf.capture.helper`) and the test fake: trim surrounding
//! whitespace, normalize both sides to NFC, then compare case-sensitively —
//! the ID first, then the exact name. NFC is not decoration: macOS returns
//! device names in NFD, so a name typed into a TOML file on one machine and
//! compared unnormalized on another silently fails to match.
//!
//! Two rules exist because their failure is invisible: a name that matches two
//! present devices is an *error* naming both IDs (never a silent pick of the
//! first), and a device that is not there at all is never quietly replaced by
//! the default — recording the built-in mic when the user asked for the desk
//! mic is exactly what pinning exists to prevent, and it is only discovered
//! when the transcript is bad.

use std::time::{Duration, Instant};

use unicode_normalization::UnicodeNormalization;

/// One microphone the machine could record from.
#[derive(Clone)]
pub struct InputDevice {
    pub id: String,
    pub name: String,
    pub is_default: bool,
}

/// How long a pinned device gets to appear before the run fails.
///
/// A USB interface can enumerate a second or two after wake-from-sleep, and
/// without this the helper would be asymmetric in the worst way: a device
/// missing at t=0 kills the meeting, while one that vanishes at t=1 s is
/// retried forever. Matches the supervisors' own retry cadence.
const GRACE: Duration = Duration::from_secs(3);
const GRACE_STEP: Duration = Duration::from_millis(250);

/// Why a pin matched nothing usable. Kept apart from the message because only
/// one of the two is worth waiting on: a device can still show up, an
/// ambiguous name cannot become unambiguous.
#[derive(Debug)]
pub enum Unresolved {
    Missing,
    Ambiguous(Vec<String>),
}

impl Unresolved {
    pub fn message(&self, pin: &str) -> String {
        match self {
            Unresolved::Missing => format!(
                "the selected microphone \"{pin}\" is not available — run `steno devices` to \
                 list what is connected, or pass --mic-device default"
            ),
            Unresolved::Ambiguous(ids) => format!(
                "the microphone name \"{pin}\" matches {} connected devices ({}) — \
                 pass one of those IDs to --mic-device instead",
                ids.len(),
                ids.join(", ")
            ),
        }
    }
}

/// Trimmed and NFC-normalized, the one form both sides of a comparison take.
pub fn normalize(value: &str) -> String {
    value.trim().nfc().collect()
}

/// The one device a pin names, by ID first and then by exact name.
pub fn resolve<'a>(devices: &'a [InputDevice], pin: &str) -> Result<&'a InputDevice, Unresolved> {
    let wanted = normalize(pin);
    if let Some(found) = devices.iter().find(|d| normalize(&d.id) == wanted) {
        return Ok(found);
    }
    let named: Vec<&InputDevice> = devices
        .iter()
        .filter(|d| normalize(&d.name) == wanted)
        .collect();
    match named.as_slice() {
        [] => Err(Unresolved::Missing),
        [only] => Ok(only),
        several => Err(Unresolved::Ambiguous(
            several.iter().map(|d| d.id.clone()).collect(),
        )),
    }
}

/// [`resolve`], retried until [`GRACE`] elapses while the device is merely absent.
///
/// `list` is re-run per attempt (the device list is what changes); an error
/// from it is the audio stack itself failing and is never retried.
pub fn resolve_with_grace(
    pin: &str,
    mut list: impl FnMut() -> Result<Vec<InputDevice>, String>,
) -> Result<InputDevice, String> {
    let deadline = Instant::now() + GRACE;
    loop {
        match resolve(&list()?, pin) {
            Ok(found) => return Ok(found.clone()),
            Err(Unresolved::Missing) if Instant::now() < deadline => {
                std::thread::sleep(GRACE_STEP);
            }
            Err(why) => return Err(why.message(pin)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{normalize, resolve, InputDevice, Unresolved};

    fn device(id: &str, name: &str) -> InputDevice {
        InputDevice { id: id.into(), name: name.into(), is_default: false }
    }

    #[test]
    fn an_id_wins_over_a_name() {
        let devices = [device("usb-1", "Yeti"), device("Yeti", "Built-in")];
        assert_eq!(resolve(&devices, "Yeti").unwrap().id, "Yeti");
    }

    #[test]
    fn a_name_matches_after_nfc_and_trimming() {
        // "Bürgel" typed precomposed against a device name in decomposed form.
        let devices = [device("usb-1", "Bu\u{0308}rgel-Mikrofon")];
        assert_eq!(resolve(&devices, "  Bürgel-Mikrofon ").unwrap().id, "usb-1");
    }

    #[test]
    fn case_is_load_bearing() {
        let devices = [device("usb-1", "Yeti")];
        assert!(matches!(resolve(&devices, "yeti"), Err(Unresolved::Missing)));
    }

    #[test]
    fn two_devices_of_the_same_name_are_an_error_naming_both() {
        let devices = [device("usb-1", "Yeti"), device("usb-2", "Yeti")];
        let Err(why) = resolve(&devices, "Yeti") else { panic!("expected an ambiguity") };
        let message = why.message("Yeti");
        assert!(message.contains("usb-1") && message.contains("usb-2"), "{message}");
    }

    #[test]
    fn a_missing_device_names_the_remedy() {
        let devices = [device("usb-1", "Yeti")];
        let Err(why) = resolve(&devices, "usb-2") else { panic!("expected a miss") };
        assert!(why.message("usb-2").contains("--mic-device default"));
    }

    #[test]
    fn normalize_is_idempotent_on_ascii() {
        assert_eq!(normalize(" Built-in Microphone\t"), "Built-in Microphone");
    }
}
