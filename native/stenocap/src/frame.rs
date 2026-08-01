//! The wire protocol and the framing arithmetic every backend shares.
//!
//! stdout carries frames only (status and errors go to stderr), little-endian:
//!
//!     frame = channel:u8  timestamp:f64  count:u32  samples:count×i16
//!
//! `timestamp` is seconds since capture start **on one clock for both
//! channels**, so equal timestamps mean simultaneous capture. That invariant is
//! the whole reason this helper exists: the echo canceller pairs the mic
//! against the system reference by timestamp, and a per-channel clock silently
//! hands it a reference that does not line up with the echo it must remove.
//!
//! A channel's timeline is therefore kept in the device's own units (100 ns,
//! the resolution WASAPI reports positions in) rather than in samples, and
//! samples are placed *onto* it:
//!
//! - [`Framer::next`] is where the next sample belongs. Every packet arrives
//!   with the device's stamp for its first sample; the difference between the
//!   two is a gap in the capture, and it is filled with silence rather than
//!   pretending the samples were contiguous. That is what keeps a channel's
//!   audio at the instant it happened after a dropout.
//! - The timeline never moves backwards. A stamp behind `next` is not acted on
//!   (it would misplace everything after it, and `GapPaddedBuffer` on the Python
//!   side rejects the frame outright); it is reported once and the channel's own
//!   count stays the authority.
//!
//! 16 kHz divides 10 MHz exactly — 625 units per sample — so none of this
//! arithmetic accumulates rounding error over a meeting.

use std::io::{self, Write};
use std::sync::mpsc;
use std::sync::Mutex;
use std::thread::JoinHandle;

pub const SAMPLE_RATE: u32 = 16_000;

/// 100-ns units per sample at 16 kHz. Exact, hence integer timelines.
pub const UNITS_PER_SAMPLE: i64 = 10_000_000 / SAMPLE_RATE as i64;

/// ~200 ms, the cadence every provider delivers to the core
/// (`capture.base.DEFAULT_FRAME_MS`).
pub const FRAME_SAMPLES: usize = SAMPLE_RATE as usize / 5;

/// Stamp/timeline disagreement absorbed rather than acted on (5 ms).
///
/// Packets arrive every ~10 ms and their stamps jitter by a fraction of that,
/// so reacting to every difference would churn the timeline by a sample or two
/// forever. Anything larger is a real gap (or a real overrun) and is treated as
/// one — including slow drift between the device's clock and QPC, which
/// accumulates until it crosses this line and is then corrected in one step.
const GAP_TOLERANCE_UNITS: i64 = 50_000;

pub const CHANNEL_MIC: u8 = 0;
pub const CHANNEL_SYSTEM: u8 = 1;

/// The one writer of the frame stream: serializes records from every channel.
///
/// Channels are pumped by independent threads and a half-written record would
/// desync the consumer permanently, so the lock is held across the whole frame.
/// Each frame is flushed as it is written — a buffered writer would hold ~10
/// frames back, and the live captions are downstream of this pipe.
pub struct FrameSink {
    out: Mutex<Box<dyn Write + Send>>,
    t0: i64,
}

impl FrameSink {
    /// `t0` is the shared origin, in the same 100-ns units the framers use.
    pub fn new(t0: i64, out: Box<dyn Write + Send>) -> Self {
        Self { out: Mutex::new(out), t0 }
    }

    pub fn write(&self, channel: u8, start: i64, samples: &[i16]) -> io::Result<()> {
        // Clamped at zero: the system tap's stamps are render-side and can
        // therefore precede the moment we read the clock at startup by a
        // buffer. Clamping is monotone, so ordering survives it.
        let timestamp = ((start - self.t0) as f64 / 10_000_000.0).max(0.0);
        let mut record = Vec::with_capacity(13 + samples.len() * 2);
        record.push(channel);
        record.extend_from_slice(&timestamp.to_le_bytes());
        record.extend_from_slice(&(samples.len() as u32).to_le_bytes());
        for sample in samples {
            record.extend_from_slice(&sample.to_le_bytes());
        }
        let mut out = self.out.lock().expect("frame sink poisoned");
        out.write_all(&record)?;
        out.flush()
    }
}

/// stdout behind a queue, plus the thread that drains it.
///
/// **A capture pump may never block on the consumer.** It cost a real bug to
/// re-learn: writing frames inline meant that when the reader stalled — the
/// first meeting on a machine stalls it for seconds while the ASR model loads —
/// the pipe filled, the pump blocked mid-write holding its channel's lock, and
/// WASAPI went on buffering. The packets that came back afterwards carried
/// honest timestamps from *nine seconds earlier*, by which time the silence
/// filler had moved the timeline past them. Frames now go into an unbounded
/// queue and one thread owns the pipe.
///
/// Unbounded is deliberate, and the same trade the consumer makes on its side:
/// a stalled reader costs ~64 KB/s of memory in a process that streams meeting
/// audio, while dropping frames would lose the meeting to a stall the reader
/// recovers from. A reader that is *gone* rather than slow breaks the pipe,
/// which ends the writer rather than growing forever.
pub fn queued_stdout() -> (Box<dyn Write + Send>, JoinHandle<()>) {
    let (tx, rx) = mpsc::channel::<Vec<u8>>();
    let writer = std::thread::spawn(move || {
        let mut out = io::stdout();
        for record in rx {
            if out.write_all(&record).and_then(|()| out.flush()).is_err() {
                eprintln!("stenocap: the frame stream's reader is gone");
                return;
            }
        }
    });
    (Box::new(Queued { tx, pending: Vec::new() }), writer)
}

/// Accumulates one record and hands it to the writer thread on flush.
///
/// [`FrameSink::write`] writes a whole record and then flushes exactly once, so
/// "flush" is the record boundary rather than a guess.
struct Queued {
    tx: mpsc::Sender<Vec<u8>>,
    pending: Vec<u8>,
}

impl Write for Queued {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.pending.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let record = std::mem::take(&mut self.pending);
        self.tx
            .send(record)
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "frame writer stopped"))
    }
}

/// One channel's timeline: packets in, whole frames out.
pub struct Framer {
    channel: u8,
    pending: Vec<i16>,
    /// Where the next sample belongs, in 100-ns units on the shared clock.
    /// `None` until the channel anchors — on its first packet, or at
    /// construction for a channel that must exist from t=0 ([`Framer::anchored`]).
    next: Option<i64>,
    warned_backwards: bool,
}

impl Framer {
    /// A channel that anchors on its first packet — the microphone.
    ///
    /// Its timeline starts where its audio does; inventing leading silence
    /// would be a claim about audio nobody captured.
    pub fn new(channel: u8) -> Self {
        Self { channel, pending: Vec::new(), next: None, warned_backwards: false }
    }

    /// A channel that exists from `at` whether or not it has delivered anything
    /// — the system reference.
    ///
    /// A meeting where nothing renders at all produces no loopback packets
    /// *ever*, and a far-end track that never starts stalls the echo canceller
    /// against a mic that is talking: it waits for a reference that will never
    /// arrive, then charges the wait to its reference-loss budget. Anchoring at
    /// the shared origin lets [`Framer::fill_silence`] deliver the digital
    /// silence that genuinely is at the endpoint.
    pub fn anchored(channel: u8, at: i64) -> Self {
        Self { channel, pending: Vec::new(), next: Some(at), warned_backwards: false }
    }

    /// Place one packet's samples at the device's stamp for its first sample.
    pub fn push(
        &mut self,
        stamp: i64,
        samples: &[i16],
        sink: &FrameSink,
    ) -> io::Result<Option<String>> {
        let next = *self.next.get_or_insert(stamp);
        let gap = stamp - next;
        let mut complaint = None;
        if gap > GAP_TOLERANCE_UNITS {
            self.append(&vec![0i16; (gap / UNITS_PER_SAMPLE) as usize]);
        } else if gap < -GAP_TOLERANCE_UNITS && !self.warned_backwards {
            self.warned_backwards = true;
            complaint = Some(format!(
                "channel {} delivered a stamp {:.0} ms behind its own timeline; \
                 keeping the sample count as the authority",
                self.channel,
                -gap as f64 / 10_000.0,
            ));
        }
        self.append(samples);
        self.emit_full(sink)?;
        Ok(complaint)
    }

    /// Bring the timeline up to `now - lead` with silence, and flush what is
    /// pending even if it is short of a whole frame.
    ///
    /// WASAPI's loopback tap delivers nothing at all while nothing renders, so
    /// without this the reference simply stops during a quiet stretch. `lead`
    /// keeps the fill behind real time by more than a packet's delivery jitter,
    /// so audio that resumes cannot land in a span already filled. The partial
    /// flush is what keeps the reference tracking real time to within `lead`
    /// instead of one frame plus `lead`.
    pub fn fill_silence(&mut self, now: i64, lead: i64, sink: &FrameSink) -> io::Result<()> {
        let Some(next) = self.next else { return Ok(()) };
        let missing = (now - lead - next) / UNITS_PER_SAMPLE;
        if missing <= 0 {
            return Ok(()); // the channel is delivering; nothing to invent
        }
        self.append(&vec![0i16; missing as usize]);
        self.flush(sink)
    }

    /// Emit everything held, whole frames and any remainder.
    pub fn flush(&mut self, sink: &FrameSink) -> io::Result<()> {
        self.emit_full(sink)?;
        if !self.pending.is_empty() {
            let start = self.start_of_pending();
            sink.write(self.channel, start, &self.pending)?;
            self.pending.clear();
        }
        Ok(())
    }

    fn append(&mut self, samples: &[i16]) {
        self.pending.extend_from_slice(samples);
        if let Some(next) = self.next.as_mut() {
            *next += samples.len() as i64 * UNITS_PER_SAMPLE;
        }
    }

    fn emit_full(&mut self, sink: &FrameSink) -> io::Result<()> {
        while self.pending.len() >= FRAME_SAMPLES {
            let start = self.start_of_pending();
            sink.write(self.channel, start, &self.pending[..FRAME_SAMPLES])?;
            self.pending.drain(..FRAME_SAMPLES);
        }
        Ok(())
    }

    fn start_of_pending(&self) -> i64 {
        self.next.unwrap_or(0) - self.pending.len() as i64 * UNITS_PER_SAMPLE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// A sink that keeps the bytes, and the parser the Python side implements.
    #[derive(Clone, Default)]
    struct Recorder(Arc<Mutex<Vec<u8>>>);

    impl Write for Recorder {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    struct Record {
        channel: u8,
        timestamp: f64,
        samples: Vec<i16>,
    }

    fn sink() -> (FrameSink, Recorder) {
        let recorder = Recorder::default();
        (FrameSink::new(0, Box::new(recorder.clone())), recorder)
    }

    fn parse(recorder: &Recorder) -> Vec<Record> {
        let bytes = recorder.0.lock().unwrap().clone();
        let mut records = Vec::new();
        let mut at = 0;
        while at < bytes.len() {
            let channel = bytes[at];
            let timestamp = f64::from_le_bytes(bytes[at + 1..at + 9].try_into().unwrap());
            let count = u32::from_le_bytes(bytes[at + 9..at + 13].try_into().unwrap()) as usize;
            at += 13;
            let samples = bytes[at..at + count * 2]
                .chunks_exact(2)
                .map(|p| i16::from_le_bytes([p[0], p[1]]))
                .collect();
            at += count * 2;
            records.push(Record { channel, timestamp, samples });
        }
        records
    }

    #[test]
    fn the_writer_queue_carries_one_message_per_record() {
        // The record boundary is the flush, so a partial write must not be
        // handed on: half a frame desyncs the consumer permanently.
        let (tx, rx) = mpsc::channel();
        let mut queued = Queued { tx, pending: Vec::new() };
        queued.write_all(b"head").unwrap();
        queued.write_all(b"body").unwrap();
        assert!(rx.try_recv().is_err(), "nothing may leave before the flush");
        queued.flush().unwrap();
        assert_eq!(rx.try_recv().unwrap(), b"headbody");
        queued.flush().unwrap(); // an empty flush is not an empty record
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn emits_whole_frames_only_until_flushed() {
        let (sink, recorder) = sink();
        let mut framer = Framer::new(CHANNEL_MIC);
        framer.push(0, &vec![7i16; FRAME_SAMPLES + 100], &sink).unwrap();

        let records = parse(&recorder);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].channel, CHANNEL_MIC);
        assert_eq!(records[0].samples.len(), FRAME_SAMPLES);

        framer.flush(&sink).unwrap();
        let records = parse(&recorder);
        assert_eq!(records.len(), 2);
        assert_eq!(records[1].samples.len(), 100);
        assert_eq!(records[1].timestamp, 0.2);
    }

    #[test]
    fn the_first_packets_stamp_becomes_the_channels_origin() {
        let (sink, recorder) = sink();
        // t0 is 0, but the mic's first packet is stamped a second in.
        let mut framer = Framer::new(CHANNEL_MIC);
        framer.push(10_000_000, &vec![1i16; FRAME_SAMPLES], &sink).unwrap();
        assert_eq!(parse(&recorder)[0].timestamp, 1.0);
    }

    #[test]
    fn a_gap_in_the_capture_is_filled_so_later_audio_keeps_its_instant() {
        let (sink, recorder) = sink();
        let mut framer = Framer::new(CHANNEL_SYSTEM);
        framer.push(0, &[1i16; 1600], &sink).unwrap(); // 100 ms
        // The next packet is stamped 500 ms in: 400 ms of the capture is missing.
        framer.push(5_000_000, &[2i16; 1600], &sink).unwrap();
        framer.flush(&sink).unwrap();

        let records = parse(&recorder);
        let samples: Vec<i16> = records.iter().flat_map(|r| r.samples.clone()).collect();
        assert_eq!(records[0].timestamp, 0.0);
        assert_eq!(samples.len(), 1600 + 6400 + 1600);
        assert_eq!(samples[1600..8000], [0i16; 6400]);
        assert_eq!(samples[8000], 2); // the late audio lands at 500 ms, not 100 ms
    }

    #[test]
    fn jitter_under_the_tolerance_does_not_move_the_timeline() {
        let (sink, recorder) = sink();
        let mut framer = Framer::new(CHANNEL_MIC);
        for packet in 0..20 {
            // Each packet is 10 ms but stamped 1 ms late — jitter, not a gap.
            let stamp = packet * 100_000 + 10_000;
            framer.push(stamp, &[3i16; 160], &sink).unwrap();
        }
        framer.flush(&sink).unwrap();
        let samples: usize = parse(&recorder).iter().map(|r| r.samples.len()).sum();
        assert_eq!(samples, 20 * 160); // no silence invented
    }

    #[test]
    fn a_backwards_stamp_is_reported_once_and_never_acted_on() {
        let (sink, _recorder) = sink();
        let mut framer = Framer::new(CHANNEL_MIC);
        framer.push(10_000_000, &[1i16; 160], &sink).unwrap();
        let first = framer.push(0, &[1i16; 160], &sink).unwrap();
        let second = framer.push(0, &[1i16; 160], &sink).unwrap();
        assert!(first.is_some());
        assert!(second.is_none(), "a broken clock must not spam the log");
    }

    #[test]
    fn an_anchored_channel_delivers_silence_it_never_received() {
        let (sink, recorder) = sink();
        let mut framer = Framer::anchored(CHANNEL_SYSTEM, 0);
        // Nothing has rendered for a second; the tap has produced no packets.
        framer.fill_silence(10_000_000, 1_000_000, &sink).unwrap();

        let records = parse(&recorder);
        let samples: usize = records.iter().map(|r| r.samples.len()).sum();
        assert_eq!(records[0].timestamp, 0.0);
        assert_eq!(samples, 14_400); // 900 ms: the lead is deliberately not filled
    }

    #[test]
    fn a_delivering_channel_is_never_filled() {
        let (sink, recorder) = sink();
        let mut framer = Framer::anchored(CHANNEL_SYSTEM, 0);
        framer.push(0, &[5i16; 16_000], &sink).unwrap(); // a second of real audio
        framer.fill_silence(10_000_000, 1_000_000, &sink).unwrap();
        framer.flush(&sink).unwrap();

        let samples: Vec<i16> = parse(&recorder).iter().flat_map(|r| r.samples.clone()).collect();
        assert_eq!(samples.len(), 16_000);
        assert!(samples.iter().all(|&s| s == 5));
    }

    #[test]
    fn frames_stay_contiguous_across_a_fill() {
        let (sink, recorder) = sink();
        let mut framer = Framer::anchored(CHANNEL_SYSTEM, 0);
        framer.fill_silence(5_000_000, 1_000_000, &sink).unwrap();
        framer.push(5_000_000, &[9i16; 1600], &sink).unwrap();
        framer.flush(&sink).unwrap();

        let records = parse(&recorder);
        let mut expected = 0.0;
        for record in &records {
            assert!(
                (record.timestamp - expected).abs() < 1e-9,
                "frame at {} broke the timeline (expected {expected})",
                record.timestamp
            );
            expected += record.samples.len() as f64 / SAMPLE_RATE as f64;
        }
        // 400 ms filled, then the 100 ms the lead had held back, then the audio.
        let samples: usize = records.iter().map(|r| r.samples.len()).sum();
        assert_eq!(samples, 6400 + 1600 + 1600);
    }
}
