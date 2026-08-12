// stenocap — stenograf's macOS capture helper (PLAN.md §2).
//
// Captures two independent channels and streams them to the Python core:
//   --system : whole-system audio via a Core Audio process tap (macOS 14.4+)
//   --mic    : the microphone via AVAudioEngine
//
// Both are downmixed + resampled to mono 16 kHz int16 and written to stdout as
// framed PCM. stdout carries frames only; all status/errors go to stderr.
//
//   frame = channel:u8  timestamp:f64le  count:u32le  samples:count×i16le
//   channel: 0 = mic, 1 = system;  timestamp: seconds since capture start.
//
// Both channels' timestamps share one Mach host-time origin, so a sample at
// time t on the mic and a sample at time t on the system tap were captured at
// the same instant. The Python echo canceller depends on that: it aligns the
// tap (far-end reference) against the mic (near-end) by timestamp.
//
// Audio is never written to disk — this process only streams it. Stop with
// SIGINT/SIGTERM; the helper flushes and exits 0. The capture APIs were proven
// in native/spike; this adds resampling, framing, and clean lifecycle.

import AVFoundation
import AudioToolbox
import CoreAudio
import Foundation

let SAMPLE_RATE = 16_000.0

// MARK: - protocol codes

enum ChannelCode: UInt8 {
    case mic = 0
    case system = 1
}

// MARK: - stderr logging

func log(_ message: String) {
    FileHandle.standardError.write(Data("stenocap: \(message)\n".utf8))
}

func fourCC(_ status: OSStatus) -> String {
    let n = UInt32(bitPattern: status)
    let bytes = [UInt8((n >> 24) & 0xff), UInt8((n >> 16) & 0xff),
                 UInt8((n >> 8) & 0xff), UInt8(n & 0xff)]
    if bytes.allSatisfy({ $0 >= 0x20 && $0 < 0x7f }) {
        return "'\(String(bytes: bytes, encoding: .ascii)!)' (\(status))"
    }
    return "\(status)"
}

func die(_ status: OSStatus, _ what: String) {
    if status != noErr {
        log("FATAL \(what): OSStatus \(fourCC(status))")
        exit(1)
    }
}

// MARK: - clock

/// The single timeline both channels are stamped against.
///
/// The mic and the tap are separate Core Audio devices that start at different
/// instants — the tap is running before `AVAudioEngine` has even opened the mic.
/// Counting each channel's samples from its own first frame would give both a
/// timestamp of 0 for audio captured hundreds of milliseconds apart. Anchoring
/// each channel to the Mach host time of its first buffer puts them on one
/// timeline instead.
enum Clock {
    static let epoch = mach_absolute_time()

    private static let scale: Double = {
        var info = mach_timebase_info_data_t()
        mach_timebase_info(&info)
        return Double(info.numer) / Double(info.denom) / 1_000_000_000.0
    }()

    /// Seconds from capture start to `hostTime`; clamped at 0 for buffers
    /// stamped fractionally before the epoch was read.
    static func seconds(since hostTime: UInt64) -> Double {
        guard hostTime > epoch else { return 0 }
        return Double(hostTime - epoch) * scale
    }

    static func now() -> UInt64 { mach_absolute_time() }
}

// MARK: - frame emitter

/// Serializes frame writes from the mic and tap callbacks onto stdout, and
/// stamps each channel against the shared clock.
final class Emitter: @unchecked Sendable {
    /// Persistent drift past this is corrected (or, when the correction would
    /// need a backward jump, warned about once). Kept under the echo
    /// canceller's 0.5 s reference hold so pairing survives between checks.
    private static let driftLimit = 0.25
    /// How long drift must hold past the limit before the anchor moves. A
    /// single reading can spike — a buffer without a valid host time is
    /// stamped with its arrival time — and a spike must never move the
    /// timeline, because it can never be moved back.
    private static let driftWindowSeconds = 2.0

    /// The extremes of (wall clock − stamped timeline) one channel showed
    /// since its current observation window opened.
    private struct DriftWindow {
        var openedAt: Double
        var minDrift: Double
        var maxDrift: Double
    }

    private let lock = NSLock()
    private let out = FileHandle.standardOutput
    private var emitted: [UInt8: Int] = [:]
    private var anchor: [UInt8: Double] = [:]
    private var floor: [UInt8: Double] = [:]
    private var driftWindows: [UInt8: DriftWindow] = [:]
    private var driftWarned: Set<UInt8> = []

    /// Forget a channel's anchor so its next buffer re-anchors on the shared
    /// clock — used when the channel's device is rebuilt mid-capture (a device
    /// switch). The old stream's end becomes a floor for the new base: the
    /// consumer (SessionStore) rejects backward timestamps, and a device whose
    /// clock ran ahead of wall time could otherwise re-anchor slightly before
    /// the samples it already emitted. The capture gap lands as silence.
    func reanchor(_ channel: ChannelCode) {
        lock.lock()
        defer { lock.unlock() }
        let code = channel.rawValue
        if let base = anchor[code] {
            floor[code] = base + Double(emitted[code, default: 0]) / SAMPLE_RATE
        }
        anchor.removeValue(forKey: code)
        emitted[code] = 0
        driftWindows.removeValue(forKey: code)
        driftWarned.remove(code)
    }

    /// Append one frame of mono 16 kHz int16 samples for `channel`. `hostTime`
    /// is when the *input* buffer behind these samples was captured; it anchors
    /// the channel on first use, after which sample counting carries the
    /// timeline (monotonic, and sample-accurate within the channel) — with the
    /// anchor nudged forward when the device's delivery falls behind wall
    /// clock (`correctDrift`).
    func emit(_ channel: ChannelCode, _ samples: UnsafeBufferPointer<Int16>, hostTime: UInt64) {
        lock.lock()
        defer { lock.unlock() }
        let code = channel.rawValue
        let priorSamples = emitted[code, default: 0]
        let base: Double
        if let existing = anchor[code] {
            base = existing
        } else {
            base = max(Clock.seconds(since: hostTime), floor[code, default: 0])
            anchor[code] = base
        }
        var timestamp = base + Double(priorSamples) / SAMPLE_RATE
        emitted[code] = priorSamples + samples.count

        if priorSamples > 0 {
            timestamp += correctDrift(code, base: base, timestamp: timestamp, hostTime: hostTime)
        }

        var header = Data(capacity: 13)
        header.append(code)
        withUnsafeBytes(of: timestamp.bitPattern.littleEndian) { header.append(contentsOf: $0) }
        withUnsafeBytes(of: UInt32(samples.count).littleEndian) { header.append(contentsOf: $0) }
        var payload = Data(count: samples.count * 2)
        payload.withUnsafeMutableBytes { raw in
            let dst = raw.bindMemory(to: Int16.self)
            for i in 0..<samples.count { dst[i] = samples[i].littleEndian }
        }
        out.write(header)
        out.write(payload)
    }

    /// Keep a channel's sample-counted timeline from walking away from wall
    /// clock. Returns the seconds to add to the current frame's timestamp
    /// (0 when nothing needs correcting); the anchor is moved by the same
    /// amount so subsequent frames stay on the corrected timeline.
    ///
    /// A device that drops buffers, or runs slower than its nominal rate
    /// (Bluetooth outputs do both), delivers fewer samples than wall time
    /// elapses, so the stamped timeline falls further and further behind.
    /// Left alone, the lag crosses the echo canceller's 0.5 s reference hold
    /// and the canceller pairs nothing for the rest of the meeting — observed
    /// 2026-07-24: a headphone session drifted 380 ms and ran its entire
    /// 102 min without a usable reference while remote transcription was fine.
    /// The fix is a forward anchor shift; the jump lands downstream as a
    /// silence-padded gap, exactly like a device-rebuild re-anchor.
    ///
    /// Correct by the observation window's *minimum* drift: only lag that
    /// persisted through every reading in the window is real under-delivery,
    /// and over-shifting can never be undone (the consumer rejects backward
    /// timestamps). A device running *fast* — persistently negative drift —
    /// would need that forbidden backward shift, so it stays a once-only
    /// warning as before.
    private func correctDrift(
        _ code: UInt8, base: Double, timestamp: Double, hostTime: UInt64
    ) -> Double {  // holding lock
        let wall = Clock.seconds(since: hostTime)
        let drift = wall - timestamp
        guard var window = driftWindows[code] else {
            driftWindows[code] = DriftWindow(openedAt: wall, minDrift: drift, maxDrift: drift)
            return 0
        }
        window.minDrift = min(window.minDrift, drift)
        window.maxDrift = max(window.maxDrift, drift)
        if wall - window.openedAt < Self.driftWindowSeconds {
            driftWindows[code] = window
            return 0
        }
        let shift = window.minDrift > Self.driftLimit ? window.minDrift : 0
        driftWindows[code] = DriftWindow(
            openedAt: wall, minDrift: drift - shift, maxDrift: drift - shift)
        if shift > 0 {
            anchor[code] = base + shift
            log("channel \(code) re-anchored +\(Int(shift * 1000)) ms "
                + "— device delivered fewer samples than wall clock; gap padded")
            return shift
        }
        if window.maxDrift < -Self.driftLimit, !driftWarned.contains(code) {
            driftWarned.insert(code)
            log("WARNING channel \(code) drifted \(Int(window.maxDrift * 1000)) ms "
                + "ahead of wall clock (device delivering too many samples)")
        }
        return 0
    }
}

// MARK: - resampler

/// Wraps an AVAudioConverter that renders arbitrary input into mono 16 kHz
/// int16, emitting each converted block as a frame. One per capture channel.
final class Resampler {
    private let converter: AVAudioConverter
    private let target: AVAudioFormat
    private let channel: ChannelCode
    private let emitter: Emitter

    init?(source: AVAudioFormat, channel: ChannelCode, emitter: Emitter) {
        guard let target = AVAudioFormat(
            commonFormat: .pcmFormatInt16, sampleRate: SAMPLE_RATE,
            channels: 1, interleaved: true),
            let converter = AVAudioConverter(from: source, to: target)
        else { return nil }
        self.target = target
        self.converter = converter
        self.channel = channel
        self.emitter = emitter
    }

    func feed(_ input: AVAudioPCMBuffer, hostTime: UInt64) {
        let ratio = SAMPLE_RATE / input.format.sampleRate
        let capacity = AVAudioFrameCount(Double(input.frameLength) * ratio) + 16
        guard capacity > 0,
              let output = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity)
        else { return }

        var supplied = false
        var error: NSError?
        let status = converter.convert(to: output, error: &error) { _, outStatus in
            if supplied {
                outStatus.pointee = .noDataNow
                return nil
            }
            supplied = true
            outStatus.pointee = .haveData
            return input
        }
        if status == .error {
            log("convert failed: \(error?.localizedDescription ?? "unknown")")
            return
        }
        guard output.frameLength > 0, let data = output.int16ChannelData else { return }
        emitter.emit(channel, UnsafeBufferPointer(start: data[0], count: Int(output.frameLength)),
                     hostTime: hostTime)
    }
}

// MARK: - input devices

/// One microphone this machine could record from.
///
/// `uid` is what gets stored and re-resolved: `AudioObjectID`s are handles that
/// change across a re-plug (and across a reboot), while
/// `kAudioDevicePropertyDeviceUID` survives both.
struct InputDevice {
    let deviceID: AudioObjectID
    let uid: String
    let name: String
    let isDefault: Bool
}

func address(_ selector: AudioObjectPropertySelector,
             scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal)
    -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress(mSelector: selector, mScope: scope,
                               mElement: kAudioObjectPropertyElementMain)
}

/// Every audio device the system knows, input and output alike.
func allDevices() -> [AudioObjectID] {
    var addr = address(kAudioHardwarePropertyDevices)
    var size: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject),
                                         &addr, 0, nil, &size) == noErr else { return [] }
    var ids = [AudioObjectID](repeating: 0, count: Int(size) / MemoryLayout<AudioObjectID>.size)
    guard !ids.isEmpty,
          AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject),
                                     &addr, 0, nil, &size, &ids) == noErr else { return [] }
    return ids
}

/// A CFString-valued device property, read the way Core Audio hands it over.
///
/// These properties return a *retained* string the caller owns, so the buffer
/// is an `Unmanaged` pointer and the reference is consumed here. Reading into a
/// plain `CFString` variable instead — what this file used to do at its two
/// call sites — writes past ARC's back; that is a leak per call, and the
/// picker calls it once per device on every enumeration.
func deviceString(_ id: AudioObjectID, _ selector: AudioObjectPropertySelector) -> String? {
    var addr = address(selector)
    var value: Unmanaged<CFString>?
    var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    guard AudioObjectGetPropertyData(id, &addr, 0, nil, &size, &value) == noErr,
          let string = value else { return nil }
    return string.takeRetainedValue() as String
}

func deviceUID(_ id: AudioObjectID) -> String? {
    deviceString(id, kAudioDevicePropertyDeviceUID)
}

func deviceName(_ id: AudioObjectID) -> String? {
    deviceString(id, kAudioObjectPropertyName)
}

/// How many input channels a device offers — 0 for a pure output device.
///
/// The stream *configuration*, not the device's name or transport: an
/// aggregate, a virtual driver and a USB interface all look alike from
/// outside, and this is the only thing that says which ones a microphone
/// channel could actually open.
func inputChannelCount(_ id: AudioObjectID) -> Int {
    var addr = address(kAudioDevicePropertyStreamConfiguration,
                       scope: kAudioObjectPropertyScopeInput)
    var size: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(id, &addr, 0, nil, &size) == noErr, size > 0 else {
        return 0
    }
    let raw = UnsafeMutableRawPointer.allocate(
        byteCount: Int(size), alignment: MemoryLayout<AudioBufferList>.alignment)
    defer { raw.deallocate() }
    guard AudioObjectGetPropertyData(id, &addr, 0, nil, &size, raw) == noErr else { return 0 }
    let list = UnsafeMutableAudioBufferListPointer(
        raw.assumingMemoryBound(to: AudioBufferList.self))
    return list.reduce(0) { $0 + Int($1.mNumberChannels) }
}

func defaultDeviceID(_ selector: AudioObjectPropertySelector) -> AudioObjectID? {
    var addr = address(selector)
    var deviceID = AudioObjectID(kAudioObjectUnknown)
    var size = UInt32(MemoryLayout<AudioObjectID>.size)
    guard AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject),
                                     &addr, 0, nil, &size, &deviceID) == noErr,
          deviceID != AudioObjectID(kAudioObjectUnknown) else { return nil }
    return deviceID
}

/// Every microphone, with the current default marked.
///
/// Enumeration is not TCC-gated (measured 2026-08-12 from a bundle whose
/// authorization status was `notDetermined`: the full list, names included, no
/// prompt), which is what lets the picker and `steno devices` list devices
/// without asking for the microphone.
func inputDevices() -> [InputDevice] {
    let defaultID = defaultDeviceID(kAudioHardwarePropertyDefaultInputDevice)
    var devices: [InputDevice] = []
    for id in allDevices() where inputChannelCount(id) > 0 {
        guard let uid = deviceUID(id) else { continue }
        devices.append(InputDevice(deviceID: id, uid: uid,
                                   name: deviceName(id) ?? uid, isDefault: id == defaultID))
    }
    return devices
}

// MARK: - choosing an input device

/// Why a `--mic-device` value matched nothing usable.
enum Unresolved: Error {
    case missing
    case ambiguous([String])

    func message(_ pin: String) -> String {
        switch self {
        case .missing:
            return "the selected microphone \"\(pin)\" is not available — run `steno devices` "
                + "to list what is connected, or pass --mic-device default"
        case .ambiguous(let uids):
            return "the microphone name \"\(pin)\" matches \(uids.count) connected devices "
                + "(\(uids.joined(separator: ", "))) — pass one of those IDs to --mic-device instead"
        }
    }
}

/// Trimmed and NFC-normalized — the one form both sides of a comparison take.
///
/// `kAudioObjectPropertyName` hands back decomposed text (NFD), so a name typed
/// into a TOML file as precomposed characters would never match an
/// unnormalized comparison; the rule is shared with the Rust helper
/// (`../stenocap/src/device.rs`) and the Python layer.
func normalizeDeviceKey(_ value: String) -> String {
    value.trimmingCharacters(in: .whitespacesAndNewlines).precomposedStringWithCanonicalMapping
}

/// The device a stored selection names: UID first, then the exact name.
///
/// Never a "closest match" and never the default on a miss — silently
/// recording the built-in mic when the user asked for the desk mic is the
/// failure this whole feature exists to prevent, and it stays invisible until
/// the transcript is bad.
func resolveInput(_ devices: [InputDevice], pin: String) -> Result<InputDevice, Unresolved> {
    let wanted = normalizeDeviceKey(pin)
    if let byUID = devices.first(where: { normalizeDeviceKey($0.uid) == wanted }) {
        return .success(byUID)
    }
    let byName = devices.filter { normalizeDeviceKey($0.name) == wanted }
    if byName.count == 1 { return .success(byName[0]) }
    if byName.isEmpty { return .failure(.missing) }
    return .failure(.ambiguous(byName.map { $0.uid }))
}

/// How long a pinned device gets to appear at startup before the run fails.
///
/// A USB interface can enumerate a second or two after wake-from-sleep. Without
/// this the helper would be asymmetric in the worst way: absent at t=0 kills the
/// meeting, vanishing at t=1 s is retried forever (`MicSupervisor`).
let pinGraceSeconds = 3.0

/// [`resolveInput`] retried for [`pinGraceSeconds`] while the device is merely absent.
func resolveInputWithGrace(pin: String) -> Result<InputDevice, Unresolved> {
    let deadline = Date().addingTimeInterval(pinGraceSeconds)
    while true {
        switch resolveInput(inputDevices(), pin: pin) {
        case .success(let device):
            return .success(device)
        case .failure(.missing) where Date() < deadline:
            Thread.sleep(forTimeInterval: 0.25)
        case .failure(let why):
            return .failure(why)
        }
    }
}

// MARK: - system-audio process tap

/// The aggregate that delivers the tap's audio, pinned to one output device.
/// The tap itself outlives rebuilds (it taps processes, not a device); this is
/// what gets torn down and recreated when its pinned device vanishes.
struct AggSession {
    var aggID: AudioObjectID
    var procID: AudioDeviceIOProcID?
    var outputUID: String
    /// The pinned microphone riding in this aggregate, if any (``PinnedMic``).
    var micUID: String?
    /// The silent keep-alive holding the output device awake, if any
    /// (``startKeepAlive``).
    var keepAlive: (deviceID: AudioObjectID, procID: AudioDeviceIOProcID)?
}

/// The default output device, as both the handle and the UID an aggregate
/// needs. Read once and used for both: a keep-alive attached to a device the
/// aggregate is not clocked by would hold the wrong one awake, and the
/// aggregate would then deliver nothing for the whole meeting.
func defaultOutputDevice() -> (deviceID: AudioObjectID, uid: String)? {
    guard let deviceID = defaultDeviceID(kAudioHardwarePropertyDefaultOutputDevice) else {
        log("no default output device")
        return nil
    }
    guard let uid = deviceUID(deviceID) else {
        log("could not read output device UID")
        return nil
    }
    return (deviceID, uid)
}

/// Is a device with this UID still present? The aggregate's pinned output
/// disappearing (AirPods disconnecting mid-meeting) takes the aggregate's
/// clock with it — the trigger for a rebuild onto the new default output.
func deviceExists(uid target: String) -> Bool {
    allDevices().contains { deviceUID($0) == target }
}

/// Tracks a channel's observed buffer layout so a mid-session change is logged
/// once rather than per buffer.
final class ChannelLayoutWatch: @unchecked Sendable {
    private let lock = NSLock()
    private let what: String
    private var channels = 0

    init(what: String) {
        self.what = what
    }

    func note(_ observed: Int) {
        lock.lock()
        defer { lock.unlock() }
        if channels != 0, channels != observed {
            log("WARNING \(what) changed from \(channels) to \(observed) channel(s) "
                + "— the device was renegotiated mid-capture")
        }
        channels = observed
    }
}

/// Create the global process tap — once per run, fatal on failure (startup).
/// Returns the tap and its description UUID (needed to wire it into every
/// aggregate this run builds).
func createSystemTap() -> (tapID: AudioObjectID, tapUUID: String) {
    let desc = CATapDescription(monoGlobalTapButExcludeProcesses: [])
    desc.name = "stenograf-tap"
    desc.muteBehavior = .unmuted
    desc.isPrivate = true

    var tapID = AudioObjectID(kAudioObjectUnknown)
    die(AudioHardwareCreateProcessTap(desc, &tapID), "AudioHardwareCreateProcessTap")
    if tapID == AudioObjectID(kAudioObjectUnknown) {
        // Core Audio reports noErr but hands back no object when the tap can't
        // be created — audio-capture permission missing, or coreaudiod wedged
        // by a concurrent capture app (measured 2026-07-20: racing another
        // mic+tap capture makes this or the aggregate build fail).
        log("FATAL: system tap unavailable — audio-capture permission missing, "
            + "or another app is capturing (OBS, a second stenograf?)")
        exit(1)
    }
    return (tapID, desc.uuid.uuidString)
}

/// Wrap the tap in an aggregate pinned to the current default output and start
/// the IO proc. Non-fatal — nil with the reason logged — so the supervisor can
/// retry a rebuild after a device vanishes; startup turns nil into a FATAL.
///
/// `micUID` rides along when a pinned microphone shares this aggregate (see
/// ``PinnedMic`` for why it must): the device joins as a drift-compensated
/// sub-device, and its audio arrives in the same buffer list as the tap's,
/// ahead of it, on the same clock.
func buildAggregate(tapID: AudioObjectID, tapUUID: String, emitter: Emitter,
                    micUID: String? = nil) -> AggSession? {
    var asbd = AudioStreamBasicDescription()
    var size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
    var addr = AudioObjectPropertyAddress(
        mSelector: kAudioTapPropertyFormat,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)
    guard AudioObjectGetPropertyData(tapID, &addr, 0, nil, &size, &asbd) == noErr else {
        log("could not read tap format — audio-capture permission missing, "
            + "or another app is capturing (OBS, a second stenograf?)")
        return nil
    }
    log("system tap format: \(asbd.mSampleRate) Hz, \(asbd.mChannelsPerFrame) ch")

    guard let output = defaultOutputDevice() else { return nil }
    let outputUID = output.uid
    var subDevices: [[String: Any]] = [[kAudioSubDeviceUIDKey: outputUID]]
    if let micUID {
        // Drift compensation because the microphone runs on its own crystal
        // while the aggregate is clocked by the main sub-device.
        subDevices.append([kAudioSubDeviceUIDKey: micUID, kAudioSubDeviceDriftCompensationKey: 1])
    }
    let aggDesc: [String: Any] = [
        kAudioAggregateDeviceNameKey: "stenograf-agg",
        kAudioAggregateDeviceUIDKey: UUID().uuidString,
        kAudioAggregateDeviceIsPrivateKey: true,
        kAudioAggregateDeviceIsStackedKey: false,
        kAudioAggregateDeviceTapAutoStartKey: true,
        kAudioAggregateDeviceMainSubDeviceKey: outputUID,
        kAudioAggregateDeviceSubDeviceListKey: subDevices,
        kAudioAggregateDeviceTapListKey: [[
            kAudioSubTapDriftCompensationKey: true,
            kAudioSubTapUIDKey: tapUUID,
        ]],
    ]
    var aggID = AudioObjectID(kAudioObjectUnknown)
    let created = AudioHardwareCreateAggregateDevice(aggDesc as CFDictionary, &aggID)
    guard created == noErr else {
        log("create aggregate device failed: OSStatus \(fourCC(created))")
        return nil
    }

    // The IO proc delivers buffers at the *aggregate's* rate — it follows the
    // main sub-device, i.e. the pinned output device — not at the rate the tap
    // advertises. On the built-in speakers both are 48 kHz and the difference is
    // invisible; a Bluetooth output runs at 44.1 kHz, and resampling those
    // buffers as 48 kHz warps the reference by 8 % and walks this channel's
    // sample-counted timestamps off the shared clock (measured: ERLE collapses
    // to ~1 dB and the canceller starves). Trust the device doing the delivering.
    // Read once per aggregate build; the pinned device *vanishing* rebuilds
    // this whole session (TapSupervisor), and a rate mismatch on a
    // still-present pinned device — a Bluetooth device's real clock straying
    // from its nominal rate — is absorbed by the emitter's drift correction.
    var sampleRate = asbd.mSampleRate
    var aggRate = 0.0
    var rateSize = UInt32(MemoryLayout<Double>.size)
    var rateAddr = AudioObjectPropertyAddress(
        mSelector: kAudioDevicePropertyNominalSampleRate,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)
    if AudioObjectGetPropertyData(aggID, &rateAddr, 0, nil, &rateSize, &aggRate) == noErr,
       aggRate > 0, aggRate != sampleRate {
        log("system buffers arrive at \(aggRate) Hz (aggregate rate), not \(sampleRate) Hz")
        sampleRate = aggRate
    }

    // The resampler always sees mono float32 at that rate; renderInputBuffer
    // downmixes whatever channel layout the buffer actually arrives in.
    guard let sourceFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32, sampleRate: sampleRate,
        channels: 1, interleaved: false),
        let resampler = Resampler(source: sourceFormat, channel: .system, emitter: emitter)
    else {
        log("could not build system-audio resampler")
        AudioHardwareDestroyAggregateDevice(aggID)
        return nil
    }
    let layout = ChannelLayoutWatch(what: "system tap")
    let split = SplitWatch()

    // The mic sub-device's buffers arrive in the same list, before the tap's:
    // the aggregate composes its input streams sub-device by sub-device (the
    // output device contributes none) and appends the taps last. Verified on
    // this Mac 2026-08-12 — the tap is one mono buffer, so it is the last one,
    // and everything before it is the microphone.
    var micResampler: Resampler?
    let micLayout = ChannelLayoutWatch(what: "mic device")
    if let micUID {
        guard let built = Resampler(source: sourceFormat, channel: .mic, emitter: emitter) else {
            log("could not build the microphone resampler")
            AudioHardwareDestroyAggregateDevice(aggID)
            return nil
        }
        micResampler = built
        // The composition is checked, not assumed: the split below trusts that
        // the microphone's streams precede the tap's single mono one, so an
        // aggregate that did not take the device (or a tap that grew a channel)
        // must be loud here rather than quietly swap the two channels.
        let wanted = inputDevices().first { $0.uid == micUID }.map { inputChannelCount($0.deviceID) }
        let got = inputChannelCount(aggID)
        if let wanted, got != wanted + 1 {
            log("WARNING the capture group carries \(got) input channel(s), not the "
                + "\(wanted + 1) expected of this microphone plus the system tap "
                + "— the microphone may be missing from the recording")
        }
    }

    var procID: AudioDeviceIOProcID?
    let queue = DispatchQueue(label: "dev.stenograf.tap")
    let procStatus = AudioDeviceCreateIOProcIDWithBlock(&procID, aggID, queue) { _, inInputData, inInputTime, _, _ in
        let stamp = inInputTime.pointee
        let hostTime = stamp.mFlags.contains(.hostTimeValid) ? stamp.mHostTime : Clock.now()
        let list = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: inInputData))
        guard let micResampler else {
            renderInputBuffer(Array(list), hostTime: hostTime, sourceFormat: sourceFormat,
                              resampler: resampler, layout: layout)
            return
        }
        // Split: everything but the last buffer is the microphone, the last is
        // the tap. A list holding only the tap means the device left the group
        // without leaving the system's device list, which nothing watches —
        // hence the warning; the channel stops rather than carrying tap audio.
        split.note(list)
        if list.count > 1 {
            renderInputBuffer(Array(list.dropLast()), hostTime: hostTime,
                              sourceFormat: sourceFormat, resampler: micResampler,
                              layout: micLayout)
        }
        renderInputBuffer(Array(list.suffix(1)), hostTime: hostTime, sourceFormat: sourceFormat,
                          resampler: resampler, layout: layout)
    }
    guard procStatus == noErr else {
        log("create tap IO proc failed: OSStatus \(fourCC(procStatus))")
        AudioHardwareDestroyAggregateDevice(aggID)
        return nil
    }
    let startStatus = AudioDeviceStart(aggID, procID)
    guard startStatus == noErr else {
        log("start aggregate device failed: OSStatus \(fourCC(startStatus))")
        if let procID { AudioDeviceDestroyIOProcID(aggID, procID) }
        AudioHardwareDestroyAggregateDevice(aggID)
        return nil
    }
    log("system capture started")
    var keepAlive: (deviceID: AudioObjectID, procID: AudioDeviceIOProcID)?
    if micUID != nil, let keepProc = startKeepAlive(output.deviceID) {
        keepAlive = (output.deviceID, keepProc)
    }
    if let micUID {
        log("mic capture started (device \(micUID), through the system aggregate)")
    }
    return AggSession(aggID: aggID, procID: procID, outputUID: outputUID, micUID: micUID,
                      keepAlive: keepAlive)
}

/// Downmix one IO-proc buffer to mono and hand it to the resampler.
///
/// The frame count is derived from the buffer we were handed, never from the
/// format read at startup: Core Audio renegotiates the tap when the output
/// device changes (headphones, AirPods, a display with speakers), and reading a
/// multi-channel buffer as mono would emit several times too many samples.
func renderInputBuffer(_ list: [AudioBuffer], hostTime: UInt64,
                       sourceFormat: AVAudioFormat, resampler: Resampler,
                       layout: ChannelLayoutWatch) {
    guard let first = list.first, first.mData != nil else { return }

    let planes = list.count
    let perPlane = Int(first.mNumberChannels)
    guard perPlane > 0 else { return }
    layout.note(planes * perPlane)

    let bytesPerFrame = MemoryLayout<Float>.size * perPlane
    let frames = Int(first.mDataByteSize) / bytesPerFrame
    guard frames > 0,
          let mono = AVAudioPCMBuffer(pcmFormat: sourceFormat,
                                      frameCapacity: AVAudioFrameCount(frames)),
          let dst = mono.floatChannelData
    else { return }
    mono.frameLength = AVAudioFrameCount(frames)
    downmix(list, into: dst[0], frames: frames)
    resampler.feed(mono, hostTime: hostTime)
}

/// Average every channel into one, for either buffer layout Core Audio uses:
/// one plane per channel (deinterleaved), or one plane of interleaved frames.
func downmix(_ list: [AudioBuffer],
             into dst: UnsafeMutablePointer<Float>, frames: Int) {
    if list.count > 1 {
        for i in 0..<frames { dst[i] = 0 }
        for plane in 0..<list.count {
            guard let src = list[plane].mData?.assumingMemoryBound(to: Float.self) else { continue }
            for i in 0..<frames { dst[i] += src[i] }
        }
        let scale = 1.0 / Float(list.count)
        for i in 0..<frames { dst[i] *= scale }
        return
    }
    guard let src = list[0].mData?.assumingMemoryBound(to: Float.self) else { return }
    let channels = Int(list[0].mNumberChannels)
    if channels <= 1 {
        dst.update(from: src, count: frames)
        return
    }
    let scale = 1.0 / Float(channels)
    for i in 0..<frames {
        var sum: Float = 0
        for c in 0..<channels { sum += src[i * channels + c] }
        dst[i] = sum * scale
    }
}

/// Render digital silence into `deviceID` for as long as capture runs.
///
/// An aggregate that contains a microphone delivers **nothing at all** until
/// something renders on its output device — measured 2026-08-12: with no audio
/// playing, not one buffer arrived in 25 s, and speech at second 10 started the
/// flow at second 12. The mic channel of a meeting cannot depend on the remote
/// side making noise, so the output device is held awake here instead. The
/// buffers written are zeros: nothing is audible, and the tap has nothing of
/// ours to capture. Only the aggregate-hosted microphone needs this; the
/// shipped default path never starts one.
func startKeepAlive(_ deviceID: AudioObjectID) -> AudioDeviceIOProcID? {
    var procID: AudioDeviceIOProcID?
    let queue = DispatchQueue(label: "dev.stenograf.keepalive")
    let created = AudioDeviceCreateIOProcIDWithBlock(&procID, deviceID, queue) { _, _, _, out, _ in
        for buffer in UnsafeMutableAudioBufferListPointer(out) {
            if let data = buffer.mData { memset(data, 0, Int(buffer.mDataByteSize)) }
        }
    }
    guard created == noErr, let procID else {
        log("could not hold the output device awake (OSStatus \(fourCC(created)))")
        return nil
    }
    let started = AudioDeviceStart(deviceID, procID)
    guard started == noErr else {
        log("could not hold the output device awake (OSStatus \(fourCC(started)))")
        AudioDeviceDestroyIOProcID(deviceID, procID)
        return nil
    }
    return procID
}

func tearDownAggregate(_ session: AggSession) {
    if let keepAlive = session.keepAlive {
        AudioDeviceStop(keepAlive.deviceID, keepAlive.procID)
        AudioDeviceDestroyIOProcID(keepAlive.deviceID, keepAlive.procID)
    }
    if let procID = session.procID {
        AudioDeviceStop(session.aggID, procID)
        AudioDeviceDestroyIOProcID(session.aggID, procID)
    }
    AudioHardwareDestroyAggregateDevice(session.aggID)
}

/// Keeps the system channel alive across its pinned device vanishing.
///
/// A default-output *switch* needs nothing: the global tap hears process audio
/// wherever it is routed (measured 2026-07-20 — content kept flowing with the
/// aggregate still pinned to the old device), so the working path is left
/// untouched. But the pinned device *disappearing* (AirPods disconnect, a
/// display unplugged) takes the aggregate's clock with it; this watches the
/// device list and rebuilds the aggregate around the surviving default output,
/// re-anchoring the channel on the shared clock (the gap lands as silence).
///
/// When a pinned microphone rides in the same aggregate (``PinnedMic``), that
/// device's coming and going is watched here too: the aggregate is rebuilt with
/// the mic when it is present and without it when it is not, so its
/// disappearance costs the mic channel rather than the whole meeting, and its
/// return picks capture back up. The pin is never resolved to a *different*
/// microphone.
final class TapSupervisor: @unchecked Sendable {
    private let emitter: Emitter
    private let tapID: AudioObjectID
    private let tapUUID: String
    private let micPin: String?
    private let queue = DispatchQueue(label: "dev.stenograf.tap-rebuild")
    private var agg: AggSession?
    private var pending: DispatchWorkItem?
    private var lost = false
    private var micLost = false
    private var stopped = false

    init(emitter: Emitter, tapID: AudioObjectID, tapUUID: String, agg: AggSession,
         micPin: String? = nil) {
        self.emitter = emitter
        self.tapID = tapID
        self.tapUUID = tapUUID
        self.micPin = micPin
        self.agg = agg
        var addr = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        AudioObjectAddPropertyListenerBlock(
            AudioObjectID(kAudioObjectSystemObject), &addr, queue
        ) { [weak self] _, _ in self?.schedule() }
    }

    /// Debounce: plugging or unplugging a device fires several list changes.
    private func schedule() {  // on queue
        pending?.cancel()
        let work = DispatchWorkItem { [weak self] in self?.rebuildIfVanished() }
        pending = work
        queue.asyncAfter(deadline: .now() + 0.5, execute: work)
    }

    private func rebuildIfVanished() {  // on queue
        guard !stopped else { return }
        if let agg, deviceExists(uid: agg.outputUID), agg.micUID == currentMicUID() {
            return  // nothing we depend on moved — leave the working path alone
        }
        rebuild()
    }

    /// The pinned microphone's UID if it is connected right now, else nil.
    private func currentMicUID() -> String? {
        guard let micPin else { return nil }
        return try? resolveInput(inputDevices(), pin: micPin).get().uid
    }

    private func rebuild() {  // on queue
        guard !stopped else { return }
        if let old = agg {
            tearDownAggregate(old)
            agg = nil
        }
        emitter.reanchor(.system)
        let micUID = currentMicUID()
        if micPin != nil {
            emitter.reanchor(.mic)
            if micUID == nil, !micLost {
                micLost = true
                log("WARNING mic capture lost — the selected microphone \"\(micPin!)\" is not "
                    + "connected; the meeting keeps recording system audio until it returns")
            } else if micUID != nil, micLost {
                micLost = false
            }
        }
        if let fresh = buildAggregate(tapID: tapID, tapUUID: tapUUID, emitter: emitter,
                                      micUID: micUID) {
            agg = fresh
            log("system capture moved to output device \(fresh.outputUID)")
            lost = false
        } else {
            if !lost {
                lost = true
                log("WARNING system capture lost (output device vanished) — retrying")
            }
            queue.asyncAfter(deadline: .now() + 2.0) { [weak self] in self?.rebuild() }
        }
    }

    func stop() {
        queue.sync {
            stopped = true
            pending?.cancel()
            if let agg { tearDownAggregate(agg) }
            agg = nil
            AudioHardwareDestroyProcessTap(tapID)
        }
    }
}

// MARK: - microphone

/// Prompt for microphone access. Called before any capture starts: the prompt
/// blocks, and doing it after the tap is running would skew the two channels'
/// start by however long the user takes to answer.
func requestMicrophoneAccess() {
    let sem = DispatchSemaphore(value: 0)
    var granted = false
    AVCaptureDevice.requestAccess(for: .audio) { granted = $0; sem.signal() }
    sem.wait()
    guard granted else {
        log("FATAL: microphone permission denied")
        exit(1)
    }
}

/// Checks, once, that the capture group's buffers are laid out as the split
/// below assumes: the microphone's streams first, the tap's single mono stream
/// last.
///
/// Two ways that can be wrong, both silent. The group may deliver the tap
/// alone, which is a meeting that records the far end perfectly and the room
/// not at all. Or the streams may not be in that order on some other macOS
/// build, which would feed the *remote* audio into the microphone channel —
/// the echo canceller then has the same audio on both sides and the transcript
/// attributes the far end to the room. The tap is created mono, so a last
/// buffer that is not mono is the tell.
final class SplitWatch: @unchecked Sendable {
    private let lock = NSLock()
    private var warned = false

    func note(_ list: UnsafeMutableAudioBufferListPointer) {
        lock.lock()
        defer { lock.unlock() }
        guard !warned else { return }
        if list.count < 2 {
            warned = true
            log("WARNING no microphone audio in the capture group — the selected device "
                + "left it; the meeting keeps recording system audio")
        } else if list.last?.mNumberChannels != 1 {
            warned = true
            log("WARNING the capture group's last stream carries "
                + "\(list.last?.mNumberChannels ?? 0) channels, not the system tap's one "
                + "— the microphone and system channels may be swapped")
        }
    }
}

/// Warns once when a device keeps handing over buffers with no valid host time.
///
/// Such a buffer is stamped with its *arrival* instead — the very thing the
/// helper transport exists to eliminate, because the echo canceller pairs the
/// two channels by timestamp. Pinning is what makes virtual and aggregate
/// devices (BlackHole, Loopback) selectable, and those are where invalid host
/// times live. A single invalid stamp is noise the emitter's drift window
/// absorbs; a device that never stamps is a different situation, so the warning
/// waits for the third one.
final class HostTimeWatch: @unchecked Sendable {
    private let lock = NSLock()
    private let device: String
    private var invalid = 0
    private var warned = false

    init(device: String) {
        self.device = device
    }

    func note(valid: Bool) {
        guard !valid else { return }
        lock.lock()
        defer { lock.unlock() }
        invalid += 1
        guard invalid >= 3, !warned else { return }
        warned = true
        log("WARNING mic device \(device) reports no capture time — its audio is stamped "
            + "on arrival, which the echo canceller cannot pair reliably "
            + "(virtual and aggregate devices do this)")
    }
}

/// Build and start a mic engine on the current default input. Non-fatal — nil
/// with the reason logged — so the supervisor can retry after a device change;
/// startup turns nil into a FATAL.
func buildMicEngine(emitter: Emitter) -> AVAudioEngine? {
    let engine = AVAudioEngine()
    let input = engine.inputNode
    let format = input.inputFormat(forBus: 0)
    guard format.sampleRate > 0, format.channelCount > 0,
          let resampler = Resampler(source: format, channel: .mic, emitter: emitter)
    else {
        log("mic unavailable: no usable input format "
            + "(\(format.sampleRate) Hz, \(format.channelCount) ch)")
        return nil
    }
    let name = deviceName(defaultDeviceID(kAudioHardwarePropertyDefaultInputDevice)
        ?? AudioObjectID(kAudioObjectUnknown)) ?? "the default input"
    log("mic format: \(format.sampleRate) Hz, \(format.channelCount) ch — \(name)")
    let hostTimes = HostTimeWatch(device: name)
    input.installTap(onBus: 0, bufferSize: 4096, format: format) { buffer, when in
        hostTimes.note(valid: when.isHostTimeValid)
        let hostTime = when.isHostTimeValid ? when.hostTime : Clock.now()
        resampler.feed(buffer, hostTime: hostTime)
    }
    do {
        try engine.start()
    } catch {
        log("mic engine start: \(error.localizedDescription)")
        return nil
    }
    log("mic capture started")
    return engine
}

func startMic(emitter: Emitter) -> AVAudioEngine {
    guard let engine = buildMicEngine(emitter: emitter) else {
        log("FATAL: mic capture failed to start "
            + "— is another app capturing? (OBS, a second stenograf)")
        exit(1)
    }
    return engine
}

// MARK: - the pinned microphone

/// One microphone captured straight from its device, for a `--mic-device` run.
///
/// **Why not AVAudioEngine.** Retargeting the engine's input node is a
/// documented dead end, and measured here on macOS 26.5.1 (2026-08-12) it is
/// worse than useless while the system-audio tap runs: setting
/// `kAudioOutputUnitProperty_CurrentDevice` to the built-in microphone with the
/// tap's aggregate live wedges `engine.start()` inside the HAL and it never
/// returns (a sample shows the main thread waiting on the shared client IO
/// thread's mutex), and starting the tap *after* a pinned engine instead leaves
/// the tap delivering not one buffer. Pinning a virtual device happened to
/// work, which is exactly the kind of luck a meeting recorder cannot ship on.
/// The engine also announces its own retargeting as a configuration change, so
/// the supervisor rebuilt itself in a loop — nine rebuilds in ten seconds.
///
/// So a pinned mic is taken the way the system channel already is: an IO proc
/// on the device, which coexists with the tap because that is what the tap
/// itself uses. The unpinned path keeps AVAudioEngine unchanged — its
/// default-following and its rebuild-on-AirPods behaviour are measured and
/// shipped, and nothing here touches them.
final class PinnedMic {
    let device: InputDevice
    private let procID: AudioDeviceIOProcID
    private var stopped = false

    private init(device: InputDevice, procID: AudioDeviceIOProcID) {
        self.device = device
        self.procID = procID
    }

    /// Open and start the device. Non-fatal — nil with the reason logged — so
    /// the supervisor can retry after a re-plug.
    static func start(device: InputDevice, emitter: Emitter) -> PinnedMic? {
        // The device's own rate, not a format read at startup: buffers arrive
        // at whatever the device is clocked to, and resampling 44.1 kHz audio
        // as if it were 48 kHz warps the timeline by 8 % (measured on the
        // system channel, `buildAggregate`).
        var rate = 0.0
        var rateSize = UInt32(MemoryLayout<Double>.size)
        var rateAddr = address(kAudioDevicePropertyNominalSampleRate)
        guard AudioObjectGetPropertyData(device.deviceID, &rateAddr, 0, nil, &rateSize, &rate)
            == noErr, rate > 0 else {
            log("mic unavailable: \(device.name) did not report a sample rate")
            return nil
        }
        guard let sourceFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: rate, channels: 1, interleaved: false),
            let resampler = Resampler(source: sourceFormat, channel: .mic, emitter: emitter)
        else {
            log("mic unavailable: could not build a resampler for \(device.name) at \(rate) Hz")
            return nil
        }
        let layout = ChannelLayoutWatch(what: "mic device \(device.name)")
        let hostTimes = HostTimeWatch(device: device.name)
        var procID: AudioDeviceIOProcID?
        let queue = DispatchQueue(label: "dev.stenograf.mic")
        let created = AudioDeviceCreateIOProcIDWithBlock(
            &procID, device.deviceID, queue
        ) { _, inInputData, inInputTime, _, _ in
            let stamp = inInputTime.pointee
            let valid = stamp.mFlags.contains(.hostTimeValid)
            hostTimes.note(valid: valid)
            let list = UnsafeMutableAudioBufferListPointer(
                UnsafeMutablePointer(mutating: inInputData))
            renderInputBuffer(Array(list), hostTime: valid ? stamp.mHostTime : Clock.now(),
                              sourceFormat: sourceFormat, resampler: resampler, layout: layout)
        }
        guard created == noErr, let procID else {
            log("mic unavailable: could not open \(device.name) (OSStatus \(fourCC(created)))")
            return nil
        }
        let started = AudioDeviceStart(device.deviceID, procID)
        guard started == noErr else {
            log("mic unavailable: could not start \(device.name) (OSStatus \(fourCC(started)))")
            AudioDeviceDestroyIOProcID(device.deviceID, procID)
            return nil
        }
        log("mic format: \(rate) Hz — \(device.name)")
        log("mic capture started")
        return PinnedMic(device: device, procID: procID)
    }

    func stop() {
        guard !stopped else { return }
        stopped = true
        AudioDeviceStop(device.deviceID, procID)
        AudioDeviceDestroyIOProcID(device.deviceID, procID)
    }
}

/// Keeps a pinned mic on *its* device, and on no other.
///
/// The device list is watched rather than the default input: a pinned run must
/// ignore a default change by definition, and what it must react to is its own
/// device disappearing and coming back (a re-plug hands out a new
/// `AudioObjectID` under the same UID, so every restart re-resolves). The
/// device's sample rate is watched too, because the resampler is built for one
/// rate and a device that renegotiates would otherwise warp the timeline
/// silently. While the device is away the channel simply stops — never falls
/// back to another microphone.
final class PinnedMicSupervisor: @unchecked Sendable {
    private let emitter: Emitter
    private let pin: String
    private let queue = DispatchQueue(label: "dev.stenograf.mic-rebuild")
    private var mic: PinnedMic?
    private var pending: DispatchWorkItem?
    private var retrying: DispatchWorkItem?
    private var rateWatch: (deviceID: AudioObjectID, block: AudioObjectPropertyListenerBlock)?
    private var lost = false
    private var stopped = false

    init(emitter: Emitter, pin: String, mic: PinnedMic) {
        self.emitter = emitter
        self.pin = pin
        self.mic = mic
        watchRate(mic.device.deviceID)
        var addr = address(kAudioHardwarePropertyDevices)
        AudioObjectAddPropertyListenerBlock(
            AudioObjectID(kAudioObjectSystemObject), &addr, queue
        ) { [weak self] _, _ in self?.schedule("the device list changed") }
    }

    /// Watch the current device's sample rate, and only it.
    ///
    /// The resampler is built for one rate, so a device that renegotiates must
    /// rebuild. The old listener is removed first: a re-plugged device is a new
    /// `AudioObjectID`, and leaving each generation's listener registered would
    /// pile up one rebuild trigger per unplug for the rest of the meeting.
    private func watchRate(_ deviceID: AudioObjectID) {  // on queue (or init)
        unwatchRate()
        var addr = address(kAudioDevicePropertyNominalSampleRate)
        let block: AudioObjectPropertyListenerBlock = { [weak self] _, _ in
            self?.schedule("the microphone changed its sample rate")
        }
        AudioObjectAddPropertyListenerBlock(deviceID, &addr, queue, block)
        rateWatch = (deviceID, block)
    }

    private func unwatchRate() {  // on queue (or init)
        guard let watch = rateWatch else { return }
        var addr = address(kAudioDevicePropertyNominalSampleRate)
        AudioObjectRemovePropertyListenerBlock(watch.deviceID, &addr, queue, watch.block)
        rateWatch = nil
    }

    /// Debounce: one plug event fires several list changes back-to-back.
    private func schedule(_ reason: String) {  // on queue
        pending?.cancel()
        let work = DispatchWorkItem { [weak self] in self?.rebuildIfNeeded(reason) }
        pending = work
        queue.asyncAfter(deadline: .now() + 0.5, execute: work)
    }

    private func rebuildIfNeeded(_ reason: String) {  // on queue
        guard !stopped else { return }
        let resolved = try? resolveInput(inputDevices(), pin: pin).get()
        // Still the same device, still running: a list change that did not
        // touch us (some other device appeared) must leave capture alone.
        if let mic, let resolved, resolved.deviceID == mic.device.deviceID, !lost { return }
        rebuild(reason, resolved: resolved)
    }

    private func rebuild(_ reason: String, resolved: InputDevice?) {  // on queue
        guard !stopped else { return }
        mic?.stop()
        mic = nil
        emitter.reanchor(.mic)
        guard let resolved, let fresh = PinnedMic.start(device: resolved, emitter: emitter) else {
            if !lost {
                lost = true
                log("WARNING mic capture lost (\(reason)) — the selected microphone "
                    + "\"\(pin)\" is not available; retrying until it returns")
            }
            // One chain only: every device-list change while the mic is away
            // would otherwise start its own, and they would all fire together
            // when it returns — each one stopping and restarting a microphone
            // that was already back, with a silence-padded gap apiece.
            retrying?.cancel()
            let again = DispatchWorkItem { [weak self] in
                guard let self else { return }
                self.rebuild(reason, resolved: try? resolveInput(inputDevices(), pin: self.pin).get())
            }
            retrying = again
            queue.asyncAfter(deadline: .now() + 2.0, execute: again)
            return
        }
        retrying?.cancel()
        if rateWatch?.deviceID != fresh.device.deviceID {
            watchRate(fresh.device.deviceID)
        }
        mic = fresh
        lost = false
        log("mic capture restarted (\(reason))")
    }

    func stop() {
        queue.sync {
            stopped = true
            pending?.cancel()
            retrying?.cancel()
            unwatchRate()
            mic?.stop()
            mic = nil
        }
    }
}

/// Keeps the *default* mic alive across device changes (the unpinned path).
///
/// A default-*output* switch (connecting AirPods!) permanently stops a running
/// AVAudioEngine — measured 2026-07-20: the mic goes silent mid-meeting with
/// no error anywhere. And a default-*input* switch should move capture to the
/// new mic the way every other app does, instead of clinging to the old one.
/// Both funnel into one debounced rebuild: stop the old engine, re-anchor the
/// channel on the shared clock (the gap lands as silence), start fresh on the
/// current default input, and retry until a device comes back.
///
/// A pinned run uses [`PinnedMicSupervisor`] instead and never gets here.
final class MicSupervisor: @unchecked Sendable {
    private let emitter: Emitter
    private let queue = DispatchQueue(label: "dev.stenograf.mic-rebuild")
    private var engine: AVAudioEngine?
    private var observer: NSObjectProtocol?
    private var pending: DispatchWorkItem?
    private var lost = false
    private var stopped = false

    init(emitter: Emitter, engine: AVAudioEngine) {
        self.emitter = emitter
        self.engine = engine
        watch(engine)
        var addr = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        AudioObjectAddPropertyListenerBlock(
            AudioObjectID(kAudioObjectSystemObject), &addr, queue
        ) { [weak self] _, _ in self?.schedule("default input changed") }
    }

    private func watch(_ engine: AVAudioEngine) {
        observer = NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange, object: engine, queue: nil
        ) { [weak self] _ in
            guard let self else { return }
            self.queue.async { self.schedule("engine configuration changed") }
        }
    }

    /// Debounce: one device switch fires several change events back-to-back.
    private func schedule(_ reason: String) {  // on queue
        pending?.cancel()
        let work = DispatchWorkItem { [weak self] in self?.rebuild(reason) }
        pending = work
        queue.asyncAfter(deadline: .now() + 0.3, execute: work)
    }

    private func rebuild(_ reason: String) {  // on queue
        guard !stopped else { return }
        if let old = engine {
            if let observer { NotificationCenter.default.removeObserver(observer) }
            old.inputNode.removeTap(onBus: 0)
            old.stop()
            engine = nil
        }
        emitter.reanchor(.mic)
        if let fresh = buildMicEngine(emitter: emitter) {
            engine = fresh
            watch(fresh)
            log("mic capture restarted (\(reason))")
            lost = false
        } else {
            if !lost {
                lost = true
                log("WARNING mic capture lost (\(reason)) — retrying until an input device returns")
            }
            queue.asyncAfter(deadline: .now() + 2.0) { [weak self] in self?.rebuild(reason) }
        }
    }

    func stop() {
        queue.sync {
            stopped = true
            pending?.cancel()
            if let observer { NotificationCenter.default.removeObserver(observer) }
            engine?.stop()
            engine = nil
        }
    }
}

// MARK: - main

let usage = "usage: stenocap [--mic] [--system] [--mic-device ID]"
    + " | --devices [--mic-device ID] | --list-inputs"

/// Everything argv can say, and nothing it cannot.
///
/// Scanning for known flags and ignoring the rest is how a new caller passing
/// `--mic-device X` to a stale binary would record the *default* microphone
/// while the UI said otherwise — and it cannot express a value-taking flag at
/// all, since `--mic-device --mic` would swallow the channel.
struct Options {
    var mic = false
    var system = false
    var devices = false
    var listInputs = false
    var help = false
    var micDevice: String?
}

/// What a caller got wrong about argv, phrased for the usage line below it.
struct UsageError: Error {
    let message: String
}

func parseArguments(_ args: [String]) -> Result<Options, UsageError> {
    var options = Options()
    var index = 0
    while index < args.count {
        switch args[index] {
        case "--mic": options.mic = true
        case "--system": options.system = true
        case "--devices": options.devices = true
        case "--list-inputs": options.listInputs = true
        case "-h", "--help": options.help = true
        case "--mic-device":
            guard index + 1 < args.count else {
                return .failure(UsageError(message: "--mic-device needs a device id or name"))
            }
            let value = args[index + 1]
            if value.hasPrefix("--") {
                return .failure(UsageError(
                    message: "--mic-device needs a device id or name, not the flag \(value)"))
            }
            // `default` is the word every failure message offers as the way
            // back to the OS default, so the binary that prints it accepts it.
            options.micDevice = normalizeDeviceKey(value) == "default" ? nil : value
            index += 1
        case let other:
            return .failure(UsageError(message: "unknown argument \(other)"))
        }
        index += 1
    }
    return .success(options)
}

/// One line of JSON on stdout, escaped by Foundation rather than by hand:
/// device names come from the driver and are not ours to trust.
func printJSON(_ value: Any) {
    guard let data = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]),
          let text = String(data: data, encoding: .utf8) else {
        log("FATAL: could not encode the device list")
        exit(1)
    }
    print(text)
}

let options: Options
switch parseArguments(Array(CommandLine.arguments.dropFirst())) {
case .success(let parsed): options = parsed
case .failure(let why):
    log(why.message)
    log(usage)
    exit(2)
}

// --help succeeds and a bare invocation does not: the second is a caller that
// forgot to name a channel, and `stenocap --help` is what a wheel's smoke test
// runs to prove the binary starts on this machine.
if options.help {
    log(usage)
    exit(0)
}

// Neither read-only query may ask for microphone access: enumeration is not
// TCC-gated (measured 2026-08-12), and prompting inside `steno doctor` or a
// setup form would be a permission dialog nobody asked for.
if options.listInputs {
    printJSON(inputDevices().map { ["id": $0.uid, "name": $0.name, "default": $0.isDefault] })
    exit(0)
}

let wantMic = options.mic
let wantSystem = options.system

if options.devices {
    var named: [String: String] = [:]
    if wantMic || !wantSystem {
        if let pin = options.micDevice {
            switch resolveInputWithGrace(pin: pin) {
            case .success(let device): named["mic"] = device.name
            case .failure(let why):
                log("FATAL: \(why.message(pin))")
                exit(1)
            }
        } else if let id = defaultDeviceID(kAudioHardwarePropertyDefaultInputDevice) {
            named["mic"] = deviceName(id) ?? "the default input"
        } else {
            log("FATAL: no default microphone is configured — check Sound settings")
            exit(1)
        }
    }
    if wantSystem || !wantMic {
        guard let id = defaultDeviceID(kAudioHardwarePropertyDefaultOutputDevice) else {
            log("FATAL: no default output device is configured — check Sound settings")
            exit(1)
        }
        named["system"] = deviceName(id) ?? "the default output"
    }
    printJSON(named)
    exit(0)
}

if !wantMic && !wantSystem {
    log("\(usage)  (at least one channel)")
    exit(2)
}

let emitter = Emitter()
var tapSupervisor: TapSupervisor?
var micSupervisor: MicSupervisor?
var pinnedMic: PinnedMicSupervisor?

/// The startup path for a pinned mic: resolve (with its grace), then open the
/// device. Both failures are fatal, and neither substitutes another microphone
/// (`resolveInput`).
func startPinnedMic(pin: String) -> PinnedMic {
    switch resolveInputWithGrace(pin: pin) {
    case .failure(let why):
        log("FATAL: \(why.message(pin))")
        exit(1)
    case .success(let device):
        guard let mic = PinnedMic.start(device: device, emitter: emitter) else {
            log("FATAL: mic capture failed to start on \(device.name) "
                + "— is another app capturing? (OBS, a second stenograf)")
            exit(1)
        }
        return mic
    }
}

if wantMic { requestMicrophoneAccess() }
// Startup watchdog, armed only after the (legitimately open-ended) permission
// prompt: device setup normally takes well under a second, but a concurrent
// capture app can wedge coreaudiod so `engine.start()` never returns (measured
// 2026-07-20 racing two mic+tap captures). Turn the hang into a loud exit the
// consumer can detect and retry instead of a meeting that records nothing.
let startupWatchdog = DispatchSource.makeTimerSource(queue: .global())
startupWatchdog.schedule(deadline: .now() + 15)
startupWatchdog.setEventHandler {
    log("FATAL: capture did not start within 15 s "
        + "— is another app capturing? (OBS, a second stenograf)")
    exit(1)
}
startupWatchdog.resume()
_ = Clock.epoch  // fix the shared origin before either channel can stamp a frame

// Three ways to take the microphone, and which one applies is decided here:
//
//   unpinned            → AVAudioEngine on the default input, as ever
//   pinned, no system   → an IO proc straight on the device
//   pinned, with system → the device joins the tap's aggregate
//
// The third exists because the second does not work beside the tap: opening a
// hardware input directly while the tap's aggregate runs wedges the HAL
// (measured 2026-08-12, `PinnedMic`). Inside the aggregate there is one IO
// context and the conflict cannot arise — and both channels then share a clock
// by construction rather than by agreement.
let micInAggregate = wantMic && wantSystem && options.micDevice != nil
// Resolved once, before the tap exists, so a missing device fails with its own
// message rather than as a broken aggregate — and the answer is *kept*:
// re-deriving it a moment later is a window in which the device can vanish and
// the run silently becomes system-audio-only.
var pinnedMicUID: String?
if let pin = options.micDevice, micInAggregate {
    switch resolveInputWithGrace(pin: pin) {
    case .failure(let why):
        log("FATAL: \(why.message(pin))")
        exit(1)
    case .success(let device):
        pinnedMicUID = device.uid
    }
}
if wantSystem {
    let (tapID, tapUUID) = createSystemTap()
    guard let agg = buildAggregate(tapID: tapID, tapUUID: tapUUID, emitter: emitter,
                                   micUID: pinnedMicUID) else {
        log("FATAL: could not start system capture "
            + "— is another app capturing? (OBS, a second stenograf)")
        exit(1)
    }
    tapSupervisor = TapSupervisor(emitter: emitter, tapID: tapID, tapUUID: tapUUID, agg: agg,
                                  micPin: micInAggregate ? options.micDevice : nil)
}
if wantMic && !micInAggregate {
    if let pin = options.micDevice {
        pinnedMic = PinnedMicSupervisor(emitter: emitter, pin: pin, mic: startPinnedMic(pin: pin))
    } else {
        micSupervisor = MicSupervisor(emitter: emitter, engine: startMic(emitter: emitter))
    }
}
startupWatchdog.cancel()

func shutdown() -> Never {
    micSupervisor?.stop()
    pinnedMic?.stop()
    tapSupervisor?.stop()
    try? FileHandle.standardOutput.synchronize()
    log("stopped")
    exit(0)
}

// Sources must outlive this scope or their handlers never fire — hold them.
var signalSources: [DispatchSourceSignal] = []
for sig in [SIGINT, SIGTERM] {
    signal(sig, SIG_IGN)  // suppress the default action; the source handles it
    let source = DispatchSource.makeSignalSource(signal: sig, queue: .main)
    source.setEventHandler { shutdown() }
    source.resume()
    signalSources.append(source)
}

log("ready")
dispatchMain()
