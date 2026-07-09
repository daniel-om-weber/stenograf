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
    private let lock = NSLock()
    private let out = FileHandle.standardOutput
    private var emitted: [UInt8: Int] = [:]
    private var anchor: [UInt8: Double] = [:]
    private var driftWarned: Set<UInt8> = []

    /// Append one frame of mono 16 kHz int16 samples for `channel`. `hostTime`
    /// is when the *input* buffer behind these samples was captured; it anchors
    /// the channel on first use, after which sample counting carries the
    /// timeline (monotonic, and sample-accurate within the channel).
    func emit(_ channel: ChannelCode, _ samples: UnsafeBufferPointer<Int16>, hostTime: UInt64) {
        lock.lock()
        defer { lock.unlock() }
        let code = channel.rawValue
        let priorSamples = emitted[code, default: 0]
        let base: Double
        if let existing = anchor[code] {
            base = existing
        } else {
            base = Clock.seconds(since: hostTime)
            anchor[code] = base
        }
        let timestamp = base + Double(priorSamples) / SAMPLE_RATE
        emitted[code] = priorSamples + samples.count

        // A device that drops or repeats buffers walks its sample count away
        // from wall clock, which silently misaligns the echo canceller. Say so
        // once rather than emitting a plausible-looking lie forever.
        if priorSamples > 0, !driftWarned.contains(code) {
            let drift = Clock.seconds(since: hostTime) - timestamp
            if abs(drift) > 0.25 {
                driftWarned.insert(code)
                log("WARNING channel \(code) drifted \(Int(drift * 1000)) ms from wall clock")
            }
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

// MARK: - system-audio process tap

/// Holds the Core Audio objects for the system tap so they can be torn down.
struct TapSession {
    var tapID: AudioObjectID
    var aggID: AudioObjectID
    var procID: AudioDeviceIOProcID?
}

func defaultOutputDeviceUID() -> String {
    var deviceID = AudioObjectID(kAudioObjectUnknown)
    var size = UInt32(MemoryLayout<AudioObjectID>.size)
    var addr = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDefaultOutputDevice,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)
    die(AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject),
                                   &addr, 0, nil, &size, &deviceID),
        "get default output device")
    var uid: CFString = "" as CFString
    size = UInt32(MemoryLayout<CFString>.size)
    addr.mSelector = kAudioDevicePropertyDeviceUID
    die(AudioObjectGetPropertyData(deviceID, &addr, 0, nil, &size, &uid),
        "get output device UID")
    return uid as String
}

/// Tracks the tap's observed channel layout so a mid-session change is logged
/// once rather than per buffer.
final class TapLayout: @unchecked Sendable {
    private let lock = NSLock()
    private var channels = 0

    func note(_ observed: Int) {
        lock.lock()
        defer { lock.unlock() }
        if channels != 0, channels != observed {
            log("WARNING system tap changed from \(channels) to \(observed) channel(s) "
                + "— the output device was renegotiated mid-capture")
        }
        channels = observed
    }
}

func startSystemTap(emitter: Emitter) -> TapSession {
    let desc = CATapDescription(monoGlobalTapButExcludeProcesses: [])
    desc.name = "stenograf-tap"
    desc.muteBehavior = .unmuted
    desc.isPrivate = true

    var tapID = AudioObjectID(kAudioObjectUnknown)
    die(AudioHardwareCreateProcessTap(desc, &tapID), "AudioHardwareCreateProcessTap")

    var asbd = AudioStreamBasicDescription()
    var size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
    var addr = AudioObjectPropertyAddress(
        mSelector: kAudioTapPropertyFormat,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)
    die(AudioObjectGetPropertyData(tapID, &addr, 0, nil, &size, &asbd), "read tap format")
    log("system tap format: \(asbd.mSampleRate) Hz, \(asbd.mChannelsPerFrame) ch")

    // The resampler always sees mono float32 at the tap's rate; renderTapBuffer
    // downmixes whatever channel layout the buffer actually arrives in.
    guard let sourceFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32, sampleRate: asbd.mSampleRate,
        channels: 1, interleaved: false),
        let resampler = Resampler(source: sourceFormat, channel: .system, emitter: emitter)
    else {
        log("FATAL: could not build system-audio resampler")
        exit(1)
    }
    let layout = TapLayout()

    let outputUID = defaultOutputDeviceUID()
    let aggDesc: [String: Any] = [
        kAudioAggregateDeviceNameKey: "stenograf-agg",
        kAudioAggregateDeviceUIDKey: UUID().uuidString,
        kAudioAggregateDeviceIsPrivateKey: true,
        kAudioAggregateDeviceIsStackedKey: false,
        kAudioAggregateDeviceTapAutoStartKey: true,
        kAudioAggregateDeviceMainSubDeviceKey: outputUID,
        kAudioAggregateDeviceSubDeviceListKey: [[kAudioSubDeviceUIDKey: outputUID]],
        kAudioAggregateDeviceTapListKey: [[
            kAudioSubTapDriftCompensationKey: true,
            kAudioSubTapUIDKey: desc.uuid.uuidString,
        ]],
    ]
    var aggID = AudioObjectID(kAudioObjectUnknown)
    die(AudioHardwareCreateAggregateDevice(aggDesc as CFDictionary, &aggID),
        "create aggregate device")

    var procID: AudioDeviceIOProcID?
    let queue = DispatchQueue(label: "dev.stenograf.tap")
    die(AudioDeviceCreateIOProcIDWithBlock(&procID, aggID, queue) { _, inInputData, inInputTime, _, _ in
        let stamp = inInputTime.pointee
        let hostTime = stamp.mFlags.contains(.hostTimeValid) ? stamp.mHostTime : Clock.now()
        renderTapBuffer(inInputData, hostTime: hostTime, sourceFormat: sourceFormat,
                        resampler: resampler, layout: layout)
    }, "create tap IO proc")
    die(AudioDeviceStart(aggID, procID), "start aggregate device")
    log("system capture started")
    return TapSession(tapID: tapID, aggID: aggID, procID: procID)
}

/// Downmix the IO-proc buffer to mono and hand it to the resampler.
///
/// The frame count is derived from the buffer we were handed, never from the
/// format read at startup: Core Audio renegotiates the tap when the output
/// device changes (headphones, AirPods, a display with speakers), and reading a
/// multi-channel buffer as mono would emit several times too many samples.
func renderTapBuffer(_ abl: UnsafePointer<AudioBufferList>, hostTime: UInt64,
                     sourceFormat: AVAudioFormat, resampler: Resampler, layout: TapLayout) {
    let list = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: abl))
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
func downmix(_ list: UnsafeMutableAudioBufferListPointer,
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

func stopSystemTap(_ session: TapSession) {
    if let procID = session.procID {
        AudioDeviceStop(session.aggID, procID)
        AudioDeviceDestroyIOProcID(session.aggID, procID)
    }
    AudioHardwareDestroyAggregateDevice(session.aggID)
    AudioHardwareDestroyProcessTap(session.tapID)
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

func startMic(emitter: Emitter) -> AVAudioEngine {
    let engine = AVAudioEngine()
    let input = engine.inputNode
    let format = input.inputFormat(forBus: 0)
    log("mic format: \(format.sampleRate) Hz, \(format.channelCount) ch")

    guard let resampler = Resampler(source: format, channel: .mic, emitter: emitter) else {
        log("FATAL: could not build mic resampler")
        exit(1)
    }
    input.installTap(onBus: 0, bufferSize: 4096, format: format) { buffer, when in
        let hostTime = when.isHostTimeValid ? when.hostTime : Clock.now()
        resampler.feed(buffer, hostTime: hostTime)
    }
    do {
        try engine.start()
    } catch {
        log("FATAL: mic engine start: \(error.localizedDescription)")
        exit(1)
    }
    log("mic capture started")
    return engine
}

// MARK: - main

let args = Array(CommandLine.arguments.dropFirst())
let wantMic = args.contains("--mic")
let wantSystem = args.contains("--system")

if !wantMic && !wantSystem {
    log("usage: stenocap [--mic] [--system]  (at least one channel)")
    exit(2)
}

let emitter = Emitter()
var tapSession: TapSession?
var micEngine: AVAudioEngine?

if wantMic { requestMicrophoneAccess() }
_ = Clock.epoch  // fix the shared origin before either channel can stamp a frame
if wantSystem { tapSession = startSystemTap(emitter: emitter) }
if wantMic { micEngine = startMic(emitter: emitter) }

func shutdown() -> Never {
    micEngine?.stop()
    if let tapSession { stopSystemTap(tapSession) }
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
