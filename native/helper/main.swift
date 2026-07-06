// stenocap — stenograf's macOS capture helper (PLAN.md §2).
//
// Captures two independent channels and streams them to the Python core:
//   --system : whole-system audio via a Core Audio process tap (macOS 14.4+)
//   --mic    : the microphone via AVAudioEngine (optional --aec echo cancel)
//
// Both are downmixed + resampled to mono 16 kHz int16 and written to stdout as
// framed PCM. stdout carries frames only; all status/errors go to stderr.
//
//   frame = channel:u8  timestamp:f64le  count:u32le  samples:count×i16le
//   channel: 0 = mic, 1 = system;  timestamp: seconds since capture start.
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

// MARK: - frame emitter

/// Serializes frame writes from the mic and tap callbacks onto stdout, and
/// stamps each channel with a monotonic timestamp from its own sample count.
final class Emitter: @unchecked Sendable {
    private let lock = NSLock()
    private let out = FileHandle.standardOutput
    private var emitted: [UInt8: Int] = [:]

    /// Append one frame of mono 16 kHz int16 samples for `channel`.
    func emit(_ channel: ChannelCode, _ samples: UnsafeBufferPointer<Int16>) {
        lock.lock()
        defer { lock.unlock() }
        let priorSamples = emitted[channel.rawValue, default: 0]
        let timestamp = Double(priorSamples) / SAMPLE_RATE
        emitted[channel.rawValue] = priorSamples + samples.count

        var header = Data(capacity: 13)
        header.append(channel.rawValue)
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

    func feed(_ input: AVAudioPCMBuffer) {
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
        emitter.emit(channel, UnsafeBufferPointer(start: data[0], count: Int(output.frameLength)))
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

    guard let sourceFormat = AVAudioFormat(streamDescription: &asbd),
          let resampler = Resampler(source: sourceFormat, channel: .system, emitter: emitter)
    else {
        log("FATAL: could not build system-audio resampler")
        exit(1)
    }

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
    die(AudioDeviceCreateIOProcIDWithBlock(&procID, aggID, queue) { _, inInputData, _, _, _ in
        renderTapBuffer(inInputData, sourceFormat: sourceFormat, resampler: resampler)
    }, "create tap IO proc")
    die(AudioDeviceStart(aggID, procID), "start aggregate device")
    log("system capture started")
    return TapSession(tapID: tapID, aggID: aggID, procID: procID)
}

/// Wrap the raw IO-proc buffer list as an AVAudioPCMBuffer and resample it.
func renderTapBuffer(_ abl: UnsafePointer<AudioBufferList>,
                     sourceFormat: AVAudioFormat, resampler: Resampler) {
    let list = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: abl))
    guard let first = list.first, let data = first.mData else { return }
    let bytesPerFrame = Int(sourceFormat.streamDescription.pointee.mBytesPerFrame)
    guard bytesPerFrame > 0 else { return }
    let frames = AVAudioFrameCount(Int(first.mDataByteSize) / bytesPerFrame)
    guard frames > 0,
          let buffer = AVAudioPCMBuffer(pcmFormat: sourceFormat, frameCapacity: frames)
    else { return }
    buffer.frameLength = frames
    // The tap delivers deinterleaved float32; copy the first (mono) channel.
    if let dst = buffer.floatChannelData {
        memcpy(dst[0], data, Int(first.mDataByteSize))
    }
    resampler.feed(buffer)
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

func startMic(emitter: Emitter, aec: Bool) -> AVAudioEngine {
    let sem = DispatchSemaphore(value: 0)
    var granted = false
    AVCaptureDevice.requestAccess(for: .audio) { granted = $0; sem.signal() }
    sem.wait()
    guard granted else {
        log("FATAL: microphone permission denied")
        exit(1)
    }

    let engine = AVAudioEngine()
    let input = engine.inputNode
    if aec {
        do {
            try input.setVoiceProcessingEnabled(true)  // AEC for speaker output
            log("mic echo cancellation enabled")
        } catch {
            log("could not enable echo cancellation: \(error.localizedDescription)")
        }
    }
    let format = input.inputFormat(forBus: 0)
    log("mic format: \(format.sampleRate) Hz, \(format.channelCount) ch")

    guard let resampler = Resampler(source: format, channel: .mic, emitter: emitter) else {
        log("FATAL: could not build mic resampler")
        exit(1)
    }
    input.installTap(onBus: 0, bufferSize: 4096, format: format) { buffer, _ in
        resampler.feed(buffer)
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
let wantAEC = args.contains("--aec")

if !wantMic && !wantSystem {
    log("usage: stenocap [--mic] [--system] [--aec]  (at least one channel)")
    exit(2)
}

let emitter = Emitter()
var tapSession: TapSession?
var micEngine: AVAudioEngine?

if wantSystem { tapSession = startSystemTap(emitter: emitter) }
if wantMic { micEngine = startMic(emitter: emitter, aec: wantAEC) }

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
