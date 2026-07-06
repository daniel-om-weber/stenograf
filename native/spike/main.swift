// tap-spike: Phase 1 capture spike (PLAN.md §2 "Deployment & distribution").
//
// Verifies, with a free ad-hoc-signed CLI binary launched from a terminal:
//   1. `tap` — AudioHardwareCreateProcessTap (macOS 14.4+) fires the
//      "System Audio Recording" TCC prompt and delivers non-silent system audio.
//   2. `mic` — AVAudioEngine fires the microphone TCC prompt and delivers audio.
//
// Usage: tap-spike [tap|mic] [seconds]

import Foundation
import CoreAudio
import AudioToolbox
import AVFoundation

func fourCC(_ status: OSStatus) -> String {
    let n = UInt32(bitPattern: status)
    let bytes = [UInt8((n >> 24) & 0xff), UInt8((n >> 16) & 0xff),
                 UInt8((n >> 8) & 0xff), UInt8(n & 0xff)]
    if bytes.allSatisfy({ $0 >= 0x20 && $0 < 0x7f }) {
        return "'\(String(bytes: bytes, encoding: .ascii)!)' (\(status))"
    }
    return "\(status)"
}

func check(_ status: OSStatus, _ what: String) {
    guard status == noErr else {
        print("FAIL: \(what): OSStatus \(fourCC(status))")
        exit(1)
    }
    print("ok: \(what)")
}

final class Stats: @unchecked Sendable {
    private let lock = NSLock()
    private var samples = 0
    private var sumSquares = 0.0
    private var peakVal: Float = 0
    private var callbackCount = 0

    func add(samples n: Int, sumSquares s: Double, peak p: Float) {
        lock.lock()
        samples += n
        sumSquares += s
        peakVal = max(peakVal, p)
        callbackCount += 1
        lock.unlock()
    }

    var snapshot: (samples: Int, rms: Double, peak: Float, callbacks: Int) {
        lock.lock()
        defer { lock.unlock() }
        let rms = samples > 0 ? (sumSquares / Double(samples)).squareRoot() : 0
        return (samples, rms, peakVal, callbackCount)
    }
}

func accumulate(_ abl: UnsafePointer<AudioBufferList>, into stats: Stats) {
    let list = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: abl))
    var n = 0
    var s = 0.0
    var p: Float = 0
    for buf in list {
        guard let data = buf.mData else { continue }
        let count = Int(buf.mDataByteSize) / MemoryLayout<Float32>.size
        let ptr = data.assumingMemoryBound(to: Float32.self)
        for i in 0..<count {
            let v = ptr[i]
            s += Double(v * v)
            p = max(p, abs(v))
        }
        n += count
    }
    stats.add(samples: n, sumSquares: s, peak: p)
}

/// Watch `stats` for `seconds`, printing one line per second; returns pass/fail.
func watch(_ label: String, _ stats: Stats, seconds: Int) -> Bool {
    for i in 1...seconds {
        Thread.sleep(forTimeInterval: 1)
        let s = stats.snapshot
        print(String(format: "  t=%2ds  callbacks=%-5d samples=%-9d rms=%.5f  peak=%.4f",
                     i, s.callbacks, s.samples, s.rms, s.peak))
    }
    let s = stats.snapshot
    let flowing = s.callbacks > 0
    let nonSilent = s.peak > 0.001
    print("\(label): callbacks arriving: \(flowing ? "YES" : "NO"); non-silent audio: \(nonSilent ? "YES" : "NO")")
    return flowing && nonSilent
}

func defaultOutputDeviceUID() -> String {
    var deviceID = AudioObjectID(kAudioObjectUnknown)
    var size = UInt32(MemoryLayout<AudioObjectID>.size)
    var addr = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDefaultOutputDevice,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)
    check(AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject),
                                     &addr, 0, nil, &size, &deviceID),
          "get default output device")

    var uid: CFString = "" as CFString
    size = UInt32(MemoryLayout<CFString>.size)
    addr.mSelector = kAudioDevicePropertyDeviceUID
    check(AudioObjectGetPropertyData(deviceID, &addr, 0, nil, &size, &uid),
          "get output device UID")
    return uid as String
}

func runTapTest(seconds: Int) {
    print("== system-audio process-tap test ==")

    // Mono mixdown of everything except no processes = whole-system tap,
    // same shape as the planned production helper.
    let desc = CATapDescription(monoGlobalTapButExcludeProcesses: [])
    desc.name = "stenograf-spike-tap"
    desc.muteBehavior = .unmuted
    desc.isPrivate = true

    var tapID = AudioObjectID(kAudioObjectUnknown)
    print("creating process tap (System Audio Recording prompt should appear on first run)...")
    check(AudioHardwareCreateProcessTap(desc, &tapID), "AudioHardwareCreateProcessTap")

    var asbd = AudioStreamBasicDescription()
    var size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
    var addr = AudioObjectPropertyAddress(
        mSelector: kAudioTapPropertyFormat,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)
    check(AudioObjectGetPropertyData(tapID, &addr, 0, nil, &size, &asbd),
          "read tap stream format")
    print("tap format: \(asbd.mSampleRate) Hz, \(asbd.mChannelsPerFrame) ch, \(asbd.mBitsPerChannel)-bit float")

    let outputUID = defaultOutputDeviceUID()
    let aggDesc: [String: Any] = [
        kAudioAggregateDeviceNameKey: "stenograf-spike-agg",
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
    check(AudioHardwareCreateAggregateDevice(aggDesc as CFDictionary, &aggID),
          "create aggregate device containing the tap")

    let stats = Stats()
    var procID: AudioDeviceIOProcID?
    let queue = DispatchQueue(label: "dev.stenograf.spike.io")
    check(AudioDeviceCreateIOProcIDWithBlock(&procID, aggID, queue) { _, inInputData, _, _, _ in
        accumulate(inInputData, into: stats)
    }, "create IO proc on aggregate device")
    check(AudioDeviceStart(aggID, procID), "start aggregate device")

    print("capturing system audio for \(seconds)s — some sound must be playing...")
    let ok = watch("tap", stats, seconds: seconds)

    AudioDeviceStop(aggID, procID)
    if let procID { AudioDeviceDestroyIOProcID(aggID, procID) }
    AudioHardwareDestroyAggregateDevice(aggID)
    AudioHardwareDestroyProcessTap(tapID)

    print(ok ? "TAP SPIKE: PASS"
             : "TAP SPIKE: FAIL (callbacks but silence usually means TCC denied; no callbacks means device setup failed)")
    exit(ok ? 0 : 1)
}

func runMicTest(seconds: Int) {
    print("== microphone test ==")
    let before = AVCaptureDevice.authorizationStatus(for: .audio)
    print("mic TCC status before: \(before.rawValue) (0=notDetermined 1=restricted 2=denied 3=authorized)")

    let sem = DispatchSemaphore(value: 0)
    var granted = false
    AVCaptureDevice.requestAccess(for: .audio) { g in
        granted = g
        sem.signal()
    }
    print("waiting for mic permission (prompt should appear if not previously decided)...")
    sem.wait()
    guard granted else {
        print("MIC SPIKE: FAIL (permission denied)")
        exit(1)
    }
    print("ok: mic permission granted")

    let engine = AVAudioEngine()
    let input = engine.inputNode
    let format = input.inputFormat(forBus: 0)
    print("mic format: \(format.sampleRate) Hz, \(format.channelCount) ch")

    let stats = Stats()
    input.installTap(onBus: 0, bufferSize: 4096, format: format) { buffer, _ in
        guard let ch = buffer.floatChannelData else { return }
        let frames = Int(buffer.frameLength)
        var s = 0.0
        var p: Float = 0
        for c in 0..<Int(buffer.format.channelCount) {
            for i in 0..<frames {
                let v = ch[c][i]
                s += Double(v * v)
                p = max(p, abs(v))
            }
        }
        stats.add(samples: frames, sumSquares: s, peak: p)
    }

    do {
        try engine.start()
    } catch {
        print("MIC SPIKE: FAIL (engine start: \(error))")
        exit(1)
    }

    print("capturing mic for \(seconds)s — make some noise...")
    let ok = watch("mic", stats, seconds: seconds)
    engine.stop()

    print(ok ? "MIC SPIKE: PASS" : "MIC SPIKE: FAIL (no audio captured)")
    exit(ok ? 0 : 1)
}

let args = CommandLine.arguments
let mode = args.count > 1 ? args[1] : "tap"
let seconds = args.count > 2 ? (Int(args[2]) ?? 10) : 10

switch mode {
case "tap": runTapTest(seconds: seconds)
case "mic": runMicTest(seconds: seconds)
default:
    print("usage: tap-spike [tap|mic] [seconds]")
    exit(2)
}
