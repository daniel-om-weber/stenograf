//! stenodiar — diarization helper for stenograf.
//!
//! Runs the speakrs pipeline (pyannote community-1-class: segmentation →
//! embeddings → PLDA → VBx clustering, which auto-estimates the speaker
//! count) over a mono 16 kHz 16-bit WAV and prints the speaker turns as one
//! JSON object on stdout. Model files download once from the ungated
//! HuggingFace mirror (`avencera/speakrs-models`) into the standard HF cache;
//! the first CoreML run additionally compiles the models (minutes, cached
//! per machine) — `--warmup` does both without diarizing anything.
//!
//! Usage:
//!   stenodiar [--mode MODE] <audio.wav>
//!   stenodiar [--mode MODE] --stdin    # raw mono 16 kHz s16le PCM on stdin
//!   stenodiar [--mode MODE] --warmup
//!
//! MODE is `cpu` everywhere, plus `coreml`/`coreml-fast` or `cuda`/`cuda-fast`
//! when the matching cargo feature was compiled in; the default is the best
//! compiled-in backend (coreml > cuda > cpu).
//!
//! stenograf itself always pipes PCM via ``--stdin``: meeting audio must
//! never touch disk (see native/README.md); the WAV path exists for
//! debugging against files.

use std::path::Path;
use std::process::ExitCode;

use speakrs::{ExecutionMode, OwnedDiarizationPipeline};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("stenodiar: {message}");
            ExitCode::FAILURE
        }
    }
}

fn default_mode() -> ExecutionMode {
    #[cfg(feature = "coreml")]
    return ExecutionMode::CoreMl;
    #[cfg(all(feature = "cuda", not(feature = "coreml")))]
    return ExecutionMode::Cuda;
    #[cfg(not(any(feature = "coreml", feature = "cuda")))]
    ExecutionMode::Cpu
}

fn parse_mode(value: &str) -> Result<ExecutionMode, String> {
    match value {
        "cpu" => Ok(ExecutionMode::Cpu),
        #[cfg(feature = "coreml")]
        "coreml" => Ok(ExecutionMode::CoreMl),
        #[cfg(feature = "coreml")]
        "coreml-fast" => Ok(ExecutionMode::CoreMlFast),
        #[cfg(feature = "cuda")]
        "cuda" => Ok(ExecutionMode::Cuda),
        #[cfg(feature = "cuda")]
        "cuda-fast" => Ok(ExecutionMode::CudaFast),
        #[cfg(not(feature = "coreml"))]
        "coreml" | "coreml-fast" => {
            Err(format!("mode '{value}' was not compiled into this binary"))
        }
        #[cfg(not(feature = "cuda"))]
        "cuda" | "cuda-fast" => {
            Err(format!("mode '{value}' was not compiled into this binary"))
        }
        other => Err(format!("unknown mode '{other}'")),
    }
}

fn run() -> Result<(), String> {
    let mut mode = default_mode();
    let mut warmup = false;
    let mut stdin_pcm = false;
    let mut wav_path: Option<String> = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--mode" => {
                let value = args.next().ok_or("--mode needs a value")?;
                mode = parse_mode(&value)?;
            }
            "--warmup" => warmup = true,
            "--stdin" => stdin_pcm = true,
            "--help" | "-h" => {
                eprintln!(
                    "usage: stenodiar [--mode cpu|coreml|coreml-fast|cuda|cuda-fast] (--warmup | --stdin | <audio.wav>)"
                );
                return Ok(());
            }
            other if wav_path.is_none() && !other.starts_with('-') => {
                wav_path = Some(other.to_owned());
            }
            other => return Err(format!("unexpected argument '{other}'")),
        }
    }

    let mut pipeline = OwnedDiarizationPipeline::from_pretrained(mode)
        .map_err(|e| format!("failed to load models: {e}"))?;

    if warmup {
        println!("{{\"ok\": true}}");
        return Ok(());
    }

    let audio = if stdin_pcm {
        read_stdin_pcm()?
    } else {
        let wav_path = wav_path.ok_or("no audio given (a WAV path, --stdin, or --warmup)")?;
        read_wav_mono_16k(Path::new(&wav_path))?
    };
    let result = pipeline
        .run(&audio)
        .map_err(|e| format!("diarization failed: {e}"))?;

    print!("{}", turns_json(&result.segments));
    Ok(())
}

/// Serialize turns as JSON by hand: the payload is floats plus speakrs'
/// fixed `SPEAKER_NN` labels, so a JSON library would be a dependency for
/// nothing. Guarded by the label check below.
fn turns_json(segments: &[speakrs::Segment]) -> String {
    let mut out = String::from("{\"turns\": [");
    for (i, seg) in segments.iter().enumerate() {
        assert!(
            seg.speaker.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'),
            "unexpected speaker label {:?}",
            seg.speaker
        );
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&format!(
            "{{\"speaker\": \"{}\", \"start\": {:.3}, \"end\": {:.3}}}",
            seg.speaker, seg.start, seg.end
        ));
    }
    out.push_str("]}\n");
    out
}

/// Raw mono 16 kHz s16le PCM until EOF — how stenograf feeds meeting audio,
/// which must never touch disk.
fn read_stdin_pcm() -> Result<Vec<f32>, String> {
    use std::io::Read;

    let mut data = Vec::new();
    std::io::stdin()
        .lock()
        .read_to_end(&mut data)
        .map_err(|e| format!("cannot read stdin: {e}"))?;
    Ok(data
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
        .collect())
}

/// Minimal RIFF reader for the one format stenograf writes: mono 16 kHz
/// 16-bit PCM. Anything else is a caller bug, reported not resampled.
fn read_wav_mono_16k(path: &Path) -> Result<Vec<f32>, String> {
    let data = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    if data.len() < 44 || &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err(format!("{} is not a WAV file", path.display()));
    }
    let channels = u16::from_le_bytes(data[22..24].try_into().unwrap());
    let sample_rate = u32::from_le_bytes(data[24..28].try_into().unwrap());
    let bits = u16::from_le_bytes(data[34..36].try_into().unwrap());
    if channels != 1 || sample_rate != 16_000 || bits != 16 {
        return Err(format!(
            "expected mono 16kHz 16-bit WAV, got {channels}ch {sample_rate}Hz {bits}-bit"
        ));
    }

    let mut pos = 12;
    while pos + 8 <= data.len() {
        let chunk_id = &data[pos..pos + 4];
        let chunk_size = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body = pos + 8;
        if chunk_id == b"data" {
            let end = (body + chunk_size).min(data.len());
            return Ok(data[body..end]
                .chunks_exact(2)
                .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
                .collect());
        }
        pos = body + chunk_size + (chunk_size & 1);
    }
    Err(format!("{} has no data chunk", path.display()))
}
