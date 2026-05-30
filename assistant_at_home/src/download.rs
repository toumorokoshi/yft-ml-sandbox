use std::fs::File;
use std::path::Path;

// Model download URL constants
pub const TOKENIZER_URL: &str =
    "https://huggingface.co/UsefulSensors/moonshine-tiny/resolve/main/tokenizer.json";
pub const ENCODER_URL: &str = "https://huggingface.co/UsefulSensors/moonshine/resolve/main/onnx/merged/tiny/float/encoder_model.onnx";
pub const DECODER_URL: &str = "https://huggingface.co/UsefulSensors/moonshine/resolve/main/onnx/merged/tiny/float/decoder_model_merged.onnx";

pub const TOKENIZER_PATH: &str = "models/tokenizer.json";
pub const ENCODER_PATH: &str = "models/tiny/encoder_model.onnx";
pub const DECODER_PATH: &str = "models/tiny/decoder_model_merged.onnx";

// Qwen3 Model constants
pub const QWEN_TOKENIZER_URL: &str =
    "https://huggingface.co/onnx-community/Qwen3-0.6B-ONNX/resolve/main/tokenizer.json";
pub const QWEN_MODEL_URL: &str =
    "https://huggingface.co/onnx-community/Qwen3-0.6B-ONNX/resolve/main/onnx/model_q4.onnx";

pub const QWEN_TOKENIZER_PATH: &str = "models/qwen3_0.6b/tokenizer.json";
pub const QWEN_MODEL_PATH: &str = "models/qwen3_0.6b/model_q4.onnx";

// Kokoro Model constants
pub const KOKORO_TOKENIZER_URL: &str =
    "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/tokenizer.json";
pub const KOKORO_MODEL_URL: &str =
    "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model_quantized.onnx";
pub const KOKORO_VOICE_URL: &str =
    "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/voices/af_bella.bin";

pub const KOKORO_TOKENIZER_PATH: &str = "models/kokoro/tokenizer.json";
pub const KOKORO_MODEL_PATH: &str = "models/kokoro/model_quantized.onnx";
pub const KOKORO_VOICE_PATH: &str = "models/kokoro/af_bella.bin";

/// Resolves a relative path to an absolute path dynamically based on environment.
pub fn get_path(rel_path: &str) -> std::path::PathBuf {
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        Path::new(&manifest_dir).join(rel_path)
    } else if let Ok(workspace_dir) = std::env::var("BUILD_WORKSPACE_DIRECTORY") {
        Path::new(&workspace_dir).join("assistant_at_home").join(rel_path)
    } else {
        Path::new(rel_path).to_path_buf()
    }
}

/// Resolves a user-provided CLI path relative to the invoking shell working directory under Bazel.
pub fn resolve_user_path(user_path: &str) -> std::path::PathBuf {
    let path = Path::new(user_path);
    if path.is_absolute() {
        path.to_path_buf()
    } else if let Ok(working_dir) = std::env::var("BUILD_WORKING_DIRECTORY") {
        Path::new(&working_dir).join(user_path)
    } else {
        path.to_path_buf()
    }
}



/// Downloads a single file from a URL to a local destination path.
pub fn download_file(url: &str, dest_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("Downloading {} -> {:?}", url, dest_path);

    let response = ureq::get(url).call()?;
    if response.status() != 200 {
        return Err(format!(
            "Failed to download from {}. HTTP status: {}",
            url,
            response.status()
        )
        .into());
    }

    if let Some(parent) = dest_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut file = File::create(dest_path)?;
    let mut reader = response.into_reader();
    std::io::copy(&mut reader, &mut file)?;

    println!("Successfully downloaded to {:?}", dest_path);
    Ok(())
}

/// Helper wrapper function to download all required Moonshine and Qwen3 ONNX model assets.
pub fn download_models_pipeline() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Download Speech-to-Text assets
    println!("--- Downloading Speech-to-Text (Moonshine) assets ---");
    download_file(TOKENIZER_URL, &get_path(TOKENIZER_PATH))?;
    download_file(ENCODER_URL, &get_path(ENCODER_PATH))?;
    download_file(DECODER_URL, &get_path(DECODER_PATH))?;

    // 2. Download LLM (Qwen3) assets
    println!("\n--- Downloading LLM (Qwen3-0.6B-ONNX) assets ---");
    download_file(QWEN_TOKENIZER_URL, &get_path(QWEN_TOKENIZER_PATH))?;
    download_file(QWEN_MODEL_URL, &get_path(QWEN_MODEL_PATH))?;

    // 3. Download TTS (Kokoro) assets
    println!("\n--- Downloading TTS (Kokoro-82M) assets ---");
    download_file(KOKORO_TOKENIZER_URL, &get_path(KOKORO_TOKENIZER_PATH))?;
    download_file(KOKORO_MODEL_URL, &get_path(KOKORO_MODEL_PATH))?;
    download_file(KOKORO_VOICE_URL, &get_path(KOKORO_VOICE_PATH))?;

    println!("\nAll models downloaded successfully!");
    Ok(())
}

