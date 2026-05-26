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
    download_file(TOKENIZER_URL, Path::new(TOKENIZER_PATH))?;
    download_file(ENCODER_URL, Path::new(ENCODER_PATH))?;
    download_file(DECODER_URL, Path::new(DECODER_PATH))?;

    // 2. Download LLM (Qwen3) assets
    println!("\n--- Downloading LLM (Qwen3-0.6B-ONNX) assets ---");
    download_file(QWEN_TOKENIZER_URL, Path::new(QWEN_TOKENIZER_PATH))?;
    download_file(QWEN_MODEL_URL, Path::new(QWEN_MODEL_PATH))?;

    println!("\nAll models downloaded successfully!");
    Ok(())
}
