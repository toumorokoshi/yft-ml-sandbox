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

/// Helper wrapper function to download all required Moonshine ONNX model assets.
pub fn download_models_pipeline() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Download tokenizer
    download_file(TOKENIZER_URL, Path::new(TOKENIZER_PATH))?;

    // 2. Download encoder
    download_file(ENCODER_URL, Path::new(ENCODER_PATH))?;

    // 3. Download decoder
    download_file(DECODER_URL, Path::new(DECODER_PATH))?;

    println!("\nAll models downloaded successfully!");
    Ok(())
}
