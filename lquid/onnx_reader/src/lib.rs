use std::error::Error;
use std::path::Path;
use protobuf::Message;
pub use onnx_protobuf::ModelProto;

/// Wrapper function for I/O: loads bytes from file and passes them to the inner parsing function.
/// Follows Rule 2: IO with network/filesystem is kept in a wrapper.
pub fn load_model_from_file<P: AsRef<Path>>(path: P) -> Result<ModelProto, Box<dyn Error>> {
    let bytes = std::fs::read(path)?;
    let model = parse_model_from_bytes(&bytes)?;
    Ok(model)
}

/// Inner function: parses the ONNX graph from raw bytes. Works on data structures.
/// Follows Rule 2: works purely on data structures.
pub fn parse_model_from_bytes(bytes: &[u8]) -> Result<ModelProto, protobuf::Error> {
    ModelProto::parse_from_bytes(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Rule 5: prefer to use constants, especially for hard-coded directories or values.
    const TEMP_TEST_FILE_PATH: &str = "test_model_temp.onnx";

    // Rule 3: unit tests should run on data structures directly.
    #[test]
    fn test_parse_model_from_bytes_success() {
        let mut model = ModelProto::new();
        model.ir_version = 8;
        model.producer_name = "test_producer".to_string();

        let bytes = model.write_to_bytes().expect("failed to serialize model");
        let parsed = parse_model_from_bytes(&bytes).expect("failed to parse model");

        assert_eq!(parsed.ir_version, 8);
        assert_eq!(parsed.producer_name, "test_producer");
    }

    #[test]
    fn test_parse_model_from_bytes_invalid() {
        let invalid_bytes = b"not a valid protobuf";
        let result = parse_model_from_bytes(invalid_bytes);
        assert!(result.is_err());
    }

    // Rule 3: only a single test on IO to serve as an integration test.
    #[test]
    fn test_load_model_from_file_io_integration() {
        let mut model = ModelProto::new();
        model.ir_version = 8;
        let bytes = model.write_to_bytes().expect("failed to serialize model");

        // Write temp file to test integration
        std::fs::write(TEMP_TEST_FILE_PATH, bytes).expect("failed to write test file");

        let result = load_model_from_file(TEMP_TEST_FILE_PATH);

        // Ensure file is deleted even if assertions fail
        let _ = std::fs::remove_file(TEMP_TEST_FILE_PATH);

        let parsed = result.expect("failed to load model from file");
        assert_eq!(parsed.ir_version, 8);
    }
}
