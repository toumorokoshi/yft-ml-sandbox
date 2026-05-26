use crate::download::{DECODER_PATH, ENCODER_PATH, TOKENIZER_PATH};
use ndarray::{Array2, Array4, ArrayViewD, Axis, Ix1, Ix4};
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;
use std::collections::HashMap;
use std::path::Path;
use tokenizers::Tokenizer;

// --- CONSTANTS ---
pub const START_TOKEN_ID: i64 = 1;
pub const EOS_TOKEN_ID: i64 = 2;
pub const MAX_LEN: usize = 192;

// Model configuration parameters
pub struct ModelParams {
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
}

pub const TINY_PARAMS: ModelParams = ModelParams {
    num_layers: 6,
    num_heads: 8,
    head_dim: 36,
};

pub enum WavSamples {
    Float(Vec<f32>),
    Int(Vec<i32>),
}

/// Normalizes and reshapes WAV audio samples into shape [1, samples]
pub fn process_wav_data(
    samples: WavSamples,
    sample_rate: u32,
    channels: u16,
    bits_per_sample: u16,
) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    if sample_rate != 16000 {
        return Err("Sample rate must be 16000 Hz".into());
    }
    if channels != 1 {
        return Err("Audio must be mono (1 channel)".into());
    }

    let normalized_samples: Vec<f32> = match samples {
        WavSamples::Float(s) => s,
        WavSamples::Int(s) => {
            let max_val = (1 << (bits_per_sample - 1)) as f32;
            s.into_iter().map(|val| val as f32 / max_val).collect()
        }
    };

    let num_samples = normalized_samples.len();
    let array = Array2::from_shape_vec((1, num_samples), normalized_samples)?;
    Ok(array)
}

/// Loads a WAV file (16kHz mono) and normalizes it to float values in shape [1, samples]
pub fn load_wav_file<P: AsRef<Path>>(path: P) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    let samples = match spec.sample_format {
        hound::SampleFormat::Float => {
            let raw_samples: Vec<f32> = reader.samples::<f32>().filter_map(Result::ok).collect();
            WavSamples::Float(raw_samples)
        }
        hound::SampleFormat::Int => {
            let raw_samples: Vec<i32> = reader.samples::<i32>().filter_map(Result::ok).collect();
            WavSamples::Int(raw_samples)
        }
    };

    process_wav_data(
        samples,
        spec.sample_rate,
        spec.channels,
        spec.bits_per_sample,
    )
}

/// Decodes sequence token IDs using the provided Tokenizer.
pub fn decode_tokens_to_string(
    tokens: &[i64],
    tokenizer: &Tokenizer,
) -> Result<String, Box<dyn std::error::Error>> {
    let u32_tokens: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
    let text = tokenizer
        .decode(&u32_tokens, true)
        .map_err(|e| e.to_string())?;
    Ok(text)
}

/// Generates names for key-value caches preserving deterministic ordering.
pub fn get_cache_keys(params: &ModelParams) -> Vec<String> {
    let mut keys = Vec::new();
    for layer in 0..params.num_layers {
        for stage in &["decoder", "encoder"] {
            for kv in &["key", "value"] {
                keys.push(format!("past_key_values.{layer}.{stage}.{kv}"));
            }
        }
    }
    keys
}

/// Initializes blank Key-Value cache arrays.
pub fn initialize_past_key_values(params: &ModelParams) -> HashMap<String, Array4<f32>> {
    let mut cache = HashMap::new();
    for key in get_cache_keys(params) {
        let tensor = Array4::zeros((0, params.num_heads, 1, params.head_dim));
        cache.insert(key, tensor);
    }
    cache
}

/// Finds the argmax index along the vocab axis for the last sequence item.
pub fn argmax_next_token(logits: &ArrayViewD<f32>) -> i64 {
    let shape = logits.shape();
    let seq_len = shape[1];

    let binding = logits.index_axis(Axis(0), 0);
    let last_token_logits = binding.index_axis(Axis(0), seq_len - 1);
    let last_token_logits_1d = last_token_logits.into_dimensionality::<Ix1>().unwrap();

    let mut max_idx = 0;
    let mut max_val = last_token_logits_1d[0];
    for (idx, &val) in last_token_logits_1d.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = idx;
        }
    }
    max_idx as i64
}

/// Updates the KV Cache with the present KV output values from the decoder.
pub fn update_kv_cache(
    past_key_values: HashMap<String, Array4<f32>>,
    present_values: Vec<Array4<f32>>,
    keys_order: &[String],
    use_cache_branch: bool,
) -> HashMap<String, Array4<f32>> {
    let mut updated = HashMap::new();
    for (k, v) in keys_order.iter().zip(present_values.into_iter()) {
        if !use_cache_branch || k.contains("decoder") {
            updated.insert(k.clone(), v);
        } else {
            updated.insert(k.clone(), past_key_values[k].clone());
        }
    }
    updated
}

/// Executes the core speech-to-text inference pipeline using loaded ONNX sessions.
pub fn transcribe_speech(
    audio: Array2<f32>,
    encoder_session: &mut Session,
    decoder_session: &mut Session,
    params: &ModelParams,
) -> Result<Vec<i64>, Box<dyn std::error::Error>> {
    // 1. Run the Encoder
    let encoder_inputs = ort::inputs![
        "input_values" => Tensor::from_array(audio)?
    ];
    let encoder_outputs = encoder_session.run(encoder_inputs)?;
    let last_hidden_state: ArrayViewD<f32> = encoder_outputs[0].try_extract_array()?;
    let encoder_hidden_states = last_hidden_state.to_owned();

    // 2. Initialize autoregressive loop state
    let cache_keys = get_cache_keys(params);
    let mut past_key_values = initialize_past_key_values(params);
    let mut tokens = vec![START_TOKEN_ID];

    let mut input_ids = Array2::from_shape_vec((1, 1), vec![START_TOKEN_ID])?;

    for i in 0..MAX_LEN {
        let use_cache_branch = i > 0;
        let mut decoder_inputs: HashMap<String, SessionInputValue> = HashMap::new();

        // Convert ndarrays to dynamic Value/Tensor inputs
        let input_ids_val = Tensor::from_array(input_ids.clone())?;
        decoder_inputs.insert("input_ids".to_string(), input_ids_val.into());

        let encoder_hidden_states_val = Tensor::from_array(encoder_hidden_states.clone())?;
        decoder_inputs.insert(
            "encoder_hidden_states".to_string(),
            encoder_hidden_states_val.into(),
        );

        let use_cache_val = Tensor::from_array(ndarray::Array1::from_elem(1, use_cache_branch))?;
        decoder_inputs.insert("use_cache_branch".to_string(), use_cache_val.into());

        // Insert key-value caches
        for k in &cache_keys {
            let kv_tensor = past_key_values.get(k).unwrap();
            let kv_val = Tensor::from_array(kv_tensor.clone())?;
            decoder_inputs.insert(k.clone(), kv_val.into());
        }

        // Run the Decoder Session
        let outputs = decoder_session.run(decoder_inputs)?;

        // Extract and process logits
        let logits: ArrayViewD<f32> = outputs[0].try_extract_array()?;
        let next_token = argmax_next_token(&logits);
        tokens.push(next_token);

        if next_token == EOS_TOKEN_ID {
            break;
        }

        // Collect present key values returned from index [1..]
        let mut present_values = Vec::new();
        for idx in 1..outputs.len() {
            let present_kv: ArrayViewD<f32> = outputs[idx].try_extract_array()?;
            let owned_kv = present_kv.to_owned().into_dimensionality::<Ix4>()?;
            present_values.push(owned_kv);
        }

        // Update sequence input and KV caches for next generation step
        input_ids = Array2::from_shape_vec((1, 1), vec![next_token])?;
        past_key_values = update_kv_cache(
            past_key_values,
            present_values,
            &cache_keys,
            use_cache_branch,
        );
    }

    Ok(tokens)
}

pub fn get_transcription(audio_path: &str) -> Result<String, Box<dyn std::error::Error>> {
    use crate::download::get_path;

    // Check if required files exist
    let tok_p = get_path(TOKENIZER_PATH);
    let enc_p = get_path(ENCODER_PATH);
    let dec_p = get_path(DECODER_PATH);

    if !tok_p.exists() || !enc_p.exists() || !dec_p.exists() {
        return Err("Model files not found. Please run the `download` command first.".into());
    }

    println!("Initializing ONNX Runtime sessions...");
    let mut encoder_session = Session::builder()?.commit_from_file(&enc_p)?;
    let mut decoder_session = Session::builder()?.commit_from_file(&dec_p)?;

    println!("Loading tokenizer...");
    let tokenizer = Tokenizer::from_file(&tok_p).map_err(|e| e.to_string())?;

    println!("Loading WAV audio file: {} ...", audio_path);
    let audio_p = crate::download::resolve_user_path(audio_path);
    let audio_data = load_wav_file(&audio_p)?;

    println!("Transcribing speech...");
    let tokens = transcribe_speech(
        audio_data,
        &mut encoder_session,
        &mut decoder_session,
        &TINY_PARAMS,
    )?;

    decode_tokens_to_string(&tokens, &tokenizer)
}

pub fn run_transcription(audio_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let transcription = get_transcription(audio_path)?;
    println!(
        "\nTranscription Results:\n----------------------\n{}\n",
        transcription
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_wav_data_float() {
        let samples = WavSamples::Float(vec![0.5, -0.5, 0.0]);
        let res = process_wav_data(samples, 16000, 1, 32).unwrap();
        assert_eq!(res.shape(), &[1, 3]);
        assert_eq!(res[[0, 0]], 0.5);
    }

    #[test]
    fn test_process_wav_data_int() {
        let samples = WavSamples::Int(vec![16384, -16384, 0]);
        // 16-bit signed int max is 32768
        let res = process_wav_data(samples, 16000, 1, 16).unwrap();
        assert_eq!(res.shape(), &[1, 3]);
        assert_eq!(res[[0, 0]], 0.5);
    }

    #[test]
    fn test_process_wav_data_invalid_rate() {
        let samples = WavSamples::Float(vec![0.0]);
        let res = process_wav_data(samples, 44100, 1, 32);
        assert!(res.is_err());
    }

    #[test]
    fn test_get_cache_keys() {
        let params = ModelParams {
            num_layers: 2,
            num_heads: 4,
            head_dim: 8,
        };
        let keys = get_cache_keys(&params);
        assert_eq!(keys.len(), 8);
        assert_eq!(keys[0], "past_key_values.0.decoder.key");
        assert_eq!(keys[7], "past_key_values.1.encoder.value");
    }

    #[test]
    fn test_initialize_past_key_values() {
        let params = ModelParams {
            num_layers: 2,
            num_heads: 4,
            head_dim: 8,
        };
        let cache = initialize_past_key_values(&params);
        assert_eq!(cache.len(), 8);
        let sample = cache.get("past_key_values.0.decoder.key").unwrap();
        assert_eq!(sample.shape(), &[0, 4, 1, 8]);
    }

    #[test]
    fn test_argmax_next_token() {
        let data = vec![1.0, 2.0, 0.5, 0.1, 0.2, 0.9];
        let array = ndarray::Array3::from_shape_vec((1, 2, 3), data)
            .unwrap()
            .into_dyn();
        let next_token = argmax_next_token(&array.view());
        assert_eq!(next_token, 2);
    }

    #[test]
    fn test_update_kv_cache() {
        let params = ModelParams {
            num_layers: 1,
            num_heads: 2,
            head_dim: 4,
        };
        let keys = get_cache_keys(&params);

        let mut past = HashMap::new();
        for key in &keys {
            past.insert(key.clone(), Array4::zeros((1, 2, 1, 4)));
        }

        let present = vec![
            Array4::from_elem((2, 2, 1, 4), 1.0),
            Array4::from_elem((2, 2, 1, 4), 2.0),
            Array4::from_elem((2, 2, 1, 4), 3.0),
            Array4::from_elem((2, 2, 1, 4), 4.0),
        ];

        let updated = update_kv_cache(past.clone(), present.clone(), &keys, true);
        assert_eq!(
            updated.get("past_key_values.0.decoder.key").unwrap()[[0, 0, 0, 0]],
            1.0
        );
        assert_eq!(
            updated.get("past_key_values.0.encoder.key").unwrap()[[0, 0, 0, 0]],
            0.0
        );

        let updated_all = update_kv_cache(past, present, &keys, false);
        assert_eq!(
            updated_all.get("past_key_values.0.decoder.key").unwrap()[[0, 0, 0, 0]],
            1.0
        );
        assert_eq!(
            updated_all.get("past_key_values.0.encoder.key").unwrap()[[0, 0, 0, 0]],
            3.0
        );
    }

    #[test]
    #[ignore]
    fn test_integration_transcribe() {
        use crate::download::{download_models_pipeline, get_path};
        let tok_p = get_path(TOKENIZER_PATH);
        if !tok_p.exists() {
            download_models_pipeline().unwrap();
        }

        let wav_path = get_path("beckett.wav");
        assert!(
            wav_path.exists(),
            "Make sure beckett.wav is downloaded"
        );

        let mut encoder_session = Session::builder()
            .unwrap()
            .commit_from_file(get_path(ENCODER_PATH))
            .unwrap();
        let mut decoder_session = Session::builder()
            .unwrap()
            .commit_from_file(get_path(DECODER_PATH))
            .unwrap();
        let tokenizer = Tokenizer::from_file(&tok_p).unwrap();
        let audio_data = load_wav_file(&wav_path).unwrap();

        let tokens = transcribe_speech(
            audio_data,
            &mut encoder_session,
            &mut decoder_session,
            &TINY_PARAMS,
        )
        .unwrap();
        let text = decode_tokens_to_string(&tokens, &tokenizer).unwrap();
        assert!(text.contains("fail") || text.contains("tried"));
    }
}
