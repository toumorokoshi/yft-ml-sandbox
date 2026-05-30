use crate::download::{QWEN_MODEL_PATH, QWEN_TOKENIZER_PATH};
use ndarray::{Array2, Array4, ArrayView1, ArrayViewD, Axis, Ix1, Ix4};
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;
use std::collections::HashMap;
use tokenizers::Tokenizer;

pub struct LlmPipeline {
    session: Session,
    tokenizer: Tokenizer,
    num_layers: usize,
    eos_token_id: i64,
    profiling_enabled: bool,
}

/// Helper function to dynamically count the number of layers in the model.
pub fn get_num_layers(session: &Session) -> usize {
    let mut num_layers = 0;
    for input in session.inputs().iter() {
        let name = input.name();
        if name.starts_with("past_key_values.") {
            let parts: Vec<&str> = name.split('.').collect();
            if let Some(idx) = parts.get(1)
                .and_then(|p| p.parse::<usize>().ok())
                .filter(|&idx| idx >= num_layers)
            {
                num_layers = idx + 1;
            }
        }
    }
    num_layers
}

/// Finds the argmax index along the logits vector.
pub fn argmax_next_token(logits_1d: &ArrayView1<f32>) -> usize {
    let mut max_idx = 0;
    let mut max_val = logits_1d[0];
    for (idx, &val) in logits_1d.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = idx;
        }
    }
    max_idx
}

/// Executes the core LLM token generation loop.
pub fn generate_tokens(
    session: &mut Session,
    prompt_tokens: &[u32],
    num_layers: usize,
    eos_token_id: i64,
    max_new_tokens: usize,
) -> Result<Vec<i64>, Box<dyn std::error::Error>> {
    let prompt_len = prompt_tokens.len();
    let mut generated = Vec::new();

    // Initialize KV cache
    // Shape: [batch_size, num_heads, past_sequence_length, head_dim]
    // Qwen3 q4 model has 8 attention heads and head dimension of 128
    let mut past_key_values: HashMap<String, Array4<f32>> = HashMap::new();
    for l in 0..num_layers {
        past_key_values.insert(
            format!("past_key_values.{}.key", l),
            Array4::zeros((1, 8, 0, 128)),
        );
        past_key_values.insert(
            format!("past_key_values.{}.value", l),
            Array4::zeros((1, 8, 0, 128)),
        );
    }

    let mut current_input_ids: Vec<i64> = prompt_tokens.iter().map(|&t| t as i64).collect();

    for i in 0..max_new_tokens {
        let mut inputs: HashMap<String, SessionInputValue> = HashMap::new();

        let seq_len = current_input_ids.len();
        let input_ids_arr = Array2::from_shape_vec((1, seq_len), current_input_ids.clone())?;

        let position_ids_vec: Vec<i64> = if i == 0 {
            (0..seq_len as i64).collect()
        } else {
            vec![(prompt_len + i - 1) as i64]
        };
        let position_ids_arr =
            Array2::from_shape_vec((1, position_ids_vec.len()), position_ids_vec)?;

        let attention_mask_len = prompt_len + i;
        let attention_mask_vec: Vec<i64> = vec![1; attention_mask_len];
        let attention_mask_arr =
            Array2::from_shape_vec((1, attention_mask_len), attention_mask_vec)?;

        inputs.insert(
            "input_ids".to_string(),
            Tensor::from_array(input_ids_arr)?.into(),
        );
        inputs.insert(
            "position_ids".to_string(),
            Tensor::from_array(position_ids_arr)?.into(),
        );
        inputs.insert(
            "attention_mask".to_string(),
            Tensor::from_array(attention_mask_arr)?.into(),
        );

        for l in 0..num_layers {
            let key_name = format!("past_key_values.{}.key", l);
            let val_name = format!("past_key_values.{}.value", l);
            inputs.insert(
                key_name.clone(),
                Tensor::from_array(past_key_values.get(&key_name).unwrap().clone())?.into(),
            );
            inputs.insert(
                val_name.clone(),
                Tensor::from_array(past_key_values.get(&val_name).unwrap().clone())?.into(),
            );
        }

        let outputs = session.run(inputs)?;

        let logits: ArrayViewD<f32> = outputs[0].try_extract_array()?;
        let logits_shape = logits.shape();
        let logits_seq_len = logits_shape[1];

        let binding = logits.index_axis(Axis(0), 0);
        let last_token_logits = binding.index_axis(Axis(0), logits_seq_len - 1);
        let last_token_logits_1d = last_token_logits.into_dimensionality::<Ix1>()?;

        let next_token_id = argmax_next_token(&last_token_logits_1d) as i64;

        if next_token_id == eos_token_id {
            break;
        }

        generated.push(next_token_id);

        for l in 0..num_layers {
            let present_key_idx = 2 * l + 1;
            let present_val_idx = 2 * l + 2;

            let present_key: ArrayViewD<f32> = outputs[present_key_idx].try_extract_array()?;
            let present_val: ArrayViewD<f32> = outputs[present_val_idx].try_extract_array()?;

            past_key_values.insert(
                format!("past_key_values.{}.key", l),
                present_key.to_owned().into_dimensionality::<Ix4>()?,
            );
            past_key_values.insert(
                format!("past_key_values.{}.value", l),
                present_val.to_owned().into_dimensionality::<Ix4>()?,
            );
        }

        current_input_ids = vec![next_token_id];
    }

    Ok(generated)
}

impl LlmPipeline {
    pub fn load(profile_config: &crate::ProfileConfig) -> Result<Self, Box<dyn std::error::Error>> {
        use crate::download::get_path;
        let tok_path = get_path(QWEN_TOKENIZER_PATH);
        let model_path = get_path(QWEN_MODEL_PATH);

        if !tok_path.exists() || !model_path.exists() {
            return Err(
                "Qwen3 model assets not found. Please run the `download` command first.".into(),
            );
        }

        println!("Initializing LLM ONNX Runtime session...");
        let mut builder = Session::builder()?;
        if profile_config.enabled {
            builder = builder.with_profiling(format!("{}/qwen_llm", profile_config.dir))?;
        }
        let session = builder.commit_from_file(&model_path)?;

        println!("Loading LLM tokenizer...");
        let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| e.to_string())?;

        let num_layers = get_num_layers(&session);
        println!("Detected {} layers in LLM model.", num_layers);

        let eos_token_id = tokenizer
            .token_to_id("<|im_end|>")
            .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
            .map(|id| id as i64)
            .unwrap_or(151643);

        Ok(Self {
            session,
            tokenizer,
            num_layers,
            eos_token_id,
            profiling_enabled: profile_config.enabled,
        })
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| e.to_string())?;
        let prompt_tokens = encoding.get_ids();

        let generated_ids = generate_tokens(
            &mut self.session,
            prompt_tokens,
            self.num_layers,
            self.eos_token_id,
            max_new_tokens,
        )?;

        let u32_generated: Vec<u32> = generated_ids.iter().map(|&id| id as u32).collect();
        let text = self
            .tokenizer
            .decode(&u32_generated, true)
            .map_err(|e| e.to_string())?;
        Ok(text)
    }
}

impl Drop for LlmPipeline {
    fn drop(&mut self) {
        if self.profiling_enabled {
            match self.session.end_profiling() {
                Ok(path) => println!("LLM profiling file written to: {}", path),
                Err(e) => eprintln!("Failed to end LLM profiling: {}", e),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argmax_next_token() {
        let logits = ndarray::Array1::from_vec(vec![-1.0, 0.0, 5.5, 2.0, -10.0]);
        let next_tok = argmax_next_token(&logits.view());
        assert_eq!(next_tok, 2);
    }

    #[test]
    #[ignore]
    fn test_integration_llm_generation() {
        let mut pipeline = LlmPipeline::load(&crate::ProfileConfig::default()).unwrap();
        let prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHow are you?<|im_end|>\n<|im_start|>assistant\n";
        let response = pipeline.generate(prompt, 30).unwrap();
        println!("Prompt: {}\nResponse: {}", prompt, response);
        assert!(!response.is_empty());
    }
}
