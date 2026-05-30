use crate::download::{get_path, KOKORO_MODEL_PATH, KOKORO_TOKENIZER_PATH, KOKORO_VOICE_PATH};
use ndarray::{Array1, Array2};
use ort::session::Session;
use ort::value::Tensor;
use std::collections::HashMap;
use std::path::Path;

pub struct TtsPipeline {
    session: Session,
    vocab: HashMap<char, i64>,
    voices_data: Vec<f32>,
}

/// Loads a binary voice style file into a flat vector of floats.
pub fn load_voice_data<P: AsRef<Path>>(path: P) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err("Voice style binary file size must be a multiple of 4 bytes".into());
    }
    let floats = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    Ok(floats)
}

/// Unescapes standard JSON string characters, including unicode escapes.
pub fn unescape(s: &str) -> String {
    let mut chars = s.chars().peekable();
    let mut result = String::new();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(&next) = chars.peek() {
                match next {
                    '"' | '\\' | '/' => {
                        result.push(next);
                        chars.next();
                    }
                    'b' => { result.push('\u{0008}'); chars.next(); }
                    'f' => { result.push('\u{000c}'); chars.next(); }
                    'n' => { result.push('\n'); chars.next(); }
                    'r' => { result.push('\r'); chars.next(); }
                    't' => { result.push('\t'); chars.next(); }
                    'u' => {
                        chars.next(); // consume 'u'
                        let mut hex = String::new();
                        for _ in 0..4 {
                            if let Some(h) = chars.next() {
                                hex.push(h);
                            }
                        }
                        if let Ok(code) = u32::from_str_radix(&hex, 16) {
                            if let Some(uc) = char::from_u32(code) {
                                result.push(uc);
                            }
                        }
                    }
                    _ => {
                        result.push(c);
                    }
                }
            } else {
                result.push(c);
            }
        } else {
            result.push(c);
        }
    }
    result
}

/// Parses the vocab block from tokenizer.json.
pub fn parse_vocab(json_str: &str) -> HashMap<char, i64> {
    let mut vocab = HashMap::new();
    if let Some(vocab_start) = json_str.find("\"vocab\": {") {
        let block = &json_str[vocab_start..];
        for line in block.lines() {
            let trimmed = line.trim();
            if trimmed.contains('}') {
                if trimmed.contains("vocab") { continue; }
                break;
            }
            if let Some(last_colon) = trimmed.rfind(':') {
                let key_part = trimmed[..last_colon].trim();
                let val_part = trimmed[last_colon + 1..].trim().trim_end_matches(',');
                
                if key_part.starts_with('"') && key_part.ends_with('"') && key_part.len() >= 2 {
                    let key_str = &key_part[1..key_part.len() - 1];
                    let unescaped = unescape(key_str);
                    if let Some(c) = unescaped.chars().next() {
                        if let Ok(val) = val_part.parse::<i64>() {
                            vocab.insert(c, val);
                        }
                    }
                }
            }
        }
    }
    vocab
}

/// Simple rule-based English grapheme-to-phoneme mapping fallback.
fn phonemize_word(word: &str) -> String {
    let mut w = word.to_string();
    
    // Apply digraph replacements
    w = w.replace("th", "ð");
    w = w.replace("sh", "ʃ");
    w = w.replace("ch", "tʃ");
    w = w.replace("ng", "ŋ");
    w = w.replace("ck", "k");
    w = w.replace("ee", "iː");
    w = w.replace("oo", "uː");
    w = w.replace("oa", "oʊ");
    w = w.replace("ai", "eɪ");
    w = w.replace("ay", "eɪ");
    w = w.replace("ou", "aʊ");
    w = w.replace("ow", "aʊ");
    
    let chars: Vec<char> = w.chars().collect();
    let mut phonemes = String::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            'c' => {
                if i + 1 < chars.len() && "eiy".contains(chars[i + 1]) {
                    phonemes.push('s');
                } else {
                    phonemes.push('k');
                }
            }
            'q' => phonemes.push('k'),
            'x' => phonemes.push_str("ks"),
            'j' => phonemes.push_str("dʒ"),
            'y' => {
                if i == chars.len() - 1 {
                    phonemes.push_str("aɪ");
                } else {
                    phonemes.push('j');
                }
            }
            'a' => phonemes.push('æ'),
            'e' => phonemes.push('ɛ'),
            'i' => phonemes.push('ɪ'),
            'o' => phonemes.push('ɑ'),
            'u' => phonemes.push('ʌ'),
            other => phonemes.push(other),
        }
        i += 1;
    }
    phonemes
}

/// Converts arbitrary English text to a phoneme representation suitable for Kokoro TTS.
pub fn phonemize(text: &str) -> String {
    let text = text.to_lowercase();
    
    // Dictionary of common conversational assistant words mapped to accurate IPA phonemes
    let dict = [
        ("hello", "həloʊ"),
        ("hi", "haɪ"),
        ("how", "haʊ"),
        ("are", "ɑː"),
        ("you", "juː"),
        ("i", "aɪ"),
        ("am", "æm"),
        ("a", "ə"),
        ("helpful", "hɛlpfʊl"),
        ("assistant", "əsɪstənt"),
        ("today", "tədeɪ"),
        ("yes", "jɛs"),
        ("no", "noʊ"),
        ("what", "wɑːt"),
        ("is", "ɪz"),
        ("the", "ðə"),
        ("good", "ɡʊd"),
        ("fine", "faɪn"),
        ("ok", "oʊkeɪ"),
        ("okay", "oʊkeɪ"),
        ("help", "hɛlp"),
        ("can", "kæn"),
        ("do", "duː"),
        ("sorry", "sɑːri"),
        ("thank", "θæŋk"),
        ("thanks", "θæŋks"),
        ("my", "maɪ"),
        ("name", "neɪm"),
        ("nice", "naɪs"),
        ("to", "tuː"),
        ("meet", "miːt"),
        ("it", "ɪt"),
        ("this", "ðɪs"),
        ("that", "ðæt"),
        ("here", "hɪə"),
        ("there", "ðɛə"),
        ("we", "wiː"),
        ("they", "ðeɪ"),
        ("he", "hiː"),
        ("she", "ʃiː"),
        ("was", "wɑːz"),
        ("were", "wɜː"),
        ("will", "wɪl"),
        ("be", "biː"),
        ("have", "hæv"),
        ("has", "hæz"),
        ("had", "hæd"),
        ("not", "nɑːt"),
        ("but", "bʌt"),
        ("or", "ɔː"),
        ("and", "ænd"),
        ("of", "ʌv"),
        ("for", "fɔː"),
        ("with", "wɪð"),
        ("about", "əbaʊt"),
        ("would", "wʊd"),
        ("could", "kʊd"),
        ("should", "ʃʊd"),
        ("some", "sʌm"),
        ("any", "ɛni"),
        ("all", "ɔːl"),
        ("one", "wʌn"),
        ("two", "tuː"),
        ("three", "θriː"),
        ("four", "fɔː"),
        ("five", "faɪv"),
        ("great", "ɡreɪt"),
        ("well", "wɛl"),
        ("time", "taɪm"),
        ("day", "deɪ"),
        ("people", "piːpəl"),
        ("work", "wɜːk"),
        ("make", "meɪk"),
        ("know", "noʊ"),
        ("think", "θɪŋk"),
        ("see", "siː"),
        ("come", "kʌm"),
        ("give", "ɡɪv"),
        ("want", "wɑːnt"),
        ("go", "ɡoʊ"),
        ("way", "weɪ"),
        ("look", "lʊk"),
        ("first", "fɜːst"),
        ("new", "njuː"),
        ("use", "juːz"),
        ("more", "mɔː"),
    ];

    let mut result = String::new();
    let mut word = String::new();
    
    for c in text.chars() {
        if c.is_alphabetic() {
            word.push(c);
        } else {
            if !word.is_empty() {
                let mut mapped = false;
                for (k, v) in dict.iter() {
                    if *k == word {
                        result.push_str(v);
                        mapped = true;
                        break;
                    }
                }
                if !mapped {
                    result.push_str(&phonemize_word(&word));
                }
                word.clear();
            }
            if ";:,.!?¡¿—…\"«»“” ".contains(c) {
                result.push(c);
            }
        }
    }
    
    if !word.is_empty() {
        let mut mapped = false;
        for (k, v) in dict.iter() {
            if *k == word {
                result.push_str(v);
                mapped = true;
                break;
            }
        }
        if !mapped {
            result.push_str(&phonemize_word(&word));
        }
    }
    
    result
}

impl TtsPipeline {
    /// Loads the Kokoro TTS ONNX model, tokenizer, and voice style embedding.
    pub fn load() -> Result<Self, Box<dyn std::error::Error>> {
        let model_path = get_path(KOKORO_MODEL_PATH);
        let tok_path = get_path(KOKORO_TOKENIZER_PATH);
        let voice_path = get_path(KOKORO_VOICE_PATH);

        if !model_path.exists() || !tok_path.exists() || !voice_path.exists() {
            return Err("Kokoro TTS assets not found. Please run the `download` command first.".into());
        }

        println!("Initializing Kokoro TTS ONNX Runtime session...");
        let session = Session::builder()?.commit_from_file(&model_path)?;

        println!("Loading Kokoro TTS tokenizer configuration...");
        let tok_json = std::fs::read_to_string(&tok_path)?;
        let vocab = parse_vocab(&tok_json);
        println!("Parsed {} tokens from vocabulary.", vocab.len());

        println!("Loading voice style data...");
        let voices_data = load_voice_data(&voice_path)?;

        Ok(Self {
            session,
            vocab,
            voices_data,
        })
    }

    /// Synthesizes English text into a float32 raw PCM audio buffer at 24kHz.
    pub fn synthesize(&mut self, text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let phonemes = phonemize(text);
        println!("Phonemes: \"{}\"", phonemes);

        // Map phonemes to token IDs using the parsed vocabulary
        let mut input_ids_vec = vec![0];
        for c in phonemes.chars() {
            if let Some(&id) = self.vocab.get(&c) {
                input_ids_vec.push(id);
            }
        }
        input_ids_vec.push(0);

        let seq_len = input_ids_vec.len();
        let input_ids_array = Array2::from_shape_vec((1, seq_len), input_ids_vec)?;

        // Select the style vector based on the phoneme token sequence length
        let voice_idx = seq_len.min(511);
        let offset = voice_idx * 256;
        let mut style_vec = vec![0.0f32; 256];
        if offset + 256 <= self.voices_data.len() {
            style_vec.copy_from_slice(&self.voices_data[offset..offset + 256]);
        }

        let speed_arr = Array1::from_vec(vec![1.0f32]);

        // Try [1, 256] style tensor shape first
        let first_run = {
            if let Ok(style_array_2d) = Array2::from_shape_vec((1, 256), style_vec.clone()) {
                let run_result = self.session.run(ort::inputs![
                    "input_ids" => Tensor::from_array(input_ids_array.clone())?,
                    "style" => Tensor::from_array(style_array_2d)?,
                    "speed" => Tensor::from_array(speed_arr.clone())?,
                ]);
                if let Ok(outputs) = run_result {
                    if let Ok(audio) = outputs[0].try_extract_array() {
                        Some(audio.iter().cloned().collect::<Vec<f32>>())
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        };

        let samples = match first_run {
            Some(s) => s,
            None => {
                // If it fails, fallback to [1, 1, 256] shape
                let style_array_3d = ndarray::Array3::from_shape_vec((1, 1, 256), style_vec)?;
                let outputs = self.session.run(ort::inputs![
                    "input_ids" => Tensor::from_array(input_ids_array)?,
                    "style" => Tensor::from_array(style_array_3d)?,
                    "speed" => Tensor::from_array(speed_arr)?,
                ])?;
                let audio: ndarray::ArrayViewD<f32> = outputs[0].try_extract_array()?;
                audio.iter().cloned().collect::<Vec<f32>>()
            }
        };

        Ok(samples)
    }
}

/// Helper function to save a float32 audio buffer as a 16-bit PCM WAV file at 24kHz.
pub fn save_audio_to_wav(samples: &[f32], path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 24000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &sample in samples {
        let amplitude = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
        writer.write_sample(amplitude)?;
    }
    writer.finalize()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unescape_quotes() {
        let escaped = "\\\"";
        let unescaped = unescape(escaped);
        assert_eq!(unescaped, "\"");
    }

    #[test]
    fn test_parse_vocab_basic() {
        let json_str = r#"
        {
            "model": {
                "vocab": {
                    "$": 0,
                    ";": 1,
                    ":": 2,
                    "a": 43,
                    "\"": 11,
                    "\u0251": 69
                }
            }
        }
        "#;
        let vocab = parse_vocab(json_str);
        assert_eq!(vocab.get(&'$'), Some(&0));
        assert_eq!(vocab.get(&';'), Some(&1));
        assert_eq!(vocab.get(&':'), Some(&2));
        assert_eq!(vocab.get(&'a'), Some(&43));
        assert_eq!(vocab.get(&'"'), Some(&11));
        assert_eq!(vocab.get(&'ɑ'), Some(&69));
    }

    #[test]
    fn test_phonemize_common_words() {
        let res = phonemize("Hello hi");
        assert_eq!(res, "həloʊ haɪ");
    }

    #[test]
    fn test_phonemize_fallback() {
        let res = phonemize("this shin");
        assert_eq!(res, "ðɪs ʃɪn");
    }

    #[test]
    #[ignore]
    fn test_integration_tts_synthesis() {
        let mut pipeline = TtsPipeline::load().unwrap();
        let samples = pipeline.synthesize("Hello, how are you?").unwrap();
        assert!(!samples.is_empty());
        let output_path = get_path("test_out.wav");
        save_audio_to_wav(&samples, &output_path).unwrap();
        assert!(output_path.exists());
        if output_path.exists() {
            let _ = std::fs::remove_file(output_path);
        }
    }
}
