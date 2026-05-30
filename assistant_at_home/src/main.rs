mod audio;
mod download;
mod llm;
mod transcribe;
mod tts;

use download::download_models_pipeline;
use transcribe::run_transcription;

const MAX_TOKENS: usize = 1024;

pub const DEFAULT_PROFILE_DIR: &str = "/tmp";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileConfig {
    pub enabled: bool,
    pub dir: String,
}

impl Default for ProfileConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            dir: String::from(DEFAULT_PROFILE_DIR),
        }
    }
}

pub fn parse_profile_args(args: &[String]) -> (ProfileConfig, Vec<String>) {
    let mut enabled = false;
    let mut dir = String::from(DEFAULT_PROFILE_DIR);
    let mut clean_args = Vec::new();

    let mut i = 0;
    while i < args.len() {
        if args[i] == "--profile" {
            enabled = true;
            i += 1;
        } else if args[i] == "--profile-dir" {
            enabled = true;
            if i + 1 < args.len() {
                dir = args[i + 1].clone();
                i += 2;
            } else {
                i += 1;
            }
        } else if args[i].starts_with("--profile-dir=") {
            enabled = true;
            dir = args[i]["--profile-dir=".len()..].to_string();
            i += 1;
        } else {
            clean_args.push(args[i].clone());
            i += 1;
        }
    }

    (ProfileConfig { enabled, dir }, clean_args)
}

fn print_help() {
    println!(
        "assistant_at_home - Local speech-to-text voice assistant interface\n\n\
         Usage:\n  \
           assistant_at_home <COMMAND> [ARGS]\n\n\
         Commands:\n  \
           download          Download the tiny model and tokenizer ONNX assets from Hugging Face\n  \
           transcribe <WAV>  Transcribe a 16kHz mono WAV file to text\n  \
           pipeline <WAV>    Transcribe a 16kHz mono WAV file and pass the text to Qwen3 LLM\n  \
           live              Run the live voice assistant loop from the microphone\n"
    );
}

fn run_pipeline(audio_path: &str, profile_config: &ProfileConfig) -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Step 1: Transcribing Audio ---");
    let transcription = transcribe::get_transcription(audio_path, profile_config)?;
    println!("Transcribed text: \"{}\"\n", transcription);

    println!("--- Step 2: Running LLM (Qwen3) ---");
    let mut llm_pipeline = llm::LlmPipeline::load(profile_config)?;

    // Format the transcribed text using ChatML template for Qwen3
    let chat_prompt = format!(
        "<|im_start|>system\nYou are a helpful voice assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        transcription.trim()
    );

    println!("Generating LLM response...");
    let response = llm_pipeline.generate(&chat_prompt, 128)?;

    println!("\nLLM Response:\n----------------------\n{}\n", response);
    Ok(())
}

fn strip_think_tags(text: &str) -> String {
    let mut cleaned = String::new();
    let mut remaining = text;

    while let Some(start_idx) = remaining.find("<think>") {
        cleaned.push_str(&remaining[..start_idx]);
        if let Some(end_idx) = remaining[start_idx..].find("</think>") {
            remaining = &remaining[start_idx + end_idx + 8..];
        } else {
            remaining = "";
            break;
        }
    }
    cleaned.push_str(remaining);
    cleaned
}

fn run_live_assistant(profile_config: &ProfileConfig) -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Initializing Voice Assistant Modules ---");
    let mut llm_pipeline = llm::LlmPipeline::load(profile_config)?;
    let mut tts_pipeline = tts::TtsPipeline::load(profile_config)?;

    let input_path = download::get_path("assistant_input.wav");
    let output_path = download::get_path("assistant_output.wav");

    loop {
        println!("\n==================================================");
        println!("=== Ready for speech! ===");

        // Step 1: Record from microphone
        if let Err(e) = audio::record_audio_to_file(input_path.to_str().unwrap()) {
            eprintln!("Failed to record audio: {}", e);
            continue;
        }

        // Step 2: Speech to Text
        println!("Transcribing audio...");
        let transcription = match transcribe::get_transcription(input_path.to_str().unwrap(), profile_config) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Transcription failed: {}", e);
                continue;
            }
        };

        let trimmed = transcription.trim();
        if trimmed.is_empty() {
            println!("No speech detected. Please try again.");
            continue;
        }
        println!("You said: \"{}\"", trimmed);

        if trimmed.eq_ignore_ascii_case("exit") || trimmed.eq_ignore_ascii_case("quit") {
            println!("Exiting live assistant. Goodbye!");
            break;
        }

        // Step 3: Run LLM
        println!("Thinking...");
        let chat_prompt = format!(
            "<|im_start|>system\nYou are a helpful voice assistant. Keep your responses brief and concise, ideal for speech.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            trimmed
        );

        let response = match llm_pipeline.generate(&chat_prompt, MAX_TOKENS) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("LLM response generation failed: {}", e);
                continue;
            }
        };
        let response_clean = response.trim();
        println!("Assistant: \"{}\"", response_clean);

        let response_speak = strip_think_tags(response_clean);
        let response_speak_trimmed = response_speak.trim();

        // Step 4: Text to Speech
        println!("Synthesizing speech response...");
        let audio_data = match tts_pipeline.synthesize(response_speak_trimmed) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("TTS synthesis failed: {}", e);
                continue;
            }
        };

        if let Err(e) = tts::save_audio_to_wav(&audio_data, &output_path) {
            eprintln!("Failed to save output WAV: {}", e);
            continue;
        }

        // Step 5: Play response audio
        if let Err(e) = audio::play_audio_file(output_path.to_str().unwrap()) {
            eprintln!("Failed to play response audio: {}", e);
        }
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (profile_config, clean_args) = parse_profile_args(&args);

    if clean_args.len() < 2 {
        print_help();
        return;
    }

    if profile_config.enabled {
        let path = std::path::Path::new(&profile_config.dir);
        if !path.exists() {
            if let Err(e) = std::fs::create_dir_all(path) {
                eprintln!("Warning: Failed to create profiling directory {}: {}", profile_config.dir, e);
            }
        }
    }

    let command = &clean_args[1];
    let result = match command.as_str() {
        "download" => download_models_pipeline(),
        "transcribe" => {
            if clean_args.len() < 3 {
                println!("Error: transcribe requires a path to a WAV audio file.\n");
                print_help();
                return;
            }
            run_transcription(&clean_args[2], &profile_config)
        }
        "pipeline" => {
            if clean_args.len() < 3 {
                println!("Error: pipeline requires a path to a WAV audio file.\n");
                print_help();
                return;
            }
            run_pipeline(&clean_args[2], &profile_config)
        }
        "live" => run_live_assistant(&profile_config),
        _ => {
            println!("Error: Unknown command '{}'\n", command);
            print_help();
            return;
        }
    };

    if let Err(e) = result {
        eprintln!("Operation failed: {}", e);
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_think_tags_empty() {
        assert_eq!(strip_think_tags(""), "");
    }

    #[test]
    fn test_strip_think_tags_no_tags() {
        assert_eq!(strip_think_tags("hello world"), "hello world");
    }

    #[test]
    fn test_strip_think_tags_with_think() {
        assert_eq!(
            strip_think_tags("hello <think>let me think</think>world"),
            "hello world"
        );
    }

    #[test]
    fn test_strip_think_tags_multiple_think() {
        assert_eq!(
            strip_think_tags("a <think>b</think> c <think>d</think> e"),
            "a  c  e"
        );
    }

    #[test]
    fn test_strip_think_tags_unclosed() {
        assert_eq!(strip_think_tags("hello <think>world"), "hello ");
    }

    #[test]
    fn test_parse_profile_args_none() {
        let args = vec!["assistant_at_home".to_string(), "transcribe".to_string(), "file.wav".to_string()];
        let (config, clean) = parse_profile_args(&args);
        assert!(!config.enabled);
        assert_eq!(config.dir, DEFAULT_PROFILE_DIR);
        assert_eq!(clean, args);
    }

    #[test]
    fn test_parse_profile_args_enable() {
        let args = vec![
            "assistant_at_home".to_string(),
            "--profile".to_string(),
            "transcribe".to_string(),
            "file.wav".to_string(),
        ];
        let (config, clean) = parse_profile_args(&args);
        assert!(config.enabled);
        assert_eq!(config.dir, DEFAULT_PROFILE_DIR);
        assert_eq!(clean, vec!["assistant_at_home".to_string(), "transcribe".to_string(), "file.wav".to_string()]);
    }

    #[test]
    fn test_parse_profile_args_dir() {
        let args = vec![
            "assistant_at_home".to_string(),
            "transcribe".to_string(),
            "--profile-dir".to_string(),
            "/tmp/custom".to_string(),
            "file.wav".to_string(),
        ];
        let (config, clean) = parse_profile_args(&args);
        assert!(config.enabled);
        assert_eq!(config.dir, "/tmp/custom");
        assert_eq!(clean, vec!["assistant_at_home".to_string(), "transcribe".to_string(), "file.wav".to_string()]);
    }

    #[test]
    fn test_parse_profile_args_dir_equals() {
        let args = vec![
            "assistant_at_home".to_string(),
            "transcribe".to_string(),
            "--profile-dir=/tmp/custom2".to_string(),
            "file.wav".to_string(),
        ];
        let (config, clean) = parse_profile_args(&args);
        assert!(config.enabled);
        assert_eq!(config.dir, "/tmp/custom2");
        assert_eq!(clean, vec!["assistant_at_home".to_string(), "transcribe".to_string(), "file.wav".to_string()]);
    }
}
