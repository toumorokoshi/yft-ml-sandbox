mod audio;
mod download;
mod llm;
mod transcribe;
mod tts;

use clap::{Parser, Subcommand};
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

#[derive(Parser, Debug)]
#[command(name = "assistant_at_home", version = "0.1.0", about = "Local speech-to-text voice assistant interface", long_about = None)]
pub struct Cli {
    #[arg(long, global = true, help = "Turn on ONNX session profiling when running each model")]
    pub profile: bool,

    #[arg(long, global = true, help = "The directory where ONNX session profiling files are written")]
    pub profile_dir: Option<String>,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug, Clone, PartialEq, Eq)]
pub enum Commands {
    #[command(about = "Download the tiny model and tokenizer ONNX assets from Hugging Face")]
    Download,

    #[command(about = "Transcribe a 16kHz mono WAV file to text")]
    Transcribe {
        #[arg(help = "Path to the 16kHz mono WAV file")]
        wav_path: String,
    },

    #[command(about = "Transcribe a 16kHz mono WAV file and pass the text to Qwen3 LLM")]
    Pipeline {
        #[arg(help = "Path to the 16kHz mono WAV file")]
        wav_path: String,
    },

    #[command(about = "Run the live voice assistant loop from the microphone")]
    Live {
        #[arg(long, help = "Path to a WAV file to use instead of recording from the microphone")]
        input: Option<String>,
    },
}

fn process_audio_flow(
    audio_path: &str,
    llm_pipeline: &mut llm::LlmPipeline,
    tts_pipeline: &mut tts::TtsPipeline,
    profile_config: &ProfileConfig,
) -> Result<Option<(String, String, Vec<f32>)>, Box<dyn std::error::Error>> {
    println!("Transcribing audio...");
    let transcription = transcribe::get_transcription(audio_path, profile_config)?;
    let trimmed = transcription.trim().to_string();
    if trimmed.is_empty() {
        return Ok(None);
    }
    println!("You said: \"{}\"", trimmed);

    if trimmed.eq_ignore_ascii_case("exit") || trimmed.eq_ignore_ascii_case("quit") {
        return Ok(Some((trimmed, String::new(), Vec::new())));
    }

    println!("Thinking...");
    let chat_prompt = format!(
        "<|im_start|>system\nYou are a helpful voice assistant. Keep your responses brief and concise, ideal for speech.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        trimmed
    );

    let response = llm_pipeline.generate(&chat_prompt, MAX_TOKENS)?;
    let response_clean = response.trim().to_string();
    println!("Assistant: \"{}\"", response_clean);

    let response_speak = strip_think_tags(&response_clean);
    let response_speak_trimmed = response_speak.trim();

    println!("Synthesizing speech response...");
    let audio_data = tts_pipeline.synthesize(response_speak_trimmed)?;

    Ok(Some((trimmed, response_clean, audio_data)))
}

fn run_pipeline(audio_path: &str, profile_config: &ProfileConfig) -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Initializing Voice Assistant Modules ---");
    let mut llm_pipeline = llm::LlmPipeline::load(profile_config)?;
    let mut tts_pipeline = tts::TtsPipeline::load(profile_config)?;
    let output_path = download::get_path("assistant_output.wav");

    if let Some((_transcription, _response, audio_data)) = process_audio_flow(
        audio_path,
        &mut llm_pipeline,
        &mut tts_pipeline,
        profile_config,
    )? {
        if !audio_data.is_empty() {
            tts::save_audio_to_wav(&audio_data, &output_path)?;
            println!("Saved synthesized response to {}", output_path.display());
        }
    } else {
        println!("No speech detected.");
    }
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

fn run_live_assistant(
    input_wav: Option<&str>,
    profile_config: &ProfileConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Initializing Voice Assistant Modules ---");
    let mut llm_pipeline = llm::LlmPipeline::load(profile_config)?;
    let mut tts_pipeline = tts::TtsPipeline::load(profile_config)?;

    let input_path = download::get_path("assistant_input.wav");
    let output_path = download::get_path("assistant_output.wav");

    if let Some(wav_file) = input_wav {
        println!("\n==================================================");
        println!("=== Processing input WAV file: {} ===", wav_file);

        let resolved_input = download::resolve_user_path(wav_file);
        if let Some((_transcription, _response, audio_data)) = process_audio_flow(
            resolved_input.to_str().unwrap(),
            &mut llm_pipeline,
            &mut tts_pipeline,
            profile_config,
        )? {
            if !audio_data.is_empty() {
                tts::save_audio_to_wav(&audio_data, &output_path)?;
                if let Err(e) = audio::play_audio_file(output_path.to_str().unwrap()) {
                    eprintln!("Failed to play response audio: {}", e);
                }
            }
        }
    } else {
        loop {
            println!("\n==================================================");
            println!("=== Ready for speech! ===");

            // Step 1: Record from microphone
            if let Err(e) = audio::record_audio_to_file(input_path.to_str().unwrap()) {
                eprintln!("Failed to record audio: {}", e);
                continue;
            }

            match process_audio_flow(
                input_path.to_str().unwrap(),
                &mut llm_pipeline,
                &mut tts_pipeline,
                profile_config,
            ) {
                Ok(Some((trimmed, _response, audio_data))) => {
                    if trimmed.eq_ignore_ascii_case("exit") || trimmed.eq_ignore_ascii_case("quit") {
                        println!("Exiting live assistant. Goodbye!");
                        break;
                    }
                    if !audio_data.is_empty() {
                        if let Err(e) = tts::save_audio_to_wav(&audio_data, &output_path) {
                            eprintln!("Failed to save output WAV: {}", e);
                            continue;
                        }
                        if let Err(e) = audio::play_audio_file(output_path.to_str().unwrap()) {
                            eprintln!("Failed to play response audio: {}", e);
                        }
                    }
                }
                Ok(None) => {
                    println!("No speech detected. Please try again.");
                }
                Err(e) => {
                    eprintln!("Pipeline processing failed: {}", e);
                }
            }
        }
    }
    Ok(())
}

fn main() {
    let cli = Cli::parse();

    let profile_config = ProfileConfig {
        enabled: cli.profile || cli.profile_dir.is_some(),
        dir: cli.profile_dir.unwrap_or_else(|| String::from(DEFAULT_PROFILE_DIR)),
    };

    if profile_config.enabled {
        let path = std::path::Path::new(&profile_config.dir);
        if !path.exists() {
            if let Err(e) = std::fs::create_dir_all(path) {
                eprintln!("Warning: Failed to create profiling directory {}: {}", profile_config.dir, e);
            }
        }
    }

    let result = match cli.command {
        Commands::Download => download_models_pipeline(),
        Commands::Transcribe { wav_path } => {
            run_transcription(&wav_path, &profile_config)
        }
        Commands::Pipeline { wav_path } => {
            run_pipeline(&wav_path, &profile_config)
        }
        Commands::Live { input } => run_live_assistant(input.as_deref(), &profile_config),
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
        let args = vec!["assistant_at_home", "transcribe", "file.wav"];
        let cli = Cli::try_parse_from(args).unwrap();
        assert!(!cli.profile);
        assert_eq!(cli.profile_dir, None);
        assert_eq!(cli.command, Commands::Transcribe { wav_path: "file.wav".to_string() });
    }

    #[test]
    fn test_parse_profile_args_enable() {
        let args = vec![
            "assistant_at_home",
            "--profile",
            "transcribe",
            "file.wav",
        ];
        let cli = Cli::try_parse_from(args).unwrap();
        assert!(cli.profile);
        assert_eq!(cli.profile_dir, None);
        assert_eq!(cli.command, Commands::Transcribe { wav_path: "file.wav".to_string() });
    }

    #[test]
    fn test_parse_profile_args_dir() {
        let args = vec![
            "assistant_at_home",
            "transcribe",
            "file.wav",
            "--profile-dir",
            "/tmp/custom",
        ];
        let cli = Cli::try_parse_from(args).unwrap();
        assert!(!cli.profile); // profile is false, but profile_dir is some
        assert_eq!(cli.profile_dir, Some("/tmp/custom".to_string()));
        assert_eq!(cli.command, Commands::Transcribe { wav_path: "file.wav".to_string() });
    }

    #[test]
    fn test_parse_live_default() {
        let args = vec!["assistant_at_home", "live"];
        let cli = Cli::try_parse_from(args).unwrap();
        assert_eq!(cli.command, Commands::Live { input: None });
    }

    #[test]
    fn test_parse_live_with_input() {
        let args = vec!["assistant_at_home", "live", "--input", "test.wav"];
        let cli = Cli::try_parse_from(args).unwrap();
        assert_eq!(cli.command, Commands::Live { input: Some("test.wav".to_string()) });
    }
}
