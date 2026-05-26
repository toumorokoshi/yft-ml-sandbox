mod download;
mod llm;
mod transcribe;

use download::download_models_pipeline;
use transcribe::run_transcription;

fn print_help() {
    println!(
        "assistant_at_home - Local speech-to-text voice assistant interface\n\n\
         Usage:\n  \
           assistant_at_home <COMMAND> [ARGS]\n\n\
         Commands:\n  \
           download          Download the tiny model and tokenizer ONNX assets from Hugging Face\n  \
           transcribe <WAV>  Transcribe a 16kHz mono WAV file to text\n  \
           pipeline <WAV>    Transcribe a 16kHz mono WAV file and pass the text to Qwen3 LLM\n"
    );
}

fn run_pipeline(audio_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Step 1: Transcribing Audio ---");
    let transcription = transcribe::get_transcription(audio_path)?;
    println!("Transcribed text: \"{}\"\n", transcription);

    println!("--- Step 2: Running LLM (Qwen3) ---");
    let mut llm_pipeline = llm::LlmPipeline::load()?;

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

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_help();
        return;
    }

    let command = &args[1];
    let result = match command.as_str() {
        "download" => download_models_pipeline(),
        "transcribe" => {
            if args.len() < 3 {
                println!("Error: transcribe requires a path to a WAV audio file.\n");
                print_help();
                return;
            }
            run_transcription(&args[2])
        }
        "pipeline" => {
            if args.len() < 3 {
                println!("Error: pipeline requires a path to a WAV audio file.\n");
                print_help();
                return;
            }
            run_pipeline(&args[2])
        }
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
