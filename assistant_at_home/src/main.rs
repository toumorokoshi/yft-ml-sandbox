mod download;
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
           transcribe <WAV>  Transcribe a 16kHz mono WAV file to text\n"
    );
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
