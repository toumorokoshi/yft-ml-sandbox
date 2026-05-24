use std::env;
pub mod triton_ir;
pub mod onnx_compiler;
use std::error::Error;


use std::path::PathBuf;

// Constants
const HELP_TEXT: &str = "\
lquid - Deep Learning and Model Tool

USAGE:
    lquid <SUBCOMMAND> [ARGS...]

SUBCOMMANDS:
    read-onnx <PATH>                     Read and print the ONNX graph from the specified file path.
    compile-onnx <ONNX_PATH> <OUT_PATH>  Compile the first Gemm node in the ONNX model to Triton IR (.ttir).
    help                                 Print this help message.
";

#[derive(Debug, PartialEq, Eq)]
pub enum Command {
    ReadOnnx(PathBuf),
    CompileOnnx { onnx_path: PathBuf, output_path: PathBuf },
    Help,
    Unknown(String),
}

/// Pure parser: parses command line arguments into a Command enum.
/// Follows Rule 2: works purely on data structures.
pub fn parse_args<I>(mut args: I) -> Command
where
    I: Iterator<Item = String>,
{
    // Skip the executable path
    let _ = args.next();

    match args.next() {
        Some(ref subcmd) if subcmd == "read-onnx" => {
            match args.next() {
                Some(path) => Command::ReadOnnx(PathBuf::from(path)),
                None => Command::Unknown("Missing path for read-onnx subcommand".to_string()),
            }
        }
        Some(ref subcmd) if subcmd == "compile-onnx" => {
            match (args.next(), args.next()) {
                (Some(onnx_path), Some(output_path)) => Command::CompileOnnx {
                    onnx_path: PathBuf::from(onnx_path),
                    output_path: PathBuf::from(output_path),
                },
                _ => Command::Unknown("Missing args for compile-onnx subcommand. Expected: compile-onnx <ONNX_PATH> <OUT_PATH>".to_string()),
            }
        }
        Some(ref subcmd) if subcmd == "help" || subcmd == "-h" || subcmd == "--help" => Command::Help,
        Some(other) => Command::Unknown(format!("Unknown subcommand: {}", other)),
        None => Command::Help,
    }
}

/// Pure formatting function: formats a ModelProto graph to a string representation.
/// Follows Rule 2: works purely on data structures.
pub fn format_graph(model: &onnx_reader::ModelProto) -> String {
    if let Some(graph) = model.graph.as_ref() {
        format!("{:#?}", graph)
    } else {
        "No graph found in the model".to_string()
    }
}

/// Wrapper function for print IO: prints to stdout.
/// Follows Rule 2: IO wrapper.
pub fn print_stdout(text: &str) {
    println!("{}", text);
}

/// Wrapper function for file write IO.
/// Follows Rule 2: IO wrapper.
pub fn write_file_io(path: &std::path::Path, content: &str) -> std::io::Result<()> {
    std::fs::write(path, content)
}

/// Main pipeline: executes the parsed command.
/// Coordinates the pure functions and the IO wrappers.
pub fn execute_command(command: Command) -> Result<(), Box<dyn Error>> {
    match command {
        Command::ReadOnnx(path) => {
            let resolved_path = if !path.is_absolute() {
                if let Ok(workspace_dir) = env::var("BUILD_WORKSPACE_DIRECTORY") {
                    PathBuf::from(workspace_dir).join(&path)
                } else {
                    path
                }
            } else {
                path
            };
            let model = onnx_reader::load_model_from_file(resolved_path)?;
            let formatted = format_graph(&model);
            print_stdout(&formatted);
            Ok(())
        }
        Command::CompileOnnx { onnx_path, output_path } => {
            let resolved_onnx = if !onnx_path.is_absolute() {
                if let Ok(workspace_dir) = env::var("BUILD_WORKSPACE_DIRECTORY") {
                    PathBuf::from(workspace_dir).join(&onnx_path)
                } else {
                    onnx_path
                }
            } else {
                onnx_path
            };
            let resolved_output = if !output_path.is_absolute() {
                if let Ok(workspace_dir) = env::var("BUILD_WORKSPACE_DIRECTORY") {
                    PathBuf::from(workspace_dir).join(&output_path)
                } else {
                    output_path
                }
            } else {
                output_path
            };

            let model = onnx_reader::load_model_from_file(resolved_onnx)?;
            let params = onnx_compiler::extract_gemm_params(&model)?;
            let module = onnx_compiler::generate_gemm_module(&params);
            let formatted_ir = triton_ir::format_module(&module);
            
            write_file_io(&resolved_output, &formatted_ir)?;
            print_stdout(&format!("Successfully compiled ONNX Gemm to Triton IR: {:?}", resolved_output));
            Ok(())
        }
        Command::Help => {
            print_stdout(HELP_TEXT);
            Ok(())
        }
        Command::Unknown(err_msg) => {
            print_stdout(&format!("Error: {}\n", err_msg));
            print_stdout(HELP_TEXT);
            Err("Invalid command line arguments".into())
        }
    }
}


fn main() {
    let command = parse_args(env::args());
    if let Err(e) = execute_command(command) {
        eprintln!("Execution failed: {}", e);
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_args_help() {
        let args = vec!["lquid".to_string(), "help".to_string()];
        assert_eq!(parse_args(args.into_iter()), Command::Help);
    }

    #[test]
    fn test_parse_args_read_onnx() {
        let args = vec!["lquid".to_string(), "read-onnx".to_string(), "model.onnx".to_string()];
        assert_eq!(parse_args(args.into_iter()), Command::ReadOnnx(PathBuf::from("model.onnx")));
    }

    #[test]
    fn test_parse_args_read_onnx_missing_path() {
        let args = vec!["lquid".to_string(), "read-onnx".to_string()];
        match parse_args(args.into_iter()) {
            Command::Unknown(msg) => assert!(msg.contains("Missing path")),
            _ => panic!("Expected Command::Unknown"),
        }
    }

    #[test]
    fn test_parse_args_compile_onnx() {
        let args = vec![
            "lquid".to_string(),
            "compile-onnx".to_string(),
            "model.onnx".to_string(),
            "output.ttir".to_string(),
        ];
        assert_eq!(
            parse_args(args.into_iter()),
            Command::CompileOnnx {
                onnx_path: PathBuf::from("model.onnx"),
                output_path: PathBuf::from("output.ttir"),
            }
        );
    }

    #[test]
    fn test_format_graph_empty() {
        let model = onnx_reader::ModelProto::new();
        assert_eq!(format_graph(&model), "No graph found in the model");
    }

}
