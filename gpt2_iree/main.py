import sys
import argparse
from typing import Sequence
import numpy as np

from gpt2_iree.tokenizer import get_tokenizer, encode_text, decode_tokens
from gpt2_iree.compile import get_model_path, compile_onnx_to_vmfb
from gpt2_iree.inference import load_iree_context, run_iree_inference, greedy_predict_next_token

# Constants (Rule 5)
DEFAULT_TEXT = "Here is some text to encode : Hello World"
DEFAULT_GEN_LENGTH = 5
VMFB_NAME = "gpt2_lm_head"

def generate_text_pipeline(
    text: str,
    gen_length: int,
    model_url: str | None = None
) -> str:
    """Orchestrates compilation and inference pipeline to generate text from prompt."""
    # 1. Load tokenizer (IO wrapper)
    print("Loading GPT-2 tokenizer...")
    tokenizer = get_tokenizer()
    
    # 2. Encode prompt (pure)
    token_ids = encode_text(tokenizer, text)
    print(f"Encoded prompt tokens: {token_ids}")
    
    # 3. Locate or download ONNX model (IO wrapper)
    if model_url:
        onnx_path = get_model_path(model_url=model_url)
    else:
        onnx_path = get_model_path()
        
    # 4. Compile ONNX model to IREE VMFB (IO wrapper)
    vmfb_path = compile_onnx_to_vmfb(onnx_path, VMFB_NAME)
    print(f"Model compiled to: {vmfb_path}")
    
    # 5. Load model into SystemContext (IO wrapper)
    ctx = load_iree_context(vmfb_path)
    
    # 6. Run generation loop (works on data structures inside)
    # We mutate the copy of tokens array in the loop to generate next tokens
    current_tokens = list(token_ids)
    
    print(f"Generating {gen_length} tokens...")
    for i in range(gen_length):
        # Convert input to numpy array shape (1, 1, len) of int64 to match the 3D ONNX export
        input_array = np.array([[current_tokens]], dtype=np.int64)
        
        # Run inference (IO wrapper)
        logits = run_iree_inference(ctx, input_array)
        
        # Predict next token (pure)
        next_token = greedy_predict_next_token(logits)
        
        # Append next token
        current_tokens.append(next_token)
        print(f"  Step {i+1}/{gen_length}: Generated token {next_token}")
        
    # 7. Decode generated tokens (pure)
    generated_text = decode_tokens(tokenizer, current_tokens)
    return generated_text

def parse_arguments(args: Sequence[str]) -> argparse.Namespace:
    """Helper to parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GPT-2 ONNX model using IREE.")
    parser.add_argument(
        "--text",
        type=str,
        default=DEFAULT_TEXT,
        help=f"Prompt text to generate from (default: '{DEFAULT_TEXT}')"
    )
    parser.add_argument(
        "--length",
        type=int,
        default=DEFAULT_GEN_LENGTH,
        help=f"Number of tokens to generate (default: {DEFAULT_GEN_LENGTH})"
    )
    parser.add_argument(
        "--model_url",
        type=str,
        default=None,
        help="Optional custom URL to download the GPT-2 ONNX model from"
    )
    return parser.parse_args(args)

def main(args: Sequence[str] = sys.argv[1:]) -> None:
    parsed_args = parse_arguments(args)
    
    print(f"Prompt: '{parsed_args.text}'")
    print(f"Target length: {parsed_args.length}")
    
    output_text = generate_text_pipeline(
        parsed_args.text,
        parsed_args.length,
        parsed_args.model_url
    )
    
    print("\n" + "=" * 40)
    print("GENERATED OUTPUT:")
    print("=" * 40)
    print(output_text)
    print("=" * 40)

if __name__ == "__main__":
    main()
