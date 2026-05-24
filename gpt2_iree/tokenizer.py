from transformers import GPT2Tokenizer

# Constants (Rule 5)
TOKENIZER_NAME = "gpt2"

def get_tokenizer(name: str = TOKENIZER_NAME) -> GPT2Tokenizer:
    """Wrapper function to load the pre-trained tokenizer (IO)."""
    return GPT2Tokenizer.from_pretrained(name)

def encode_text(tokenizer: GPT2Tokenizer, text: str) -> list[int]:
    """Pure function: encodes input text to token IDs (works on data structures)."""
    return tokenizer.encode(text)

def decode_tokens(tokenizer: GPT2Tokenizer, token_ids: list[int]) -> str:
    """Pure function: decodes token IDs back to a string (works on data structures)."""
    return tokenizer.decode(token_ids)
