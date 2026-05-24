import unittest
import numpy as np

from gpt2_iree.inference import greedy_predict_next_token
from gpt2_iree.tokenizer import encode_text, decode_tokens
from gpt2_iree.main import generate_text_pipeline

class TestGPT2IREE(unittest.TestCase):
    def test_greedy_predict_next_token_pure(self) -> None:
        """Unit test: Test greedy prediction on logits data structure directly."""
        # Shape: (batch_size=1, sequence_length=2, vocab_size=5)
        # We look at the logits of the last token (index 1)
        mock_logits = np.array([
            [
                [1.0, 5.0, 2.0, 0.0, -1.0],  # first token (should be ignored)
                [0.1, -1.5, 3.4, 0.5, 2.1]   # last token (max is index 2, value 3.4)
            ]
        ], dtype=np.float32)
        
        predicted = greedy_predict_next_token(mock_logits)
        self.assertEqual(predicted, 2)

    def test_tokenizer_pure_wrappers(self) -> None:
        """Unit test: Test tokenizer wrappers using mock objects (no IO)."""
        class MockTokenizer:
            def encode(self, text: str) -> list[int]:
                return [42, 43]
            def decode(self, token_ids: list[int]) -> str:
                return "hello world"
                
        mock_tok = MockTokenizer()
        # Test encode_text
        self.assertEqual(encode_text(mock_tok, "test"), [42, 43])
        # Test decode_tokens
        self.assertEqual(decode_tokens(mock_tok, [42, 43]), "hello world")

    def test_pipeline_integration_io(self) -> None:
        """Single integration test verifying end-to-end compilation and inference (IO)."""
        prompt = "Once upon a"
        gen_len = 1
        
        # Run pipeline
        output_text = generate_text_pipeline(prompt, gen_len)
        
        # Verify output
        self.assertIsInstance(output_text, str)
        self.assertTrue(output_text.startswith(prompt))
        self.assertGreater(len(output_text), len(prompt))

if __name__ == "__main__":
    unittest.main()
