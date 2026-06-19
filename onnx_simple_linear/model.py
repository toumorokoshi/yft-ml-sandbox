import io
import torch
import torch.nn as nn

# Constants
INPUT_DIM = 2**12
OUTPUT_DIM = 2**12

class SimpleLinearModel(nn.Module):
    """A simple linear model for ONNX export."""
    def __init__(self, input_dim: int = INPUT_DIM, output_dim: int = OUTPUT_DIM) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

def create_model(input_dim: int = INPUT_DIM, output_dim: int = OUTPUT_DIM) -> nn.Module:
    """Creates and returns the linear model instance."""
    return SimpleLinearModel(input_dim, output_dim)

def generate_dummy_input(batch_size: int = 1, input_dim: int = INPUT_DIM) -> torch.Tensor:
    """Generates dummy input tensor for ONNX export."""
    return torch.randn(batch_size, input_dim)

def export_to_onnx_bytes(model: nn.Module, dummy_input: torch.Tensor) -> bytes:
    """Exports a PyTorch model to ONNX format in-memory, returning the bytes."""
    model.eval()
    buffer = io.BytesIO()
    torch.onnx.export(
        model,
        dummy_input,
        buffer,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
    )
    return buffer.getvalue()
