import io
import os
import sys
import argparse
import torch
import torch.nn as nn
import asciichartpy
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.transforms import v2
from torch.utils.data import DataLoader

from alexnet_core.model import NeuralNetwork, IMAGE_HEIGHT, IMAGE_WIDTH, SIZE, BATCH_SIZE
from alexnet_core.recipe import train, test, TRAIN_TRANSFORM, TEST_TRANSFORM

# Constants
NUM_CLASSIFICATIONS = 10
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DATA_ROOT = "data"
DEFAULT_EPOCHS = 60
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005


def get_model_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Gets the state dict from the model (Inner function working on data structures)."""
    return model.state_dict()


def save_state_dict_to_file(state_dict: dict[str, torch.Tensor], file_path: str) -> None:
    """Saves the state dict to a file (IO Wrapper)."""
    torch.save(state_dict, file_path)


def save_model_state_to_file(model: nn.Module, file_path: str) -> None:
    """Gets the state dict and saves it to a file (IO Wrapper)."""
    state_dict = get_model_state_dict(model)
    save_state_dict_to_file(state_dict, file_path)


def load_state_dict_from_file(file_path: str) -> dict[str, torch.Tensor]:
    """Loads the model state dict from a file (IO Wrapper)."""
    expanded_path = os.path.expanduser(file_path)
    try:
        return torch.load(expanded_path, map_location="cpu", weights_only=True)
    except Exception:
        # Fallback to weights_only=False in case the checkpoint has custom structures
        return torch.load(expanded_path, map_location="cpu", weights_only=False)


def apply_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor] | nn.Module) -> nn.Module:
    """Applies a state dict to a model (Inner function working on data structures)."""
    if isinstance(state_dict, nn.Module):
        state_dict = state_dict.state_dict()
    elif isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    return model


def export_model_to_onnx_bytes(model: nn.Module, input_size: tuple[int, int, int, int]) -> bytes:
    """Exports the PyTorch model to ONNX format in-memory and returns the bytes (Inner function working on data structures)."""
    model.eval()
    example_inputs = torch.randn(*input_size)
    buffer = io.BytesIO()
    torch.onnx.export(
        model,
        example_inputs,
        buffer,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
    return buffer.getvalue()


def write_bytes_to_file(data: bytes, file_path: str) -> None:
    """Writes bytes to a file (IO Wrapper)."""
    expanded_path = os.path.expanduser(file_path)
    with open(expanded_path, "wb") as f:
        f.write(data)


def train_model(args: argparse.Namespace) -> None:
    print("This is the main function of AlexNet.")

    # Download training data from open datasets.
    # this produces tensors of size BATCH_SIZE, 3 (per color channel), IMAGE_HEIGHT, IMAGE_WIDTH
    training_data = datasets.Imagenette(
        root=DATA_ROOT,
        split="train",
        size=SIZE,
        download=True,
        transform=TRAIN_TRANSFORM,
    )

    ## Download test data from open datasets.
    test_data = datasets.Imagenette(
        root=DATA_ROOT,
        split="val",
        size=SIZE,
        download=True,
        transform=TEST_TRANSFORM,
    )

    # shuffling the dataset is critical here, otherwise
    # it overfits to one at a time and does not progress past 10% accuracy.
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    loss_fn = nn.CrossEntropyLoss()
    print(f"Using device: {device}")
    model = NeuralNetwork().to(device)
    print(f"Model structure: {model}")
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )
    epochs = args.epochs if hasattr(args, "epochs") else DEFAULT_EPOCHS
    model.train()
    test_accuracies = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        test_accuracy = test(test_dataloader, model, loss_fn, device)
        test_accuracies.append(test_accuracy)
    print(f"Test accuracies: {test_accuracies}")
    ascii_chart = asciichartpy.plot(test_accuracies, {"height": 20})
    print(ascii_chart)
    model.eval()

    if args.save_model:
        save_model_state_to_file(model, args.save_model)
        print(f"Model saved to {args.save_model}")

    print("Done!")


def export_model(args: argparse.Namespace) -> None:
    assert args.output_path.endswith(".onnx"), "Output path must end with .onnx"
    print(f"Exporting model to {args.output_path}")

    model = NeuralNetwork()

    if args.model_path:
        print(f"Loading weights from {args.model_path}")
        state_dict = load_state_dict_from_file(args.model_path)
        model = apply_state_dict(model, state_dict)

    input_size = (1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
    onnx_bytes = export_model_to_onnx_bytes(model, input_size)
    write_bytes_to_file(onnx_bytes, args.output_path)
    print(f"ONNX model successfully exported to {args.output_path}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AlexNet training and export")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS, help=f"Number of training epochs (default: {DEFAULT_EPOCHS})"
    )
    train_parser.add_argument(
        "--save-model", type=str, default=None, help="Path to save the trained model"
    )

    export_parser = subparsers.add_parser("export", help="Export a trained model")
    export_parser.add_argument(
        "output_path", type=str, help="Path to save the exported model"
    )
    export_parser.add_argument(
        "--model-path", type=str, default=None, help="Path to the pretrained model weights"
    )

    args = parser.parse_args(argv[1:])

    if args.command is None:
        args.command = "train"
        args.epochs = DEFAULT_EPOCHS
        args.save_model = None
        args.model_path = None

    return args


def main(argv: list[str]) -> None:
    args = parse_args(argv)

    if args.command == "train":
        train_model(args)
    elif args.command == "export":
        export_model(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    from alexnet_core.model import NeuralNetwork
    main(sys.argv)
