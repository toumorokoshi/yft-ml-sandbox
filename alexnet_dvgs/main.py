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

# this is the number of classifications within the dataset.
# I believe this has to match the number of classifications in the original
# data set to enable validation
#
# Imagenette has 10 simple classifications.
NUM_CLASSIFICATIONS = 10
# not all images from the Imagenette dataset are the same size,
# so we have to crop them with a a custom transform, and request a s
# specific size
# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def train_model(args: argparse.Namespace) -> None:
    print("This is the main function of AlexNet.")

    # Download training data from open datasets.
    # this produces tensors of size BATCH_SIZE, 3 (per color channel), IMAGE_HEIGHT, IMAGE_WIDTH
    training_data = datasets.Imagenette(
        root="data",
        split="train",
        size=SIZE,
        download=True,
        transform=TRAIN_TRANSFORM,
    )

    ## Download test data from open datasets.
    test_data = datasets.Imagenette(
        root="data",
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
        model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0005
    )
    epochs = args.epochs if hasattr(args, "epochs") else 60
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
        torch.save(model.state_dict(), args.save_model)
        print(f"Model saved to {args.save_model}")

    print("Done!")


def run_dvgs(args: argparse.Namespace, model_cls=NeuralNetwork, transform=TEST_TRANSFORM) -> None:
    print(f"Running DVGS with threshold {args.threshold} on {args.samples} samples...")
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    model = model_cls().to(device)
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model, weights_only=True))
        print(f"Model loaded from {args.load_model}")
    model.eval()

    dataset = datasets.Imagenette(
        root="data",
        split="train",
        size=SIZE,
        download=True,
        transform=transform,
    )

    # Use a small subset if samples < len(dataset)
    if args.samples < len(dataset):
        indices = torch.randperm(len(dataset))[: args.samples]
        dataset = torch.utils.data.Subset(dataset, indices)

    dataloader = DataLoader(dataset, batch_size=1)
    loss_fn = nn.CrossEntropyLoss()

    selected_indices = []
    gradient_vectors = []

    for i, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        model.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()

        # Capture gradient of the 2nd to last linear layer (linear_stack[5])
        grad = model.linear_stack[5].weight.grad
        if grad is None:
            continue

        grad_vec = grad.flatten()

        is_distinct = True
        if gradient_vectors:
            # Stack existing vectors for batch cosine similarity
            ref_vectors = torch.stack(gradient_vectors)
            similarities = torch.nn.functional.cosine_similarity(
                grad_vec.unsqueeze(0), ref_vectors
            )
            if torch.any(similarities > args.threshold):
                is_distinct = False

        if is_distinct:
            gradient_vectors.append(grad_vec.detach())
            # For data.Subset, dataset.indices[i] gives the original index
            original_idx = dataset.indices[i].item() if hasattr(dataset, "indices") else i
            selected_indices.append(original_idx)
            if len(selected_indices) % 10 == 0:
                print(f"Found {len(selected_indices)} distinct data points so far...")

    print("\nFinal set of representative data points (original indices):")
    print(selected_indices)
    print(f"Total selected: {len(selected_indices)} out of {len(dataset)}")


def export_model(args: argparse.Namespace) -> None:
    assert args.output_path.endswith(".onnx"), "Output path must end with .onnx"
    print(f"Exporting model to {args.output_path}")
    model = NeuralNetwork()
    example_inputs = torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
    onnx_program = torch.onnx.export(model, example_inputs, dynamo=True, strict=True)
    import pdb

    pdb.set_trace()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AlexNet training and export")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--epochs", type=int, default=60, help="Number of training epochs (default: 60)"
    )
    train_parser.add_argument(
        "--save-model", type=str, default=None, help="Path to save the trained model"
    )

    export_parser = subparsers.add_parser("export", help="Export a trained model")
    export_parser.add_argument(
        "output_path", type=str, help="Path to save the exported model"
    )

    dvgs_parser = subparsers.add_parser("dvgs", help="Data Valuation using Gradients")
    dvgs_parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Cosine similarity threshold (default: 0.3)",
    )
    dvgs_parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of samples to evaluate (default: 1000)",
    )
    dvgs_parser.add_argument(
        "--load-model", type=str, default=None, help="Path to a pretrained model"
    )

    args = parser.parse_args(argv[1:])

    if args.command is None:
        args.command = "train"
        args.epochs = 60
        args.save_model = None

    return args


def main(argv: list[str]) -> None:
    args = parse_args(argv)

    if args.command == "train":
        train_model(args)
    elif args.command == "export":
        export_model(args)
    elif args.command == "dvgs":
        run_dvgs(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv)
