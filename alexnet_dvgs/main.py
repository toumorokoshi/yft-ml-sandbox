import os
import sys
import argparse
import math
import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from alexnet_core.model import NeuralNetwork, SIZE
from alexnet_core.recipe import TEST_TRANSFORM

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# DVGS Optimization Constants
# The desired Johnson-Lindestrauss Error
DATASET_SIZE = 10000
DESIRED_JL_ERROR = 0.002
DUGS_PROJECTIONS_SIZE_BASE_2 = math.ceil(math.log(8 * math.log(DATASET_SIZE) / (DESIRED_JL_ERROR)**2, 2))
DUGS_PROJECTION_SIZE = 2**DUGS_PROJECTIONS_SIZE_BASE_2


def run_dvgs(args: argparse.Namespace, model_cls=NeuralNetwork, transform=TEST_TRANSFORM) -> None:
    print(f"Running DVGS with threshold {args.threshold} on {args.samples} samples...")
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    model = model_cls().to(device)
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, weights_only=True))
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

    # Dimensionality reduction: pick fixed random indices
    # We need to know the gradient dimension. For linear_stack[5], it's 4096*4096
    total_dim = model.linear_stack[5].weight.numel()
    proj_indices = torch.randperm(total_dim, device=device)[:DUGS_PROJECTION_SIZE]

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        # Statistics for de-normalization
        mean = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1)

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

        # Dimensionality reduction via random projection (subsampling)
        grad_vec = grad.flatten()[proj_indices]

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

            if args.output_dir:
                # De-normalize image
                img = X[0] * std + mean
                img = torch.clamp(img, 0, 1)

                label = y[0].item()
                save_path = os.path.join(args.output_dir, f"idx_{original_idx}_class_{label}.png")
                save_image(img, save_path)

            if len(selected_indices) % 10 == 0:
                print(f"Found {len(selected_indices)} distinct data points so far...")

    print("\nFinal set of representative data points (original indices):")
    print(selected_indices)
    print(f"Total selected: {len(selected_indices)} out of {len(dataset)}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Data Valuation using Gradients (DVGS)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Cosine similarity threshold (default: 0.3)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of samples to evaluate (default: 1000)",
    )
    parser.add_argument(
        "--model-path", type=str, default=None, help="Path to the pretrained model weights"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Directory to save selected images"
    )

    return parser.parse_args(argv[1:])


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    run_dvgs(args)


if __name__ == "__main__":
    main(sys.argv)
