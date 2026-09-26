"""Generic PyTorch checkpointing, persistence, and training lifecycle management."""

from __future__ import annotations

import os
import signal
import threading
from typing import Any, Final, Optional

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]
    HAS_TORCH = False

# Generic Checkpoint Dictionary Keys (Rule 5)
CHECKPOINT_KEY_MODEL: Final[str] = "model_state_dict"
CHECKPOINT_KEY_TARGET_MODEL: Final[str] = "target_model_state_dict"
CHECKPOINT_KEY_OPTIMIZER: Final[str] = "optimizer_state_dict"
CHECKPOINT_KEY_EPSILON: Final[str] = "epsilon"
CHECKPOINT_KEY_EPISODES: Final[str] = "episodes"
CHECKPOINT_KEY_EPOCH: Final[str] = "epoch"
CHECKPOINT_KEY_STEP: Final[str] = "step"


def create_checkpoint(
    model_state_dict: dict[str, Any],
    optimizer_state_dict: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Pure function to construct a generic checkpoint dictionary."""
    checkpoint: dict[str, Any] = {
        CHECKPOINT_KEY_MODEL: model_state_dict,
    }
    if optimizer_state_dict is not None:
        checkpoint[CHECKPOINT_KEY_OPTIMIZER] = optimizer_state_dict
    checkpoint.update(kwargs)
    return checkpoint


def extract_checkpoint(
    checkpoint: dict[str, Any],
    model_key: str = CHECKPOINT_KEY_MODEL,
    optimizer_key: str = CHECKPOINT_KEY_OPTIMIZER,
) -> tuple[dict[str, Any], Optional[dict[str, Any]], dict[str, Any]]:
    """Pure function to extract model state, optimizer state, and remaining metadata from a checkpoint dictionary."""
    if model_key not in checkpoint:
        raise KeyError(f"Checkpoint missing required key: '{model_key}'")
    model_state = checkpoint[model_key]
    optimizer_state = checkpoint.get(optimizer_key)
    metadata = {k: v for k, v in checkpoint.items() if k not in (model_key, optimizer_key)}
    return model_state, optimizer_state, metadata


def extract_checkpoint_model_state(
    checkpoint: dict[str, Any],
    model_key: str = CHECKPOINT_KEY_MODEL,
) -> dict[str, Any]:
    """Pure function to validate and extract model state dict from checkpoint."""
    if model_key not in checkpoint:
        raise KeyError(f"Checkpoint missing required key: '{model_key}'")
    return checkpoint[model_key]


def apply_checkpoint_state(
    checkpoint: dict[str, Any],
    model: Optional[nn.Module] = None,
    optimizer: Optional[optim.Optimizer] = None,
    model_key: str = CHECKPOINT_KEY_MODEL,
    optimizer_key: str = CHECKPOINT_KEY_OPTIMIZER,
    strict: bool = True,
) -> dict[str, Any]:
    """Pure helper function to apply model and optimizer state dicts from checkpoint.

    Returns the remaining metadata dictionary (all keys other than model_key and optimizer_key).
    """
    model_state, optimizer_state, metadata = extract_checkpoint(
        checkpoint, model_key=model_key, optimizer_key=optimizer_key
    )
    if model is not None:
        model.load_state_dict(model_state, strict=strict)
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    return metadata


def apply_checkpoint_models(
    checkpoint: dict[str, Any],
    models: dict[str, nn.Module],
    strict: bool = True,
) -> None:
    """Helper function to load multiple models from checkpoint by dictionary key."""
    for key, model in models.items():
        if key in checkpoint and checkpoint[key] is not None:
            model.load_state_dict(checkpoint[key], strict=strict)


def resolve_checkpoint_save_path(
    save_checkpoint_arg: Optional[str],
    load_checkpoint_arg: Optional[str],
    is_eval: bool,
) -> Optional[str]:
    """Pure function to determine the target path for saving checkpoints."""
    if save_checkpoint_arg:
        return save_checkpoint_arg
    if load_checkpoint_arg and not is_eval:
        return load_checkpoint_arg
    return None


def compute_total_episodes(prior_episodes: Optional[int], episodes_completed: int) -> int:
    """Pure function to calculate total completed episodes across sessions."""
    return (prior_episodes or 0) + episodes_completed


class GracefulInterruptHandler:
    """Context manager for handling SIGINT gracefully to complete the current unit of work."""

    def __init__(self) -> None:
        self.interrupted: bool = False
        self._original_handler: Any = None
        self._installed: bool = False

    def __enter__(self) -> GracefulInterruptHandler:
        self.interrupted = False
        if threading.current_thread() is threading.main_thread():
            try:
                self._original_handler = signal.getsignal(signal.SIGINT)
                signal.signal(signal.SIGINT, self._handle_signal)
                self._installed = True
            except (ValueError, AttributeError):
                self._installed = False
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._installed and self._original_handler is not None:
            try:
                signal.signal(signal.SIGINT, self._original_handler)
            except (ValueError, AttributeError):
                pass

    def _handle_signal(self, signum: int, frame: Any) -> None:
        if self.interrupted:
            print("\nForcefully interrupting immediately...")
            raise KeyboardInterrupt
        self.interrupted = True
        print(
            "\nInterrupt received (Ctrl+C). Completing current episode before saving and exiting... "
            "(Press Ctrl+C again to abort immediately)"
        )


def save_checkpoint(
    checkpoint_path: str,
    checkpoint: dict[str, Any],
) -> None:
    """IO wrapper function to save checkpoint data structure to filesystem."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required to save checkpoints.")
    parent_dir = os.path.dirname(checkpoint_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> dict[str, Any]:
    """IO wrapper function to load checkpoint data structure from filesystem."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required to load checkpoints.")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device, weights_only=False)
