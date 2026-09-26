"""Training script with interactive rendering option for Mario Env."""

from __future__ import annotations

import argparse
import os
import random
from typing import Any, Final, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from jepa_rl_mario.mario_env import MarioEnv
from jepa_rl_mario.model import MarioModel
from yft_utils import detect_device

# Constants (Rule 5)
NUM_ACTIONS: Final[int] = 7
GAMMA: Final[float] = 0.99
BATCH_SIZE: Final[int] = 32
LR: Final[float] = 1e-3
REPLAY_SIZE: Final[int] = 10000
HEIGHT: Final[int] = 240
WIDTH: Final[int] = 256
DEFAULT_EPISODES: Final[int] = 5
DEFAULT_STEPS: Final[int] = 500

CHECKPOINT_KEY_MODEL: Final[str] = "model_state_dict"
CHECKPOINT_KEY_TARGET_MODEL: Final[str] = "target_model_state_dict"
CHECKPOINT_KEY_OPTIMIZER: Final[str] = "optimizer_state_dict"
CHECKPOINT_KEY_EPSILON: Final[str] = "epsilon"
CHECKPOINT_KEY_EPISODES: Final[str] = "episodes"


def create_checkpoint(
    model_state_dict: dict[str, Any],
    target_state_dict: dict[str, Any],
    optimizer_state_dict: Optional[dict[str, Any]],
    epsilon: float,
    episodes: int,
) -> dict[str, Any]:
    """Pure function to construct a checkpoint dictionary from model and training states."""
    checkpoint: dict[str, Any] = {
        CHECKPOINT_KEY_MODEL: model_state_dict,
        CHECKPOINT_KEY_TARGET_MODEL: target_state_dict,
        CHECKPOINT_KEY_EPSILON: epsilon,
        CHECKPOINT_KEY_EPISODES: episodes,
    }
    if optimizer_state_dict is not None:
        checkpoint[CHECKPOINT_KEY_OPTIMIZER] = optimizer_state_dict
    return checkpoint


def extract_checkpoint(
    checkpoint: dict[str, Any],
) -> tuple[dict[str, Any], Optional[dict[str, Any]], Optional[dict[str, Any]], Optional[float], Optional[int]]:
    """Pure function to validate and extract state dicts and metadata from a checkpoint dictionary."""
    if CHECKPOINT_KEY_MODEL not in checkpoint:
        raise KeyError(f"Checkpoint missing required key: '{CHECKPOINT_KEY_MODEL}'")
    model_state = checkpoint[CHECKPOINT_KEY_MODEL]
    target_state = checkpoint.get(CHECKPOINT_KEY_TARGET_MODEL)
    optimizer_state = checkpoint.get(CHECKPOINT_KEY_OPTIMIZER)
    epsilon = checkpoint.get(CHECKPOINT_KEY_EPSILON)
    episodes = checkpoint.get(CHECKPOINT_KEY_EPISODES)
    return model_state, target_state, optimizer_state, epsilon, episodes


def apply_checkpoint_state(
    q_network: nn.Module,
    checkpoint: dict[str, Any],
    target_network: Optional[nn.Module] = None,
    optimizer: Optional[optim.Optimizer] = None,
) -> tuple[Optional[float], Optional[int]]:
    """Pure helper function to apply checkpoint state dicts to networks and optimizer."""
    model_state, target_state, optimizer_state, epsilon, episodes = extract_checkpoint(checkpoint)
    q_network.load_state_dict(model_state)
    if target_network is not None:
        if target_state is not None:
            target_network.load_state_dict(target_state)
        else:
            target_network.load_state_dict(model_state)
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    return epsilon, episodes


def save_checkpoint(
    checkpoint_path: str,
    checkpoint: dict[str, Any],
) -> None:
    """IO wrapper function to save checkpoint data structure to filesystem."""
    parent_dir = os.path.dirname(checkpoint_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> dict[str, Any]:
    """IO wrapper function to load checkpoint data structure from filesystem."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def preprocess_observation(obs: np.ndarray) -> np.ndarray:
    """Pure function to downsample and grayscale an observation from (240, 256, 3) to (240, 256)."""
    gray = obs.mean(axis=2)
    return gray


def select_action(q_values: torch.Tensor, epsilon: float, num_actions: int) -> int:
    """Pure function for epsilon-greedy action selection."""
    if random.random() < epsilon:
        return random.randint(0, num_actions - 1)
    return int(q_values.argmax().item())


def compute_loss(
    q_network: nn.Module,
    target_network: nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Pure function to calculate Mean Squared Error loss between Q-value predictions and targets."""
    q_values = q_network(states)
    state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_values = target_network(next_states)
        max_next_q_values = next_q_values.max(1)[0]
        expected_state_action_values = rewards + (gamma * max_next_q_values * (1.0 - dones))

    return nn.MSELoss()(state_action_values, expected_state_action_values)


def run_episode(
    env: MarioEnv,
    q_network: nn.Module,
    target_network: nn.Module,
    optimizer: optim.Optimizer,
    replay_buffer: list,
    epsilon: float,
    max_steps: int,
    batch_size: int,
    gamma: float,
    device: torch.device,
    is_training: bool = True,
) -> float:
    """Wrapper function executing environment interactions (IO) and training steps."""
    obs, _ = env.reset()
    state = preprocess_observation(obs)
    total_reward = 0.0

    for _ in range(max_steps):
        # 1. Choose action
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            q_values = q_network(state_tensor)
        action = select_action(q_values, epsilon, NUM_ACTIONS)

        # 2. Environment step (IO)
        next_obs, reward, terminated, truncated, _ = env.step(action)

        # Render if human mode
        if env.render_mode == "human":
            env.render()

        next_state = preprocess_observation(next_obs)
        done = terminated or truncated

        # 3. Store in replay buffer & train if enabled
        if is_training:
            replay_buffer.append((state, action, reward, next_state, float(done)))
            if len(replay_buffer) > REPLAY_SIZE:
                replay_buffer.pop(0)

            # 4. Optimize network
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                b_states, b_actions, b_rewards, b_next_states, b_dones = zip(*batch)

                loss = compute_loss(
                    q_network,
                    target_network,
                    torch.tensor(np.array(b_states), dtype=torch.float32, device=device).unsqueeze(1),
                    torch.tensor(b_actions, dtype=torch.long, device=device),
                    torch.tensor(b_rewards, dtype=torch.float32, device=device),
                    torch.tensor(np.array(b_next_states), dtype=torch.float32, device=device).unsqueeze(1),
                    torch.tensor(b_dones, dtype=torch.float32, device=device),
                    gamma,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        state = next_state
        total_reward += reward

        if done:
            break

    return total_reward


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for Mario training."""
    parser = argparse.ArgumentParser(description="Train DQN on wrapped Mario environment")
    parser.add_argument(
        "--render-mode",
        type=str,
        default="human",
        choices=["human", "rgb_array", "none"],
        help="Interactive rendering mode",
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES, help="Number of training episodes")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Max steps per episode")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Target accelerator device (auto, cuda, mps, cpu)",
    )
    parser.add_argument(
        "--save-checkpoint",
        type=str,
        default=None,
        help="Optional path to save final training checkpoint",
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="Optional path to load initial checkpoint before training or evaluation",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run in evaluation mode (no training updates, greedy actions)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Override exploration epsilon (float between 0.0 and 1.0)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

    # Determine render mode passed to Gymnasium
    gym_render_mode = args.render_mode if args.render_mode in ["human", "rgb_array"] else "rgb_array"
    env = MarioEnv(render_mode=gym_render_mode)

    # If the user chose "none", we override self.render_mode so env.render() is not called
    if args.render_mode == "none":
        env.render_mode = "none"

    dev_info = detect_device(args.device)
    print(f"Using device: {dev_info.device} ({dev_info.device_name}) on platform '{dev_info.platform}'")

    q_network = MarioModel(num_actions=NUM_ACTIONS, height=HEIGHT, width=WIDTH).to(dev_info.device)
    target_network = MarioModel(num_actions=NUM_ACTIONS, height=HEIGHT, width=WIDTH).to(dev_info.device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=LR)

    replay_buffer: list = []
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.95

    if args.load_checkpoint:
        print(f"Loading checkpoint from: {args.load_checkpoint}")
        checkpoint = load_checkpoint(args.load_checkpoint, device=dev_info.device)
        loaded_eps, loaded_episodes = apply_checkpoint_state(
            q_network=q_network,
            checkpoint=checkpoint,
            target_network=target_network,
            optimizer=optimizer if not args.eval else None,
        )
        if loaded_eps is not None:
            epsilon = loaded_eps
        print(
            f"Loaded checkpoint successfully (resumed epsilon={epsilon:.2f}, "
            f"prior episodes={loaded_episodes or 0})"
        )

    if args.epsilon is not None:
        epsilon = args.epsilon

    if args.eval:
        epsilon = 0.0
        q_network.eval()
        target_network.eval()
        print("Running in evaluation mode (training disabled, epsilon=0.0)")

    try:
        for ep in range(args.episodes):
            reward = run_episode(
                env=env,
                q_network=q_network,
                target_network=target_network,
                optimizer=optimizer,
                replay_buffer=replay_buffer,
                epsilon=epsilon,
                max_steps=args.steps,
                batch_size=BATCH_SIZE,
                gamma=GAMMA,
                device=dev_info.device,
                is_training=not args.eval,
            )
            if not args.eval:
                epsilon = max(epsilon_min, epsilon * epsilon_decay)
            print(f"Episode {ep+1}/{args.episodes} | Total Reward: {reward:.1f} | Epsilon: {epsilon:.2f}")

            # Soft update target network
            if not args.eval and ep % 2 == 0:
                target_network.load_state_dict(q_network.state_dict())
    finally:
        env.close()

    if args.save_checkpoint:
        print(f"Saving final training checkpoint to: {args.save_checkpoint}")
        checkpoint = create_checkpoint(
            model_state_dict=q_network.state_dict(),
            target_state_dict=target_network.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            epsilon=epsilon,
            episodes=args.episodes,
        )
        save_checkpoint(args.save_checkpoint, checkpoint)
        print("Checkpoint saved successfully.")


if __name__ == "__main__":
    main()
