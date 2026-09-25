"""Training script with interactive rendering option for Mario Env."""

from __future__ import annotations

import argparse
import random
from typing import Final, Optional

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

        # 3. Store in replay buffer
        replay_buffer.append((state, action, reward, next_state, float(done)))
        if len(replay_buffer) > REPLAY_SIZE:
            replay_buffer.pop(0)

        state = next_state
        total_reward += reward

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

        if done:
            break

    return total_reward


def main() -> None:
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

    args = parser.parse_args()

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

    try:
        for ep in range(args.episodes):
            reward = run_episode(
                env,
                q_network,
                target_network,
                optimizer,
                replay_buffer,
                epsilon,
                args.steps,
                BATCH_SIZE,
                GAMMA,
                dev_info.device,
            )
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            print(f"Episode {ep+1}/{args.episodes} | Total Reward: {reward:.1f} | Epsilon: {epsilon:.2f}")

            # Soft update target network
            if ep % 2 == 0:
                target_network.load_state_dict(q_network.state_dict())
    finally:
        env.close()


if __name__ == "__main__":
    main()
