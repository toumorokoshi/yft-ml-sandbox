"""Training script with interactive rendering option for Mario Env."""

from __future__ import annotations

import argparse
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from jepa_rl_mario.mario_env import MarioEnv

# Constants
NUM_ACTIONS = 7
GAMMA = 0.99
BATCH_SIZE = 32
LR = 1e-3
REPLAY_SIZE = 10000


def preprocess_observation(obs: np.ndarray) -> np.ndarray:
    """Pure function to downsample and grayscale an observation from (240, 256, 3) to (80, 80)."""
    # Simple color channel average
    gray = obs.mean(axis=2)
    # Downsample by slicing with stride 3 to get size (80, 85) then crop to (80, 80)
    resized = gray[::3, ::3][:80, :80]
    return (resized / 255.0).astype(np.float32)


def select_action(q_values: torch.Tensor, epsilon: float, num_actions: int) -> int:
    """Pure function for epsilon-greedy action selection."""
    if random.random() < epsilon:
        return random.randint(0, num_actions - 1)
    return int(q_values.argmax().item())


class QNetwork(nn.Module):
    """Convolutional Neural Network to evaluate Q-values for Mario actions."""

    def __init__(self, num_actions: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch, 1, 80, 80)
        conv_out = self.conv(x)
        conv_out = conv_out.view(conv_out.size(0), -1)
        return self.fc(conv_out)


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
) -> float:
    """Wrapper function executing environment interactions (IO) and training steps."""
    obs, _ = env.reset()
    state = preprocess_observation(obs)
    total_reward = 0.0

    for step in range(max_steps):
        # 1. Choose action
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
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
                torch.tensor(np.array(b_states), dtype=torch.float32).unsqueeze(1),
                torch.tensor(b_actions, dtype=torch.long),
                torch.tensor(b_rewards, dtype=torch.float32),
                torch.tensor(np.array(b_next_states), dtype=torch.float32).unsqueeze(1),
                torch.tensor(b_dones, dtype=torch.float32),
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
        help="Interactive rendering mode"
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of training episodes")
    parser.add_argument("--steps", type=int, default=500, help="Max steps per episode")

    args = parser.parse_args()

    # Determine render mode passed to Gymnasium
    gym_render_mode = args.render_mode if args.render_mode in ["human", "rgb_array"] else "rgb_array"
    env = MarioEnv(render_mode=gym_render_mode)

    # If the user chose "none", we override self.render_mode so env.render() is not called
    if args.render_mode == "none":
        env.render_mode = "none"

    q_network = QNetwork(NUM_ACTIONS)
    target_network = QNetwork(NUM_ACTIONS)
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
