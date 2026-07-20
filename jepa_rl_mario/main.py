"""A simple hello-world simulation using the Mario RL environment."""

from __future__ import annotations

import gymnasium as gym

# Import the environment directly
from jepa_rl_mario.mario_env import ACTION_JUMP, ACTION_RIGHT, MarioEnv


def simulate_episode(env: gym.Env) -> None:
    """Runs a single simulation episode in the environment with predefined actions."""
    obs, info = env.reset()
    print("Initial State:")
    print(env.render())

    # Actions: Right, Right, Right, Jump (at obstacle 3.0), Right, Right, Right...
    actions = [
        ACTION_RIGHT,  # Pos: 0.5
        ACTION_RIGHT,  # Pos: 1.0
        ACTION_RIGHT,  # Pos: 1.5
        ACTION_RIGHT,  # Pos: 2.0
        ACTION_RIGHT,  # Pos: 2.5
        ACTION_JUMP,   # Jump triggered, Pos: 2.5, Height: 1.0
        ACTION_RIGHT,  # Pos: 3.0 (overlapping obstacle 3.0, but height is 0.75, so safe!), Height: 0.75
        ACTION_RIGHT,  # Pos: 3.5, Height: 0.5
        ACTION_RIGHT,  # Pos: 4.0, Height: 0.25
        ACTION_RIGHT,  # Pos: 4.5, Height: 0.0 (jump ends)
        ACTION_RIGHT,  # Pos: 5.0
        ACTION_RIGHT,  # Pos: 5.5
        ACTION_RIGHT,  # Pos: 6.0
        ACTION_RIGHT,  # Pos: 6.5
        ACTION_JUMP,   # Jump triggered, Pos: 6.5, Height: 1.0
        ACTION_RIGHT,  # Pos: 7.0 (overlapping obstacle 7.0, height is 0.75, so safe!), Height: 0.75
        ACTION_RIGHT,  # Pos: 7.5, Height: 0.5
        ACTION_RIGHT,  # Pos: 8.0, Height: 0.25
        ACTION_RIGHT,  # Pos: 8.5, Height: 0.0
        ACTION_RIGHT,  # Pos: 9.0
        ACTION_RIGHT,  # Pos: 9.5
        ACTION_RIGHT,  # Pos: 10.0 (Goal!)
    ]

    for i, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1} | Action: {action} | Reward: {reward:.1f}")
        print(env.render())
        if terminated or truncated:
            print("Episode finished!")
            break


def main() -> None:
    env = MarioEnv()
    simulate_episode(env)


if __name__ == "__main__":
    main()
