"""A simple hello-world simulation using gym-super-mario-bros wrapped in Gymnasium."""

from __future__ import annotations

import argparse
import os
import sys
from matplotlib import pyplot as plt
import numpy as np

from jepa_rl_mario.mario_env import MarioEnv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the Mario simulation."""
    parser = argparse.ArgumentParser(
        description="Run a simulation using gym-super-mario-bros wrapped in Gymnasium."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Maximum number of steps to run per episode (default: 20)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to simulate (default: 1)",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default="rgb_array",
        choices=["rgb_array", "human", "none"],
        help="Rendering mode for Gymnasium (default: rgb_array)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.environ.get("MARIO_HELLO_OUT", "mario_hello_last_frame.png"),
        help="Output image path for saving the final frame (default: MARIO_HELLO_OUT or mario_hello_last_frame.png)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for environment action space and resetting (default: None)",
    )
    return parser.parse_args(argv)


def run_simulation(
    env: MarioEnv,
    steps: int = 20,
    seed: int | None = None,
) -> tuple[float, int, np.ndarray | None]:
    """Runs a single episode simulation up to max steps taking random actions.

    Returns:
        (total_reward, step_count, final_frame)
    """
    if seed is not None:
        env.action_space.seed(seed)
        obs, info = env.reset(seed=seed)
    else:
        obs, info = env.reset()

    total_reward = 0.0
    step_count = 0

    for step in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        print(f"Step {step+1} | Action: {action} | Reward: {reward:.1f} | Terminated: {terminated}")

        if env.render_mode == "human":
            env.render()

        if terminated or truncated:
            break

    frame = env.render() if env.render_mode != "none" else None
    return total_reward, step_count, frame


def save_frame(frame: np.ndarray | None, output_path: str) -> bool:
    """Saves the given frame to the specified output image path."""
    if frame is None:
        print("Warning: Render returned None or render-mode is 'none', couldn't save frame.")
        return False
    plt.imsave(output_path, frame)
    print(f"Saved final frame of shape {frame.shape} to {output_path}")
    return True


def main(argv: list[str] | None = None) -> None:
    """Main execution function for running Mario simulation with CLI args."""
    args = parse_args(argv)

    gym_render_mode = args.render_mode if args.render_mode in ["human", "rgb_array"] else "rgb_array"
    env = MarioEnv(render_mode=gym_render_mode)
    if args.render_mode == "none":
        env.render_mode = "none"

    try:
        final_frame: np.ndarray | None = None
        for episode in range(args.episodes):
            print(f"--- Episode {episode + 1}/{args.episodes} ---")
            seed = args.seed + episode if args.seed is not None else None
            total_reward, steps_taken, frame = run_simulation(env, steps=args.steps, seed=seed)
            print(f"Episode {episode + 1} finished in {steps_taken} steps | Total Reward: {total_reward:.1f}")
            if frame is not None:
                final_frame = frame

        if args.render_mode != "none" and final_frame is not None:
            save_frame(final_frame, args.output)
    finally:
        env.close()


if __name__ == "__main__":
    main(sys.argv[1:])

