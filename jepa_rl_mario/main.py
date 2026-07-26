"""A simple hello-world simulation using gym-super-mario-bros wrapped in Gymnasium."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from matplotlib import pyplot as plt
import numpy as np

from jepa_rl_mario.mario_env import MarioEnv, RenderMode

logger = logging.getLogger(__name__)

DEFAULT_STEPS = 1000
DEFAULT_EPISODES = 10


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the Mario simulation."""
    parser = argparse.ArgumentParser(
        description="Run a simulation using gym-super-mario-bros wrapped in Gymnasium."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Maximum number of steps to run per episode (default: {DEFAULT_STEPS})",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
        help=f"Number of episodes to simulate (default: {DEFAULT_EPISODES})",
    )
    parser.add_argument(
        "--render-mode",
        type=RenderMode.from_str,
        default=RenderMode.RGB_ARRAY,
        choices=list(RenderMode),
        help=f"Rendering mode for Gymnasium (default: {RenderMode.RGB_ARRAY.value})",
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
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
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
        logger.debug("Step %d | Action: %s | Reward: %.1f | Terminated: %s", step + 1, action, reward, terminated)

        if env.render_mode == RenderMode.HUMAN:
            env.render()

        if terminated or truncated:
            break

    frame = env.render() if env.render_mode != RenderMode.NONE else None
    return total_reward, step_count, frame


def save_frame(frame: np.ndarray | None, output_path: str) -> bool:
    """Saves the given frame to the specified output image path."""
    if frame is None:
        logger.warning("Render returned None or render-mode is 'none', couldn't save frame.")
        return False
    plt.imsave(output_path, frame)
    logger.info("Saved final frame of shape %s to %s", frame.shape, output_path)
    return True


def main(argv: list[str] | None = None) -> None:
    """Main execution function for running Mario simulation with CLI args."""
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    env = MarioEnv(render_mode=args.render_mode)

    try:
        final_frame: np.ndarray | None = None
        for episode in range(args.episodes):
            logger.info("--- Episode %d/%d ---", episode + 1, args.episodes)
            seed = args.seed + episode if args.seed is not None else None
            total_reward, steps_taken, frame = run_simulation(env, steps=args.steps, seed=seed)
            logger.info("Episode %d finished in %d steps | Total Reward: %.1f", episode + 1, steps_taken, total_reward)
            if frame is not None:
                final_frame = frame

        if args.render_mode != RenderMode.NONE and final_frame is not None:
            save_frame(final_frame, args.output)
    finally:
        env.close()


if __name__ == "__main__":
    main(sys.argv[1:])



