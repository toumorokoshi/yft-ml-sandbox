"""A simple hello-world simulation using gym-super-mario-bros wrapped in Gymnasium."""

from __future__ import annotations

import os
from matplotlib import pyplot as plt
import numpy as np

from jepa_rl_mario.mario_env import MarioEnv


def main() -> None:
    # Initialize the environment in rgb_array mode to capture frames
    env = MarioEnv(render_mode="rgb_array")

    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")

    # Run for 20 steps taking random actions
    total_reward = 0.0
    for step in range(20):
        # Sample action from simplified action space (SIMPLE_MOVEMENT has 7 actions)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step {step+1} | Action: {action} | Reward: {reward:.1f} | Terminated: {terminated}")

        if terminated or truncated:
            break

    # Save the final frame
    frame = env.render()
    if frame is not None:
        out = os.environ.get("MARIO_HELLO_OUT", "mario_hello_last_frame.png")
        plt.imsave(out, frame)
        print(f"Saved final frame of shape {frame.shape} to {out}")
    else:
        print("Warning: Render returned None, couldn't save frame.")

    env.close()


if __name__ == "__main__":
    main()
