"""Minimal highway-env quickstart (https://highway-env.farama.org/quickstart/)."""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import gymnasium
import highway_env  # noqa: F401 — registers env ids with Gymnasium
from matplotlib import pyplot as plt


def main() -> None:
    env = gymnasium.make("highway-v0", render_mode="rgb_array")
    env.reset()
    for _ in range(3):
        action = env.unwrapped.action_type.actions_indexes["IDLE"]
        env.step(action)
        env.render()

    frame = env.render()
    out = os.environ.get("HIGHWAY_HELLO_OUT", "highway_hello_last_frame.png")
    plt.imsave(out, frame)
    print(f"Wrote last frame to {out}")


if __name__ == "__main__":
    main()
