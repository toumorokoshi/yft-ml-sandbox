"""A wrapper to expose the gym_super_mario_bros CLI via python3 -m nes_py.play."""

from __future__ import annotations

import argparse
import os
import sys

# Remove local directory from path to allow importing the installed packages when running outside Bazel
sys_path_backup = list(sys.path)
if "RUNFILES_DIR" not in os.environ:
    local_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if local_dir in sys.path:
        sys.path.remove(local_dir)
    if "" in sys.path:
        sys.path.remove("")

try:
    from gym_super_mario_bros._app.cli import main as gym_mario_main
except ImportError as e:
    print("Error: Could not import gym_super_mario_bros. Ensure the package is installed.", file=sys.stderr)
    sys.exit(1)


finally:
    # Restore original path
    sys.path = sys_path_backup


def main() -> None:
    parser = argparse.ArgumentParser(description="Wrapper to run gym_super_mario_bros CLI")
    parser.add_argument("--env", "-e", default="SuperMarioBros-v0", help="The environment ID")
    parser.add_argument("--mode", "-m", default="human", choices=["human", "random"], help="The control mode")

    # Parse args to intercept them and translate if necessary
    args, unknown = parser.parse_known_args()

    # Reconstruct sys.argv to call gym_super_mario_bros CLI main
    # gym_super_mario_bros expects: -e <env> -m <mode>
    sys.argv = [sys.argv[0], "-e", args.env, "-m", args.mode] + unknown

    # Delegate execution to gym_super_mario_bros CLI main
    gym_mario_main()


if __name__ == "__main__":
    main()
