"""A wrapper to expose the nes_py emulator using gym-super-mario-bros actions via python3 -m nes_py.play."""

from __future__ import annotations

import argparse
import os
import sys

# Remove local directory from path to allow importing the installed third-party nes_py package
local_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys_path_backup = list(sys.path)
if local_dir in sys.path:
    sys.path.remove(local_dir)
if "" in sys.path:
    sys.path.remove("")

try:
    from nes_py import NESEnv
    from nes_py.wrappers import JoypadSpace
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    from nes_py.app.play_human import play_human
except ImportError as e:
    print("Error: Could not import nes_py or gym_super_mario_bros. Ensure packages are installed.", file=sys.stderr)
    sys.exit(1)
finally:
    # Restore original path
    sys.path = sys_path_backup


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Super Mario Bros using gym-super-mario-bros actions")
    parser.add_argument("--rom", required=True, help="Path to the NES ROM file")

    args = parser.parse_args()

    # 1. Create raw NESEnv with the ROM
    raw_env = NESEnv(args.rom)

    # 2. Wrap it with JoypadSpace and SIMPLE_MOVEMENT (from gym_super_mario_bros)
    env = JoypadSpace(raw_env, SIMPLE_MOVEMENT)

    # 3. Launch human-play loop!
    play_human(env)


if __name__ == "__main__":
    main()
