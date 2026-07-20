"""A wrapper to expose the nes_py emulator via python3 -m nes_py.play."""

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
    from nes_py.app.cli import main as nes_py_main
except ImportError as e:
    print("Error: Could not import nes_py. Ensure that the nes-py package is installed.", file=sys.stderr)
    sys.exit(1)
finally:
    # Restore original path
    sys.path = sys_path_backup


def main() -> None:
    parser = argparse.ArgumentParser(description="Wrapper for nes_py CLI")
    parser.add_argument("--rom", required=True, help="Path to the NES ROM file")

    # Parse only known args to preserve extra flags that the user might pass
    args, unknown_args = parser.parse_known_args()

    # Translate --rom <path> into -r <path> expected by nes_py cli
    sys.argv = [sys.argv[0], "-r", args.rom] + unknown_args

    # Delegate execution to nes_py CLI main
    nes_py_main()


if __name__ == "__main__":
    main()
