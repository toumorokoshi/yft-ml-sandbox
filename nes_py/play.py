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
    import gym
    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace
    from nes_py.app.play_human import play_human
    from nes_py.app.play_random import play_random
    from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
except ImportError as e:
    print("Error: Could not import dependencies. Ensure packages are installed.", file=sys.stderr)
    sys.exit(1)
finally:
    # Restore original path
    sys.path = sys_path_backup

# Key mapping of action spaces
_ACTION_SPACES = {
    'right': RIGHT_ONLY,
    'simple': SIMPLE_MOVEMENT,
    'complex': COMPLEX_MOVEMENT,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Super Mario Bros using gym-super-mario-bros")
    parser.add_argument('--env', '-e',
        type=str,
        default='SuperMarioBros-v3',
        help='The name of the environment to play'
    )
    parser.add_argument('--mode', '-m',
        type=str,
        default='human',
        choices=['human', 'random'],
        help='The execution mode for the emulation'
    )
    parser.add_argument('--actionspace', '-a',
        type=str,
        default='nes',
        choices=['nes', 'right', 'simple', 'complex'],
        help='the action space wrapper to use'
    )
    parser.add_argument('--steps', '-s',
        type=int,
        default=500,
        help='The number of random steps to take.',
    )
    parser.add_argument('--stages', '-S',
        type=str,
        nargs='+',
        help='The random stages to sample from for a random stage env'
    )

    args = parser.parse_args()

    if args.stages is not None and 'RandomStages' not in args.env:
        print('--stages,-S should only be specified for RandomStages environments', file=sys.stderr)
        sys.exit(1)

    # Replicate gym.make call but FIX the stages parameter issue!
    # If stages is None, do NOT pass it to gym.make!
    if args.stages is not None:
        env = gym.make(args.env, stages=args.stages)
    else:
        env = gym.make(args.env)

    # Unwrap the environment to bypass gym 0.26+'s TimeLimit wrapper,
    # which expects 5-tuple step returns and causes a ValueError when
    # stepping the underlying 4-tuple environment.
    env = env.unwrapped


    # Wrap the environment with action space if specified
    if args.actionspace != 'nes':
        actions = _ACTION_SPACES[args.actionspace]
        env = JoypadSpace(env, actions)

    # Play the environment
    if args.mode == 'human':
        play_human(env)
    else:
        play_random(env, args.steps)


if __name__ == "__main__":
    main()
