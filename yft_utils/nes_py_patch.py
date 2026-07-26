"""Helper module to patch nes_py and gym_super_mario_bros for NumPy 2.x compatibility.

NumPy 2.0 type promotion rules preserve uint8 dtypes during scalar multiplication
(e.g., 16 * header[4] -> uint8(64), or ram[0x6d] * 0x100 -> uint8 overflow),
which causes an OverflowError when computing ROM slice sizes or Mario positions.

Importing this module patches `nes_py._rom.ROM` and `gym_super_mario_bros.smb_env.SuperMarioBrosEnv`
properties to explicitly cast uint8 bytes to standard Python integers.
"""

from __future__ import annotations

import os
import sys


def patch_nes_py() -> None:
    """Patch nes_py and gym_super_mario_bros to ensure NumPy 2.x compatibility."""
    # Temporarily remove project root directory from sys.path to ensure
    # we import the installed PyPI nes_py package rather than any shadowing directory.
    sys_path_backup = list(sys.path)
    local_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if local_dir in sys.path:
        sys.path.remove(local_dir)
    if "" in sys.path:
        sys.path.remove("")


    try:
        import nes_py  # noqa: F401
        import nes_py._rom as _rom  # type: ignore[import-not-found]
        import gym_super_mario_bros.smb_env as smb_env  # type: ignore[import-not-found]

        # Patch ROM header properties to return Python standard ints
        _rom.ROM.prg_rom_size = property(lambda self: 16 * int(self.header[4]))
        _rom.ROM.chr_rom_size = property(lambda self: 8 * int(self.header[5]))
        _rom.ROM.prg_ram_size = property(lambda self: 8 * (int(self.header[8]) or 1))
        _rom.ROM._zero_fill = property(lambda self: int(self.header[11:].sum()))

        # Patch SuperMarioBrosEnv RAM properties for NumPy 2.x uint8 promotion
        smb_env.SuperMarioBrosEnv._x_position = property(
            lambda self: int(self.ram[0x6D]) * 0x100 + int(self.ram[0x86])
        )
        smb_env.SuperMarioBrosEnv._y_position = property(
            lambda self: int(self.ram[0x00CE])
        )
        smb_env.SuperMarioBrosEnv._x_position_screen = property(
            lambda self: int(self.ram[0x03AD])
        )
        smb_env.SuperMarioBrosEnv._y_position_screen = property(
            lambda self: int(self.ram[0x03B8])
        )
        smb_env.SuperMarioBrosEnv._world = property(
            lambda self: int(self.ram[0x075F]) + 1
        )
        smb_env.SuperMarioBrosEnv._stage = property(
            lambda self: int(self.ram[0x075C]) + 1
        )
        smb_env.SuperMarioBrosEnv._lives = property(
            lambda self: int(self.ram[0x075A]) + 1
        )
    except ImportError:
        pass
    finally:
        sys.path = sys_path_backup


# Automatically apply patch upon module import
patch_nes_py()
