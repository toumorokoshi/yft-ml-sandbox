"""Helper module to patch nes_py for NumPy 2.x compatibility.

NumPy 2.0 type promotion rules preserve uint8 dtypes during scalar multiplication
(e.g., 16 * header[4] -> uint8(64)), which causes an OverflowError when nes_py._rom
computes ROM slice sizes like `64 * 1024`.

Importing this module patches `nes_py._rom.ROM` properties to explicitly cast header uint8
bytes to standard Python integers.
"""

from __future__ import annotations

import os
import sys


def patch_nes_py() -> None:
    """Patch nes_py._rom.ROM to ensure NumPy 2.x compatibility."""
    # Temporarily remove project root directory from sys.path to ensure
    # we import the installed PyPI nes_py package rather than any shadowing directory.
    sys_path_backup = list(sys.path)
    if "RUNFILES_DIR" not in os.environ:
        local_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if local_dir in sys.path:
            sys.path.remove(local_dir)
        if "" in sys.path:
            sys.path.remove("")

    try:
        import nes_py._rom as _rom  # type: ignore[import-not-found]

        # Patch ROM header properties to return Python standard ints
        _rom.ROM.prg_rom_size = property(lambda self: 16 * int(self.header[4]))
        _rom.ROM.chr_rom_size = property(lambda self: 8 * int(self.header[5]))
        _rom.ROM.prg_ram_size = property(lambda self: 8 * (int(self.header[8]) or 1))
        _rom.ROM._zero_fill = property(lambda self: int(self.header[11:].sum()))
    except ImportError:
        pass
    finally:
        sys.path = sys_path_backup


# Automatically apply patch upon module import
patch_nes_py()
