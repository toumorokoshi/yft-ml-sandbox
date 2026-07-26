"""Tests for yft_utils.nes_py_patch."""

import unittest
from yft_utils.nes_py_patch import patch_nes_py


class NesPyPatchTest(unittest.TestCase):
    def test_patch_nes_py(self):
        patch_nes_py()
        try:
            import nes_py._rom as _rom

            # Verify ROM properties are patched to return Python standard ints
            class DummyROM(_rom.ROM):
                def __init__(self):
                    import numpy as np

                    self.raw_data = np.array(
                        [0x4E, 0x45, 0x53, 0x1A, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        dtype="uint8",
                    )

            rom = DummyROM()
            self.assertEqual(type(rom.prg_rom_size), int)
            self.assertEqual(rom.prg_rom_size, 64)
            self.assertEqual(type(rom.chr_rom_size), int)
            self.assertEqual(rom.chr_rom_size, 16)
        except ImportError:
            # nes_py might not be installed in test env
            pass


if __name__ == "__main__":
    unittest.main()
