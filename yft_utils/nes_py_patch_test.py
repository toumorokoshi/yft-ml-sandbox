"""Tests for yft_utils.nes_py_patch."""

import unittest
import numpy as np
import yft_utils.nes_py_patch  # noqa: F401


class NesPyPatchTest(unittest.TestCase):
    def test_patch_nes_py(self):
        try:
            import nes_py._rom as _rom

            class DummyROM(_rom.ROM):
                def __init__(self):
                    self.raw_data = np.array(
                        [0x4E, 0x45, 0x53, 0x1A, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        dtype="uint8",
                    )

            rom = DummyROM()
            self.assertTrue(isinstance(rom.prg_rom_size, (int, np.integer)))
            self.assertEqual(int(rom.prg_rom_size), 64)
            self.assertTrue(isinstance(rom.chr_rom_size, (int, np.integer)))
            self.assertEqual(int(rom.chr_rom_size), 16)
        except ImportError:
            pass


if __name__ == "__main__":
    unittest.main()
