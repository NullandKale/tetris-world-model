"""World NES Tetris stream: index frames, the tick border, and a live window check."""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from token_world.data.rom import tetris_rom
from token_world.data.model_frames import BAND, GAME_ROWS, NES_TO_INDEX, border_codes, model_index_frames
from token_world.data.nes_palette import NES_COLOURS, NES_MASTER_PALETTE

ROM = tetris_rom()


class ModelFrameTests(unittest.TestCase):
    def test_colour_map_agrees_with_the_palette(self):
        for p in range(64):
            self.assertEqual(NES_COLOURS[NES_TO_INDEX[p]], NES_MASTER_PALETTE[p])

    def test_border_cells_hold_the_counter_and_state_bytes(self):
        ticks, state = np.array([0, 1, 5, 531440]), np.array([[0, 0], [37, 16], [255, 3], [9, 200]])
        codes = border_codes(ticks, state)                                         # [T, 32, 256]
        self.assertEqual(codes.shape, (4, 2 * BAND, 256))

        def cell(c):                                                           # token c // 4, quarter c % 4
            token, quarter = divmod(c, 4)
            band, column = divmod(token, 16)
            block = codes[:, band * BAND + (quarter // 2) * 8:][:, :8, column * 16 + (quarter % 2) * 8:][:, :, :8]
            self.assertTrue((block == block[:, :1, :1]).all())                 # each cell is one solid shade
            return block[:, 0, 0]
        read = lambda first, digits: sum(cell(first + i).astype(np.int64) * 3 ** i for i in range(digits))
        self.assertTrue(np.array_equal(read(0, 12), ticks))
        self.assertTrue(np.array_equal(read(12, 6), state[:, 0]))
        self.assertTrue(np.array_equal(read(20, 6), state[:, 1]))
        plain = border_codes(ticks)
        self.assertTrue(np.array_equal(plain[:, :, 7 * 16:], codes[:, :, 7 * 16:]))   # state stays in its cells
        self.assertTrue(np.array_equal(plain[:, BAND:], codes[:, BAND:]))

    def test_game_image_fills_the_tokens_between_the_bands(self):
        g = np.random.default_rng(0)
        frames = g.integers(0, 64, (2, 240, 256), dtype=np.uint8)
        out = model_index_frames(frames, np.zeros_like(frames), np.array([5, 6]), np.array([55, 56, 57], np.uint8),
                                 np.array([[3, 4], [5, 6]]))
        self.assertTrue(np.array_equal(out[:, GAME_ROWS], NES_TO_INDEX[frames[:, 8:232]]))   # NES line y at row y + 8
        self.assertTrue(((out[:, :BAND] >= 55) & (out[:, :BAND] <= 57)).all())
        self.assertTrue(((out[:, 256 - BAND:] >= 55) & (out[:, 256 - BAND:] <= 57)).all())
        self.assertEqual((GAME_ROWS.start % 16, GAME_ROWS.stop % 16, BAND % 16), (0, 0, 0))   # whole tokens

    def test_emphasis_is_refused(self):
        frames = np.zeros((1, 240, 256), np.uint8)
        emphasis = frames.copy()
        emphasis[0, 10, 10] = 1
        with self.assertRaises(ValueError):
            model_index_frames(frames, emphasis, np.array([0]), np.array([55, 56, 57], np.uint8))


@unittest.skipUnless(ROM.is_file() and importlib.util.find_spec("world_nes"), "needs the Tetris ROM and world_nes")
class LiveStreamTests(unittest.TestCase):
    def test_boots_into_play_with_consecutive_windows(self):
        from token_world.data.world_nes_tetris import WorldNesTetrisStreams
        stream = WorldNesTetrisStreams(64, seed=5)
        parts = [part for _, part in zip(range(20), iter(stream))]
        self.assertEqual(stream.new_frames, 63)
        for a, b in zip(parts, parts[1:]):
            self.assertEqual(b["tick"] - a["tick"], 63)
            self.assertTrue(torch.equal(a["x"][-1], b["x"][0]))                 # one shared observation
        self.assertEqual(parts[0]["x"].shape, (64, 256, 256))
        self.assertEqual(parts[0]["x"].dtype, torch.uint8)
        self.assertEqual(parts[0]["action"].shape, (64,))
        actions = torch.cat([p["action"][:-1] for p in parts[8:]])
        self.assertGreater(len(set(actions.tolist()) - {0, 8}), 1)             # in a game, not only menus


if __name__ == "__main__":
    unittest.main()
