"""Tetris events: detectors on synthetic RAM."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from token_world.data import tetris_bot as T
from token_world.data import tetris_events as E


def ram_run(length=120, piece=0x02, x=5, y=3):
    """[length, 2048] RAM of a falling piece in a game, nothing changing."""
    ram = np.zeros((length, 2048), np.uint8)
    ram[:, E.MODE], ram[:, E.STATE] = E.IN_GAME, E.FALLING
    ram[:, T.PIECE], ram[:, T.X], ram[:, T.Y] = piece, x, y
    return ram


class DetectorTests(unittest.TestCase):
    def test_piece_moves(self):
        ram, acts = ram_run(), np.zeros(119, np.int64)
        ram[10:, T.Y] = 4                                                 # falls a row, no buttons
        self.assertEqual(E.events_at(ram, acts, 10), ["gravity (no buttons)"])
        ram, acts = ram_run(), np.zeros(119, np.int64)
        ram[10:, T.Y], acts[9] = 4, T.DOWN
        self.assertEqual(E.events_at(ram, acts, 10), ["soft drop"])
        ram, acts = ram_run(), np.zeros(119, np.int64)
        ram[10:, T.X], acts[9] = 4, T.LEFT
        self.assertEqual(E.events_at(ram, acts, 10), ["shift"])
        ram, acts = ram_run(), np.zeros(119, np.int64)
        ram[10:, T.PIECE], acts[9] = 0x03, T.A                            # T down -> T left
        self.assertEqual(E.events_at(ram, acts, 10), ["rotate"])

    def test_falling_alone_starts_when_the_player_lets_go(self):
        ram, acts = ram_run(), np.zeros(119, np.int64)
        acts[:8] = T.LEFT                                                  # moving, then hands off at frame 8
        self.assertIn("falling alone", E.events_at(ram, acts, 9))
        self.assertNotIn("falling alone", E.events_at(ram, acts, 12))      # only where the stretch starts
        acts[15] = T.DOWN                                                  # pressed again within 16 frames
        self.assertNotIn("falling alone", E.events_at(ram, acts, 9))

    def test_spawn_line_clear_level_up_and_top_out(self):
        ram, acts = ram_run(), np.zeros(119, np.int64)
        ram[:10, E.STATE] = 8
        self.assertEqual(E.events_at(ram, acts, 10), ["spawn"])
        ram = ram_run()
        ram[:10, E.STATE], ram[10:30, E.STATE] = E.CHECK_ROWS, E.CLEARING
        ram[:28, E.LINES], ram[28:, E.LINES] = 0x08, 0x12                 # BCD 8 -> 12: a tetris
        self.assertEqual(E.events_at(ram, acts, 10), ["line clear: tetris"])
        ram = ram_run()
        ram[:10, E.LEVEL], ram[10:, E.LEVEL] = 4, 5
        self.assertEqual(E.events_at(ram, acts, 10), ["level up"])
        ram = ram_run()
        ram[10:, E.STATE] = E.TOP_OUT
        self.assertEqual(E.events_at(ram, acts, 10), ["top-out"])
        ram = ram_run()
        ram[:10, E.MODE], ram[:10, E.LEVEL] = 3, 0                        # a new game resets the level
        ram[10:, E.LEVEL] = 7
        self.assertEqual(E.events_at(ram, acts, 10), ["game start"])

    def test_nothing_outside_a_game(self):
        ram, acts = ram_run(), np.zeros(119, np.int64)
        ram[:, E.MODE] = 3
        ram[10:, T.Y] = 4
        self.assertEqual(E.events_at(ram, acts, 10), [])


if __name__ == "__main__":
    unittest.main()
