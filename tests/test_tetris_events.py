"""Tetris events: detectors on synthetic RAM, and which windows EventToss keeps."""
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


class TossTests(unittest.TestCase):
    def windows(self, ram, frames=64, stride=63):
        """Consecutive windows of one stream, each starting `stride` frames after the one before."""
        acts = np.full(len(ram) - 1, T.DOWN, np.int64)             # pressed throughout: nothing falls alone
        for w, lo in enumerate(range(0, len(ram) - frames + 1, stride)):
            yield w, ram[lo:lo + frames], acts[lo:lo + frames - 1]

    def kept(self, ram, seed=0, stride=63):
        toss, kept = E.EventToss(np.random.default_rng(seed), stride), []
        for w, r, a in self.windows(ram, stride=stride):
            out = toss.push(w, r, a)
            if out is not None:
                kept.append(out)
        return kept

    def test_rare_events_are_always_kept_and_their_aftermath_too(self):
        ram = ram_run(64 + 63 * 9)
        event = 63 * 4 + 60                                         # near the end of window 4
        ram[:event, E.LEVEL], ram[event:, E.LEVEL] = 2, 3
        for seed in range(20):
            kept = self.kept(ram, seed)
            self.assertIn(4, kept)
            self.assertIn(5, kept)                                  # within AFTERMATH of window 4's end
        early = 63 * 2 + 5                                          # early in window 2: no aftermath
        ram = ram_run(64 + 63 * 9)
        ram[:early, E.LEVEL], ram[early:, E.LEVEL] = 2, 3
        self.assertLess(np.mean([3 in self.kept(ram, seed) for seed in range(200)]), 0.3)

    def test_a_window_is_judged_with_the_next_ones_lookahead(self):
        ram = ram_run(64 + 63 * 3)
        e = 63 + 62                                                 # line clear on window 1's last step
        ram[:e, E.STATE], ram[e:e + 20, E.STATE] = E.CHECK_ROWS, E.CLEARING
        ram[:e + 18, E.LINES], ram[e + 18:, E.LINES] = 0x08, 0x10  # counted 18 frames later: a double
        for seed in range(20):
            self.assertIn(1, self.kept(ram, seed))

    def test_overlapping_windows_see_the_event_and_its_lookahead(self):
        ram = ram_run(64 + 32 * 12)
        e = 32 * 5 + 60                                             # line clear near the end of window 5
        ram[:e, E.STATE], ram[e:e + 20, E.STATE] = E.CHECK_ROWS, E.CLEARING
        ram[:e + 18, E.LINES], ram[e + 18:, E.LINES] = 0x08, 0x10  # a double, counted 18 frames later
        for seed in range(20):
            kept = self.kept(ram, seed, stride=32)
            self.assertTrue({5, 6} <= set(kept))                   # frames 220-223 lie in windows 5 and 6

    def test_ordinary_windows_are_kept_at_the_plain_rate(self):
        ram = ram_run(64 + 63 * 2000)
        self.assertAlmostEqual(len(self.kept(ram)) / 1999, E.PLAIN_KEEP, delta=0.03)


if __name__ == "__main__":
    unittest.main()
