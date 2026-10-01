"""The RAM-driven Tetris bot: drops, line clears, the placement search and the button driver."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from token_world.data import tetris_bot as T


def ram_with(board: np.ndarray, orientation: int, x: int, y: int, next_id: int = 0x0A) -> np.ndarray:
    ram = np.zeros(2048, np.uint8)
    ram[T.BOARD:T.BOARD + 200] = np.where(board, 0x7B, T.EMPTY).ravel()
    ram[T.PIECE], ram[T.X], ram[T.Y], ram[T.NEXT] = orientation, x, y, next_id
    return ram


class TetrisBotTests(unittest.TestCase):
    def test_moves_cover_every_on_board_column(self):
        dy, columns, moves = T.MOVES["I"]
        self.assertEqual(len(moves), 7 + 10)                     # horizontal x 7 columns, vertical x 10
        self.assertTrue((columns >= 0).all() and (columns <= 9).all())

    def test_drop_rests_on_column_tops_and_clears_lines(self):
        board = np.zeros((20, 10), bool)
        board[19, :9] = True                                     # one gap, far right
        board[18, 0] = True
        after, lines, ok = T.place_all(board[None], "I")
        moves = T.MOVES["I"][2]
        k = moves.index((0x11, 9))                               # vertical I into the gap
        self.assertTrue(ok[0, k])
        self.assertEqual(lines[0, k], 1)
        expected = np.zeros((20, 10), bool)
        expected[17:, 9] = True                                  # the I's top 3 cells, shifted down one row
        expected[19, 0] = True                                   # the cell that sat on the cleared row
        self.assertTrue(np.array_equal(after[0, k], expected))

    def test_placement_above_the_top_is_refused(self):
        board = np.ones((20, 10), bool)
        board[:, 4] = False
        board[0, :] = True
        _, _, ok = T.place_all(board[None], "O")
        self.assertFalse(ok.any())
        self.assertIsNone(T.best_placement(board, "O", None))

    def test_search_fills_the_well(self):
        board = np.zeros((20, 10), bool)
        board[16:, :9] = True
        self.assertEqual(T.best_placement(board, "I", "O"), (0x11, 9))

    def test_perfect_brain_rotates_and_shifts_on_press_edges_then_drops(self):
        brain, rng = T.PerfectBrain(), np.random.default_rng(0)
        board = np.zeros((20, 10), bool)
        board[16:, :9] = True
        self.assertEqual(brain(ram_with(board, 0x12, 5, 0), rng), 0)          # plans; Down released
        self.assertEqual(brain.target, (0x11, 9))
        self.assertEqual(brain(ram_with(board, 0x12, 5, 0), rng), T.A)
        self.assertEqual(brain(ram_with(board, 0x12, 5, 0), rng), 0)          # A released before the next tap
        self.assertEqual(brain(ram_with(board, 0x11, 5, 0), rng), T.RIGHT)
        self.assertEqual(brain(ram_with(board, 0x11, 6, 0), rng), 0)
        self.assertEqual(brain(ram_with(board, 0x11, 6, 1), rng), T.RIGHT)
        for x in (7, 8):
            brain(ram_with(board, 0x11, x, 1), rng)
            brain(ram_with(board, 0x11, x, 1), rng)
        self.assertEqual(brain(ram_with(board, 0x11, 9, 1), rng), T.DOWN)
        self.assertEqual(brain(ram_with(board, 0x11, 9, 2), rng), T.DOWN)     # Down is held, not tapped

    def test_switcher_plays_one_brain_until_its_interval_ends(self):
        switcher = T.BrainSwitcher(np.random.default_rng(3), long=False)
        switcher.game_left = 10 ** 6
        first, left = switcher.current, switcher.left
        ram = ram_with(np.zeros((20, 10), bool), 0x02, 5, 0)
        for _ in range(left - 1):
            switcher(ram, np.random.default_rng(0))
            self.assertEqual(switcher.current, first)

    def test_switcher_gives_up_when_the_game_length_runs_out(self):
        rng = np.random.default_rng(5)
        switcher = T.BrainSwitcher(rng, long=False)
        switcher.game_left = 3
        ram = ram_with(np.zeros((20, 10), bool), 0x02, 5, 0)
        seen = set()
        for _ in range(20_000):
            switcher(ram, rng)
            if switcher.game_left <= 0:
                seen.add(switcher.current)
        self.assertEqual(seen, {"reckless", "random"})

    def test_long_games_are_perfect_play_short_games_mixed(self):
        rng = np.random.default_rng(0)
        long, short = T.BrainSwitcher(rng, long=True), T.BrainSwitcher(rng, long=False)
        self.assertGreaterEqual(long.game_left, T.BrainSwitcher.LONG[0])
        self.assertLessEqual(short.game_left, T.BrainSwitcher.SHORT[1])
        ram = ram_with(np.zeros((20, 10), bool), 0x02, 5, 0)
        seen = set()
        for _ in range(10_000):
            long(ram, rng)
            seen.add(long.current)
        self.assertEqual(seen, {"perfect"})

    def test_brain_taking_over_replans_the_falling_piece(self):
        """The stale-target bug: a perfect brain that played a T and takes over a J mid-fall
        must replan instead of tapping rotate toward a T orientation."""
        rng = np.random.default_rng(0)
        brain = T.PerfectBrain()
        brain(ram_with(np.zeros((20, 10), bool), 0x02, 5, 0), rng)
        for y in range(1, 13):
            brain(ram_with(np.zeros((20, 10), bool), brain.target[0], brain.target[1], y), rng)
        brain.reset()                                                            # BrainSwitcher does this
        brain(ram_with(np.zeros((20, 10), bool), 0x07, 5, 14), rng)
        self.assertEqual(T.TYPE_OF[brain.target[0]], "J")

    def test_patient_pieces_wait_then_soft_drop(self):
        board = np.zeros((20, 10), bool)
        board[16:, :9] = True
        brain, rng = T.PerfectBrain(patience=1.0), np.random.default_rng(0)
        brain(ram_with(board, 0x11, 9, 0), rng)                                    # plans: vertical I, column 9
        wait = brain.wait
        self.assertTrue(T.PATIENT_FRAMES[0] <= wait < T.PATIENT_FRAMES[1])
        pressed = [brain(ram_with(board, 0x11, 9, 1), rng) for _ in range(wait + 3)]
        self.assertEqual(pressed[:wait], [0] * wait)                               # gravity alone
        self.assertEqual(pressed[wait:], [T.DOWN] * 3)                             # then soft drop
        brain, rng = T.PerfectBrain(patience=0.3), np.random.default_rng(1)
        patient = []
        for piece in range(2000):
            brain.reset()
            brain(ram_with(board, 0x11, 9, 0), rng)
            patient.append(brain.wait > 0)
        self.assertAlmostEqual(np.mean(patient), 0.3, delta=0.03)

    def test_reckless_brain_picks_legal_random_placements(self):
        rng = np.random.default_rng(1)
        board = np.zeros((20, 10), bool)
        board[10:, :] = True
        board[10:, 4] = False
        brain = T.RecklessBrain()
        targets = set()
        for _ in range(50):
            brain.reset()
            brain(ram_with(board, 0x12, 5, 0), rng)
            targets.add(brain.target)
        ok = T.place_all(board[None], "I")[2][0]
        legal = {T.MOVES["I"][2][i] for i in np.flatnonzero(ok)}
        self.assertTrue(targets <= legal)
        self.assertGreater(len(targets), 5)

    def test_well_builder_keeps_the_well_and_takes_the_tetris(self):
        board = np.zeros((20, 10), bool)
        board[16:, :9] = True                                                    # four rows ready, well open
        self.assertEqual(T.best_placement(board, "I", None, well=True), (0x11, 9))   # vertical I: a tetris
        board = np.zeros((20, 10), bool)
        board[18:, :7] = True                                                    # covering the well clears nothing
        self.assertNotEqual(T.best_placement(board, "O", None, well=True)[1], 9)  # O at x 9 fills columns 8-9
        self.assertEqual(T.best_placement(board, "O", None, well=False)[1], 8)   # the classic brain fills 7-8


class StartLevelTests(unittest.TestCase):
    def test_long_games_start_low_short_games_anywhere(self):
        from token_world.data.world_nes_tetris import TetrisSession
        session = TetrisSession(0)
        drawn = [(session._start_level(), session.long_game) for _ in range(20_000)]
        long = [lv for lv, is_long in drawn if is_long]
        short = [lv for lv, is_long in drawn if not is_long]
        self.assertAlmostEqual(len(long) / len(drawn), T.BrainSwitcher.LONG_SHARE, delta=0.01)
        self.assertEqual(set(long), set(range(10)))
        self.assertEqual(set(short), set(range(20)))

if __name__ == "__main__":
    unittest.main()
