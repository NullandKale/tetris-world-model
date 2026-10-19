"""The play check's scores (diagnostics/play.py) on made-up wells: a real clear and a quiet board score 0; a row
vanishing that was not full, a piece losing a block, a piece born too big or too small and a full row that
stays are caught (a top-out is not); a piece keeping its
tetromino while it falls and turns scores shape 1, one turning into another tetromino 0."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from token_world.diagnostics.play import SETTLE, Play, score


def frame(cells) -> np.ndarray:
    """Filled cells (row, col) of the well -> a model frame: each a 7 x 7 block of colour 5 on colour 0."""
    out = np.zeros((256, 256), np.uint8)
    for r, c in cells:
        out[56 + 8 * r:56 + 8 * r + 7, 96 + 8 * c:96 + 8 * c + 7] = 5
    return out


STACK = [(19, c) for c in range(9)] + [(18, 0), (18, 1)]        # the bottom row one short of full
PIECE = [(5, 4), (5, 5), (6, 4), (6, 5)]
L_PIECE = [(0, 0), (1, 0), (2, 0), (2, 1)]                     # an L, at (row, col) offsets
L_TURNED = [(0, 0), (0, 1), (0, 2), (1, 0)]                    # the same L turned a quarter
S_PIECE = [(0, 1), (0, 2), (1, 0), (1, 1)]


def at(piece, row, col):
    return [(r + row, c + col) for r, c in piece]


def run(frames, dreamed=None) -> dict:
    frames = np.stack(frames)
    play = Play(frames[:1], np.zeros(1, np.int64), None, frames[1:], np.zeros(len(frames) - 1, np.int64))
    return score(play, play.future if dreamed is None else np.stack(dreamed)[1:])


class PlayScoreTests(unittest.TestCase):
    def test_a_quiet_board_and_a_falling_piece_score_zero(self):
        frames = [frame(STACK + [(r + t // 10, c) for r, c in PIECE]) for t in range(100)]
        result = run(frames)
        self.assertEqual({k: result[k] for k in ("lost", "morph", "full")}, {"lost": 0.0, "morph": 0.0, "full": 0.0})
        self.assertEqual(result["shape"], 1.0)

    def test_a_full_row_clearing_scores_zero(self):
        full = STACK + [(19, 9)]
        frames = [frame(full)] * 5 + [frame([(18, 0), (18, 1)])] * 5 + [frame([(19, 0), (19, 1)])] * 50
        self.assertEqual(run(frames)["lost"], 0.0)

    def test_a_row_that_was_not_full_vanishing_is_lost(self):
        frames = [frame(STACK)] * 40 + [frame([(18, 0), (18, 1)])] * 60
        result = run(frames)
        self.assertAlmostEqual(result["lost"], 9 * 100 / 99)
        self.assertGreater(result["morph"], 0)

    def test_a_piece_losing_a_block_morphs(self):
        frames = [frame(STACK + PIECE)] * 50 + [frame(STACK + PIECE[:3])] * 50
        result = run(frames)
        self.assertGreater(result["morph"], 0)
        self.assertGreater(result["lost"], 0)

    def test_a_spawn_is_no_morph(self):
        frames = [frame(STACK)] * 50 + [frame(STACK + [(0, 3), (0, 4), (0, 5), (1, 4)])] * 50
        self.assertEqual(run(frames)["morph"], 0.0)

    def test_a_piece_coming_into_view_at_the_top_is_no_morph(self):
        frames = [frame(STACK)] * 20 + [frame(STACK + [(0, 4)])] * 5 + [frame(STACK + [(0, 4), (1, 4)])] * 5 +             [frame(STACK + at(L_PIECE, t // 10, 4)) for t in range(70)]
        self.assertEqual(run(frames)["morph"], 0.0)

    def test_a_piece_turning_into_another_tetromino_loses_shape(self):
        real = [frame(STACK + at(L_PIECE if t < 50 else L_TURNED, 3 + t // 10, 4)) for t in range(100)]
        self.assertEqual(run(real)["shape"], 1.0)                      # turning keeps the tetromino
        dreamed = real[:40] + [frame(STACK + at(S_PIECE, 3 + t // 10, 4)) for t in range(40, 100)]
        # scored from frame 30 (the piece's first cells count as stack until it falls clear of them): S for 10 of 70
        self.assertAlmostEqual(run(real, dreamed)["shape"], 10 / 70)

    def test_a_full_row_that_never_clears(self):
        frames = [frame(STACK + [(19, 9)])] * 100
        self.assertAlmostEqual(run(frames)["full"], (100 - SETTLE) * 100 / 99)        # it stood SETTLE + 1.. 100 frames


    def test_a_piece_born_too_big_or_too_small_is_a_size(self):
        def spawned(piece):                         # it comes into view at the top, then falls clear of it
            return ([frame(STACK)] * 40 + [frame(STACK + at(piece, 0, 4))] * 10
                    + [frame(STACK + at(piece, 1 + t // 10, 4)) for t in range(60)])
        self.assertEqual(run(spawned(L_PIECE))["size"], 0.0)
        for wrong in (L_PIECE + [(2, 2)], L_PIECE[:3]):                  # five blocks, three
            result = run(spawned(wrong))
            self.assertAlmostEqual(result["size"], 100 / 109)
            self.assertEqual(result["morph"], 0.0)                       # already wrong when it left the top

    def test_a_top_out_and_a_new_game_are_no_size_or_full(self):
        curtain = [frame([(r, c) for r in range(20 - n, 20) for c in range(10)]) for n in range(5, 21)]
        frames = [frame(STACK)] * 40 + [c for c in curtain for _ in range(5)] + [frame([])] * 40
        result = run(frames)
        self.assertEqual((result["size"], result["full"]), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
