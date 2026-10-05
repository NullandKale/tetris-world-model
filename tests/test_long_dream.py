"""Long-dream scores: expected wrong pixels, block mass, the falling piece (kept, hit, ghost, fall), the next
piece's spawn and the fall timer, on made-up trials."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from token_world.diagnostics.long_dream import FUTURE, PLAYFIELD, Dream, Trial, score

PALETTE = np.zeros((58, 3), np.uint8)


def trial_with_piece(rows_per_frame: float = 0.25, respawn: int | None = None) -> Trial:
    """A walled well (static) with an 8 x 24 piece falling from the top of the playfield; respawn: the
    frame of the future a next piece appears at the top (the first one is then gone)."""
    frames = np.zeros((48 + FUTURE, 256, 256), np.uint8)
    frames[:, PLAYFIELD[0], PLAYFIELD[1].start] = 5                     # a wall that never changes
    for i in range(48, 48 + FUTURE):
        if respawn is not None and i - 48 >= respawn:
            frames[i, 60:68, 120:144] = 9
            continue
        top = 60 + int((i - 48) * rows_per_frame)
        if top < 204:
            frames[i, top:top + 8, 120:144] = 9
    return Trial(frames[:48], np.zeros(48, np.int64), frames[48:], level=0)


def frozen(trial: Trial) -> np.ndarray:
    """A dream that holds the first future frame: the piece never moves."""
    return np.repeat(trial.future[:1], FUTURE, 0)


class LongDreamScoreTests(unittest.TestCase):
    def test_the_real_game_scores_perfectly(self):
        trial = trial_with_piece()
        out = score(trial, Dream.of_frames(trial, trial.future, PALETTE))
        self.assertEqual(out["wrong_128"], 0.0)
        self.assertEqual(out["mass"], 1.0)
        self.assertEqual((out["piece_16"], out["piece_32"]), (1.0, 1.0))
        self.assertEqual((out["piece_hit_32"], out["piece_ghost_32"], out["fall"]), (1.0, 0.0, 1.0))
        self.assertEqual(out["timer_wrong_128"], 0.0)
        self.assertTrue(np.isnan(out["spawn"]))                         # no next piece in this trial

    def test_an_erased_piece_keeps_nothing_though_few_pixels_are_wrong(self):
        trial = trial_with_piece()
        erased = trial.future.copy()
        erased[:, 52:140, 92:172][erased[:, 52:140, 92:172] == 9] = 0
        out = score(trial, Dream.of_frames(trial, erased, PALETTE))
        self.assertEqual(out["piece_32"], 0.0)
        self.assertLess(out["wrong_32"], 0.02)                          # the wrong-pixel score hardly sees it

    def test_no_piece_score_once_the_real_piece_has_left_the_upper_playfield(self):
        trial = trial_with_piece(rows_per_frame=4)
        self.assertTrue(np.isnan(score(trial, Dream.of_frames(trial, trial.future, PALETTE))["piece_32"]))

    def test_a_haze_is_not_a_piece_but_a_sure_piece_is(self):
        trial = trial_with_piece()
        dreamed = Dream.of_frames(trial, trial.future, PALETTE)
        piece = trial.future == 9
        sure = np.where(piece, 0.9, dreamed.filled).astype(np.float16)
        dreamed.filled = np.where(piece, 0.0, np.maximum(dreamed.filled, 0.05)).astype(np.float16)  # erased, a haze
        self.assertEqual(score(trial, dreamed)["piece_32"], 0.0)
        dreamed.filled = sure
        self.assertEqual(score(trial, dreamed)["piece_32"], 1.0)

    def test_a_frozen_piece_is_kept_but_neither_hits_nor_falls(self):
        trial = trial_with_piece()
        out = score(trial, Dream.of_frames(trial, frozen(trial), PALETTE))
        self.assertEqual(out["piece_32"], 1.0)                          # all its pixels are still there
        self.assertLess(out["piece_hit_32"], 0.2)                       # a row left where the real piece is
        self.assertGreater(out["piece_ghost_32"], 0.8)
        self.assertEqual(out["fall"], 0.0)

    def test_the_next_piece_spawns_when_the_real_one_does(self):
        trial = trial_with_piece(rows_per_frame=4, respawn=60)
        out = score(trial, Dream.of_frames(trial, trial.future, PALETTE))
        self.assertEqual((out["spawn"], out["spawn_lag"]), (1.0, 0.0))
        late = trial.future.copy()
        late[60:80] = late[59]                                          # the next piece 20 frames late
        self.assertEqual(score(trial, Dream.of_frames(trial, late, PALETTE))["spawn"], 0.0)
        out = score(trial, Dream.of_frames(trial, frozen(trial), PALETTE))
        self.assertEqual(out["spawn"], 0.0)                             # its piece never left: no next one
        self.assertTrue(np.isnan(out["spawn_lag"]))

if __name__ == "__main__":
    unittest.main()
