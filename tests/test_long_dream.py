"""Long-dream scores: expected wrong pixels, block mass, the falling piece (kept, hit, ghost, fall), the next
piece's spawn and the fall timer, on made-up trials."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from token_world.diagnostics.long_dream import (EVERY, FIRST, FUTURE, PLAYFIELD, Dream, Trial, animation, dreams,
                                                score, sheet, skip_gaps, stacked)
from token_world.models.dynamics import Dynamics

PALETTE = np.zeros((58, 3), np.uint8)


def trial_with_piece(rows_per_frame: float = 0.25, respawn: int | None = None) -> Trial:
    """A walled well (static) with an 8 x 24 piece falling from the top of the playfield; respawn: the
    frame of the future a next piece appears at the top (the first one is then gone)."""
    frames = np.zeros((48 + FIRST, 256, 256), np.uint8)
    frames[:, PLAYFIELD[0], PLAYFIELD[1].start] = 5                     # a wall that never changes
    for i in range(48, 48 + FIRST):
        if respawn is not None and i - 48 >= respawn:
            frames[i, 60:68, 120:144] = 9
            continue
        top = 60 + int((i - 48) * rows_per_frame)
        if top < 204:
            frames[i, top:top + 8, 120:144] = 9
    return Trial(frames[:48], np.zeros(48, np.int64), frames[48:], level=0)


def trial_with_tetromino(cells=((0, 0), (1, 0), (2, 0), (2, 1))) -> Trial:
    """An L of 7 x 7 blocks on the well's cell grid (x = 96 + 8 col, y = 56 + 8 row), falling a row every 8
    frames from row 0."""
    frames = np.zeros((48 + FIRST, 256, 256), np.uint8)
    for i in range(48, 48 + FIRST):
        drop = (i - 48) // 8
        for r, c in cells:
            if r + drop < 20:
                y, x = 56 + 8 * (r + drop), 96 + 8 * (4 + c)
                frames[i, y:y + 7, x:x + 7] = 9
    return Trial(frames[:48], np.zeros(48, np.int64), frames[48:], level=0)


def frozen(trial: Trial) -> np.ndarray:
    """A dream that holds the first future frame: the piece never moves."""
    return np.repeat(trial.future[:1], FIRST, 0)


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
        self.assertEqual((out["presence"], out["activation"]), (0.0, 0.0))     # the stall: nothing drawn

    def test_a_wrong_piece_is_present_and_a_faint_one_active(self):
        trial = trial_with_piece()
        out = score(trial, Dream.of_frames(trial, frozen(trial), PALETTE))
        self.assertEqual(out["presence"], 1.0)                          # wrong, but a piece to correct
        self.assertAlmostEqual(out["activation"], 1.0, places=2)
        dreamed = Dream.of_frames(trial, trial.future, PALETTE)
        dreamed.filled = np.where(trial.future == 9, 0.2, dreamed.filled).astype(np.float16)  # a faint piece
        out = score(trial, dreamed)
        self.assertEqual(out["presence"], 0.0)
        self.assertAlmostEqual(out["activation"], 0.2, places=2)

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

    def test_a_piece_keeps_its_shape_wherever_it_is_but_not_when_it_changes(self):
        trial = trial_with_tetromino()
        self.assertEqual(score(trial, Dream.of_frames(trial, trial.future, PALETTE))["shape"], 1.0)
        self.assertEqual(score(trial, Dream.of_frames(trial, frozen(trial), PALETTE))["shape"], 1.0)   # misplaced
        square = trial_with_tetromino(((0, 0), (0, 1), (1, 0), (1, 1)))                 # an O where an L falls
        self.assertEqual(score(trial, Dream.of_frames(trial, square.future, PALETTE))["shape"], 0.0)
        erased = np.zeros_like(trial.future)
        self.assertEqual(score(trial, Dream.of_frames(trial, erased, PALETTE))["shape"], 0.0)
        three = trial_with_tetromino(((0, 0), (1, 0), (2, 0)))                           # never a whole piece
        self.assertTrue(np.isnan(score(three, Dream.of_frames(three, three.future, PALETTE))["shape"]))

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


class PressedTrialTests(unittest.TestCase):
    def test_a_trial_without_buttons_presses_none_and_variants_stack(self):
        trial = trial_with_piece()
        self.assertEqual(trial.buttons.shape, (FUTURE,))
        self.assertFalse(trial.pressed)
        self.assertIn("no buttons", trial.what)
        played = Trial(trial.context, trial.actions, trial.future, 0, buttons=np.full(FUTURE, 64))
        self.assertTrue(played.pressed)
        self.assertIn("the bot's buttons", played.what)
        real = Dream.of_frames(trial, trial.future, PALETTE)
        sheets = [sheet(t, {"dream": real}, PALETTE) for t in (trial, played)]
        both = stacked(*sheets)
        self.assertEqual(both.size, (sheets[0].width, 2 * sheets[0].height))
        gifs = [animation(t, {"dream": real}, PALETTE) for t in (trial, played)]
        self.assertEqual([f.height for f in stacked(*gifs)], [2 * gifs[0][0].height] * len(gifs[0]))

    def test_a_pressed_dream_must_turn_the_piece_when_the_real_one_turns(self):
        upright = trial_with_tetromino(((0, 0), (1, 0), (2, 0), (2, 1)))
        turned = trial_with_tetromino(((0, 0), (0, 1), (0, 2), (1, 0)))
        out = score(turned, Dream.of_frames(turned, upright.future, PALETTE))
        self.assertEqual(out["shape"], 0.0)                              # an unturned L is not the turned one
        self.assertEqual(score(turned, Dream.of_frames(turned, turned.future, PALETTE))["shape"], 1.0)


class ThousandFrameTests(unittest.TestCase):
    def test_the_whole_dream_is_scored_for_lost_cells_and_morphs(self):
        trial = trial_with_tetromino()
        out = score(trial, Dream.of_frames(trial, trial.future, PALETTE))
        self.assertEqual((out["lost_long"], out["morph_long"], out["full_long"]), (0.0, 0.0, 0.0))
        melted = trial.future.copy()
        melted[64:, 56:216, 96:176] = 0                                     # the piece vanishes mid-fall
        out = score(trial, Dream.of_frames(trial, melted, PALETTE))
        self.assertGreater(out["lost_long"], 0)

    def test_trials_dreamed_side_by_side_dream_as_alone(self):
        torch.manual_seed(0)
        model = Dynamics(dim=32, layers=2, heads=2, patch=16, frames=6, colour_dim=8).eval()
        palette = np.random.default_rng(0).integers(0, 256, (58, 3)).astype(np.uint8)
        made = [trial_with_piece(), trial_with_tetromino()]
        made[1].buttons = np.random.default_rng(1).integers(0, 256, FUTURE)
        together = list(dreams(model, made, palette, pictures=1, batch=2))
        alone = [next(dreams(model, [t], palette, pictures=1, batch=1)) for t in made]
        self.assertEqual(together[0].cells.shape, (FUTURE, 20, 10))
        self.assertEqual(together[0].seen.shape, (FUTURE // EVERY, 256, 256, 3))
        self.assertIsNone(together[1].seen)                                 # pictures for the first only
        for a, b in zip(together, alone):                  # the same, up to float noise on a near tie
            self.assertGreater((a.likely == b.likely).mean(), 0.9999)
            self.assertLess(np.abs(a.sure.astype(np.float32) - b.sure.astype(np.float32)).max(), 1e-3)
            self.assertGreater((a.cells == b.cells).mean(), 0.999)


class SkipDreamTests(unittest.TestCase):
    """2026-11-03: a skip model's dreams (dreams gaps) and the scores of the frames its steps drew."""

    def test_steps_cover_the_dream_one_way_or_at_random(self):
        fixed = skip_gaps(4)
        self.assertEqual((fixed[0], set(fixed[1:].tolist())), (1, {4}))
        self.assertGreaterEqual(fixed.sum(), FUTURE)
        drawn = skip_gaps(6, np.random.default_rng(0))
        self.assertEqual(drawn[0], 1)                                       # a frame to hold from the start
        self.assertGreaterEqual(drawn.sum(), FUTURE)
        self.assertEqual(set(drawn.tolist()), set(range(1, 7)))
        self.assertTrue(np.array_equal(drawn, skip_gaps(6, np.random.default_rng(0))))   # the same every test

    def test_a_held_dream_is_behind_between_steps_and_right_on_them(self):
        for trial in (trial_with_piece(), trial_with_tetromino()):
            steps = np.zeros(FUTURE, bool)
            steps[np.cumsum(skip_gaps(6))[np.cumsum(skip_gaps(6)) <= FUTURE] - 1] = True
            last = np.maximum.accumulate(np.where(steps[:FIRST], np.arange(FIRST), 0))
            held = Dream.of_frames(trial, trial.future[last], PALETTE)     # the real game, shown at each step
            behind = score(trial, held)
            held.steps = steps
            out = score(trial, held)
            self.assertEqual(out["wrong_steps"], 0.0)
            self.assertGreater(behind["wrong_steps"], 0.0)                  # as if every frame were drawn
            if not np.isnan(out["shape_steps"]):
                self.assertEqual(out["shape_steps"], 1.0)

    def test_a_skip_dream_draws_at_its_steps_and_holds_between(self):
        torch.manual_seed(0)
        model = Dynamics(dim=32, layers=2, heads=2, patch=16, frames=6, colour_dim=8, skip=6).eval()
        torch.nn.init.normal_(model.skipped_action.weight, std=0.1)
        palette = np.random.default_rng(0).integers(0, 256, (58, 3)).astype(np.uint8)
        trial = trial_with_tetromino()
        trial.buttons = np.random.default_rng(1).integers(0, 256, FUTURE)
        gaps = skip_gaps(6, np.random.default_rng(2))
        made = next(dreams(model, [trial], palette, pictures=1, gaps=gaps))
        at = np.cumsum(gaps)
        self.assertTrue(np.array_equal(np.nonzero(made.steps)[0], at[at <= FUTURE] - 1))
        for h in range(1, FIRST):                                           # frame h + 1 held unless a step drew it
            if not made.steps[h]:
                self.assertTrue(np.array_equal(made.likely[h], made.likely[h - 1]))
        self.assertIsNone(next(dreams(model, [trial], palette)).steps)       # one frame a step: every frame drawn


if __name__ == "__main__":
    unittest.main()
