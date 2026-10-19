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


@unittest.skipUnless(ROM.is_file() and importlib.util.find_spec("world_nes"), "needs the Tetris ROM and world_nes")
class BranchTests(unittest.TestCase):
    def test_a_replay_from_a_saved_state_plays_the_window_again(self):
        from token_world.data.model_frames import border_slots
        from token_world.data.nes_palette import tetris_palette
        from token_world.data.world_nes_tetris import BORDER_STATE, TetrisSession
        from token_world.data.world_nes_windows import world_nes_windows
        checked = []

        def again(window, replay):                  # the window's own buttons from observation 20 on
            x, ram = replay(20, window["action"].numpy()[20:-1])
            checked.append(np.array_equal(x, window["x"][21:].numpy()) and np.array_equal(ram, window["ram"][21:].numpy()))
            return None
        windows = world_nes_windows(tetris_rom(), TetrisSession(5), border_slots(tetris_palette()), BORDER_STATE,
                                    64, 16, 0, branch=again)
        for _, window in zip(range(12), windows):
            self.assertNotIn("branch", window)
        self.assertEqual(len(checked), 12)
        self.assertTrue(all(checked))                       # the border's counter and state bytes too

    def test_a_twin_differs_from_its_window_by_the_turn_button_alone(self):
        from token_world.data.tetris_bot import A, B
        from token_world.data.world_nes_tetris import BRANCH_FROM, KEEP_ALL, WorldNesTetrisStreams
        stream = WorldNesTetrisStreams(64, seed=5, stride=16, branches=1, motion=True, toss=KEEP_ALL)  # as trained
        parts = [part for _, part in zip(range(40), iter(stream))]
        self.assertTrue(all("branch" in p for p in parts))  # held back until the first twin
        kinds, differed = set(), 0
        for p in parts:
            twin, at = p["branch"], p["branch"]["at"]
            if twin["ram"][at].tolist() != p["ram"][at].tolist():
                continue                                    # a reused twin: from an earlier window
            self.assertGreaterEqual(at, BRANCH_FROM)
            self.assertTrue(torch.equal(twin["x"][:at + 1], p["x"][:at + 1]))
            self.assertTrue(torch.equal(twin["action"][:at], p["action"][:at]))
            real, made = int(p["action"][at]), int(twin["action"][at])
            self.assertIn(real & (A | B), (A, B))
            self.assertEqual(real & ~(A | B), made & ~(A | B))  # the other buttons as played
            self.assertEqual(made & (A | B), (A | B) ^ (real & (A | B)) if twin["kind"] == "swapped" else 0)
            self.assertEqual(twin["motion"].shape, (64, 32, 32))
            kinds.add(twin["kind"])
            differed += not torch.equal(twin["x"], p["x"])
        self.assertEqual(kinds, {"swapped", "released"})
        self.assertGreater(differed, 0)


class ScreensTests(unittest.TestCase):
    def test_no_wait_passes_level_0_gravity(self):
        from token_world.data.world_nes_tetris import STILL, VAE, WORLD_MODEL, Screens
        self.assertEqual(STILL, 48)
        for setting in (WORLD_MODEL, VAE):
            self.assertLessEqual(max(setting.pause[1], setting.wait[1]), STILL)
        with self.assertRaises(ValueError):
            Screens(pause_every=100, pause=(4, STILL + 1), end_reset=0.1, reset_every=100)
        with self.assertRaises(ValueError):
            Screens(pause_every=100, pause=(4, 8), end_reset=0.1, reset_every=100, wait=(0, STILL + 1))

    def test_toss_keeps_the_windows_with_what_it_asks_for(self):
        from token_world.data.world_nes_tetris import GAME_MODE, PAUSE_FLAG, Toss, WindowToss
        ram = np.zeros((300, 2048), np.uint8)
        ram[:, GAME_MODE] = 4
        ram[50:60, GAME_MODE] = 3                    # a menu, then a game starts at frame 60
        ram[120:126, PAUSE_FLAG] = 7                 # a pause
        toss = WindowToss(Toss(keep={"game start": 1.0, "paused": 1.0}, plain=0.0, aftermath=8),
                          np.random.default_rng(0), stride=8)
        kept, judged = [], 0
        for start in range(0, 300 - 16 + 1, 8):
            window = toss.push({"start": start}, ram[start:start + 16], np.zeros(15, np.int64))
            if window is not None:
                kept.append((window["start"], window["toss_names"]))
                judged += window["toss_judged"]
        self.assertEqual([s for s, _ in kept], [48, 56, 64, 112, 120])      # 64: the game start's aftermath
        self.assertIn("game start", kept[0][1])
        self.assertIn("paused", kept[3][1])
        self.assertEqual(judged, 121 // 8 + 1)       # every window up to the last kept one was judged

    def test_toss_can_judge_only_a_long_windows_last_frames(self):
        from token_world.data.world_nes_tetris import GAME_MODE, PAUSE_FLAG, Toss, WindowToss
        ram = np.zeros((300, 2048), np.uint8)
        ram[:, GAME_MODE] = 4
        ram[50:60, GAME_MODE] = 3                    # a menu, then a game starts at frame 60
        ram[120:126, PAUSE_FLAG] = 7                 # a pause
        toss = WindowToss(Toss(keep={"game start": 1.0, "paused": 1.0}, plain=0.0, aftermath=8),
                          np.random.default_rng(0), stride=8, judged=8)
        kept = []
        for start in range(0, 300 - 40 + 1, 8):     # 40 frames: 32 of history, then the 8 judged
            window = toss.push({"start": start}, ram[start:start + 40], np.zeros(39, np.int64))
            if window is not None:
                kept.append(window["start"])
        self.assertEqual(kept, [24, 32, 88])         # the start in frames 56-71 (with its aftermath), the pause in 120-127

    def test_a_marked_toss_hands_back_every_window_with_its_choice(self):
        from token_world.data.world_nes_tetris import GAME_MODE, PAUSE_FLAG, Toss, WindowToss
        ram = np.zeros((300, 2048), np.uint8)
        ram[:, GAME_MODE] = 4
        ram[50:60, GAME_MODE] = 3
        ram[120:126, PAUSE_FLAG] = 7

        def toss(marked):
            return WindowToss(Toss(keep={"game start": 1.0, "paused": 1.0}, plain=0.0, aftermath=8),
                              np.random.default_rng(0), stride=8, marked=marked)

        dropping, marking, starts, kept = toss(False), toss(True), [], []
        for start in range(0, 300 - 16 + 1, 8):
            window = {"start": start}
            dropped = dropping.push(window, ram[start:start + 16], np.zeros(15, np.int64))
            marked = marking.push(window, ram[start:start + 16], np.zeros(15, np.int64))
            if marked is not None:
                starts.append(marked["start"])
                if marked["keep"]:
                    kept.append(marked["start"])
            if dropped is not None:
                self.assertEqual(dropped["start"], marked["start"])
        self.assertEqual(starts, list(range(0, starts[-1] + 1, 8)))                   # every window, in order
        self.assertEqual(kept, [48, 56, 64, 112, 120])                                # the same choices

    def test_screens_label_from_ram(self):
        from token_world.data.world_nes_tetris import GAME_MODE, PAUSE_FLAG, PLAY_STATE, SCREENS, screens
        ram = np.zeros((5, 2048), np.uint8)
        ram[:, GAME_MODE] = (4, 4, 4, 0, 3)
        ram[1, PAUSE_FLAG] = 7
        ram[2, PLAY_STATE] = 10
        self.assertEqual([SCREENS[i] for i in screens(ram)], ["play", "paused", "game over", "legal", "menu"])


if __name__ == "__main__":
    unittest.main()
