"""Tetris scenarios: windows that put the event at generated +LEAD, spread in time."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from token_world.data import tetris_bot as T
from token_world.data import tetris_events as E
from token_world.diagnostics import tetris_scenarios as S


def ram_run(length):
    """[length, 2048] RAM of a game with a piece falling, nothing changing."""
    ram = np.zeros((length, 2048), np.uint8)
    ram[:, E.MODE], ram[:, E.STATE], ram[:, T.PIECE] = E.IN_GAME, E.FALLING, 0x02
    return ram


class TimelineTests(unittest.TestCase):
    def test_event_lands_at_lead_with_aligned_actions(self):
        """Frames and actions carry their absolute index; a level up at RAM frame 200 that first
        changes the screen at picture 207 must put 207 at CONTEXT - 1 + LEAD, with the right buttons."""
        total = 64 + 63 * 12
        ram = ram_run(total)
        event, shown = 200, 207                                           # the screen changes 7 frames later
        ram[:event, E.LEVEL], ram[event:, E.LEVEL] = 2, 3
        index = np.arange(total)
        frames = np.zeros((total, 256, 256), np.uint8)
        frames[:, 0, 0] = index % 251                                      # tick border: ignored by first_change
        frames[shown:, 100, 20] = 7                                        # the first visible change
        actions = (index % 97).astype(np.int64)
        timeline, found = S.Timeline(), []
        for w in range(13):
            lo = 63 * w
            window_actions = np.append(actions[lo:lo + 63], actions[lo + 62])
            timeline.add(frames[lo:lo + 64], window_actions, ram[lo:lo + 64], game=w // 4)
            found += timeline.instances()
        levels = [f for f in found if f.scenario == "level up"]
        self.assertEqual(len(levels), 1)
        window = levels[0]
        first = shown - S.LEAD - (S.CONTEXT - 1)
        self.assertEqual(int(window.frames[S.CONTEXT - 1 + S.LEAD, 0, 0]), shown % 251)
        self.assertEqual(int(window.frames[S.CONTEXT - 2 + S.LEAD, 100, 20]), 0)   # unchanged just before
        self.assertEqual(int(window.frames[S.CONTEXT - 1 + S.LEAD, 100, 20]), 7)
        self.assertEqual(int(window.frames[0, 0, 0]), first % 251)
        self.assertEqual(window.frames.shape[0], S.FRAMES)
        self.assertTrue(np.array_equal(window.actions[:-1], actions[first:first + S.FRAMES - 1]))
        self.assertEqual((window.level, window.frame), (3, shown))

    def test_layered_windows_carry_their_layers_and_any_model_scores_them(self):
        """A layered stream's windows keep each frame's layers aligned with its picture, and a layered World
        (diagnostics/worlds.py) is scored on them like any model."""
        import tempfile
        import torch
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from test_layered import window as layered_window
        from token_world.data.nes_palette import tetris_palette
        from token_world.diagnostics.worlds import load
        from token_world.models.layered import build
        total = 64 + 63 * 5
        ram = ram_run(total)
        ram[:150, E.LEVEL], ram[150:, E.LEVEL] = 2, 3
        index = np.arange(total)
        frames = np.zeros((total, 256, 256), np.uint8)
        frames[155:, 100, 20] = 7
        layers = {k: np.repeat(v[0, :1].numpy(), total, 0) for k, v in layered_window(1, 1, 4).items()}
        layers["backdrop"] = (index % 50).astype(np.uint8)                 # each frame's own backdrop
        actions = np.zeros(total, np.int64)
        timeline, found = S.Timeline(), []
        for w in range(6):
            lo = 63 * w
            timeline.add(frames[lo:lo + 64], np.append(actions[lo:lo + 63], 0), ram[lo:lo + 64],
                         layers={k: v[lo:lo + 64] for k, v in layers.items()})
            found += timeline.instances()
        level = next(f for f in found if f.scenario == "level up")
        first = level.frame - S.LEAD - (S.CONTEXT - 1)
        self.assertTrue(np.array_equal(level.layers["backdrop"], index[first:first + S.FRAMES] % 50))
        args = {"dim": 32, "layers": 2, "heads": 2, "frames": 64, "colour_dim": 8, "kind": "layered", "model": "pixels"}
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as d:
            model = build(args)
            torch.save({"step": 7, "ema": model.state_dict(), "args": args, "palettes": {"tetris": tetris_palette()}},
                       Path(d) / "model_latest.pt")
            world = load(d, device="cpu")
        scores, rgb, wrong = S.evaluate(world, [level])
        self.assertIn(f"event_wrong_{S.LEAD}", scores[0])
        self.assertEqual(rgb.shape, (1, S.FRAMES - S.CONTEXT, 256, 256, 3))
        self.assertTrue(set(np.unique(wrong)) <= {0, 1})                   # committed: right or not

    def test_spread_spaces_instances_per_stream_in_time(self):
        now = [0.0]
        make = lambda name, stream: S.Instance(name, None, None, stream, 0, 0, 0)
        def instances():
            for t in range(0, 100):                     # one shift per stream per second, for 100 s
                now[0] = float(t)
                yield make("shift", 0)
                yield make("shift", 1)
        taken = list(S.spread(instances(), per_scenario=10, seconds=100, streams=2, clock=lambda: now[0]))
        self.assertEqual(len(taken), 10)                # gap = 100 s * 2 streams / 10 = 20 s per stream
        self.assertEqual(sorted({i.stream for i in taken}), [0, 1])

if __name__ == "__main__":
    unittest.main()
