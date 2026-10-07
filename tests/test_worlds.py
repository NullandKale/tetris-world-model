"""Every model behind one interface (diagnostics/worlds.py): checkpoints of the pixel model and of both layered
models load, roll out batches of windows and dream live, frame for frame as their own dreamers do."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

from token_world.data.model_frames import BORDER_VERSION
from token_world.data.nes_palette import tetris_palette
from token_world.diagnostics.worlds import composed, load
from token_world.models import dynamics
from token_world.models.layered import build as build_layered


def save(folder: Path, model, args: dict) -> Path:
    torch.save({"step": 1234, "model": model.state_dict(), "ema": model.state_dict(), "args": args,
                "palettes": {"tetris": tetris_palette()}}, folder / "model_latest.pt")
    return folder


class WorldTests(unittest.TestCase):
    def test_a_pixel_model_rolls_out_and_dreams(self):
        args = {"dim": 32, "layers": 2, "heads": 2, "patch": 16, "frames": 8, "colour_dim": 8, "border": BORDER_VERSION}
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as d:
            world = load(save(Path(d), dynamics.build(args), args), device="cpu")
        self.assertFalse(world.layered)
        self.assertEqual(world.step, 1234)
        frames = np.random.default_rng(0).integers(0, 55, (2, 8, 256, 256)).astype(np.uint8)
        actions = np.zeros((2, 8), np.int64)
        made, right, same = world.rollout(frames, actions, 5)
        self.assertEqual(made.shape, (2, 3, 256, 256))
        for p in (right, same):
            self.assertEqual(p.shape, (2, 3, 256, 256))
            self.assertTrue(((p >= 0) & (p <= 1)).all())
        live = world.dreaming(frames[0, :5], actions[0, :5])
        frame, probs = live.step(0)
        self.assertEqual(frame.shape, (256, 256))
        self.assertEqual(probs.shape[:2], (256, 256))

    def test_layered_models_roll_out_as_their_dreamers_dream(self):
        from test_layered import window
        layers = window(2, 8, 3)
        frames = np.stack([np.stack([composed({k: v[b, t].numpy() for k, v in layers.items()}) for t in range(8)])
                           for b in range(2)])
        actions = np.zeros((2, 8), np.int64)
        for kind in ("pixels", "slots"):
            args = {"dim": 32, "layers": 2, "heads": 2, "frames": 8, "colour_dim": 8, "kind": "layered", "model": kind}
            torch.manual_seed(0)
            with tempfile.TemporaryDirectory() as d:
                world = load(save(Path(d), build_layered(args), args), device="cpu")
            self.assertTrue(world.layered)
            with self.assertRaises(ValueError):
                world.rollout(frames, actions, 5)                        # a layered model needs layered context
            made, right, same = world.rollout(frames, actions, 5, {k: v.numpy() for k, v in layers.items()},
                                              temperature=0.0)
            self.assertTrue(((right == 1) == (made == frames[:, 5:])).all())       # committed: 0 or 1
            self.assertTrue(((same == 1) == (made == frames[:, 4:5])).all())
            self.assertEqual(made.shape, (2, 3, 256, 256))
            dreamer = world.model.Dreamer(world.model, {k: v[:, :5] for k, v in layers.items()},
                                          torch.as_tensor(actions[:, :5]), temperature=0.0)
            for i in range(3):
                step = {k: v[1].numpy() for k, v in dreamer.step(torch.zeros(2, dtype=torch.long)).items()
                        if k != "probs"}
                self.assertTrue((composed(step) == made[1, i]).all(), (kind, i))
            live = world.dreaming(frames[0, :5], actions[0, :5], {k: v[0, :5].numpy() for k, v in layers.items()},
                                  temperature=0.0)
            self.assertTrue((live.step(0)[0] == made[0, 0]).all(), kind)


if __name__ == "__main__":
    unittest.main()
