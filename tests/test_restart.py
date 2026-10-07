"""The restart check (diagnostics/restart.py): milestones are scored where the restart says, and any model dreams
a restart fed its real buttons."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

from token_world.data.nes_palette import tetris_palette
from token_world.diagnostics.restart import FUTURE, Restart, marks, score, sheet
from token_world.diagnostics.worlds import composed, load
from token_world.models.layered import build as build_layered


def restart(context: np.ndarray, layers=None) -> Restart:
    future = np.zeros((FUTURE, 256, 256), np.uint8)
    future[100:] = 3                                       # the menu and the game look different from the top-out
    return Restart(context, np.zeros(len(context), np.int64), layers, future, np.zeros(FUTURE, np.int64), 100, 200)


class RestartTests(unittest.TestCase):
    def test_milestones(self):
        r = restart(np.zeros((8, 256, 256), np.uint8))
        dreamed = r.future.copy()
        dreamed[r.play + 8, :128] = 9                       # half the screen wrong 8 frames into the new game
        s = score(r, dreamed)
        self.assertEqual(s["play"], 0.5)
        self.assertEqual((s["curtain"], s["menu"], s["play64"]), (0.0, 0.0, 0.0))
        self.assertEqual((s["copy_menu"], s["copy_play"]), (1.0, 1.0))     # copying the top-out is all wrong
        palette = tetris_palette().numpy().astype(np.uint8)
        self.assertEqual(sheet(r, {"a": dreamed}, palette).shape, (512, len(marks(r)) * 256, 3))

    def test_a_layered_model_dreams_a_restart(self):
        from test_layered import window
        layers = window(1, 8, 3)
        context = np.stack([composed({k: v[0, t].numpy() for k, v in layers.items()}) for t in range(8)])
        args = {"dim": 32, "layers": 2, "heads": 2, "frames": 8, "colour_dim": 8, "kind": "layered", "model": "pixels"}
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as d:
            model = build_layered(args)
            torch.save({"step": 5, "model": model.state_dict(), "ema": model.state_dict(), "args": args,
                        "palettes": {"tetris": tetris_palette()}}, Path(d) / "model_latest.pt")
            world = load(d, device="cpu")
        r = restart(context, {k: v[0].numpy() for k, v in layers.items()})
        r.future, r.future_actions = r.future[:12], r.future_actions[:12]
        made = world.follow(r.context, r.actions, r.layers, r.future_actions)
        self.assertEqual(made.shape, (12, 256, 256))


if __name__ == "__main__":
    unittest.main()
