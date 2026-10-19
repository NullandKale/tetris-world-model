import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from token_world.data.rom import tetris_rom
from token_world.data.tetris_bot import A, B, PIECE, TYPE_OF, X, Y
from token_world.data.tetris_events import EVENTS, FALLING, IN_GAME, MODE, STATE, frame_events
from token_world.models.dynamics import COLOURS, Dynamics

J = [o for o, t in TYPE_OF.items() if t == "J"]


def turned_window(shown_at: int, t: int = 8):
    """A falling J that turns between RAM frames 3 and 4, its picture changing first at `shown_at`."""
    ram = np.zeros((t, 2048), np.uint8)
    ram[:, MODE], ram[:, STATE], ram[:, X], ram[:, Y] = IN_GAME, FALLING, 5, 6
    ram[:4, PIECE], ram[4:, PIECE] = J[0], J[1]
    frames = np.zeros((t, 256, 256), np.uint8)
    frames[shown_at:, 100, 120] = 7                                    # in the well
    frames[1:, 3, 3] = np.arange(1, t) % 2                     # the border ticks every frame (ignored)
    actions = np.zeros(t, np.int64)
    actions[3] = A
    return frames, ram, actions


class FrameEventTests(unittest.TestCase):
    def test_an_event_is_labelled_on_the_first_picture_that_shows_it(self):
        rotate = 1 << EVENTS.index("rotate")
        for shown in (5, 6):                                           # the picture after RAM frame 4, or later
            out = frame_events(*turned_window(shown))
            self.assertEqual(out[shown] & rotate, rotate, shown)
            self.assertEqual(int((out & rotate).astype(bool).sum()), 1)
        out = frame_events(*turned_window(20))                         # never shown in the window: no label
        self.assertFalse((out & rotate).any())


class LiveScoreTests(unittest.TestCase):
    def test_the_batch_scores_its_event_frames_and_the_swapped_turn(self):
        import train_dynamics_ui as D
        torch.manual_seed(0)
        model = Dynamics(dim=32, layers=2, heads=2, patch=16, frames=6, colour_dim=8).eval()
        x = torch.randint(0, COLOURS, (2, 6, 256, 256)).to(torch.uint8)
        x[:, 4] = x[:, 3]
        x[:, 4, 100:110, 120:130] = (x[:, 3, 100:110, 120:130] + 1) % COLOURS   # what changes at frame 4
        events = torch.zeros(2, 6, dtype=torch.long)
        events[0, 4] = 1 << EVENTS.index("rotate")
        events[1, 4] = 1 << EVENTS.index("spawn")
        events[1, 3] = 1 << EVENTS.index("gravity (no buttons)")      # the rarer spawn is scored instead
        piece = torch.full((2, 6), J[0], dtype=torch.long)
        incoming = torch.zeros(2, 6, dtype=torch.long)
        incoming[0, 3] = A
        with torch.no_grad():
            out = D.live_event_scores(model, model, x, incoming, events, piece=piece,
                                      generator=torch.Generator().manual_seed(0))
        self.assertEqual(out["rotate n"], 1.0)
        self.assertEqual(out["spawn n"], 1.0)
        self.assertNotIn("gravity (no buttons) n", out)
        self.assertEqual(out["turn n"], 1.0)
        for k in ("rotate exact", "rotate wrong", "rotate false", "turn exact swapped", "turn wrong swapped"):
            self.assertTrue(0 <= out[k] <= 1, k)
        means = D.live_means({**out, "spawn n": 2.0, "spawn exact": 1.0})
        self.assertEqual(means["live_spawn_exact"], 0.5)
        self.assertTrue(math.isnan(means["live_level_up_exact"]))
        self.assertAlmostEqual(means["live_turn_gap"], out["turn exact"] - out["turn exact swapped"])
        self.assertEqual(set(means), set(D.LIVE_FIELDS))
        none = D.live_event_scores(model, model, x, incoming, torch.zeros(2, 6, dtype=torch.long))
        self.assertEqual(none, {})


@unittest.skipUnless(tetris_rom().is_file() and importlib.util.find_spec("world_nes"), "needs the ROM and world_nes")
class LiveStreamEventTests(unittest.TestCase):
    def test_the_stream_labels_its_moves(self):
        from token_world.data.world_nes_tetris import WORLD_MODEL, WorldNesTetrisStreams
        stream = WorldNesTetrisStreams(64, seed=5, stride=63, screens=WORLD_MODEL)
        parts = [part for _, part in zip(range(30), iter(stream))]
        events = torch.stack([p["events"] for p in parts])
        self.assertEqual(events.shape, (30, 64))
        for name in ("shift", "rotate", "spawn"):
            self.assertTrue(((events >> EVENTS.index(name)) & 1).any(), name)


if __name__ == "__main__":
    unittest.main()
