"""What a run writes for its window (diagnostics/run_outputs.py), from frames alone, and the layered
trainer's previews through it (scripts/train_layered.py)."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "tests")]

from token_world.data.model_frames import BAND
from token_world.data.nes_palette import tetris_palette
from token_world.diagnostics.run_outputs import (HORIZONS, KINDS, PREVIEW_GIFS, append_long_dream, horizon_fields,
                                                 horizon_outputs, open_metrics, write_horizon_curve)
from token_world.ui.dynamics_app import stalling


def frames(b=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    real = torch.randint(0, 55, (b, max(HORIZONS), 256, 256), generator=g, dtype=torch.uint8)
    last = torch.randint(0, 55, (b, 256, 256), generator=g, dtype=torch.uint8)
    return real, last


class RunOutputTests(unittest.TestCase):
    def test_exact_frames_score_zero_and_every_file_is_written(self):
        real, last = frames()
        palette = tetris_palette()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            m = horizon_outputs(real, palette[real.long()], torch.zeros(real.shape), last, palette, out, "tetris")
            write_horizon_curve(m, ("tetris",), out)
            self.assertEqual(set(m), set(horizon_fields(("tetris",))))
            for h in HORIZONS:
                self.assertEqual(m[f"tetris_h{h}_wrong"], 0.0)
                self.assertEqual(m[f"tetris_h{h}_wrong_changed"], 0.0)
                self.assertGreater(m[f"tetris_h{h}_border_copy"], 0.9)        # random frames: copying fails
            self.assertTrue((out / "rollout_tetris.png").is_file())
            self.assertEqual(len(list((out / "previews").glob("tetris_*.gif"))), PREVIEW_GIFS)
            with (out / "horizon_curve.csv").open() as f:
                rows = list(csv.reader(f))
            self.assertEqual(rows[0], ["game", "horizon", *KINDS])
            self.assertEqual(len(rows), 1 + len(HORIZONS))

    def test_wrong_pixels_are_counted_where_they_are(self):
        real, last = frames()
        wrong = torch.zeros(real.shape)
        wrong[:, 0, :BAND] = 1                                              # the top border wrong at +1
        palette = tetris_palette()
        with tempfile.TemporaryDirectory() as d:
            m = horizon_outputs(real, palette[real.long()], wrong, last, palette, Path(d), "tetris")
        self.assertAlmostEqual(m["tetris_h1_border_wrong"], 0.5)
        self.assertEqual(m["tetris_h1_wrong"], 0.0)                           # game rows only
        self.assertEqual(m["tetris_h2_border_wrong"], 0.0)

    def test_metrics_with_other_columns_are_kept_aside(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "metrics.csv"
            path.write_text("step,old\n1,2\n")
            f, writer = open_metrics(path, ["step", "new"])
            writer.writerow({"step": 3, "new": 4, "ignored": 5})
            f.close()
            self.assertEqual(path.read_text().splitlines(), ["step,new", "3,4"])
            kept = list(Path(d).glob("metrics_schema_*.csv"))
            self.assertEqual(len(kept), 1)
            self.assertEqual(kept[0].read_text().splitlines()[0], "step,old")


class LayeredPreviewTests(unittest.TestCase):
    def test_the_layered_trainer_writes_the_window_s_previews(self):
        import train_layered
        from test_layered import KINDS, tiny, window
        layers, incoming = window(2, 24, 7), torch.randint(0, 256, (2, 24))
        for kind in KINDS:
            model = tiny(kind, frames=24).eval()
            with tempfile.TemporaryDirectory() as d:
                out = Path(d)
                m = train_layered.preview(model, layers, incoming, out, 1.0, steps=2)
                self.assertEqual(set(m), set(horizon_fields(("tetris",))))
                for value in m.values():
                    self.assertTrue(0.0 <= value <= 1.0)
                self.assertTrue((out / "rollout_tetris.png").is_file())
                self.assertTrue((out / "horizon_curve.csv").is_file())
                self.assertTrue(list((out / "previews").glob("tetris_*.gif")))
            self.assertTrue(model.training)                                 # put back to training


    def test_a_stage_b_step_trains_on_its_own_rollout(self):
        import train_layered
        from test_layered import KINDS, tiny, window
        layers, incoming = window(2, 12, 11), torch.randint(0, 256, (2, 12))
        for kind in KINDS:
            torch.manual_seed(3)
            model, drawer = tiny(kind, frames=12), tiny(kind, frames=12).eval()
            drawer.load_state_dict(model.state_dict())
            for blend in (0, 4):
                losses = train_layered.train_step(model, model, layers, incoming, depth=6, blend=blend, drawer=drawer,
                                                  steps=2)
                self.assertTrue(torch.isfinite(losses["total"]), kind.__name__)
                self.assertTrue(1 <= losses["rollout_frames"].item() <= 6)
                model.zero_grad()
                losses["total"].backward()
                self.assertGreater(model.cell_pixels.weight.grad.abs().sum().item(), 0)
            base = train_layered.train_step(model, model, layers, incoming)
            self.assertEqual(base["rollout_frames"].item(), 0)
            self.assertEqual(set(base) - {"total", "masked", "rollout_frames", "blend"}, set(model.PARTS))
            self.assertTrue(set(kind.PARTS) <= set(train_layered.PARTS))


class LongDreamOutputTests(unittest.TestCase):
    def test_new_columns_start_a_new_file_and_keep_the_old(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            append_long_dream(out, [{"step": 1, "wrong_128": 0.1}])
            append_long_dream(out, [{"step": 2, "wrong_128": 0.1}])
            append_long_dream(out, [{"step": 3, "wrong_128": 0.1, "presence": 0.9}])
            with (out / "long_dream.csv").open(newline="") as f:
                self.assertEqual([r["step"] for r in csv.DictReader(f)], ["3"])
            old = list(out.glob("long_dream_schema_*.csv"))
            self.assertEqual(len(old), 1)
            with old[0].open(newline="") as f:
                self.assertEqual([r["step"] for r in csv.DictReader(f)], ["1", "2"])

    def test_the_stall_warning(self):
        self.assertEqual(stalling(None, {"presence": 0.8, "wrong_128": 0.03}), "")
        self.assertIn("STALLING", stalling(None, {"presence": 0.1, "wrong_128": 0.03}))
        falling = stalling({"presence": 0.8, "wrong_128": 0.04}, {"presence": 0.5, "wrong_128": 0.03})
        self.assertIn("erasing pieces", falling)
        self.assertEqual(stalling(None, {"presence": float("nan"), "wrong_128": 0.03}), "")


class OptimizerStateTests(unittest.TestCase):
    def test_a_resume_matches_optimizer_state_by_name_whatever_the_order(self):
        import train_layered
        torch.manual_seed(0)
        a, b = torch.nn.Parameter(torch.randn(3, 4)), torch.nn.Parameter(torch.randn(5))
        first = torch.optim.AdamW([a, b])
        (a.sum() + b.sum() * 2).backward()
        first.step()
        saved = train_layered.optimizer_state(first, ["a", "b"])
        again = torch.optim.AdamW([b, a])                                   # the model registers them the other way
        train_layered.load_optimizer(again, saved, ["b", "a"])
        self.assertTrue(torch.equal(again.state[a]["exp_avg"], first.state[a]["exp_avg"]))
        self.assertTrue(torch.equal(again.state[b]["exp_avg"], first.state[b]["exp_avg"]))
        positional = {k: v for k, v in saved.items() if k != "names"}      # saved before names: by position
        with self.assertRaises(ValueError):
            train_layered.load_optimizer(torch.optim.AdamW([b, a]), positional, ["b", "a"])


if __name__ == "__main__":
    unittest.main()
