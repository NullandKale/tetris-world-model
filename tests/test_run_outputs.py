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
                                                 horizon_outputs, load_optimizer, open_metrics, optimizer_state,
                                                 write_horizon_curve)
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


class GifTests(unittest.TestCase):
    def test_the_real_panel_keeps_a_small_sprite_s_exact_colours(self):
        # 2026-10-31: PIL gave each preview frame its own 256 colours, and beside the generated panel's blends
        # small Mario (red, brown, skin) came out one orange in the REAL panel (28% of his pixels exact)
        import numpy as np
        from PIL import Image
        from PIL import ImageSequence
        try:
            from token_world.data.world_nes_smb import smb_palette
        except ImportError:                                  # the public repository ships no SMB code
            self.skipTest("needs data/world_nes_smb.py")
        palette = smb_palette()
        g = torch.Generator().manual_seed(0)
        real = torch.zeros(1, max(HORIZONS), 256, 256, dtype=torch.uint8)               # night sky
        real[:, :, 200:] = 40                                                            # bricks ($17)
        for t in range(real.shape[1]):                                                   # Mario: $16, $18, $27
            real[0, t, 180:196, 100 + t:112 + t] = torch.tensor([42, 30, 52])[
                torch.randint(0, 3, (16, 12), generator=g)].to(torch.uint8)
        last = real[:, 0].clone()
        probs = torch.softmax(torch.randn(*real.shape, 58, generator=g) * 0.5, -1)
        probs = 0.5 * probs + 0.5 * torch.nn.functional.one_hot(real.long(), 58).float()   # an early model
        rgb = (probs @ palette.float()).round().clamp(0, 255).byte()
        wrong = 1 - probs.gather(-1, real[..., None].long())[..., 0]
        want = palette.numpy()[real[0].numpy()]
        with tempfile.TemporaryDirectory() as d:
            horizon_outputs(real, rgb, wrong, last, palette, Path(d), "smb")
            back = [np.asarray(f.convert("RGB")) for f in ImageSequence.Iterator(Image.open(Path(d) / "previews/smb_0.gif"))]
        for t, frame in enumerate(back[1:]):                                             # REAL +1 ..
            self.assertTrue(np.array_equal(frame[18:274, :256], want[t]), t)


class LayeredPreviewTests(unittest.TestCase):
    def test_the_layered_trainer_writes_the_window_s_previews(self):
        import train_layered
        from test_layered import KINDS, tiny, window
        layers, incoming = window(2, 24, 7), torch.randint(0, 256, (2, 24))
        for kind in KINDS:
            model = tiny(kind, frames=24).eval()
            with tempfile.TemporaryDirectory() as d:
                out = Path(d)
                m = train_layered.preview(model, layers, incoming, out, 1.0)
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
                losses = train_layered.train_step(model, model, layers, incoming, depth=6, blend=blend, drawer=drawer)
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

    def test_the_window_shows_no_button_and_pressed_dreams_apart(self):
        from matplotlib.figure import Figure
        from token_world.ui.run_viewer import (CURVE_VIEWS, RunData, compare_table, draw_curves, fit_steps,
                                               long_dream_table, pressed)
        self.assertEqual((pressed("1 pass"), pressed("layered t=1"), pressed("buttons"), pressed("buttons t=1")),
                         (False, False, True, True))
        row = {"level": 0, "wrong_16": 0.01, "wrong_32": 0.02, "wrong_64": 0.03, "wrong_128": 0.04, "mass": 1.0,
               "piece_32": 1.0, "piece_hit_32": 0.5, "presence": 1.0, "shape": 0.8}
        with tempfile.TemporaryDirectory() as d:
            mine, theirs = Path(d) / "mine", Path(d) / "theirs"
            mine.mkdir(), theirs.mkdir()
            append_long_dream(mine, [{"step": 2000, "variant": v, **row} for v in ("1 pass", "buttons")])
            append_long_dream(theirs, [{"step": 2000, "variant": "layered t=1", **row}])
            runs = [RunData(mine), RunData(theirs)]
            for run in runs:
                run.refresh()
            for view in CURVE_VIEWS:                                  # every view draws, with or without data
                fig = Figure()
                draw_curves(fig, runs[:1], runs[1:], view)
                fit_steps(fig, runs[:1], view)
            table = compare_table(*runs)
            self.assertIn("no buttons:", table)
            self.assertIn("2,000", table.split("the bot's buttons")[0])
            self.assertIn("the bot's buttons: not tested in both runs yet", table)
            self.assertIn("buttons", long_dream_table(runs))

    def test_the_stall_warning(self):
        self.assertEqual(stalling(None, {"presence": 0.8, "wrong_128": 0.03}), "")
        self.assertIn("STALLING", stalling(None, {"presence": 0.1, "wrong_128": 0.03}))
        falling = stalling({"presence": 0.8, "wrong_128": 0.04}, {"presence": 0.5, "wrong_128": 0.03})
        self.assertIn("erasing pieces", falling)
        self.assertEqual(stalling(None, {"presence": float("nan"), "wrong_128": 0.03}), "")


class OptimizerStateTests(unittest.TestCase):
    def test_a_resume_matches_optimizer_state_by_name_whatever_the_order(self):
        torch.manual_seed(0)
        a, b = torch.nn.Parameter(torch.randn(3, 4)), torch.nn.Parameter(torch.randn(5))
        first = torch.optim.AdamW([a, b])
        (a.sum() + b.sum() * 2).backward()
        first.step()
        saved = optimizer_state(first, ["a", "b"])
        again = torch.optim.AdamW([b, a])                                   # the model registers them the other way
        load_optimizer(again, saved, ["b", "a"])
        self.assertTrue(torch.equal(again.state[a]["exp_avg"], first.state[a]["exp_avg"]))
        self.assertTrue(torch.equal(again.state[b]["exp_avg"], first.state[b]["exp_avg"]))
        positional = {k: v for k, v in saved.items() if k != "names"}      # saved before names: by position
        with self.assertRaises(ValueError):
            load_optimizer(torch.optim.AdamW([b, a]), positional, ["b", "a"])

    def test_batched_muon_is_torchs_muon(self):
        """Newton-Schulz on same-shaped matrices at once: torch's Muon's steps (tall, wide and square, several of
        a shape), within bf16's rounding of its one-matrix matmuls."""
        from token_world.diagnostics.run_outputs import BatchedMuon
        torch.manual_seed(0)
        shapes = [(48, 16), (48, 16), (16, 48), (16, 16), (16, 16), (16, 16), (32, 24)]
        start = [torch.randn(*s) for s in shapes]
        runs = []
        for kind in (torch.optim.Muon, BatchedMuon):
            ps = [torch.nn.Parameter(t.clone()) for t in start]
            opt = kind(ps, lr=0.02, weight_decay=0.1, adjust_lr_fn="match_rms_adamw")
            g = torch.Generator().manual_seed(1)
            for _ in range(3):
                for p in ps:
                    p.grad = torch.randn(p.shape, generator=g)
                opt.step()
            runs.append((ps, opt))
        (want, torch_opt), (got, ours) = runs
        for a, b in zip(got, want):
            self.assertLess((a - b).abs().max().item(), 2e-3)
            self.assertTrue(torch.equal(ours.state[a]["momentum_buffer"], torch_opt.state[b]["momentum_buffer"]))

    def test_muon_and_adamw_resume_each_part_by_name(self):
        from token_world.diagnostics.run_outputs import MuonAdamW
        torch.manual_seed(0)
        w, e, bias = (torch.nn.Parameter(torch.randn(*s)) for s in ((6, 4), (3, 4), (6,)))

        def made(order):
            groups = [{"names": ["e"], "params": [e], "weight_decay": 0.1},
                      {"names": ["bias"], "params": [bias], "weight_decay": 0.0}][::order]
            return MuonAdamW([("w", w)], groups, 1e-3, 0.1)
        first = made(1)
        (w.sum() + e.sum() + bias.sum()).backward()
        first.step()
        self.assertIn("momentum_buffer", first.muon.state[w])
        self.assertEqual(len(first.param_groups), 3)                       # a schedule sets all of them
        saved = optimizer_state(first)
        again = made(-1)
        load_optimizer(again, saved)
        self.assertTrue(torch.equal(again.muon.state[w]["momentum_buffer"], first.muon.state[w]["momentum_buffer"]))
        self.assertTrue(torch.equal(again.adamw.state[e]["exp_avg"], first.adamw.state[e]["exp_avg"]))
        with self.assertRaises(ValueError):                                # an AdamW-only state: refused
            load_optimizer(made(1), optimizer_state(torch.optim.AdamW([w]), ["w"]))
        adamw = torch.optim.AdamW([{"params": [w, e], "weight_decay": 0.1}, {"params": [bias], "weight_decay": 0.0}])
        adamw.step()                                                       # an AdamW-only run, post-trained:
        post = made(1)
        load_optimizer(post, optimizer_state(adamw, ["w", "e", "bias"]), from_adamw=True)
        self.assertEqual(len(post.muon.state), 0)                          # Muon's matrix starts fresh,
        self.assertTrue(torch.equal(post.adamw.state[e]["exp_avg"], adamw.state[e]["exp_avg"]))   # the rest kept
        self.assertTrue(torch.equal(post.adamw.state[bias]["exp_avg_sq"], adamw.state[bias]["exp_avg_sq"]))
        with self.assertRaises(ValueError):
            MuonAdamW([("bias", bias)], [], 1e-3, 0.1)                      # Muon: matrices only


class ScheduleTests(unittest.TestCase):
    def test_the_cooldown_falls_as_one_minus_sqrt_to_zero(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import train_dynamics_ui as D
        from types import SimpleNamespace
        args = SimpleNamespace(lr=3e-4, warmup=1000, cooldown=(88000, 12000), hold_lr=None, stairs=0)
        self.assertAlmostEqual(D.learning_rate(50_000, args), 3e-4)
        self.assertAlmostEqual(D.learning_rate(91_000, args), 1.5e-4)          # a quarter through: 1 - 0.5
        self.assertEqual(D.learning_rate(100_000, args), 0.0)

    def test_muon_takes_the_hidden_matrices_and_adamw_the_edges(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import train_dynamics_ui as D
        from token_world.models.dynamics import Dynamics
        model = Dynamics(dim=32, layers=2, heads=2, patch=16, frames=6, colour_dim=8, two_layers=True,
                         conv_ends=(8, 12, 16), shared_heads=True, latent=(1, 8))
        o = D.make_optimizer(model, 3e-4, 0.05)
        self.assertIn("blocks.0.mlp.0.weight", o.muon_names)
        self.assertIn("prior_blocks.1.qkv.weight", o.muon_names)
        for edge in ("action.weight", "pixel.weight", "sprite_pixel.weight", "choice.weight", "prior_out.3.weight",
                     "posterior_out.weight", "colour.weight", "space_pos", "patchify.weight"):
            self.assertIn(edge, o.adamw_names)
        self.assertEqual(len(o.muon_names) + len(o.adamw_names), len(list(model.parameters())))


if __name__ == "__main__":
    unittest.main()
