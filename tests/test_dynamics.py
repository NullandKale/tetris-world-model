"""World model: exact pixel patches, causality, masking, the masked loss, action conditioning, decoding."""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from token_world.models.dynamics import (COLOURS, GENERATED_LEVEL, LEVELS, Decoding, Dreamer, corrupt_context, Dynamics, build, decode_frame, deepen, incoming_actions,
                                         masked_loss, patch_pixels, rollout, training_mask,
                                         unpatch_pixels)


def tiny():
    torch.manual_seed(0)
    model = Dynamics(dim=32, layers=2, heads=2, patch=16, frames=6, colour_dim=8)
    with torch.no_grad():
        model.level.weight.normal_(0, 0.5)                   # trained levels are not zero
        model.colour_bias.normal_(0, 0.5)
    return model


def window(batch=2, frames=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, COLOURS, (batch, frames, 256, 256), generator=g).to(torch.uint8)


class DynamicsTests(unittest.TestCase):
    def test_patches_round_trip_and_order(self):
        index = window(1, 1)[0, 0]
        pixels = patch_pixels(index, 16)
        self.assertEqual(pixels.shape, (256, 256))
        self.assertTrue(torch.equal(unpatch_pixels(pixels, 16), index))
        self.assertTrue(torch.equal(pixels[1], index[:16, 16:32].flatten()))      # token 1: row 0, column 1

    def test_frames_never_see_later_frames(self):
        model = tiny().eval()
        a, b = window(1, 6, 1), window(1, 6, 1)
        b[0, 4:] = window(1, 2, 2)[0]                                          # change frames 4 and 5
        mask = torch.zeros(1, 6, 256, dtype=torch.bool)
        acts = torch.zeros(1, 6, dtype=torch.long)
        fa, fb = model(a, acts, mask), model(b, acts, mask)
        self.assertTrue(torch.allclose(fa[:, :4], fb[:, :4], atol=1e-5))
        self.assertFalse(torch.allclose(fa[:, 4], fb[:, 4]))

    def test_masked_tokens_hide_their_pixels(self):
        model = tiny().eval()
        a, b = window(1, 6, 3), window(1, 6, 3)
        b[0, 3, :16, :16] = (b[0, 3, :16, :16].long() + 1) % COLOURS           # token 0 of frame 3
        mask = torch.zeros(1, 6, 256, dtype=torch.bool)
        mask[0, 3, 0] = True
        acts = torch.zeros(1, 6, dtype=torch.long)
        self.assertTrue(torch.allclose(model(a, acts, mask), model(b, acts, mask), atol=1e-5))

    def test_actions_condition_their_frame(self):
        model = tiny().eval()
        x = window(1, 6, 4)
        mask = torch.zeros(1, 6, 256, dtype=torch.bool)
        quiet, pressed = torch.zeros(1, 6, dtype=torch.long), torch.zeros(1, 6, dtype=torch.long)
        pressed[0, 2] = 1
        fa, fb = model(x, quiet, mask), model(x, pressed, mask)
        self.assertTrue(torch.allclose(fa[:, :2], fb[:, :2], atol=1e-5))
        self.assertFalse(torch.allclose(fa[:, 2], fb[:, 2]))
        self.assertEqual(incoming_actions(torch.tensor([[5, 6, 7]])).tolist(), [[-1, 5, 6]])

    def test_training_mask_rates_per_frame_and_clean_contexts(self):
        mask = training_mask(400, 8, 256, "cpu", torch.Generator().manual_seed(0))
        self.assertFalse(mask[:, 0].any())
        rate = mask.float().mean(2)                                             # [B, T]
        self.assertGreater((rate > 0.9).float().mean().item(), 0.15)             # cosine schedule: many high rates
        self.assertGreater(rate[:, 1:].std(1).mean().item(), 0.15)              # rates differ within a window
        # Clean-context windows: a fully visible prefix longer than frame 0, then masked frames.
        visible = rate == 0
        prefix = visible.cumprod(1).sum(1)                                       # leading fully visible frames
        clean = prefix >= 2
        self.assertTrue(0.3 < clean.float().mean().item() < 0.7)
        after = rate[clean][torch.arange(int(clean.sum())), prefix[clean].clamp_max(7)]
        self.assertGreater((after > 0).float().mean().item(), 0.8)              # the frame after the cut is masked
        self.assertTrue((rate[:, 1:] > 0).any(1).all())                         # every window has something to score

    def test_training_mask_forced_cut_keeps_the_rollout_visible(self):
        cut = torch.tensor([0, 3, 5] * 50)
        mask = training_mask(150, 8, 256, "cpu", torch.Generator().manual_seed(7), cut=cut)
        for c in (3, 5):
            rows = mask[cut == c]
            self.assertFalse(rows[:, :c].any())                                  # real start + rollout visible
            self.assertGreater(rows[:, c].float().mean().item(), 0.3)            # the frame after is predicted
        free = mask[cut == 0].float().mean(2)
        self.assertTrue(0.2 < ((free == 0).cumprod(1).sum(1) >= 2).float().mean().item() < 0.8)   # usual cuts

    def test_masked_loss_scores_only_masked_tokens_and_trains(self):
        model = tiny()
        x = window(2, 6, 5)
        mask = training_mask(2, 6, 256, "cpu", torch.Generator().manual_seed(1))
        features = model(x, torch.zeros(2, 6, dtype=torch.long), mask)
        target = patch_pixels(x, 16)
        total, per_pixel, where = masked_loss(model, features, target, mask, chunk=100)
        self.assertEqual(per_pixel.shape, (int(mask.sum()), 256))
        self.assertTrue(torch.allclose(total, per_pixel.mean()))
        total.backward()
        self.assertGreater(model.pixel.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.colour_bias.grad.abs().sum().item(), 0)
        self.assertGreater(model.blocks[0].temporal.qkv.weight.grad.abs().sum().item(), 0)

    def test_fused_head_loss_matches_autograd_cross_entropy(self):
        x, acts = window(2, 6, 11), torch.randint(0, 256, (2, 6))
        mask = training_mask(2, 6, 256, "cpu", torch.Generator().manual_seed(6))
        target = patch_pixels(x, 16)
        results = []
        for fused in (True, False):
            model = tiny()
            features = model(x, acts, mask)
            if fused:
                total = masked_loss(model, features, target, mask, chunk=100)[0]
            else:
                where = mask.nonzero(as_tuple=True)
                logits = model.logits(features[where]).float()
                total = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target[where].long().flatten())
            total.backward()
            results.append((total.detach(), [p.grad.clone() for p in model.parameters() if p.grad is not None]))
        (l0, g0), (l1, g1) = results
        self.assertTrue(torch.allclose(l0, l1, atol=1e-6))
        self.assertEqual(len(g0), len(g1))
        for a, b in zip(g0, g1):
            self.assertTrue(torch.allclose(a, b, atol=1e-6, rtol=1e-4))

    def test_tied_head_scores_pixels_against_the_colour_embedding(self):
        model = tiny()
        h = torch.randn(3, 32)
        logits = model.logits(h)                                                  # [3, 256, COLOURS]
        pixel = (model.pixel(h).view(3, 256, 8) @ model.colour.weight.T) / 8 ** 0.5 + model.colour_bias
        self.assertTrue(torch.allclose(logits, pixel, atol=1e-5))

    def test_colour_embedding_gradient_matches_embedding(self):
        from token_world.models.dynamics import ColourEmbedding
        index = window(2, 3, 13)
        grad = torch.randn(2, 3, 256, 256, 8)
        table = torch.randn(COLOURS, 8, requires_grad=True)
        ColourEmbedding.apply(index, table).backward(grad)
        reference = table.detach().clone().requires_grad_()
        torch.nn.functional.embedding(index.long(), reference).backward(grad)
        self.assertTrue(torch.equal(ColourEmbedding.apply(index, table), reference[index.long()]))
        self.assertTrue(torch.allclose(table.grad, reference.grad, atol=1e-3, rtol=1e-4))

    def test_load_run_refuses_another_frame_layout(self):
        import tempfile
        from token_world.data.model_frames import BORDER_VERSION
        from token_world.models.dynamics import load_run
        config = {"dim": 32, "layers": 2, "heads": 2, "patch": 16, "frames": 6, "colour_dim": 8}
        model = build(config)
        with tempfile.TemporaryDirectory() as folder:
            for border in (BORDER_VERSION, None):
                torch.save({"step": 1, "args": {**config, "border": border}, "ema": model.state_dict()},
                           Path(folder) / "model_latest.pt")
                if border is None:
                    with self.assertRaises(ValueError):
                        load_run(folder, "cpu")
                else:
                    loaded, saved = load_run(folder, "cpu")
                    self.assertTrue(torch.equal(loaded.pixel.weight, model.pixel.weight))

    def test_deepened_model_computes_what_the_shallow_one_did(self):
        shallow = tiny().eval()
        torch.manual_seed(7)
        deep = Dynamics(dim=32, layers=4, heads=2, patch=16, frames=6, colour_dim=8).eval()
        deepen(deep, shallow.state_dict())
        x, acts = window(1, 6, 9), torch.randint(0, 256, (1, 6))
        mask = torch.rand(1, 6, 256, generator=torch.Generator().manual_seed(4)) < 0.3
        with torch.no_grad():
            self.assertTrue(torch.allclose(shallow(x, acts, mask), deep(x, acts, mask), atol=1e-5))
        self.assertFalse(torch.equal(deep.blocks[1].spatial.qkv.weight, shallow.blocks[0].spatial.qkv.weight))

    def test_cached_frame_matches_the_full_window(self):
        model = tiny().eval()
        x, acts = window(2, 6, 6), torch.randint(0, 256, (2, 6))
        mask = torch.zeros(2, 6, 256, dtype=torch.bool)
        mask[:, 4] = torch.rand(2, 256, generator=torch.Generator().manual_seed(2)) < 0.5
        level = torch.randint(0, LEVELS, (2, 6), generator=torch.Generator().manual_seed(3))
        with torch.no_grad():
            full = model(x, acts, mask, level)[:, 4]
            cache = model.new_cache()
            model(x[:, :4], acts[:, :4], mask[:, :4], level[:, :4], cache)
            cached = model(x[:, 4:5], acts[:, 4:5], mask[:, 4:5], level[:, 4:5], cache, at=4)[:, 0]
            pair = model.new_cache()                        # two frames in one cached call, as the export steps
            model(x[:, :3], acts[:, :3], mask[:, :3], level[:, :3], pair)
            both = model(x[:, 3:5], acts[:, 3:5], mask[:, 3:5], level[:, 3:5], pair, at=torch.tensor(3))
        self.assertTrue(torch.allclose(full, cached, atol=1e-5))
        self.assertTrue(torch.allclose(model(x, acts, mask, level)[:, 3:5], both, atol=1e-5))

    def test_cached_rollout_matches_decoding_the_full_window(self):
        model = tiny().eval()
        x, acts = window(2, 6, 6), torch.randint(0, 256, (2, 6))
        generated = rollout(model, x, acts, start=3, decoding=Decoding(steps=3))
        self.assertEqual(generated.shape, (2, 3, 256, 256))
        reference = x.clone()
        level = torch.zeros(2, 6, dtype=torch.long)
        with torch.no_grad():
            for at in range(3, 6):
                def logits_of(frame, hidden):
                    reference[:, at] = frame
                    mask = torch.zeros(2, 6, 256, dtype=torch.bool)
                    mask[:, at] = hidden
                    level[:, at] = GENERATED_LEVEL                   # the frame being made is generated
                    return model.logits(model(reference, acts, mask, level)[:, at])
                reference[:, at] = decode_frame(logits_of, 2, 256, 16, Decoding(steps=3), "cpu")
        self.assertTrue(torch.equal(generated, reference[:, 3:]))
        self.assertTrue(torch.equal(rollout(model, x, acts, start=3, decoding=Decoding(steps=3)), generated))   # argmax decoding

    def test_refinement_remasks_the_least_confident_tokens(self):
        seen = []

        def logits_of(frame, hidden):                    # token n prefers colour n % COLOURS, sure of tokens < 128
            seen.append(hidden.clone())
            logits = torch.zeros(1, 256, 256, COLOURS)
            logits[0, torch.arange(256), :, torch.arange(256) % COLOURS] = torch.where(torch.arange(256) < 128, 9.0, 1.0)[:, None]
            return logits
        plain = decode_frame(logits_of, 1, 256, 16, Decoding(steps=2), "cpu")
        seen.clear()
        refined = decode_frame(logits_of, 1, 256, 16, Decoding(steps=2, refine=0.25), "cpu")
        self.assertEqual(len(seen), 3)                                              # 2 steps + 1 refinement pass
        self.assertEqual(int(seen[2].sum()), 64)                                    # a quarter of the tokens
        self.assertTrue(seen[2][0, 128:].sum() == 64 and not seen[2][0, :128].any())   # the unsure ones
        self.assertTrue(torch.equal(plain, refined))                                # same picks here

    def test_temperature_samples_and_zero_is_argmax(self):
        logits = torch.randn(2, 256, 256, COLOURS)
        best, pick = __import__("token_world.models.dynamics", fromlist=["_pick"])._pick(logits, 0.0)
        self.assertTrue(torch.equal(pick, logits.argmax(-1)))
        _, hot = __import__("token_world.models.dynamics", fromlist=["_pick"])._pick(logits, 1.0)
        self.assertFalse(torch.equal(hot, pick))
        self.assertTrue(((hot >= 0) & (hot < COLOURS)).all())

    def test_generated_level_changes_the_dream(self):
        model = tiny().eval()
        x, acts = window(1, 6, 14), torch.randint(0, 256, (1, 6))
        flagged = rollout(model, x, acts, start=3, decoding=Decoding(steps=2))
        plain = rollout(model, x, acts, start=3, decoding=Decoding(steps=2, level=0))
        self.assertEqual(plain.shape, flagged.shape)
        self.assertFalse(torch.equal(plain, flagged))

    @unittest.skipUnless(torch.cuda.device_count() > 0, "needs CUDA")
    def test_cuda_graph_decoding_matches_its_step_and_follows_weight_updates(self):
        from token_world.models.dynamics import FrameDecoder
        model = tiny().cuda().eval()
        x, acts = window(2, 6, 6).cuda(), torch.randint(0, 256, (2, 6)).cuda()

        @torch.no_grad()
        def eager():
            decoder = FrameDecoder(model, 2, Decoding(steps=3), x.device, (False, torch.bfloat16))
            decoder.prefill(x[:, :2], acts[:, :2])
            frames = []
            for at in range(2, 6):
                decoder.act.copy_(acts[:, at:at + 1])
                decoder.at.fill_(at)
                frames.append(decoder._step())                                  # the same step, no graph
            return torch.stack(frames, 1)
        first = rollout(model, x, acts, start=2, decoding=Decoding(steps=3))                          # captures the graph
        self.assertTrue(torch.equal(first, eager()))
        self.assertTrue(torch.equal(rollout(model, x, acts, start=2, decoding=Decoding(steps=3)), first))   # replayed
        with torch.no_grad():
            model.colour_bias.add_(torch.randn_like(model.colour_bias))            # an optimizer step, in place
            model.blocks[0].mlp[0].weight.mul_(1.5)
        after = rollout(model, x, acts, start=2, decoding=Decoding(steps=3))
        self.assertFalse(torch.equal(after, first))
        self.assertTrue(torch.equal(after, eager()))

    def test_dreamer_matches_rollout_and_slides_its_window(self):
        model = tiny().eval()
        x, acts = window(2, 6, 12), torch.randint(0, 256, (2, 6))
        expected = rollout(model, x[:1], acts[:1], start=3, decoding=Decoding(steps=2))[0]
        dreamer = Dreamer(model, x[0, :3], acts[0, :3], Decoding(steps=2), keep=3)
        got = torch.stack([dreamer.step(int(a)) for a in acts[0, 3:]])
        self.assertTrue(torch.equal(got, expected))                                # same frames until the window fills
        for _ in range(5):                                                         # then it slides and keeps going
            frame = dreamer.step(0)
        self.assertEqual(frame.shape, (256, 256))
        self.assertLessEqual(len(dreamer.frames), model.frames)

    def test_dreamers_with_the_same_settings_do_not_share_a_cache(self):
        model = tiny().eval()
        x, acts = window(2, 6, 12), torch.randint(0, 256, (2, 6))
        alone = Dreamer(model, x[0, :3], acts[0, :3], Decoding(steps=2), keep=3)
        expected = [alone.step(int(a)) for a in acts[0, 3:]]
        first = Dreamer(model, x[0, :3], acts[0, :3], Decoding(steps=2), keep=3)
        other = Dreamer(model, x[1, :3], acts[1, :3], Decoding(steps=2), keep=3)
        for a, b, want in zip(acts[0, 3:], acts[1, 3:], expected):
            got = first.step(int(a))
            other.step(int(b))                                                     # interleaved, as side by side
            self.assertTrue(torch.equal(got, want))

    def test_level_marks_its_frame_and_later_ones_only(self):
        model = tiny().eval()
        x, acts = window(1, 6, 8), torch.zeros(1, 6, dtype=torch.long)
        mask, level = torch.zeros(1, 6, 256, dtype=torch.bool), torch.zeros(1, 6, dtype=torch.long)
        level[0, 2] = 5
        plain, marked = model(x, acts, mask), model(x, acts, mask, level)
        self.assertTrue(torch.allclose(plain[:, :2], marked[:, :2], atol=1e-5))
        self.assertFalse(torch.allclose(plain[:, 2], marked[:, 2]))

    def test_corrupt_context_swaps_in_earlier_or_same_frame_content(self):
        x = window(4, 12, 15)
        corrupted, level, changed = corrupt_context(x, 16, torch.Generator().manual_seed(0))
        self.assertTrue((level[:, 0] == 0).all() and ((level >= 0) & (level < LEVELS)).all())
        tokens, out = patch_pixels(x, 16), patch_pixels(corrupted, 16)
        differs = (tokens != out).any(-1)
        self.assertTrue(torch.equal(differs, changed))
        self.assertFalse(differs[:, 0].any())                                       # frame 0 stays real
        for b, f, n in differs.nonzero().tolist():                                  # each from this or earlier frames
            sources = torch.cat([tokens[b, max(0, f - 8):f, n], tokens[b, f]])
            self.assertTrue((sources == out[b, f, n]).all(-1).any())
        for value in range(LEVELS):                                                 # share grows with the level
            rows = level == value
            if rows.any() and value:
                self.assertLess(changed[rows].float().mean().item(), value / (LEVELS - 1) * 0.3 + 0.12)
        self.assertFalse(changed[level == 0].any())

    def test_recompute_keeps_loss_and_gradients(self):
        x, acts = window(2, 6, 10), torch.randint(0, 256, (2, 6))
        mask = training_mask(2, 6, 256, "cpu", torch.Generator().manual_seed(5))
        results = []
        for recompute in (False, True):
            model = tiny()
            model.recompute = recompute
            total = masked_loss(model, model(x, acts, mask), patch_pixels(x, 16), mask)[0]
            total.backward()
            results.append((total.detach(), [p.grad.clone() for p in model.parameters() if p.grad is not None]))
        (l0, g0), (l1, g1) = results
        self.assertTrue(torch.allclose(l0, l1))
        self.assertEqual(len(g0), len(g1))
        for a, b in zip(g0, g1):
            self.assertTrue(torch.allclose(a, b, atol=1e-6))

    def test_train_step_reports_border_and_changed_parts(self):
        import train_dynamics_ui as D
        model = tiny()
        x = window(2, 6, 7)
        x[:, 1:, 100:120] = x[:, :1, 100:120]                                      # some static pixels
        losses = D.train_step(model, model, x, torch.randint(0, 256, (2, 6)))
        for key in ("total", "game", "border", "changed", "corrupted", "rollout_wrong"):
            self.assertTrue(torch.isfinite(losses[key]), key)
        losses["total"].backward()


if __name__ == "__main__":
    unittest.main()
