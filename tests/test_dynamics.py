"""World model: exact pixel patches, causality, masking, the masked loss, action conditioning, soft decoding,
the two training stages (teacher forcing; rollouts with the pred-to-real blend) and the sampled choice per
frame (the latent)."""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from token_world.models.dynamics import (COLOURS, Dreamer, Dynamics, build, choice_kl, deepen, incoming_actions,
                                         masked_loss, own_share, patch_pixels, rollout, sample_choice, training_mask,
                                         unpatch_pixels, weigh_changes)


def tiny():
    torch.manual_seed(0)
    model = Dynamics(dim=32, layers=2, heads=2, patch=16, frames=6, colour_dim=8)
    with torch.no_grad():
        model.colour_bias.normal_(0, 0.5)
    return model


def window(batch=2, frames=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, COLOURS, (batch, frames, 256, 256), generator=g).to(torch.uint8)


def soft_window(model, x):
    """Real frames [B, T, 256, 256] as soft frames (their colours' embeddings)."""
    return model.colour.weight.detach()[x.long()]


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
        total, per_pixel, where = masked_loss(model, features, patch_pixels(x, 16), mask, chunk=100)
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
            for border, ok in ((BORDER_VERSION, True), ("blank-v1", False), (None, False)):
                torch.save({"step": 1, "args": {**config, "border": border}, "ema": model.state_dict()},
                           Path(folder) / "model_latest.pt")
                if not ok:
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
        with torch.no_grad():
            full = model(x, acts, mask)[:, 4]
            cache = model.new_cache()
            model(x[:, :4], acts[:, :4], mask[:, :4], cache)
            cached = model(x[:, 4:5], acts[:, 4:5], mask[:, 4:5], cache, at=4)[:, 0]
            pair = model.new_cache()                        # two frames in one cached call, as the export steps
            model(x[:, :3], acts[:, :3], mask[:, :3], pair)
            both = model(x[:, 3:5], acts[:, 3:5], mask[:, 3:5], pair, at=torch.tensor(3))
            self.assertTrue(torch.allclose(full, cached, atol=1e-5))
            self.assertTrue(torch.allclose(model(x, acts, mask)[:, 3:5], both, atol=1e-5))

    def test_modern_blocks_train_and_cache_like_the_full_window(self):
        torch.manual_seed(0)
        model = Dynamics(dim=32, layers=2, heads=2, patch=16, frames=6, colour_dim=8, modern=True).eval()
        self.assertIsNotNone(model.blocks[0].spatial.q_norm)
        x, acts = window(1, 6, 9), torch.randint(0, 256, (1, 6))
        mask = torch.zeros(1, 6, 256, dtype=torch.bool)
        with torch.no_grad():
            full = model(x, acts, mask)
            cache = model.new_cache()
            model(x[:, :5], acts[:, :5], mask[:, :5], cache=cache)
            step = model(x[:, 5:], acts[:, 5:], mask[:, 5:], cache=cache, at=5)
        self.assertTrue(torch.allclose(full[:, 5], step[:, 0], atol=1e-4))
        model.train()
        mask = training_mask(1, 6, 256, "cpu", torch.Generator().manual_seed(1))
        masked_loss(model, model(x, acts, mask), patch_pixels(x, 16), mask)[0].backward()
        self.assertTrue(all(p.grad is not None for p in model.blocks[0].mlp.parameters()))

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


class SoftDecodingTests(unittest.TestCase):
    """Frames are colour distributions: generated frames go back into the context as soft frames."""

    def test_a_frame_given_as_its_colour_embeddings_reads_as_the_frame(self):
        model = tiny().eval()
        x, acts = window(1, 6, 31), torch.randint(0, 256, (1, 6))
        mask = torch.zeros(1, 6, 256, dtype=torch.bool)
        with torch.no_grad():
            embedded = soft_window(model, x)                                       # [1, 6, 256, 256, colour_dim]
            self.assertTrue(torch.allclose(model(x, acts, mask), model(embedded, acts, mask), atol=1e-5))
            one_hot = torch.nn.functional.one_hot(patch_pixels(x[0], 16).long(), COLOURS).float()
            self.assertTrue(torch.allclose(model.embed_probs(one_hot), embedded[0], atol=1e-6))

    def test_cached_rollout_matches_the_full_window(self):
        model = tiny().eval()
        x, acts = window(2, 6, 6), torch.randint(0, 256, (2, 6))
        tokens, unsure, wrong, drawn = rollout(model, x, acts, start=3)
        self.assertEqual(tokens.shape, (2, 3, 256, 32))
        self.assertEqual((unsure.shape, wrong.shape), ((2, 3), (2, 3)))
        self.assertTrue(((wrong > 0) & (wrong <= 1)).all())
        self.assertIsNone(drawn)                                                   # no latent, no choices
        reference = soft_window(model, x)
        with torch.no_grad():
            for at in range(3, 6):                        # each frame from the real start and the soft frames
                mask = torch.zeros(2, 6, 256, dtype=torch.bool)
                mask[:, at] = True
                reference[:, at] = model.soft_frame(model(reference, acts, mask)[:, at])
        self.assertTrue(torch.allclose(tokens, model._patches(reference[:, 3:]), atol=1e-4))

    def test_a_soft_token_is_the_patch_embedding_of_its_soft_pixels(self):
        model = tiny().eval()
        features = torch.randn(1, 256, 32)
        with torch.no_grad():
            soft = model.soft_frame(features)                                      # [1, 256, 256, colour_dim]
            self.assertTrue(torch.allclose(model.soft_tokens(features[0], chunk=100),
                                           model._patches(soft[:, None])[0, 0], atol=1e-4))

    @unittest.skipUnless(torch.cuda.device_count() > 0, "needs CUDA")
    def test_cuda_graph_decoding_matches_its_step_and_follows_weight_updates(self):
        from token_world.models.dynamics import FrameDecoder
        model = tiny().cuda().eval()
        x, acts = window(2, 6, 6).cuda(), torch.randint(0, 256, (2, 6)).cuda()

        @torch.no_grad()
        def eager():
            decoder = FrameDecoder(model, 2, x.device, (False, torch.bfloat16), scored=True)
            decoder.prefill(x[:, :2], acts[:, :2])
            frames = []
            for at in range(2, 6):
                decoder.act.copy_(acts[:, at:at + 1])
                decoder.at.fill_(at)
                decoder.target.copy_(x[:, at])
                frames.append(decoder._step()["tokens"])                       # the same step, no graph
            return torch.stack(frames, 1)
        first = rollout(model, x, acts, start=2)[0]                               # captures the graph
        self.assertTrue(torch.allclose(first, eager(), atol=1e-5))
        self.assertTrue(torch.equal(rollout(model, x, acts, start=2)[0], first))   # replayed
        with torch.no_grad():
            model.colour_bias.add_(torch.randn_like(model.colour_bias))            # an optimizer step, in place
            model.blocks[0].mlp[0].weight.mul_(1.5)
        after = rollout(model, x, acts, start=2)[0]
        self.assertFalse(torch.allclose(after, first))
        self.assertTrue(torch.allclose(after, eager(), atol=1e-5))

    def test_dreamer_matches_rollout_and_slides_its_window(self):
        model = tiny().eval()
        x, acts = window(2, 6, 12), torch.randint(0, 256, (2, 6))
        expected = rollout(model, x[:1], acts[:1], start=3)[0][0]
        dreamer = Dreamer(model, x[0, :3], acts[0, :3], keep=3)
        probs = [dreamer.step(int(a)) for a in acts[0, 3:]]
        self.assertTrue(torch.allclose(torch.stack(dreamer.frames[3:]), expected, atol=1e-5))   # same frames
        self.assertEqual(probs[0].shape, (256, 256, COLOURS))
        self.assertTrue(torch.allclose(probs[0].sum(-1), torch.ones(256, 256), atol=1e-4))
        for _ in range(5):                                                         # then it slides and keeps going
            dreamer.step(0)
        self.assertLessEqual(len(dreamer.frames), model.frames)

    def test_weighing_changes_multiplies_their_odds(self):
        probs = torch.tensor([[0.75, 0.15, 0.10], [0.2, 0.3, 0.5]])
        last = torch.tensor([0, 2])                                                # the colour each pixel had
        out = weigh_changes(probs, last, 3.0)
        self.assertTrue(torch.allclose(out[0], torch.tensor([0.5, 0.3, 0.2])))   # change 0.25 -> 0.5, split as before
        self.assertTrue(torch.allclose(out.sum(-1), torch.ones(2)))
        self.assertTrue(torch.allclose(weigh_changes(probs, last, 1.0), probs))

    def test_a_change_weight_draws_more_change(self):
        model = tiny().eval()
        x, acts = window(1, 6, 13), torch.randint(0, 256, (1, 6))
        changed = []
        for weight in (1.0, 4.0):
            dreamer = Dreamer(model, x[0, :3], acts[0, :3], keep=3, change_weight=weight)
            probs = dreamer.step(0)
            changed.append(1 - probs.gather(-1, x[0, 2].long()[..., None]).mean().item())
        self.assertGreater(changed[1], changed[0])

    def test_dreamers_with_the_same_settings_do_not_share_a_cache(self):
        model = tiny().eval()
        x, acts = window(2, 6, 12), torch.randint(0, 256, (2, 6))
        alone = Dreamer(model, x[0, :3], acts[0, :3], keep=3)
        expected = [alone.step(int(a)) for a in acts[0, 3:]]
        first = Dreamer(model, x[0, :3], acts[0, :3], keep=3)
        other = Dreamer(model, x[1, :3], acts[1, :3], keep=3)
        for a, b, want in zip(acts[0, 3:], acts[1, 3:], expected):
            got = first.step(int(a))
            other.step(int(b))                                                     # interleaved, as side by side
            self.assertTrue(torch.equal(got, want))


class TrainingTests(unittest.TestCase):
    def test_own_share_fades_the_last_history_frames(self):
        share = own_share(2, 5, 2)                             # history frames 2..4, boundary 5, radius 2
        self.assertTrue(torch.allclose(share, torch.tensor([1.0, 2 / 3, 1 / 3])))   # the last: mostly real
        self.assertEqual(own_share(2, 5, 0).tolist(), [1.0, 1.0, 1.0])               # no blend: all its own
        self.assertTrue(torch.allclose(own_share(3, 5, 8), torch.tensor([2 / 9, 1 / 9])))   # a short rollout

    def test_base_step_is_teacher_forcing(self):
        import train_dynamics_ui as D
        model = tiny()
        x = window(4, 6, 34)
        x[:, 1:, 100:120] = x[:, :1, 100:120]                                      # some static pixels
        losses = D.train_step(model, model, x, torch.randint(0, 256, (4, 6)), generator=torch.Generator().manual_seed(0))
        for key in ("total", "game", "border", "changed"):
            self.assertTrue(torch.isfinite(losses[key]), key)
        self.assertTrue(torch.isnan(losses["rollout_wrong"]))
        self.assertEqual(float(losses["rollout_frames"]), 0.0)
        losses["total"].backward()
        self.assertGreater(model.blocks[0].temporal.qkv.weight.grad.abs().sum().item(), 0)

    def test_rollout_step_with_a_blend_reports_and_trains(self):
        import train_dynamics_ui as D
        model = tiny()
        x = window(4, 6, 35)
        for blend in (0, 2, 8):                                                    # 8: wider than the window
            model.zero_grad()
            losses = D.train_step(model, model, x, torch.randint(0, 256, (4, 6)), depth=4, blend=blend,
                                  generator=torch.Generator().manual_seed(blend))
            for key in ("total", "game", "changed", "rollout_wrong", "rollout_ghost_px"):
                self.assertTrue(torch.isfinite(losses[key]), (blend, key))
            self.assertTrue(1 <= float(losses["rollout_frames"]) <= 4)
            self.assertTrue(0 <= float(losses["blend"]) <= blend)
            losses["total"].backward()
            self.assertGreater(model.blocks[0].temporal.qkv.weight.grad.abs().sum().item(), 0)

    def test_rollouts_are_drawn_by_the_drawer(self):
        import train_dynamics_ui as D
        model, drawer = tiny(), tiny()
        with torch.no_grad():
            drawer.blocks[0].mlp[0].weight.mul_(2.0)                               # a different model
        x, acts = window(4, 6, 36), torch.randint(0, 256, (4, 6))
        losses = [D.train_step(model, model, x, acts, depth=4, blend=2, generator=torch.Generator().manual_seed(5),
                               drawer=d)["rollout_wrong"] for d in (None, drawer)]
        self.assertNotEqual(float(losses[0]), float(losses[1]))
        self.assertIsNone(drawer.blocks[0].mlp[0].weight.grad)                     # no gradient reaches it

    def test_each_step_draws_its_blend_radius(self):
        import train_dynamics_ui as D
        model, x, acts = tiny().eval(), window(2, 6, 37), torch.randint(0, 256, (2, 6))
        with torch.no_grad():
            radii = {float(D.train_step(model, model, x, acts, depth=3, blend=2,
                                        generator=torch.Generator().manual_seed(seed))["blend"]) for seed in range(12)}
        self.assertEqual(radii, {0.0, 1.0, 2.0})                                    # 0 (a dream's history) .. MAX

    def test_the_depth_shrinks_over_the_run(self):
        import argparse
        import train_dynamics_ui as D
        args = argparse.Namespace(rollouts=[62, 16, 8000])
        self.assertEqual([D.rollout_depth(s, args) for s in (0, 4000, 8000, 20000)], [62, 39, 16, 16])
        self.assertEqual(D.rollout_depth(0, argparse.Namespace(rollouts=None)), 0)


def tiny_latent():
    torch.manual_seed(1)
    return Dynamics(dim=32, layers=2, heads=2, patch=16, frames=6, colour_dim=8, latent=(2, 4))


class LatentTests(unittest.TestCase):
    def test_a_choice_is_one_option_per_group_with_a_straight_through_gradient(self):
        logits = torch.randn(3, 2, 4, requires_grad=True)
        z = sample_choice(logits, straight_through=True)
        self.assertEqual(z.shape, (3, 8))
        self.assertTrue(torch.equal(z.detach().view(3, 2, 4).sum(-1), torch.ones(3, 2)))
        (z * torch.arange(8.0)).sum().backward()
        self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_kl_is_zero_for_equal_choices_and_positive_otherwise(self):
        a, b = torch.randn(5, 2, 4), torch.randn(5, 2, 4)
        self.assertTrue(torch.allclose(choice_kl(a, a)[0], torch.zeros(5), atol=1e-6))
        self.assertTrue((choice_kl(a * 5, b * 5)[0] > 0).all())

    def test_grown_from_a_model_without_it_starts_as_that_model(self):
        base = tiny().eval()
        model = tiny_latent().eval()
        deepen(model, base.state_dict())
        x, acts = window(1, 6, 40), torch.randint(0, 256, (1, 6))
        mask = training_mask(1, 6, base.grid ** 2, "cpu", torch.Generator().manual_seed(0))
        z = sample_choice(torch.randn(1, 6, 2, 4))
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(x, acts, mask, z=z), base(x, acts, mask), atol=1e-5))

    def test_the_latent_step_trains_its_choice_and_both_reads(self):
        import train_dynamics_ui as D
        model = tiny_latent()
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        for step in range(2):                                   # the choice starts at zero: step 1 opens it
            x = window(2, 6, 41 + step)
            losses = D.train_step(model, model, x, torch.randint(0, 256, (2, 6)),
                                  generator=torch.Generator().manual_seed(step))
            self.assertTrue(torch.isfinite(losses["latent_kl"]))
            optimizer.zero_grad()
            losses["total"].backward()
            self.assertGreater(model.choice.weight.grad.abs().sum().item(), 0)
            optimizer.step()
        self.assertGreater(model.posterior_out.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.prior_out[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.prior_query.grad.abs().sum().item(), 0)

    def test_a_rollout_keeps_the_choices_it_drew(self):
        model = tiny_latent().eval()
        x, acts = window(2, 6, 44), torch.randint(0, 256, (2, 6))
        drawn = rollout(model, x, acts, start=3)[3]
        self.assertEqual(drawn.shape, (2, 3, 8))
        self.assertTrue(torch.equal(drawn.view(2, 3, 2, 4).sum(-1), torch.ones(2, 3, 2)))   # one per group

    def test_the_latent_trains_with_rollouts(self):
        import train_dynamics_ui as D
        model, drawer = tiny_latent(), tiny_latent().eval()
        x = window(4, 6, 45)
        losses = D.train_step(model, model, x, torch.randint(0, 256, (4, 6)), depth=3, blend=2,
                              generator=torch.Generator().manual_seed(1), drawer=drawer)
        for key in ("total", "latent_kl", "rollout_wrong"):
            self.assertTrue(torch.isfinite(losses[key]), key)
        losses["total"].backward()
        self.assertGreater(model.prior_query.grad.abs().sum().item(), 0)

    def test_the_prior_sees_the_incoming_buttons(self):
        model = tiny_latent().eval()
        features = torch.randn(1, model.grid ** 2, 32)
        with torch.no_grad():
            idle, start = model.prior(features, torch.tensor([0])), model.prior(features, torch.tensor([8]))
        self.assertEqual(idle.shape, (1, 2, 4))
        self.assertFalse(torch.allclose(idle, start))

    def test_a_dream_samples_its_choices(self):
        model = tiny_latent().eval()
        with torch.no_grad():
            model.choice.weight.normal_(0, 1.0)                 # a trained choice changes the frame
        x, acts = window(1, 6, 43), torch.randint(0, 256, (1, 6))
        dreams = []
        for seed in (0, 1):
            torch.manual_seed(seed)
            dreamer = Dreamer(model, x[0, :3], acts[0, :3], keep=3)
            dreams.append(torch.stack([dreamer.step(0) for _ in range(3)]))
        self.assertTrue(torch.allclose(dreams[0].sum(-1), torch.ones(3, 256, 256), atol=1e-4))
        self.assertFalse(torch.allclose(dreams[0], dreams[1]))  # another draw, another dream


if __name__ == "__main__":
    unittest.main()
