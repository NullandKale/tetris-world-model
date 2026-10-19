"""World model: exact pixel patches, causality, masking, the masked loss, action conditioning, soft decoding,
the two training stages (teacher forcing; rollouts with the pred-to-real blend) and the sampled choice per
frame (the latent)."""
import math
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from token_world.models.dynamics import (COLOURS, Attention, Dreamer, Dynamics, border_pixels, build, choice_kl, deepen,
                                         incoming_actions, masked_loss, motion_loss, motion_tokens, own_share,
                                         patch_pixels, rollout, sample_choice, training_mask, unpatch_pixels,
                                         weigh_changes)


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
    def test_time_is_rotary_so_only_frames_apart_count(self):
        torch.manual_seed(0)
        attention = Attention(32, 4, rotate=True).eval()
        x = torch.randn(3, 6, 32)                                   # 3 positions' 6 frames
        with torch.no_grad():
            first, later = attention(x, causal=True, at=0), attention(x, causal=True, at=40)
            self.assertTrue(torch.allclose(first, later, atol=1e-5))
            self.assertFalse(torch.allclose(first, attention(x.flip(1), causal=True).flip(1), atol=1e-3))
        self.assertFalse(any("time" in name for name, _ in Dynamics(32, 2, 2, frames=8, colour_dim=4).named_parameters()))

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

    def test_training_mask_hides_a_share_of_frames_entirely(self):
        cut = torch.tensor([0, 4] * 200)
        plain = training_mask(400, 8, 256, "cpu", torch.Generator().manual_seed(3), cut=cut)
        mask = training_mask(400, 8, 256, "cpu", torch.Generator().manual_seed(3), cut=cut, hidden=0.5)
        self.assertTrue((mask | ~plain).all())                                  # only adds hidden tokens
        self.assertFalse(mask[:, 0].any())
        self.assertFalse(mask[cut == 4][:, :4].any())                           # the clean past stays visible
        maskable = plain.any(2)                                                 # frames that are predicted
        whole = mask.all(2)[maskable].float().mean().item()
        self.assertTrue(0.42 < whole < 0.62)                                    # half, plus the cosine's few
        self.assertLess(plain.all(2)[maskable].float().mean().item(), 0.12)

    def test_training_mask_hides_the_given_frames_whenever_predicted(self):
        cut = torch.tensor([0, 4] * 200)
        hide = torch.zeros(400, 8, dtype=torch.bool)
        hide[:, 2] = hide[:, 6] = True
        plain = training_mask(400, 8, 256, "cpu", torch.Generator().manual_seed(3), cut=cut)
        mask = training_mask(400, 8, 256, "cpu", torch.Generator().manual_seed(3), cut=cut, hide=hide)
        self.assertTrue((mask | ~plain).all())                                  # only adds hidden tokens
        self.assertTrue(torch.equal(mask[:, [1, 3, 4, 5, 7]], plain[:, [1, 3, 4, 5, 7]]))
        for f in (2, 6):                                                        # all of a frame or none
            self.assertTrue((mask[:, f].all(1) | ~mask[:, f].any(1)).all())
        self.assertTrue(mask[cut == 4][:, 6].all())                             # after the cut: hidden
        self.assertFalse(mask[cut == 4][:, 2].any())                            # in a clean past: visible
        self.assertTrue(mask[cut == 0][:, 2].all(1).float().mean() > 0.4)       # predicted: hidden

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

    def test_a_weighted_loss_is_the_weighted_mean_with_its_gradient(self):
        x, acts = window(2, 6, 11), torch.randint(0, 256, (2, 6))
        mask = training_mask(2, 6, 256, "cpu", torch.Generator().manual_seed(6))
        target = patch_pixels(x, 16)
        weight = 1 + 9 * (torch.rand(target.shape, generator=torch.Generator().manual_seed(2)) < 0.1).float()
        results = []
        for fused in (True, False):
            model = tiny()
            features = model(x, acts, mask)
            if fused:
                total, per_pixel, _ = masked_loss(model, features, target, mask, chunk=100, weight=weight)
            else:
                where = mask.nonzero(as_tuple=True)
                logits = model.logits(features[where]).float()
                ce = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target[where].long().flatten(),
                                                       reduction="none")
                total = (ce * weight[where].flatten()).sum() / weight[where].sum()
                per_pixel = ce.detach().view(-1, 256)
            total.backward()
            results.append((total.detach(), per_pixel, [p.grad.clone() for p in model.parameters() if p.grad is not None]))
        (l0, pp0, g0), (l1, pp1, g1) = results
        self.assertTrue(torch.allclose(l0, l1, atol=1e-6))
        self.assertTrue(torch.allclose(pp0, pp1, atol=1e-5))                      # per pixel: unweighted
        for a, b in zip(g0, g1):
            self.assertTrue(torch.allclose(a, b, atol=1e-6, rtol=1e-4))

    def test_the_change_weight_falls_on_game_pixels_that_change(self):
        self.assertIsNone(tiny().loss_weights(patch_pixels(window(1, 3), 16)))     # unweighted: none
        model = Dynamics(dim=32, layers=2, heads=2, patch=16, frames=6, colour_dim=8, change_weight=10.0)
        x = torch.zeros(1, 3, 256, 256, dtype=torch.uint8)
        x[:, 1:, 100, 50] = 3                                                       # a game pixel changes at frame 1
        x[:, 1:, 2, 50] = 3                                                         # and a border pixel
        x[:, 2, 100, 60] = 4                                                        # another at frame 2
        weight = unpatch_pixels(model.loss_weights(patch_pixels(x, 16)), 16)
        self.assertEqual(weight[0, 1, 100, 50].item(), 10.0)
        self.assertEqual(weight[0, 2, 100, 50].item(), 1.0)                        # stayed from 1 to 2
        self.assertEqual(weight[0, 2, 100, 60].item(), 10.0)
        self.assertEqual(weight[0, 1, 2, 50].item(), 1.0)                          # the border is not weighted
        self.assertEqual(int((weight == 10).sum()), 2)
        self.assertEqual(model.loss_weights(patch_pixels(x, 16))[:, 0].max().item(), 1.0)   # frame 0: nothing before

    def test_decoding_undoes_the_change_weight(self):
        plain = tiny().eval()
        weighted = Dynamics(dim=32, layers=2, heads=2, patch=16, frames=6, colour_dim=8, change_weight=4.0).eval()
        weighted.load_state_dict(plain.state_dict())
        x, acts = window(1, 6, 13), torch.randint(0, 256, (1, 6))
        last = patch_pixels(x[0, 2].long(), 16)
        border = border_pixels(16)
        with torch.no_grad():
            mask = torch.zeros(1, 6, 256, dtype=torch.bool)
            mask[:, 3] = True
            raw = plain.logits(plain(x, acts, mask)[:, 3]).float().softmax(-1)[0]  # [N, P*P, COLOURS]
        undone = weigh_changes(raw, last, torch.where(border, 1.0, 0.25)[..., None])
        # a dream: the plain model's probabilities with the odds of change on game pixels times 1/4
        dreamed = Dreamer(weighted, x[0, :3], acts[0, :3], keep=3).step(int(acts[0, 3]))
        self.assertTrue(torch.allclose(dreamed, unpatch_pixels(undone.permute(2, 0, 1), 16).permute(1, 2, 0),
                                       atol=1e-5))
        self.assertTrue(torch.allclose(Dreamer(plain, x[0, :3], acts[0, :3], keep=3).step(int(acts[0, 3])),
                                       unpatch_pixels(raw.permute(2, 0, 1), 16).permute(1, 2, 0), atol=1e-5))
        # a scored rollout too: its pixels expected wrong, one frame ahead of the real history
        wrong = rollout(weighted, x[:, :4], acts[:, :4], start=3)[2][0, 0]
        real = patch_pixels(x[0, 3].long(), 16)
        self.assertAlmostEqual(wrong.item(), 1 - undone.gather(-1, real[..., None]).mean().item(), places=5)
        # it inverts the lean a weighted loss trains: odds of change times 4 on game pixels, then undone
        leaned = weigh_changes(raw, last, torch.where(border, 1.0, 4.0)[..., None])
        self.assertTrue(torch.allclose(weigh_changes(leaned, last, weighted.change_undo("cpu")), raw, atol=1e-6))

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

    def test_one_more_layer_computes_what_the_shallow_one_did(self):
        torch.manual_seed(3)
        shallow = Dynamics(dim=32, layers=3, heads=2, patch=16, frames=6, colour_dim=8).eval()
        torch.manual_seed(8)
        deep = Dynamics(dim=32, layers=4, heads=2, patch=16, frames=6, colour_dim=8).eval()
        deepen(deep, shallow.state_dict())
        x, acts = window(1, 6, 11), torch.randint(0, 256, (1, 6))
        mask = torch.rand(1, 6, 256, generator=torch.Generator().manual_seed(5)) < 0.3
        with torch.no_grad():
            self.assertTrue(torch.allclose(shallow(x, acts, mask), deep(x, acts, mask), atol=1e-5))
        for i, j in ((0, 0), (1, 1), (2, 3)):                                   # the new block is third
            for a, b in zip(shallow.blocks[i].state_dict().values(), deep.blocks[j].state_dict().values()):
                self.assertTrue(torch.equal(a, b))
        self.assertEqual(deep.blocks[2].spatial.out.weight.abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            deepen(shallow, deep.state_dict())

    def test_the_new_layer_goes_where_it_is_asked(self):
        torch.manual_seed(3)
        shallow = Dynamics(dim=32, layers=3, heads=2, patch=16, frames=6, colour_dim=8).eval()
        x, acts = window(1, 6, 12), torch.randint(0, 256, (1, 6))
        mask = torch.rand(1, 6, 256, generator=torch.Generator().manual_seed(6)) < 0.3
        for at in range(4):
            torch.manual_seed(9)
            deep = Dynamics(dim=32, layers=4, heads=2, patch=16, frames=6, colour_dim=8).eval()
            deepen(deep, shallow.state_dict(), new_at=at)
            with torch.no_grad():
                self.assertTrue(torch.allclose(shallow(x, acts, mask), deep(x, acts, mask), atol=1e-5))
            self.assertEqual(deep.blocks[at].spatial.out.weight.abs().sum().item(), 0)
            old = [j for j in range(4) if j != at]
            for i, j in enumerate(old):
                for a, b in zip(shallow.blocks[i].state_dict().values(), deep.blocks[j].state_dict().values()):
                    self.assertTrue(torch.equal(a, b))
        with self.assertRaises(ValueError):
            deepen(Dynamics(dim=32, layers=5, heads=2, patch=16, frames=6, colour_dim=8), shallow.state_dict(), new_at=1)

    def test_a_copied_layer_starts_near_the_shallow_model_and_is_not_a_no_op(self):
        torch.manual_seed(3)
        shallow = Dynamics(dim=32, layers=3, heads=2, patch=16, frames=6, colour_dim=8).eval()
        x, acts = window(1, 6, 13), torch.randint(0, 256, (1, 6))
        mask = torch.rand(1, 6, 256, generator=torch.Generator().manual_seed(7)) < 0.3
        with torch.no_grad():
            want = shallow(x, acts, mask)
        for at in range(4):
            deep = Dynamics(dim=32, layers=4, heads=2, patch=16, frames=6, colour_dim=8).eval()
            deepen(deep, shallow.state_dict(), new_at=at, copy=True)
            neighbour = at - 1 if at > 0 else 1
            new, near = deep.blocks[at].spatial.out.weight, deep.blocks[neighbour].spatial.out.weight
            self.assertTrue(torch.equal(new, near))
            source = shallow.blocks[at - 1 if at > 0 else 0].spatial.out.weight
            self.assertTrue(torch.allclose(new, 0.5 * source))
            with torch.no_grad():
                got = deep(x, acts, mask)
            self.assertLess(((got - want).norm() / want.norm()).item(), 0.3)          # close, not exact (s3_plain: 0.17-0.25)
        with self.assertRaises(ValueError):
            deepen(Dynamics(dim=32, layers=4, heads=2, patch=16, frames=6, colour_dim=8), shallow.state_dict(), copy=True)

    def test_buttons_reach_only_the_chosen_blocks(self):
        torch.manual_seed(4)
        model = Dynamics(dim=32, layers=3, heads=2, patch=16, frames=6, colour_dim=8, action_norm=True,
                         action_blocks=(2,)).eval()
        self.assertIsNone(model.blocks[0].modulation)
        self.assertIsNone(model.blocks[1].modulation)
        self.assertIsNotNone(model.blocks[2].modulation)
        with torch.no_grad():
            model.blocks[2].modulation[1].weight.normal_(0, 0.1)
        x = window(1, 6, 14)
        mask = torch.zeros(1, 6, 256, dtype=torch.bool)
        seen = {}
        hook = model.blocks[2].register_forward_hook(lambda m, args, out: seen.__setitem__("in", args[0]))
        with torch.no_grad():
            a = model(x, torch.zeros(1, 6, dtype=torch.long), mask)
            first = seen["in"].clone()
            b = model(x, torch.full((1, 6), 64, dtype=torch.long), mask)
        hook.remove()
        self.assertTrue(torch.equal(first, seen["in"]))          # the blocks before see no buttons
        self.assertFalse(torch.allclose(a, b))                    # the last block does
        with self.assertRaises(ValueError):
            Dynamics(dim=32, layers=3, heads=2, patch=16, frames=6, colour_dim=8, action_blocks=(2,))

    def test_local_attention_starts_as_without_and_biases_by_offset(self):
        torch.manual_seed(0)
        plain = Dynamics(dim=32, layers=2, heads=2, patch=16, frames=6, colour_dim=8, space_rope=True).eval()
        local = Dynamics(dim=32, layers=2, heads=2, patch=16, frames=6, colour_dim=8, space_rope=True,
                         local_attention=True).eval()
        missing, extra = local.load_state_dict(plain.state_dict(), strict=False)
        self.assertEqual(sorted(missing), [f"blocks.{i}.spatial.local_bias" for i in range(2)])
        x, acts = window(1, 6, 7), torch.randint(0, 256, (1, 6))
        mask = training_mask(1, 6, 256, "cpu", torch.Generator().manual_seed(1))
        with torch.no_grad():
            self.assertTrue(torch.allclose(local(x, acts, mask), plain(x, acts, mask), atol=1e-6))   # from zero
            attention = local.blocks[0].spatial
            offsets = attention.offsets                                              # [256, 256]
            self.assertEqual(offsets[0, 17].item(), offsets[100, 117].item())        # the same (dy, dx) = (1, 1)
            self.assertNotEqual(offsets[0, 17].item(), offsets[0, 18].item())
            self.assertEqual(offsets[5, 5].item(), offsets[200, 200].item())        # (0, 0)
            attention.local_bias[:, offsets[0, 0]] = 20.0                            # a head that only sees itself
            q = torch.randn(1, 256, 32)
            y = attention(q, local.place)
            v = attention.qkv(q).view(1, 256, 3, 2, 16)[:, :, 2].reshape(1, 256, 32)
            self.assertTrue(torch.allclose(y, attention.out(v), atol=1e-3))        # each token its own value
        loss = masked_loss(local, local(x, acts, mask), patch_pixels(x, 16), mask)[0]
        loss.backward()
        self.assertGreater(local.blocks[0].spatial.local_bias.grad.abs().sum().item(), 0)

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

    def test_space_rope_and_qk_norm_train_and_cache_like_the_full_window(self):
        torch.manual_seed(0)
        model = Dynamics(dim=32, layers=2, heads=2, patch=16, frames=6, colour_dim=8, space_rope=True,
                         qk_norm=True).eval()
        names = [n for n, _ in model.named_parameters()]
        self.assertTrue(any("space_pos" in n for n in names))                     # the learned place stays (+ APE)
        self.assertTrue(any("rope_freq" in n for n in names))
        self.assertIsNotNone(model.blocks[0].temporal.q_norm)
        self.assertIsInstance(model.blocks[0].mlp, torch.nn.Sequential)            # not modern's other changes
        self.assertEqual(model.place[0, 17].tolist(), [1.0, 1.0])                 # (row, column) of patch 17
        x, acts = window(1, 6, 9), torch.randint(0, 256, (1, 6))
        mask = torch.zeros(1, 6, 256, dtype=torch.bool)
        with torch.no_grad():
            full = model(x, acts, mask)
            cache = model.new_cache()
            model(x[:, :5], acts[:, :5], mask[:, :5], cache=cache)
            step = model(x[:, 5:], acts[:, 5:], mask[:, 5:], cache=cache, at=5)
            self.assertTrue(torch.allclose(full[:, 5], step[:, 0], atol=1e-4))
            plain = Dynamics(dim=32, layers=2, heads=2, patch=16, frames=6, colour_dim=8, qk_norm=True).eval()
            self.assertFalse(any("rope_freq" in n for n, _ in plain.named_parameters()))
        model.train()
        mask = training_mask(1, 6, 256, "cpu", torch.Generator().manual_seed(1))
        masked_loss(model, model(x, acts, mask), patch_pixels(x, 16), mask)[0].backward()
        self.assertTrue(all(p.grad is not None for p in model.blocks[0].spatial.parameters()))
        self.assertEqual(type(build({"dim": 32, "layers": 2, "heads": 2, "patch": 16, "frames": 6, "colour_dim": 8,
                                     "space_rope": True}).place), torch.Tensor)

    def test_action_norm_starts_as_without_then_conditions_every_block(self):
        torch.manual_seed(0)
        model = Dynamics(dim=32, layers=2, heads=2, patch=16, frames=6, colour_dim=8, action_norm=True).eval()
        plain = Dynamics(dim=32, layers=2, heads=2, patch=16, frames=6, colour_dim=8).eval()
        plain.load_state_dict({k: v for k, v in model.state_dict().items() if "modulation" not in k})
        x, acts = window(1, 6, 9), torch.randint(0, 256, (1, 6))
        mask = torch.zeros(1, 6, 256, dtype=torch.bool)
        mask[:, 5] = True
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(x, acts, mask), plain(x, acts, mask), atol=1e-6))   # zero at first
            for block in model.blocks:
                torch.nn.init.normal_(block.modulation[1].weight, std=0.1)
            self.assertFalse(torch.allclose(model(x, acts, mask), plain(x, acts, mask), atol=1e-4))  # the norms move
            other = acts.clone()
            other[0, 5] = (int(acts[0, 5]) + 1) % 256
            self.assertFalse(torch.allclose(model(x, acts, mask)[:, 5], model(x, other, mask)[:, 5], atol=1e-4))
            full = model(x, acts, mask)
            cache = model.new_cache()
            model(x[:, :5], acts[:, :5], mask[:, :5], cache=cache)
            step = model(x[:, 5:], acts[:, 5:], mask[:, 5:], cache=cache, at=5)
            self.assertTrue(torch.allclose(full[:, 5], step[:, 0], atol=1e-4))
        model.train()
        mask = training_mask(1, 6, 256, "cpu", torch.Generator().manual_seed(1))
        masked_loss(model, model(x, acts, mask), patch_pixels(x, 16), mask)[0].backward()
        self.assertTrue(all(b.modulation[1].weight.grad is not None for b in model.blocks))
        self.assertTrue(build({"dim": 32, "layers": 2, "heads": 2, "patch": 16, "frames": 6, "colour_dim": 8,
                               "action_norm": True}).blocks[0].modulation is not None)

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
        tokens, unsure, wrong, drawn, resynced, motion = rollout(model, x, acts, start=3)
        self.assertIsNone(resynced)                                                # no resync asked for
        self.assertIsNone(motion)                                                  # no motion model
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

    def test_a_resynced_rollout_replaces_only_the_patches_that_went_off(self):
        model = tiny().eval()
        x, acts = window(2, 6, 6), torch.randint(0, 256, (2, 6))
        own = rollout(model, x, acts, start=3)[0]
        tokens, _, _, _, resynced, _ = rollout(model, x, acts, start=3, resync=0)   # every patch is off by >= 0
        self.assertTrue(torch.allclose(tokens, model._patches(x[:, 3:]), atol=1e-5))   # the real frames
        self.assertTrue(torch.equal(resynced, torch.ones(2, 3)))
        tokens, _, _, _, resynced, _ = rollout(model, x, acts, start=3, resync=257)  # no patch has 257 pixels
        self.assertTrue(torch.allclose(tokens, own, atol=1e-5))                   # the model's own frames
        self.assertTrue(torch.equal(resynced, torch.zeros(2, 3)))
        with torch.no_grad():                         # in between: a patch is real exactly where it went off
            mask = torch.zeros(2, 6, 256, dtype=torch.bool)
            mask[:, 3] = True
            probs = model.logits(model(x, acts, mask)[:, 3]).float().softmax(-1)
        off = (probs.argmax(-1) != patch_pixels(x[:, 3], 16)).sum(-1)              # [2, 256] pixels off
        threshold = int(off.float().median())
        tokens, _, _, _, resynced, _ = rollout(model, x, acts, start=3, resync=threshold)
        diverged = off >= threshold
        real, mine = model._patches(x[:, 3:4])[:, 0], own[:, 0]
        self.assertTrue(torch.allclose(tokens[:, 0][diverged], real[diverged], atol=1e-5))
        self.assertTrue(torch.allclose(tokens[:, 0][~diverged], mine[~diverged], atol=1e-5))
        self.assertTrue(torch.allclose(resynced[:, 0], diverged.float().mean(1)))

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
        self.assertTrue(torch.allclose(torch.stack(dreamer.frames[3:], 1)[0], expected, atol=1e-5))   # same frames
        self.assertEqual(probs[0].shape, (256, 256, COLOURS))
        self.assertTrue(torch.allclose(probs[0].sum(-1), torch.ones(256, 256), atol=1e-4))
        for _ in range(5):                                                         # then it slides and keeps going
            dreamer.step(0)
        self.assertLessEqual(len(dreamer.frames), model.frames)

    def test_a_batch_of_dreams_dreams_each_as_alone(self):
        model = tiny().eval()
        x, acts = window(2, 6, 12), torch.randint(0, 256, (2, 6))
        both = Dreamer(model, x[:, :3], acts[:, :3], keep=3)
        alone = [Dreamer(model, x[i, :3], acts[i, :3], keep=3) for i in range(2)]
        for t in range(3, 9):                                     # past the window (6): both slide alike
            pressed = acts[:, t % 6]
            together = both.step(pressed)
            self.assertEqual(together.shape, (2, 256, 256, COLOURS))
            for i in range(2):
                self.assertTrue(torch.allclose(together[i], alone[i].step(int(pressed[i])), atol=1e-5))

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

    def test_a_direction_press_weighs_the_frames_it_reaches(self):
        import train_dynamics_ui as D
        from token_world.data.tetris_bot import A, DOWN, LEFT, START
        incoming = torch.tensor([[-1, 0, LEFT, 0, 0, DOWN, 0, A | DOWN, A, 0, START, 0]])
        weight = D.press_weights(incoming, 10.0)[0]
        # Left reaches its frame and the next; Down alone and Start nothing; A held two frames reaches three
        self.assertEqual(weight.tolist(), [1, 1, 10, 10, 1, 1, 1, 10, 10, 10, 1, 1])
        self.assertIsNone(D.press_weights(incoming, 1.0))

    def test_pressed_frames_and_a_twins_branch_frames(self):
        import train_dynamics_ui as D
        from token_world.data.tetris_bot import A, DOWN, LEFT
        incoming = torch.tensor([[-1, 0, LEFT, 0, 0, DOWN, 0, A, 0, 0],
                                 [-1, 0, LEFT, 0, 0, DOWN, 0, 0, 0, 0]])      # the twin: its A at 7 released
        self.assertEqual(D.pressed_frames(incoming)[0].tolist(), [0, 0, 1, 1, 0, 0, 0, 1, 1, 0])
        # the twin's branch press is action 6 (incoming at frame 7): frames 7 and 8 count as the real one's do
        reached = D.pressed_frames(incoming, torch.tensor([-1, 6]))
        self.assertEqual(reached[0].tolist(), [0, 0, 1, 1, 0, 0, 0, 1, 1, 0])
        self.assertEqual(reached[1].tolist(), [0, 0, 1, 1, 0, 0, 0, 1, 1, 0])

    def test_a_press_weighted_step_changes_only_the_loss_weighting(self):
        import train_dynamics_ui as D
        from token_world.data.tetris_bot import RIGHT
        x = window(4, 6, 34)
        acts = torch.zeros(4, 6, dtype=torch.long)
        acts[:, 2] = RIGHT
        out = [D.train_step(m, m, x, acts, generator=torch.Generator().manual_seed(0), press_weight=w)
               for m, w in ((tiny(), 1.0), (tiny(), 10.0))]
        self.assertTrue(torch.allclose(out[0]["game"], out[1]["game"]))           # the same unweighted losses
        self.assertFalse(torch.allclose(out[0]["total"], out[1]["total"]))
        out[1]["total"].backward()

    def test_motion_starts_as_without_and_goes_in_wherever_it_is_known(self):
        from token_world.data.tetris_motion import CODES, UNKNOWN
        torch.manual_seed(0)
        model = Dynamics(dim=32, layers=2, heads=2, patch=16, frames=6, colour_dim=8, motion=True).eval()
        x, acts = window(1, 6, 21), torch.randint(0, 256, (1, 6))
        mask = torch.zeros(1, 6, 256, dtype=torch.bool)
        mask[:, 4:] = True
        motion = motion_tokens(torch.randint(0, CODES, (1, 6, 32, 32)), 16)
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(x, acts, mask, motion=motion), model(x, acts, mask)))   # from zero
            model.motion_in.weight.normal_()
            added = model.motion_embedding(motion)
            self.assertTrue((added.abs().sum(-1) > 0).all())                         # masked tokens too
            unknown = torch.full_like(motion, UNKNOWN)
            self.assertTrue((model.motion_embedding(unknown) == 0).all())          # withheld: nothing
            self.assertFalse(torch.allclose(model(x, acts, mask, motion=motion), model(x, acts, mask)))

    def test_motion_loss_scores_the_masked_picture_cells_and_trains_the_head(self):
        from token_world.data.tetris_motion import UNKNOWN, code_of
        torch.manual_seed(0)
        model = Dynamics(dim=32, layers=2, heads=2, patch=16, frames=6, colour_dim=8, motion=True)
        x, acts = window(2, 6, 22), torch.randint(0, 256, (2, 6))
        mask = training_mask(2, 6, 256, "cpu", torch.Generator().manual_seed(3))
        cells = torch.zeros(2, 6, 32, 32, dtype=torch.uint8)
        cells[:, :, 10, 12] = code_of(1, -1)                                         # a turn's cell
        cells[:, :, 10, 14] = code_of(0, 1)                                          # a shift's
        cells[:, :2] = UNKNOWN
        cell_mask = mask[..., None].expand(-1, -1, -1, 4)
        out = motion_loss(model, model(x, acts, mask), motion_tokens(cells, 16), cell_mask)
        self.assertTrue(torch.isfinite(out["motion"]))
        self.assertTrue(0 <= out["motion_turned"] <= 1 and 0 <= out["motion_moved"] <= 1)
        out["motion"].backward()
        self.assertGreater(model.motion_out.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.blocks[0].temporal.qkv.weight.grad.abs().sum().item(), 0)
        border = torch.zeros_like(cells)
        border[:, :, :2] = code_of(1, 0)                                             # the border's rows only
        none = motion_loss(model, model(x, acts, mask), motion_tokens(border, 16), cell_mask)
        clean = motion_loss(model, model(x, acts, mask), motion_tokens(torch.zeros_like(cells), 16), cell_mask)
        self.assertAlmostEqual(none["motion"].item(), clean["motion"].item(), places=6)   # not scored

    def test_a_dream_writes_its_own_motion_into_its_history(self):
        from token_world.models.dynamics import FrameDecoder, motion_cells
        torch.manual_seed(0)
        model = Dynamics(dim=32, layers=2, heads=2, patch=16, frames=6, colour_dim=8, motion=True).eval()
        with torch.no_grad():
            model.motion_in.weight.normal_()
        x, acts = window(1, 6, 23), torch.randint(0, 256, (1, 6))
        decoder = FrameDecoder(model, 1, x.device, (False, torch.bfloat16))
        decoder.prefill(x[:, :3], acts[:, :3])
        first = decoder.next(acts[:, 3:4], 3)
        second = decoder.next(acts[:, 4:5], 4)
        # the same from the whole window: frame 3 its soft frame and its own motion, frame 4 masked
        soft = soft_window(model, x)
        soft[:, 3] = first["soft"]
        given = torch.full((1, 6, 256, 4), 255, dtype=torch.long)
        given[:, 3] = motion_tokens(first["motion"], 16)
        mask = torch.zeros(1, 6, 256, dtype=torch.bool)
        mask[:, 4] = True
        with torch.no_grad():
            whole = model.logits(model(soft, acts, mask, motion=given)[:, 4]).float().softmax(-1)
        from token_world.models.dynamics import unpatch_pixels as unpatch
        self.assertTrue(torch.allclose(second["probs"], unpatch(whole.permute(0, 3, 1, 2), 16).permute(0, 2, 3, 1),
                                       atol=1e-4))
        dreamer = Dreamer(model, x[0, :3], acts[0, :3], keep=3)
        for t in range(3, 9):                                                        # past the window: it slides
            dreamer.step(int(acts[0, t % 6]))
            self.assertEqual(dreamer.motion.shape, (32, 32))
        self.assertEqual(len(dreamer.motions), len(dreamer.frames))
        self.assertTrue((dreamer.motion[:2] == 0).all() and (dreamer.motion[-2:] == 0).all())   # the border: still

    def test_motion_first_draws_the_pixels_given_the_motion_it_drew(self):
        from token_world.models.dynamics import FrameDecoder, motion_tokens as tokens_of, unpatch_pixels as unpatch
        torch.manual_seed(0)
        model = Dynamics(dim=32, layers=2, heads=2, patch=16, frames=6, colour_dim=8, motion=True,
                         motion_first=True).eval()
        with torch.no_grad():
            model.motion_in.weight.normal_()
        x, acts = window(1, 6, 24), torch.randint(0, 256, (1, 6))
        decoder = FrameDecoder(model, 1, x.device, (False, torch.bfloat16))
        decoder.prefill(x[:, :3], acts[:, :3])
        out = decoder.next(acts[:, 3:4], 3)
        mask = torch.zeros(1, 6, 256, dtype=torch.bool)
        mask[:, 3] = True
        given = torch.full((1, 6, 256, 4), 255, dtype=torch.long)
        with torch.no_grad():
            first = model(x, acts, mask)[:, 3]                                       # motion, step 1
            committed = model.committed_motion(first)
            given[:, 3] = committed
            second = model(x, acts, mask, motion=given)[:, 3]                       # motion, step 2
            moved = torch.where(committed != 255, committed, model.drawn_motion(second))
            self.assertTrue(torch.equal(tokens_of(out["motion"], 16), moved))
            given[:, 3] = moved
            probs = model.logits(model(x, acts, mask, motion=given)[:, 3]).float().softmax(-1)   # the pixels
        self.assertTrue(torch.allclose(out["probs"], unpatch(probs.permute(0, 3, 1, 2), 16).permute(0, 2, 3, 1),
                                       atol=1e-4))
        with self.assertRaises(ValueError):
            Dynamics(dim=32, layers=2, heads=2, patch=16, frames=6, colour_dim=8, motion_first=True)

    def test_one_step_motion_draws_every_cell_from_the_first_pass_then_the_pixels_given_it(self):
        from token_world.models.dynamics import FrameDecoder, motion_tokens as tokens_of, unpatch_pixels as unpatch
        torch.manual_seed(0)
        model = Dynamics(dim=32, layers=2, heads=2, patch=16, frames=6, colour_dim=8, motion=True,
                         motion_first=True).eval()
        model.motion_steps = 1
        with torch.no_grad():
            model.motion_in.weight.normal_()
        x, acts = window(1, 6, 25), torch.randint(0, 256, (1, 6))
        decoder = FrameDecoder(model, 1, x.device, (False, torch.bfloat16))
        decoder.prefill(x[:, :3], acts[:, :3])
        out = decoder.next(acts[:, 3:4], 3)
        mask = torch.zeros(1, 6, 256, dtype=torch.bool)
        mask[:, 3] = True
        given = torch.full((1, 6, 256, 4), 255, dtype=torch.long)
        with torch.no_grad():
            moved = model.drawn_motion(model(x, acts, mask)[:, 3])                     # the one motion pass
            self.assertTrue(torch.equal(tokens_of(out["motion"], 16), moved))
            given[:, 3] = moved
            probs = model.logits(model(x, acts, mask, motion=given)[:, 3]).float().softmax(-1)   # the pixels
        self.assertTrue(torch.allclose(out["probs"], unpatch(probs.permute(0, 3, 1, 2), 16).permute(0, 2, 3, 1),
                                       atol=1e-4))

    def test_motion_first_training_gives_a_share_of_masked_frames_their_motion_and_does_not_score_it(self):
        import train_dynamics_ui as D
        from token_world.data.tetris_motion import code_of
        torch.manual_seed(0)
        x = window(4, 6, 34)
        cells = torch.zeros(4, 6, 32, 32, dtype=torch.uint8)
        cells[:, 2:, 12, 14] = code_of(1, 0)
        seen = []
        for first in (False, True):
            model = Dynamics(dim=32, layers=2, heads=2, patch=16, frames=6, colour_dim=8, motion=True,
                             motion_first=first)
            calls = []
            forward = lambda *a, **k: (calls.append(k["motion"]), model(*a, **k))[1]
            losses = D.train_step(model, forward, x, torch.randint(0, 256, (4, 6)),
                                  generator=torch.Generator().manual_seed(0), motion=cells)
            self.assertTrue(torch.isfinite(losses["motion"]))
            mask = D.training_mask(4, 6, 256, "cpu", torch.Generator().manual_seed(0))
            given = (calls[0] != 255) & mask[..., None]                              # masked cells given
            partly = given.any(-1) & ~given.all(-1)
            seen.append((given.float().sum() / (4 * mask.sum()), partly.any()))
        self.assertEqual(seen[0][0].item(), 0.0)                                     # plain: masked cells never
        self.assertGreater(seen[1][0].item(), 0.1)                                   # motion first: a share,
        self.assertTrue(seen[1][1])                                                  # some tokens only partly

    def test_a_resynced_motion_rollout_gives_the_replaced_patches_their_real_motion(self):
        from token_world.data.tetris_motion import CODES
        torch.manual_seed(0)
        model = Dynamics(dim=32, layers=2, heads=2, patch=16, frames=6, colour_dim=8, motion=True,
                         motion_first=True).eval()
        x, acts = window(2, 6, 7), torch.randint(0, 256, (2, 6))
        cells = torch.randint(0, CODES, (2, 6, 32, 32), dtype=torch.uint8)
        own = rollout(model, x, acts, start=3, motion=cells)[5]                    # [2, 3, 32, 32] its own motion
        self.assertEqual(own.shape, (2, 3, 32, 32))
        self.assertFalse(torch.equal(own.to(torch.uint8), cells[:, 3:]))          # not the real motion
        every = rollout(model, x, acts, start=3, resync=0, motion=cells)[5]       # every patch replaced
        self.assertTrue(torch.equal(every.to(torch.uint8), cells[:, 3:]))
        none = rollout(model, x, acts, start=3, resync=257, motion=cells)[5]      # none replaced
        self.assertTrue(torch.equal(none, own))

    def test_a_motion_model_trains_on_resynced_rollouts_with_the_motion_they_were_drawn_with(self):
        import train_dynamics_ui as D
        from token_world.data.tetris_motion import CODES
        torch.manual_seed(0)
        model = Dynamics(dim=32, layers=2, heads=2, patch=16, frames=6, colour_dim=8, motion=True,
                         motion_first=True)
        x, acts = window(2, 6, 8), torch.randint(0, 256, (2, 6))
        cells = torch.randint(0, CODES, (2, 6, 32, 32), dtype=torch.uint8)
        calls = []
        forward = lambda *a, **k: (calls.append(k["motion"]), model(*a, **k))[1]
        losses = D.train_step(model, forward, x, acts, depth=3, generator=torch.Generator().manual_seed(1),
                              resync=257, motion=cells)                         # the history keeps its own motion
        k = int(losses["rollout_frames"])
        self.assertGreaterEqual(k, 1)
        self.assertTrue(torch.isfinite(losses["total"]) and torch.isfinite(losses["motion"]))
        losses["total"].backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        given, real = calls[0], D.motion_tokens(cells, 16)
        known = given != 255                                                      # what was not withheld
        drawn = (given != real) & known                                            # given, and not the real motion
        frames = drawn.flatten(2).any(-1).any(0).nonzero()[:, 0].tolist()
        self.assertTrue(frames, "the dreamed history's motion should differ from the real")
        self.assertEqual(len(frames), len(set(frames)))
        self.assertTrue(all(1 <= f < 5 for f in frames))                           # only rollout frames, never the last
        self.assertLessEqual(max(frames) - min(frames) + 1, k)

    def test_the_first_motion_step_commits_the_sure_cells_and_one_more(self):
        from token_world.data.tetris_motion import STILL, UNKNOWN, code_of
        torch.manual_seed(0)
        model = Dynamics(dim=32, layers=2, heads=2, patch=16, frames=6, colour_dim=8, motion=True,
                         motion_first=True).eval()
        cw, ccw = code_of(1, -1), code_of(-1, 1)
        with torch.no_grad():
            model.motion_out.weight.zero_()
            bias = torch.full((4, 46), -30.0)
            bias[:, STILL] = 0.0                                                     # every cell sure it stays
            model.motion_out.bias.copy_(bias.flatten())
            logits = model.motion_logits(torch.zeros(1, 256, 32))
            features = torch.zeros(1, 256, 32)
        # a piece over tokens 100 and 101, unsure between two turns: cell (100, 0) the most confident of them
        unsure = {(100, 0): 0.55, (100, 1): 0.52, (101, 2): 0.51}

        def logits_of(f):
            out = logits.clone()
            for (n, c), p in unsure.items():
                out[0, n, c] = -30.0
                out[0, n, c, cw], out[0, n, c, ccw] = math.log(p), math.log(1 - p)
            return out
        model.motion_logits = logits_of
        committed = model.committed_motion(features)
        self.assertEqual(committed[0, 100, 0].item(), cw)                           # the most confident unsure cell
        self.assertEqual(committed[0, 100, 1].item(), UNKNOWN)                      # the other unsure ones: step 2
        self.assertEqual(committed[0, 101, 2].item(), UNKNOWN)
        self.assertEqual(int((committed == UNKNOWN).sum()), 2)                      # everything else is sure
        self.assertTrue((committed[0, :16] == STILL).all())                         # the border

    def test_motion_is_withheld_from_whole_windows_or_before_a_cut(self):
        import train_dynamics_ui as D
        from token_world.data.tetris_motion import UNKNOWN
        motion = torch.zeros(4000, 8, 1, 1, dtype=torch.long)
        given = D.withheld_motion(motion, torch.Generator().manual_seed(0))[:, :, 0, 0] == UNKNOWN   # [B, T]
        none, full = given.all(1), ~given.any(1)
        self.assertAlmostEqual(none.float().mean().item(), D.MOTION_DROP, delta=0.03)
        self.assertAlmostEqual(full.float().mean().item(), 1 - 2 * D.MOTION_DROP, delta=0.03)
        part = given[~none & ~full]
        self.assertTrue((part[:, :-1] >= part[:, 1:]).all())                      # withheld up to a cut, then given

    def test_a_motion_step_trains_on_the_motion_and_reports_it(self):
        import train_dynamics_ui as D
        from token_world.data.tetris_motion import code_of
        torch.manual_seed(0)
        model = Dynamics(dim=32, layers=2, heads=2, patch=16, frames=6, colour_dim=8, motion=True)
        x = window(4, 6, 34)
        cells = torch.zeros(4, 6, 32, 32, dtype=torch.uint8)
        cells[:, 2:, 12, 14] = code_of(1, 0)
        losses = D.train_step(model, model, x, torch.randint(0, 256, (4, 6)), generator=torch.Generator().manual_seed(0),
                              motion=cells)
        self.assertTrue(torch.isfinite(losses["motion"]))
        losses["total"].backward()
        self.assertGreater(model.motion_out.weight.grad.abs().sum().item(), 0)
        with self.assertRaises(ValueError):                                          # it needs the batch's motion
            D.train_step(model, model, x, torch.randint(0, 256, (4, 6)))

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
