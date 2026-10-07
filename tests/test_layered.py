"""The layered world models (models/layered.py, layered_pixels.py, layered_slots.py): temporal attention
follows identities and rotary time, frames never see later frames, a dream's cache matches the full window,
the losses train, dreams compose and batch, a rollout is the dream of those frames, and movement is scored
from the history the model was given. Shared behaviour is checked on both models."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from token_world.data.nes_layers import BANDS, CELL, LAYER_COLOURS, SPRITES, TRANSPARENT, VIEW, compose
from token_world.models.layered import (CELLS, SpatialAttention, build, mixed_rotary, patches, rollout, rotary,
                                        slots_of, unpatch)
from token_world.models.layered_pixels import SPRITE_TOKENS, PixelLayers
from token_world.models.layered_slots import TOKENS as SLOT_TOKENS, SlotLayers


class PixelsWithOptions(PixelLayers):
    """Every option: shared pixel head, registers, rotary by screen place."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, share_pixels=True, registers=4, space_rope=True, fixed_camera=True, **kwargs)


class SlotsWithOptions(SlotLayers):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, registers=4, space_rope=True, **kwargs)


KINDS = (PixelLayers, SlotLayers, PixelsWithOptions, SlotsWithOptions)


def tiny(kind=PixelLayers, frames=6):
    torch.manual_seed(0)
    return kind(dim=32, layers=2, heads=2, frames=frames, colour_dim=8)


def window(b=2, t=6, seed=0, scroll=0):
    """Synthetic layered frames: random pixels, a camera moving `scroll` px a frame, and a sprite block moving
    right 2 px a frame, as a sprite picture and as six sprite slots."""
    g = torch.Generator().manual_seed(seed)
    camera = torch.zeros(b, t, BANDS, 2, dtype=torch.int16)
    camera[..., 0] = (256 + scroll * torch.arange(t))[None, :, None]
    camera[..., 1] = 240 + CELL * torch.arange(BANDS)
    picture = torch.full((b, t, 240, 256), TRANSPARENT, dtype=torch.uint8)
    block = torch.randint(0, 55, (b, 16, 16), generator=g).to(torch.uint8)
    for i in range(t):
        picture[:, i, 100:116, 40 + 2 * i:56 + 2 * i] = block
    flags = torch.zeros(b, t, SPRITES, dtype=torch.uint8)
    flags[:, :, :6] = 1
    xy = torch.randint(0, 200, (b, 1, SPRITES, 2), generator=g).repeat(1, t, 1, 1)
    xy[..., 0] += torch.arange(t)[None, :, None] * 2                       # sprites move right 2 px a frame
    sprites = torch.randint(0, LAYER_COLOURS, (b, t, SPRITES, 8, 8), generator=g)
    sprites[:, :, 6:] = TRANSPARENT
    return {"cells": torch.randint(0, LAYER_COLOURS, (b, t, BANDS, VIEW, CELL, CELL), generator=g).to(torch.uint8),
            "cell_known": torch.ones(b, t, BANDS, VIEW, CELL, CELL, dtype=torch.bool),
            "cell_row": (camera[..., 1] % 480 // CELL).to(torch.int16),
            "cell_col": (camera[..., 0] // CELL).to(torch.int16), "camera": camera, "sprite_layer": picture,
            "sprites": sprites.to(torch.uint8), "sprite_xy": (xy * (flags[..., None] > 0)).to(torch.uint8),
            "sprite_flags": flags, "border": torch.randint(55, 58, (b, t, 32, 256), generator=g).to(torch.uint8),
            "backdrop": torch.randint(0, 55, (b, t), generator=g).to(torch.uint8)}


class SharedTests(unittest.TestCase):
    def test_cells_keep_their_world_identity_as_the_camera_moves(self):
        layers = window(1, 3, scroll=16)
        for kind in KINDS:
            model = tiny(kind)
            s = slots_of(layers, model.tokens)
            self.assertEqual(s.shape, (1, 3, model.tokens))
            self.assertTrue(all(len(set(s[0, i].tolist())) == model.tokens for i in range(3)))
            self.assertTrue(int(s.max()) < model.slots)
            # one cell of scroll: frame 1's view column 0 is frame 0's view column 1 (the same world cell)
            self.assertEqual(int(s[0, 1, 0]), int(s[0, 0, 1]))
            self.assertTrue(torch.equal(s[0, 0, CELLS:], s[0, 2, CELLS:]))  # the rest by index

    def test_the_token_layouts(self):
        self.assertEqual(tiny(PixelLayers).tokens, 543)
        self.assertEqual(tiny(SlotLayers).tokens, SLOT_TOKENS)
        self.assertEqual(SLOT_TOKENS, 367)

    def test_patches_cut_a_picture_into_row_major_tokens_and_back(self):
        picture = torch.arange(240 * 256).view(240, 256)
        tokens = patches(picture)
        self.assertEqual(tokens.shape, (SPRITE_TOKENS, 16, 16))
        self.assertTrue(torch.equal(tokens[17], picture[16:32, 16:32]))   # row 1, column 1
        self.assertTrue(torch.equal(unpatch(tokens), picture))
        border = torch.arange(32 * 256).view(32, 256)
        self.assertTrue(torch.equal(unpatch(patches(border)), border))

    def test_rotary_time_scores_by_how_far_apart_frames_are(self):
        q, k = torch.randn(1, 1, 1, 2, 24), torch.randn(1, 1, 1, 2, 24)
        score = lambda tq, tk: (rotary(q, torch.tensor([tq])) * rotary(k, torch.tensor([tk]))).sum()
        self.assertTrue(torch.allclose(score(5, 3), score(40, 38), atol=1e-4))
        self.assertFalse(torch.allclose(score(5, 3), score(5, 4), atol=1e-4))

    def test_recompute_gives_the_same_losses_and_gradients(self):
        layers, acts = window(1, 4, 12), torch.randint(0, 256, (1, 4))
        for kind in KINDS:
            grads = []
            for recompute in (False, True):
                model = tiny(kind, frames=4)
                model.recompute = recompute
                mask = model.mask(1, 4, "cpu", torch.Generator().manual_seed(1))
                loss = model.loss(model(layers, acts, mask), layers, acts, mask)["total"]
                loss.backward()
                grads.append((loss.item(), model.cell_pixels.weight.grad.clone()))
            self.assertAlmostEqual(grads[0][0], grads[1][0], places=5)
            self.assertTrue(torch.allclose(grads[0][1], grads[1][1], atol=1e-6), kind.__name__)

    def test_routed_attention_is_attention_per_identity(self):
        model = tiny(PixelLayers).eval()
        layer = model.blocks[0].temporal
        x = torch.randn(1, 4, model.tokens, 32)
        slots = slots_of(window(1, 4, scroll=8), model.tokens)
        with torch.no_grad():
            routed = layer(x, slots)
            b, t, n, d = x.shape
            q, k, v = layer.qkv(x).view(b, t, n, 3, layer.heads, d // layer.heads).unbind(3)
            q, k = rotary(q, torch.arange(t)), rotary(k, torch.arange(t))
            naive = torch.zeros(t, n, d)
            for ti in range(t):
                for i in range(0, n, 37):                                    # a sample of tokens
                    keys = [(tj, j) for tj in range(ti + 1) for j in range(n) if slots[0, tj, j] == slots[0, ti, i]]
                    kk = torch.stack([k[0, tj, j] for tj, j in keys], 1)    # [h, K, hd]
                    vv = torch.stack([v[0, tj, j] for tj, j in keys], 1)
                    a = torch.softmax((q[0, ti, i][:, None] * kk).sum(-1) / (d // layer.heads) ** 0.5, -1)
                    naive[ti, i] = (a[..., None] * vv).sum(1).flatten()
            naive = layer.out(naive)
        for ti in range(4):
            for i in range(0, n, 37):
                self.assertTrue(torch.allclose(routed[0, ti, i], naive[ti, i], atol=1e-5), (ti, i))

    def test_frames_never_see_later_frames(self):
        for kind in KINDS:
            model = tiny(kind).eval()
            a, b = window(1, 6, 1, scroll=4), window(1, 6, 1, scroll=4)
            b["cells"][:, 4:] = 0
            b["sprite_layer"][:, 4:] = 7
            b["sprite_xy"][:, 4:] = 7
            acts = torch.randint(0, 256, (1, 6))
            mask = torch.zeros(1, 6, model.tokens, dtype=torch.bool)
            with torch.no_grad():
                fa, fb = model(a, acts, mask), model(b, acts, mask)
            self.assertTrue(torch.allclose(fa[:, :4], fb[:, :4], atol=1e-5), kind.__name__)
            self.assertFalse(torch.allclose(fa[:, 4:], fb[:, 4:], atol=1e-5), kind.__name__)

    def test_the_cache_matches_the_full_window(self):
        for kind in KINDS:
            model = tiny(kind).eval()
            layers, acts = window(1, 6, 2, scroll=5), torch.randint(0, 256, (1, 6))
            mask = torch.zeros(1, 6, model.tokens, dtype=torch.bool)
            with torch.no_grad():
                full = model(layers, acts, mask)
                cache = model.new_cache()
                first = model({k: v[:, :3] for k, v in layers.items()}, acts[:, :3], mask[:, :3], cache, 0)
                steps = [model({k: v[:, i:i + 1] for k, v in layers.items()}, acts[:, i:i + 1], mask[:, i:i + 1],
                               cache, i) for i in range(3, 6)]
            self.assertTrue(torch.allclose(first, full[:, :3], atol=1e-5), kind.__name__)
            self.assertTrue(torch.allclose(torch.cat(steps, 1), full[:, 3:], atol=1e-4), kind.__name__)

    def test_the_camera_is_never_hidden(self):
        for kind in KINDS:
            model = tiny(kind)
            mask = model.mask(4, 6, "cpu", torch.Generator().manual_seed(0))
            self.assertFalse(mask[:, :, model.camera_at:model.frame_at].any())
            self.assertTrue(mask[:, 1:].any())

    def test_the_losses_train(self):
        layers, acts = window(2, 6, 3, scroll=3), torch.randint(0, 256, (2, 6))
        for kind in KINDS:
            sprite_part = "sprites" if issubclass(kind, PixelLayers) else "sprite_flags"
            model = tiny(kind)
            optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
            first = None
            for step in range(30):
                mask = model.mask(2, 6, "cpu", torch.Generator().manual_seed(step))
                losses = model.loss(model(layers, acts, mask), layers, acts, mask)
                self.assertEqual(set(losses) - {"total"}, set(model.PARTS))
                for key, value in losses.items():
                    self.assertTrue(torch.isfinite(value), key)
                optimizer.zero_grad()
                losses["total"].backward()
                optimizer.step()
                first = first if first is not None else {k: float(v) for k, v in losses.items()}
            if "camera" in model.PARTS:
                self.assertLess(float(losses["camera"]), first["camera"], kind.__name__)
            self.assertLess(float(losses[sprite_part]), first[sprite_part], kind.__name__)
            self.assertGreater(model.sprite_head().weight.grad.abs().sum().item(), 0)

    def test_a_dream_runs_and_composes(self):
        for kind in KINDS:
            model = tiny(kind, frames=6).eval()
            layers, acts = window(1, 3, 4), torch.randint(0, 256, (1, 3))
            dreamer = model.Dreamer(model, layers, acts, keep=3, generator=torch.Generator().manual_seed(0))
            for a in range(5):                                               # past the window: it slides
                frame = dreamer.step(torch.tensor([a]))
            self.assertEqual(set(frame) - {"probs"}, set(kind.KEYS))
            picture = compose({k: v[0].numpy() for k, v in frame.items() if k != "probs"})
            self.assertEqual(picture.shape, (240, 256))
            self.assertEqual(frame["probs"]["cells"].shape[-1], LAYER_COLOURS)
            self.assertEqual(frame["probs"]["border"].shape[1:], (32, 256, LAYER_COLOURS))
            self.assertEqual(frame["probs"]["sprites"].shape[1], model.SPRITE_TOKENS)
            self.assertLessEqual(len(dreamer.history), model.frames)

    def test_a_batch_of_dreams_is_each_dream_on_its_own(self):
        for kind in KINDS:
            model = tiny(kind, frames=6).eval()
            layers, acts = window(2, 3, 6), torch.randint(0, 256, (2, 3))
            both = model.Dreamer(model, layers, acts, keep=3, temperature=0.0)
            one = model.Dreamer(model, {k: v[1:] for k, v in layers.items()}, acts[1:], keep=3, temperature=0.0)
            for a in range(4):
                fb, f1 = both.step(torch.tensor([a, a])), one.step(torch.tensor([a]))
                for key in kind.KEYS:
                    self.assertTrue(torch.equal(fb[key][1], f1[key][0]), (kind.__name__, a, key))

    def test_a_rollout_is_the_dream_of_those_frames(self):
        for kind in KINDS:
            model = tiny(kind, frames=8).eval()
            layers, acts = window(2, 8, 8), torch.randint(0, 256, (2, 8))
            frames, tokens = rollout(model, layers, acts, 3, 6, temperature=0.0)
            self.assertEqual(set(frames), set(kind.KEYS))
            self.assertEqual(tokens["cells"].shape, (2, 3, CELLS, 32))
            self.assertEqual(tokens["sprites"].shape, (2, 3, model.SPRITE_TOKENS, 32))
            self.assertEqual(tokens["border"].shape[2], 32)
            dream = model.Dreamer(model, {k: v[:, :3] for k, v in layers.items()}, acts[:, :3], temperature=0.0)
            for i, at in enumerate(range(3, 6)):
                f = dream.step(acts[:, at])
                for key in kind.KEYS:
                    self.assertTrue(torch.equal(f[key], frames[key][:, i]), (kind.__name__, key))

    def test_the_camera_is_scored_from_the_camera_it_was_given(self):
        model = tiny(PixelLayers)
        layers, acts = window(1, 6, 9, scroll=2), torch.randint(0, 256, (1, 6))
        mask = model.mask(1, 6, "cpu", torch.Generator().manual_seed(0))
        features = model(layers, acts, mask)
        same = model.loss(features, layers, acts, mask)
        given = model.loss(features, layers, acts, mask, seen=layers)
        self.assertTrue(torch.allclose(same["camera"], given["camera"]))
        moved = {k: v.clone() for k, v in layers.items()}
        moved["camera"][:, 2, :, 0] += 4                                   # the history's camera 4 px off
        self.assertFalse(torch.allclose(model.loss(features, layers, acts, mask, seen=moved)["camera"],
                                        same["camera"]))


class SlotTests(unittest.TestCase):
    def test_a_stepped_dream_decides_every_sprite(self):
        model = tiny(SlotLayers, frames=6).eval()
        layers, acts = window(1, 3, 5), torch.randint(0, 256, (1, 3))
        dreamer = model.Dreamer(model, layers, acts, keep=3, steps=4, generator=torch.Generator().manual_seed(1))
        frame = {k: v[0] for k, v in dreamer.step(torch.zeros(1)).items() if k != "probs"}
        on = (frame["sprite_flags"] & 1).bool()
        self.assertTrue(set(np.unique(frame["sprite_flags"].numpy())) <= {0, 1, 3})
        self.assertTrue((frame["sprites"][~on] == TRANSPARENT).all())
        self.assertTrue((frame["sprite_xy"][~on] == 0).all())

    def test_movement_is_scored_from_where_the_history_put_a_sprite(self):
        model = tiny(SlotLayers)
        layers, acts = window(1, 6, 9), torch.randint(0, 256, (1, 6))
        mask = torch.ones(1, 6, model.tokens, dtype=torch.bool)
        mask[:, 0] = False
        features = model(layers, acts, mask)
        same = model.loss(features, layers, acts, mask)
        moved = {k: v.clone() for k, v in layers.items()}
        moved["sprite_xy"][:, 2] += 4                                      # the history's sprites 4 px off
        self.assertFalse(torch.allclose(model.loss(features, layers, acts, mask, seen=moved)["sprite_move"],
                                        same["sprite_move"]))


class OptionTests(unittest.TestCase):
    def test_space_rope_scores_by_offset_on_screen(self):
        torch.manual_seed(0)
        attention = SpatialAttention(32, 2, rope=True).eval()
        x, place = torch.randn(1, 6, 32), torch.rand(1, 6, 2) * 14
        placed = torch.tensor([True, True, True, True, False, False])
        with torch.no_grad():
            y = attention(x, place, placed)
            shifted = attention(x, place + torch.tensor([3.0, -2.5]), placed)      # everything moves together
            moved = place.clone()
            moved[0, 4:] += 5                                                       # tokens with no place move
            unplaced = attention(x, moved, placed)
            moved[0, 0] += 1                                                        # one placed token moves
            other = attention(x, moved, placed)
        self.assertTrue(torch.allclose(y, shifted, atol=1e-5))
        self.assertTrue(torch.allclose(y, unplaced, atol=1e-6))
        self.assertFalse(torch.allclose(y, other, atol=1e-4))

    def test_mixed_rotary_turns_only_its_pairs(self):
        x = torch.randn(3, 2, 16)
        self.assertTrue(torch.allclose(mixed_rotary(x, torch.zeros(3, 2, 4)), x))
        turned = mixed_rotary(x, torch.full((3, 2, 4), 0.7))
        self.assertTrue(torch.equal(turned[..., 8:], x[..., 8:]))
        self.assertTrue(torch.allclose(turned[..., :8].norm(dim=-1), x[..., :8].norm(dim=-1), atol=1e-5))

    def test_screen_places_cells_under_the_sprite_picture(self):
        model = tiny(PixelsWithOptions)
        layers = window(1, 2, scroll=4)
        place, placed = model.screen(layers)
        self.assertEqual(place.shape, (1, 2, model.tokens, 2))
        self.assertTrue(torch.allclose(place[0, 0, :CELLS].view(BANDS, VIEW, 2)[3, 5], torch.tensor([3.0, 5.0])))
        self.assertTrue(torch.allclose(place[0, 1, :CELLS].view(BANDS, VIEW, 2)[3, 5], torch.tensor([3.0, 4.75])))
        sprite = place[0, 0, model.sprite_at + 3 * 16 + 5]                         # the patch at row 3, column 5
        self.assertTrue(torch.allclose(sprite, torch.tensor([3.0, 5.0])))
        self.assertTrue(placed[:CELLS].all() and placed[model.sprite_at:model.camera_at].all())
        self.assertFalse(placed[model.camera_at:model.border_at].any() or placed[model.register_at:].any())
        self.assertTrue(placed[model.border_at:model.border_end].all())
        slots = tiny(SlotsWithOptions)
        self.assertFalse(slots.screen(layers)[1][slots.sprite_at:slots.camera_at].any())  # their place is content

    def test_registers_are_never_hidden_and_have_no_target(self):
        model = tiny(PixelsWithOptions)
        self.assertEqual(model.tokens, 543 - BANDS + 4)
        mask = model.mask(2, 6, "cpu", torch.Generator().manual_seed(0))
        self.assertFalse(mask[:, :, model.register_at:].any())
        layers, acts = window(2, 6, 3), torch.randint(0, 256, (2, 6))
        losses = model.loss(model(layers, acts, mask), layers, acts, mask)
        losses["total"].backward()
        self.assertGreater(model.register.grad.abs().sum().item(), 0)             # used, through attention

    def test_options_build_from_a_runs_config_and_old_runs_have_none(self):
        base = {"dim": 96, "layers": 8, "heads": 4, "frames": 64, "colour_dim": 16, "kind": "layered"}
        old = build({**base, "model": None})
        self.assertEqual(sum(p.numel() for p in old.parameters()), 3_029_914)     # world_model_tetris_layered_px
        self.assertFalse(any("register" in k or "rope_freq" in k for k in old.state_dict()))
        options = {"share_pixels": True, "registers": 8, "space_rope": True, "fixed_camera": True}
        new = build({**base, "model": "pixels", **options})
        self.assertEqual(new.tokens, 543 - BANDS + 8)
        self.assertLess(sum(p.numel() for p in new.parameters()), 2_100_000)
        self.assertTrue(any("rope_freq" in k for k in new.state_dict()))
        self.assertFalse(any("camera" in k for k in new.state_dict()))
        new.load_state_dict(build({**base, "model": "pixels", **options}).state_dict())

    def test_a_fixed_camera_stays_where_the_real_frames_had_it(self):
        model = tiny(PixelsWithOptions).eval()
        self.assertNotIn("camera", model.PARTS)
        layers, acts = window(1, 3, 4), torch.randint(0, 256, (1, 3))
        dreamer = model.Dreamer(model, layers, acts, generator=torch.Generator().manual_seed(0))
        for a in range(4):
            self.assertTrue(torch.equal(dreamer.step(torch.tensor([a]))["camera"], layers["camera"][:, -1]))


if __name__ == "__main__":
    unittest.main()
