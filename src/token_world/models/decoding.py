"""Inference settings for the pixel model's dreams, chosen by the non-training tests of 2026-10-30
(output/tests_2026-10-30/results.md). The defaults are the browser's decoder (models/dynamics.py FrameDecoder and
Dreamer: one pass, nothing hidden; every dream's history a ring, never re-encoded: dynamics.TemporalCache).

- steps / order: a frame's patches committed over `steps` passes (MaskGIT's cosine schedule, after pass s
  1 - cos(pi/2 s/S) of them): each pass predicts the patches not yet committed, given the committed ones (their
  soft pixels; the rest hidden, as in training). "halton" commits them in a fixed 2D Halton order, spread over the
  frame (Besnier et al. 2503.17076); "confident" the most confident first. Test 6: 4 Halton passes cut lost and
  morphed pieces 3-4x at 1.85x the time a frame; confident-first gave nothing.
- past: a frame's patches whose mean top probability is under this go into the history hidden (a hole where the
  model was unsure, as training's hidden context patches). Test 6: < 0.99 helped a little (and beat hiding as
  many at random, random_past: the test's control); test 5b: the model's warnings live above 0.99.

OptionDreamer also keeps the newest frame's features (`newest`: the frame as the model reads it, for a policy
head, models/policy_head.py).
"""
from __future__ import annotations

import math

import torch

from token_world.models.dynamics import (GHOST, MOTION_UNKNOWN, Dreamer, FrameDecoder, compose_probs, motion_cells,
                                         motion_tokens, patch_pixels, unpatch_pixels, weigh_changes)


def halton_rank(grid: int) -> torch.Tensor:
    """[grid * grid]: each patch's place in a 2D Halton order (bases 2 and 3) over the grid, 0 first."""
    def radical(i: int, base: int) -> float:
        out, f = 0.0, 1.0 / base
        while i:
            out += f * (i % base)
            i //= base
            f /= base
        return out
    order, seen, i = [], set(), 1
    while len(order) < grid * grid:
        cell = int(radical(i, 2) * grid) * grid + int(radical(i, 3) * grid)
        if cell not in seen:
            seen.add(cell)
            order.append(cell)
        i += 1
    rank = torch.empty(grid * grid, dtype=torch.long)
    rank[torch.tensor(order)] = torch.arange(grid * grid)
    return rank


class OptionDecoder(FrameDecoder):
    """FrameDecoder with pixel steps and a hidden past (module docstring); dreams only (not scored, no resync).
    Its output adds "features" (the frame as written into the history [B, N, dim]) and "past" (its patches that
    went in hidden [B, N]); prefill takes the context's hidden patches and keeps the last context frame's
    features (prefilled). With a latent, each frame's choice is drawn as FrameDecoder draws it (draw_choice)
    and goes into every pass of the frame; the output adds "z"."""

    def __init__(self, model, batch, device, autocast, steps: int = 1, order: str = "halton",
                 past: float | None = None, random_past: bool = False):
        super().__init__(model, batch, device, autocast)
        if order not in ("halton", "confident"):
            raise ValueError(f"order {order!r}")
        self.steps, self.order, self.past, self.random_past = steps, order, past, random_past
        self.halton = halton_rank(model.grid).to(device)
        self.prefilled = None

    def prefill(self, frames, actions, motion=None, hidden=None, z=None):
        b, t = frames.shape[:2]
        if self.last is not None and not frames.is_floating_point():
            self.last.copy_(frames[:, -1, 0] if frames.dim() == 5 else frames[:, -1])
            if self.sprite_last is not None and frames.dim() == 5:
                self.sprite_last.copy_(frames[:, -1, 1])
        mask = hidden if hidden is not None else torch.zeros(b, t, self.model.grid ** 2, dtype=torch.bool,
                                                             device=frames.device)
        with torch.no_grad():
            tokens = motion_tokens(motion, self.model.patch) if motion is not None else None
            z = self.start_choices(frames, z)
            features = self.model(frames, actions, mask, cache=self.cache, motion=tokens, z=z)
            self.start_history(features, actions)
            self.prefilled = features[:, -1]
        return z

    def _probs(self, features, weight):
        model = self.model
        probs = model.logits(features).float().softmax(-1)                      # [B, N, P*P, COLOURS]
        if self.last is not None:
            probs = weigh_changes(probs, patch_pixels(self.last, model.patch), weight)
        sprite = None
        if model.two_layers:
            sprite = model.sprite_logits(features).float().softmax(-1)
            if self.sprite_last is not None:
                sprite = weigh_changes(sprite, patch_pixels(self.sprite_last, model.patch), weight)
        return probs, sprite

    def _soft(self, probs, sprite):
        soft = self.model.embed_probs(probs)
        if sprite is not None:
            soft = soft + self.model.embed_probs(sprite, sprite=True)
        return soft

    def _decode(self) -> dict:
        model, b, n = self.model, self.batch, self.model.grid ** 2
        enabled, dtype = self.autocast
        device = self.act.device
        with torch.autocast(device.type, dtype=dtype, enabled=enabled, cache_enabled=False):
            blank = torch.zeros(b, 1, 256, 256, dtype=torch.uint8, device=device)
            hidden = torch.ones(b, 1, n, dtype=torch.bool, device=device)
            z = self.draw_choice()                       # with a latent: this frame's choice, in every pass
            features = model(blank, self.act, hidden, self.cache, self.at, z=z)[:, 0]
            moved = model.drawn_motion(features) if model.motion else None
            if model.motion_first:
                if model.motion_steps == 2:
                    committed_m = model.committed_motion(features)
                    second = model(blank, self.act, hidden, self.cache, self.at, z=z,
                                   motion=committed_m[:, None])[:, 0]
                    moved = torch.where(committed_m != MOTION_UNKNOWN, committed_m, model.drawn_motion(second))
                features = model(blank, self.act, hidden, self.cache, self.at, z=z, motion=moved[:, None])[:, 0]
            motion = moved[:, None] if moved is not None else None
            weight = self.change_weight if self.undo is None else self.change_weight * self.undo
            probs, sprite = self._probs(features, weight)
            committed = torch.zeros(b, n, dtype=torch.bool, device=device)
            for s in range(1, self.steps):
                shown = compose_probs(probs, sprite) if sprite is not None else probs
                conf = shown.amax(-1).mean(-1)                                     # [B, N]
                k = round(n * (1 - math.cos(math.pi / 2 * s / self.steps)))
                score = conf if self.order == "confident" else -self.halton.float().expand(b, n)
                score = torch.where(committed, torch.full_like(score, float("inf")), score)
                committed = torch.zeros_like(committed).scatter_(1, score.topk(k, -1).indices, True)
                tokens = model._patches(self._soft(probs, sprite)[:, None])
                again = model(tokens, self.act, ~committed[:, None], self.cache, self.at, z=z, motion=motion)[:, 0]
                p2, s2 = self._probs(again, weight)
                keep = committed[..., None, None]
                probs = torch.where(keep, probs, p2)
                sprite = torch.where(keep, sprite, s2) if sprite is not None else None
            if self.last is not None:
                self.last.copy_(unpatch_pixels(probs.argmax(-1), model.patch))
            if self.sprite_last is not None:
                self.sprite_last.copy_(unpatch_pixels(sprite.argmax(-1), model.patch))
            soft = self._soft(probs, sprite)
            if sprite is not None:
                probs = compose_probs(probs, sprite)
            tokens = model._patches(soft[:, None])
            past = torch.zeros_like(hidden)
            if self.past is not None:
                unsure = probs.amax(-1).mean(-1) < self.past                              # [B, N]
                if self.random_past:                                          # as many, at random places
                    count = unsure.sum(-1, keepdim=True)
                    rank = torch.rand(b, n, device=device).argsort(-1).argsort(-1)
                    unsure = rank < count
                past = unsure[:, None]
            written = model(tokens, self.act, past, self.cache, self.at, z=z, motion=motion)
            self.remember(written)
            out = {"soft": soft, "tokens": tokens[:, 0], "unsure": (probs.amax(-1) < GHOST).sum((1, 2)).float(),
                   "probs": unpatch_pixels(probs.permute(0, 3, 1, 2), model.patch).permute(0, 2, 3, 1),
                   "features": written[:, 0], "past": past[:, 0]}
            if moved is not None:
                out["motion"] = motion_cells(moved, model.patch)
            if z is not None:
                out["z"] = z[:, 0]
        return out


class OptionDreamer(Dreamer):
    """A Dreamer with the settings of the module docstring (OptionDecoder's steps, order, past). newest: the
    newest frame's features [B, N, dim] (the last start frame's before the first step)."""

    def __init__(self, model, frames, actions, change_weight: float = 1.0, motion: torch.Tensor | None = None,
                 steps: int = 1, order: str = "halton", past: float | None = None, random_past: bool = False):
        self.options = dict(steps=steps, order=order, past=past, random_past=random_past)
        super().__init__(model, frames, actions, change_weight=change_weight, motion=motion)
        self.newest = self.decoder.prefilled

    def new_decoder(self, batch, device, autocast):
        return OptionDecoder(self.model, batch, device, autocast, **self.options)

    @torch.no_grad()
    def step(self, action) -> torch.Tensor:
        out = self.advance(action)
        self.newest = out["features"]
        return out["probs"] if self.batched else out["probs"][0]
