"""The layered model whose sprites are the PPU's 64 OAM slots (models/layered.py for what the layered models
share).

A sprite slot's token holds its 8 x 8 pixels, its screen x and y and its flags (off, in front, behind).
Masked slots predict their flags, their movement since the last frame in x and y (MOVE_CLASSES: -MOVE..MOVE
pixels, or JUMP, then the absolute position from the place heads) and their pixels. Movement and presence
are categories, so a dream commits to one value of each (snapping): the sprites are decided over `steps`
passes, the surest first, so a piece's cells can agree on one movement. The idea of 2026-10-06, first run
without a working clock (its time was one learned embedding per window slot) and retried with rotary time
(docs/guides/recipe_attempts.md).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from token_world.data.nes_layers import BANDS, LAYER_COLOURS, LINES, SPRITE, SPRITES, TRANSPARENT, WIDTH
from token_world.models.dynamics import HeadCrossEntropy
from token_world.models.layered import (BORDER_TOKENS, CELLS, FRAME_TOKENS, JUMP, MOVE, MOVE_CLASSES, LayeredBase,
                                        LayeredDreamer, choose, mean_ce, move_class)

TOKENS = CELLS + SPRITES + BANDS + FRAME_TOKENS + BORDER_TOKENS          # 367
FLAG_CLASSES = 4                                        # sprite_flags: 0 off, 1 in front, 3 behind


class SlotDreamer(LayeredDreamer):
    def decide(self, frame, last, act, at):
        """The sprites decided over `steps` passes, the surest first (MaskGIT's order): each pass commits the
        slots surely empty and its surest share of the shown sprites, and the next pass sees them."""
        m, temp, g = self.model, self.temperature, self.generator
        b = act.shape[0]
        hidden = torch.ones(b, 1, m.tokens, dtype=torch.bool, device=self.device)
        hidden[:, :, m.camera_at:m.frame_at] = False
        prev_xy, prev_on = last["sprite_xy"].long(), (last["sprite_flags"] & 1).bool()
        decided = torch.zeros(b, SPRITES, dtype=torch.bool, device=self.device)
        flags_d = torch.zeros(b, SPRITES, dtype=torch.long, device=self.device)
        xy_d = torch.zeros(b, SPRITES, 2, dtype=torch.long, device=self.device)
        probs_d = torch.zeros(b, SPRITES, SPRITE * SPRITE, LAYER_COLOURS, device=self.device)
        off = F.one_hot(torch.tensor(TRANSPARENT, device=self.device), LAYER_COLOURS).float()
        limit = torch.tensor([WIDTH - 1, LINES - 1], device=self.device)
        for step in range(self.steps):
            frame["sprite_flags"] = flags_d.to(last["sprite_flags"].dtype)
            frame["sprite_xy"] = xy_d.to(last["sprite_xy"].dtype)
            hidden[:, 0, m.sprite_at:m.camera_at] = ~decided
            soft = None
            if decided.any():
                shown = torch.where(((flags_d & 1).bool() & decided)[..., None, None], probs_d, off)
                soft = {"sprites": m.embed_probs(shown).view(b, 1, SPRITES, SPRITE, SPRITE, -1)}
            f = m({k: v[:, None] for k, v in frame.items()}, act, hidden, self.cache, at, soft)[:, 0]
            s = f[:, m.sprite_at:m.camera_at]
            flag_logits = m.flag_head(s).float()
            flags = choose(flag_logits, temp, g)
            flags = torch.where(flags == 2, torch.ones_like(flags), flags)              # 2 is never a flag
            move_logits = m.move_head(s).view(b, SPRITES, 2, MOVE_CLASSES).float()
            mv = choose(move_logits, temp, g)
            place = torch.stack((choose(m.place_x(s), temp, g), choose(m.place_y(s), temp, g)), -1)
            jump = (mv == JUMP) | ~prev_on[..., None]
            xy = torch.minimum(torch.where(jump, place, prev_xy + mv - MOVE).clamp_min(0), limit)
            probs = m.pixel_logits(s, sprite=True).float().softmax(-1)                 # [B, SPRITES, 64, C]
            sure = flag_logits.log_softmax(-1).gather(-1, flags[..., None])[..., 0] + torch.where(
                (flags & 1).bool(), move_logits.log_softmax(-1).gather(-1, mv[..., None])[..., 0].sum(-1), 0.0)
            take = ~decided
            if step < self.steps - 1:
                empty = take & (flags == 0) & (sure > math.log(0.9))
                shown_now = take & (flags != 0)
                passes = self.steps - step
                k = (shown_now.sum(1, keepdim=True) + passes - 1) // passes                # [B, 1]
                rank = torch.where(shown_now, sure, -math.inf).argsort(1, descending=True).argsort(1)
                take = empty | (shown_now & (rank < k))
            flags_d = torch.where(take, flags, flags_d)
            xy_d = torch.where(take[..., None], xy, xy_d)
            probs_d = torch.where(take[..., None, None], probs, probs_d)
            decided |= take
        on = (flags_d & 1).bool()
        sprite_probs = torch.where(on[..., None, None], probs_d, off)
        frame["sprite_flags"] = flags_d.to(last["sprite_flags"].dtype)
        frame["sprite_xy"] = torch.where(on[..., None], xy_d, 0).to(last["sprite_xy"].dtype)
        frame["sprites"] = sprite_probs.argmax(-1).view(b, SPRITES, SPRITE, SPRITE).to(last["sprites"].dtype)
        return f, m.embed_probs(sprite_probs).view(b, SPRITES, SPRITE, SPRITE, -1), sprite_probs


class SlotLayers(LayeredBase):
    SPRITE_TOKENS = SPRITES
    KEYS = ("cells", "cell_known", "cell_row", "cell_col", "camera", "sprites", "sprite_xy", "sprite_flags",
            "border", "backdrop")
    PARTS = ("cells", "sprite_flags", "sprite_move", "sprite_place", "sprite_pixels", "backdrop", "camera")
    Dreamer = SlotDreamer

    def __init__(self, dim: int = 96, layers: int = 8, heads: int = 4, frames: int = 64, colour_dim: int = 16,
                 recompute: bool = False, **options):
        super().__init__(dim, layers, heads, frames, colour_dim, recompute, **options)
        e = lambda n: nn.Embedding(n, dim)
        self.sprite_patch = nn.Conv2d(colour_dim, dim, SPRITE, stride=SPRITE)
        self.slot, self.sprite_x, self.sprite_y, self.flags = e(SPRITES), e(WIDTH), e(LINES), e(FLAG_CLASSES)
        self.sprite_pixels = nn.Linear(dim, SPRITE * SPRITE * colour_dim)
        self.flag_head = nn.Linear(dim, FLAG_CLASSES)
        self.move_head = nn.Linear(dim, 2 * MOVE_CLASSES)
        self.place_x, self.place_y = nn.Linear(dim, WIDTH), nn.Linear(dim, LINES)

    def sprite_kind(self):
        return self.sprite_patch, 2

    def sprite_index(self, layers: dict) -> torch.Tensor:
        return layers["sprites"].long()

    def sprite_extra(self, layers: dict) -> torch.Tensor:
        xy = layers["sprite_xy"].long()
        return self.sprite_x(xy[..., 0]) + self.sprite_y(xy[..., 1]) + self.flags(layers["sprite_flags"].long())

    def sprite_place(self, layers: dict) -> torch.Tensor:
        return self.slot.weight

    def sprite_loss(self, s, s_mask, layers, seen, chunk):
        """Masked slots' flags, movement (from the position the history gave them: a rollout's own frames
        lead back to the real one), places and pixels."""
        out = {}
        flags = layers["sprite_flags"].long()
        on = (flags & 1).bool()
        out["sprite_flags"] = mean_ce(self.flag_head(s), flags, s_mask)
        xy = layers["sprite_xy"].long()
        shown = s_mask & on
        moves = self.move_head(s).unflatten(-1, (2, MOVE_CLASSES))                      # [B, T, S, 2, C]
        seen_xy, seen_on = seen["sprite_xy"].long(), (seen["sprite_flags"].long() & 1).bool()
        before_on = torch.cat((torch.zeros_like(seen_on[:, :1]), seen_on[:, :-1]), 1)
        before = torch.cat((seen_xy[:, :1], seen_xy[:, :-1]), 1)
        target = torch.where(before_on[..., None], move_class(xy, before), JUMP)     # [B, T, S, 2]
        out["sprite_move"] = mean_ce(moves[..., 0, :], target[..., 0], shown) + \
            mean_ce(moves[..., 1, :], target[..., 1], shown)
        out["sprite_place"] = mean_ce(self.place_x(s), xy[..., 0], shown) + mean_ce(self.place_y(s), xy[..., 1], shown)
        if shown.any():
            out["sprite_pixels"], _ = HeadCrossEntropy.apply(s[shown], self.sprite_pixels.weight,
                                                             self.sprite_pixels.bias, self.colour_table(),
                                                             self.colour_bias,
                                                             layers["sprites"].long().flatten(-2)[shown], chunk)
        else:
            out["sprite_pixels"] = s.sum() * 0
        return out
