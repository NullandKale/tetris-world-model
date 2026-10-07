"""The layered model whose sprites are a picture (models/layered.py for what the layered models share).

The sprites are the screen's sprite layer (data/nes_layers.py sprite_layer: the sprite pixels that show,
TRANSPARENT elsewhere) in SPRITE_TOKENS 16 x 16 patches, predicted by their own pixel head like the
background's cells. Everything on screen is pixels, as in the pixel model (models/dynamics.py): a falling
piece is pixels in the sprite picture, and a dream feeds its frames back soft (each pixel's colour
probabilities). share_pixels (an option): the sprite picture's patches go in and out through the cells'
patch and pixel head (both are 16 x 16 pixels of the same colours; the token kind tells them apart): 0.79M
fewer parameters.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from token_world.data.nes_layers import BANDS, CELL
from token_world.models.dynamics import HeadCrossEntropy
from token_world.models.layered import (BORDER_TOKENS, CELLS, COLUMNS, FRAME_TOKENS, LayeredBase, LayeredDreamer,
                                        patches, unpatch)

SPRITE_TOKENS = BANDS * COLUMNS                         # 240: the sprite picture's 16 x 16 patches
TOKENS = CELLS + SPRITE_TOKENS + BANDS + FRAME_TOKENS + BORDER_TOKENS     # 543


class PixelDreamer(LayeredDreamer):
    def decide(self, frame, last, act, at):
        """One pass with the frame's content hidden; the sprite picture is each pixel's most likely colour,
        fed back soft."""
        m = self.model
        b = act.shape[0]
        hidden = torch.ones(b, 1, m.tokens, dtype=torch.bool, device=self.device)
        hidden[:, :, m.camera_at:m.frame_at] = False
        f = m({k: v[:, None] for k, v in frame.items()}, act, hidden, self.cache, at)[:, 0]
        probs = m.pixel_logits(f[:, m.sprite_at:m.camera_at], sprite=True).float().softmax(-1)   # [B, 240, 256, C]
        frame["sprite_layer"] = unpatch(probs.argmax(-1).view(b, SPRITE_TOKENS, CELL, CELL)).to(
            last["sprite_layer"].dtype)
        return f, m.embed_probs(probs).view(b, SPRITE_TOKENS, CELL, CELL, -1), probs


class PixelLayers(LayeredBase):
    SPRITE_TOKENS = SPRITE_TOKENS
    KEYS = ("cells", "cell_known", "cell_row", "cell_col", "camera", "sprite_layer", "border", "backdrop")
    PARTS = ("cells", "sprites", "backdrop", "camera")
    Dreamer = PixelDreamer

    def __init__(self, dim: int = 96, layers: int = 8, heads: int = 4, frames: int = 64, colour_dim: int = 16,
                 recompute: bool = False, share_pixels: bool = False, **options):
        super().__init__(dim, layers, heads, frames, colour_dim, recompute, **options)
        self.share_pixels = share_pixels
        # registration order is the optimizer state's order: a resumed run's must not change
        if not share_pixels:
            self.sprite_patch = nn.Conv2d(colour_dim, dim, CELL, stride=CELL)
        self.sprite_pos = nn.Embedding(SPRITE_TOKENS, dim)
        if not share_pixels:
            self.sprite_pixels = nn.Linear(dim, CELL * CELL * colour_dim)

    def sprite_kind(self):
        return (self.cell_patch if self.share_pixels else self.sprite_patch), 2

    def sprite_head(self) -> nn.Linear:
        return self.cell_pixels if self.share_pixels else self.sprite_pixels

    def sprite_screen(self, device) -> torch.Tensor:
        """The picture's patches: row-major, 16 across (y, x in 16-pixel units)."""
        i = torch.arange(SPRITE_TOKENS, device=device)
        return torch.stack((i // COLUMNS, i % COLUMNS), -1).float()

    def sprite_index(self, layers: dict) -> torch.Tensor:
        return patches(layers["sprite_layer"].long())

    def sprite_place(self, layers: dict) -> torch.Tensor:
        return self.sprite_pos.weight

    def sprite_loss(self, s, s_mask, layers, seen, chunk):
        """The pixels of masked sprite-picture tokens ("sprites")."""
        if not s_mask.any():
            return {"sprites": s.sum() * 0}
        target = patches(layers["sprite_layer"].long()).flatten(-2)                    # [B, T, 240, 256]
        head = self.sprite_head()
        loss, _ = HeadCrossEntropy.apply(s[s_mask], head.weight, head.bias,
                                         self.colour_table(), self.colour_bias, target[s_mask], chunk)
        return {"sprites": loss}
