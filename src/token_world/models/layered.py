"""World models over layered NES frames (data/nes_layers.py, docs/guides/layered_tokens.md): what they share.

A frame is tokens of five kinds, in this order: CELLS background cells (the world-grid cells in view, BANDS
x VIEW), the sprites (each model's own: models/layered_pixels.py a screen-space sprite picture in 16 x 16
patches, models/layered_slots.py the 64 OAM slots), BANDS camera rows, one frame token (the backdrop colour)
and BORDER_TOKENS border tokens. The spatiotemporal transformer of models/dynamics.py runs over them, with
two changes:

- temporal attention follows each token's identity, not its index: a background cell attends to the same
  world cell in earlier frames (its band and canvas column: it lines up across camera moves; a cell new to
  the view has no history); the other tokens to their own index;
- time is rotary (RoPE) inside the temporal attention, as in Open-Oasis: a query and a key score by how many
  frames apart they are, in every layer (with one learned embedding per window slot, as before, the slot
  model never learned the border's frame counter: docs/guides/recipe_attempts.md).

Options (a run's config, off by default: the first layered-pixels run has none):
- space_rope: rotary in the spatial attention too, by screen place (RoPE-Mixed, Heo et al. 2403.13298: each
  channel pair turns along a learned 2D direction, so a score depends on the offset between two tokens,
  diagonals included). Cells, sprite-picture patches and border tokens have a place on screen (in 16-pixel
  units: a scrolled cell's x includes its fine offset); the camera, frame and register tokens, and the slot
  model's sprites (their place is content, hidden when masked), have none: their rotated channels are zero
  (DeepSeek-V2's decoupled RoPE: position in some channels, content in the rest);
- registers: that many learned tokens after the border (ViTs need registers, 2309.16588), with no target and
  never hidden: room for what a frame is about (the next piece) that no pixel holds, carried across frames by
  their own temporal identity;
- fixed_camera: for a game whose camera never moves (Tetris): no camera tokens and no camera head; a dream
  keeps the camera its real frames had (the first layered-pixels run's head learned "no move", and its
  dreams never moved the camera in 1,024 frames). Games that scroll (SMB) leave it off.

Masked tokens (MaskGIT masking, as models/dynamics.py) predict their content: cells' and border tokens'
16 x 16 pixels with the tied pixel head and fused loss of dynamics.py, over the LAYER_COLOURS colours
(TRANSPARENT: nothing drawn in that layer), the frame's backdrop colour, and each model's sprites. The camera
is never masked: the next frame's camera is predicted from this frame's camera tokens and the next frame's
buttons (camera_logits, a movement category per band), so a frame's layout (which world cells are in view)
is fixed before its content is predicted, and nothing in a frame tells its own camera.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from token_world.data.nes_layers import BANDS, CANVAS_COLS, CANVAS_ROWS, CELL, LAYER_COLOURS, VIEW, WIDTH
from token_world.models.dynamics import (Attention, HeadCrossEntropy, button_bits, pixel_logits,
                                         training_mask)

CELLS = BANDS * VIEW                                    # 255
COLUMNS = WIDTH // CELL                                 # 16 tokens across the screen
BORDER_TOKENS = 2 * COLUMNS                             # 32: the two border bands' 16 x 16 tokens
FRAME_TOKENS = 1
CELL_SLOTS = BANDS * CANVAS_COLS                        # cells' temporal identities: (band, canvas column)
UNKNOWN = LAYER_COLOURS                                 # an input pixel never seen (cells scrolling in)
ROPE_BASE = 10_000.0                                    # rotary's longest wavelength scale (frames)
SPACE_ROPE_BASE = 10.0                                  # RoPE-Mixed's frequency scale (its code's), places in 16 px
MOVE = 16
MOVE_CLASSES = 2 * MOVE + 2                             # -MOVE..MOVE, then JUMP
JUMP = MOVE_CLASSES - 1
CAMERA_X, CAMERA_Y = CANVAS_COLS * CELL, CANVAS_ROWS * CELL


def patches(picture: torch.Tensor) -> torch.Tensor:
    """Pictures [..., rows * 16, 256] -> their 16 x 16 patches, row-major [..., rows * 16, 16, 16]."""
    lead, rows = picture.shape[:-2], picture.shape[-2] // CELL
    return picture.reshape(*lead, rows, CELL, COLUMNS, CELL).transpose(-3, -2).reshape(
        *lead, rows * COLUMNS, CELL, CELL)


def unpatch(tokens: torch.Tensor) -> torch.Tensor:
    """patches' inverse: [..., rows * 16, 16, 16] -> [..., rows * 16, 256]."""
    lead, rows = tokens.shape[:-3], tokens.shape[-3] // COLUMNS
    return tokens.reshape(*lead, rows, COLUMNS, CELL, CELL).transpose(-3, -2).reshape(*lead, rows * CELL, WIDTH)


def move_class(now: torch.Tensor, before: torch.Tensor, wrap: int | None = None) -> torch.Tensor:
    """Positions now and the frame before -> the movement class (JUMP when it is more than MOVE)."""
    d = now.long() - before.long()
    if wrap:
        d = (d + wrap // 2) % wrap - wrap // 2
    return torch.where(d.abs() <= MOVE, d + MOVE, torch.full_like(d, JUMP))


def slots_of(layers: dict, tokens: int) -> torch.Tensor:
    """Layered frames (cell_col [..., BANDS]) -> each of a frame's `tokens` tokens' temporal identity [...,
    tokens]: cells by (band, canvas column), the rest by index after them."""
    col = layers["cell_col"].long()
    lead = col.shape[:-1]
    band = torch.arange(BANDS, device=col.device)
    cells = (band[:, None] * CANVAS_COLS + (col[..., None] + torch.arange(VIEW, device=col.device)) % CANVAS_COLS)
    rest = CELL_SLOTS + torch.arange(tokens - CELLS, device=col.device)
    return torch.cat((cells.flatten(-2), rest.expand(*lead, -1)), -1)


class SlotCache:
    """One layer's temporal keys and values by identity: [B * slots, heads, frames, head_dim], with which
    identities each frame had [B, slots, frames] (a dream's frames are written one at a time)."""

    def __init__(self, frames: int, slots: int):
        self.frames, self.slots, self.k, self.v, self.present = frames, slots, None, None, None

    def write(self, k: torch.Tensor, v: torch.Tensor, slots: torch.Tensor, at: int):
        """k, v [B, T, N, heads, hd] for frames at..at+T-1, slots [B, T, N] -> nothing (stored)."""
        b, t, n, h, d = k.shape
        if self.k is None:
            self.k = k.new_zeros(b * self.slots, h, self.frames, d)
            self.v = torch.zeros_like(self.k)
            self.present = torch.zeros(b, self.slots, self.frames, dtype=torch.bool, device=k.device)
        rows = (torch.arange(b, device=k.device)[:, None, None] * self.slots + slots).flatten()     # [B*T*N]
        times = (at + torch.arange(t, device=k.device))[None, :, None].expand(b, t, n).flatten()
        self.k[rows, :, times] = k.reshape(-1, h, d)
        self.v[rows, :, times] = v.reshape(-1, h, d)
        self.present.view(-1, self.frames)[rows, times] = True

    def read(self, slots: torch.Tensor, at: int):
        """slots [B, N] of the frame at `at` -> its tokens' keys, values [B * N, heads, frames, hd] and the
        attention mask [B * N, 1, 1, frames]: the identity's frames up to `at`."""
        b, n = slots.shape
        rows = (torch.arange(b, device=slots.device)[:, None] * self.slots + slots).flatten()
        seen = self.present.view(-1, self.frames)[rows] & (torch.arange(self.frames, device=slots.device) <= at)
        return self.k[rows], self.v[rows], seen[:, None, None]


def rotary(x: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    """RoPE over time (Su et al., 2104.09864; Open-Oasis's temporal attention): x [B, T, N, h, hd] at frames
    `times` [T] -> x with each pair of channels rotated by an angle proportional to its frame, so a query and
    a key score by how many frames apart they are, whatever window slot they sit in."""
    half = x.shape[-1] // 2
    freq = ROPE_BASE ** -(torch.arange(half, device=x.device, dtype=torch.float32) / half)
    angle = times.to(torch.float32)[:, None] * freq                                   # [T, hd / 2]
    cos, sin = angle.cos()[None, :, None, None].to(x.dtype), angle.sin()[None, :, None, None].to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), -1)


def mixed_rotary(x: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """x [..., h, hd] whose first 2P channels are rotated as P pairs (i, P + i) by angle [..., h, P]; the rest
    pass unchanged."""
    p = angle.shape[-1]
    cos, sin = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
    x1, x2, rest = x[..., :p], x[..., p:2 * p], x[..., 2 * p:]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos, rest), -1)


class SpatialAttention(Attention):
    """Attention within a frame; with rope (the space_rope option), half its channels rotated by each token's
    place on screen along learned 2D directions (RoPE-Mixed: per layer and head, initialised as its code does,
    two perpendicular directions at a random angle per head, magnitudes SPACE_ROPE_BASE ** -(i / P))."""

    def __init__(self, dim: int, heads: int, rope: bool = False):
        super().__init__(dim, heads)
        self.rope = rope
        if rope:
            pairs = dim // heads // 4                                                  # half the channels
            mag = SPACE_ROPE_BASE ** -(torch.arange(pairs // 2 + pairs % 2).float() / (pairs // 2 + pairs % 2))
            turn = torch.rand(heads, 1) * 2 * torch.pi
            angles = torch.cat((turn.expand(-1, len(mag)), turn.expand(-1, len(mag)) + torch.pi / 2), 1)[:, :pairs]
            mags = torch.cat((mag, mag))[:pairs]
            self.rope_freq = nn.Parameter(torch.stack((mags * angles.sin(), mags * angles.cos()), -1))  # [h, P, (y, x)]

    def forward(self, x: torch.Tensor, place: torch.Tensor | None = None, placed: torch.Tensor | None = None):
        """x [M, N, d]; place [M, N, 2] (y, x in 16-pixel units) and placed [N] (rope only)."""
        if not self.rope:
            return super().forward(x, causal=False)
        m, n, d = x.shape
        h = self.heads
        q, k, v = self.qkv(x).view(m, n, 3, h, d // h).unbind(2)                        # [M, N, h, hd]
        angle = torch.einsum("mnc,hpc->mnhp", place.float(), self.rope_freq.float())
        rotated = 2 * self.rope_freq.shape[1]
        keep = torch.ones(n, 1, d // h, device=x.device, dtype=x.dtype)
        keep[~placed, :, :rotated] = 0                                                  # no place: content only
        q, k = mixed_rotary(q, angle) * keep, mixed_rotary(k, angle) * keep
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.out(y.transpose(1, 2).reshape(m, n, d))


class RoutedTemporal(Attention):
    """Causal temporal attention per identity: each token attends to the earlier frames' tokens that share
    its identity (and to itself), with queries and keys rotated by their frame (rotary): the model's only
    sense of time, relative, in every layer."""

    def __init__(self, dim: int, heads: int, slots: int):
        super().__init__(dim, heads)
        self.slots = slots

    def forward(self, x: torch.Tensor, slots: torch.Tensor, cache: SlotCache | None = None, at: int = 0):
        """x [B, T, N, d] for frames at..at+T-1, slots [B, T, N] -> [B, T, N, d]. With a cache, the frames
        before `at` come from it (a dream: T = 1)."""
        b, t, n, d = x.shape
        h = self.heads
        q, k, v = self.qkv(x).view(b, t, n, 3, h, d // h).unbind(3)                   # [B, T, N, h, hd]
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        times = at + torch.arange(t, device=x.device)
        q, k = rotary(q, times), rotary(k, times)
        if cache is not None:
            cache.write(k, v, slots, at)
            outs = []
            for i in range(t):
                keys, values, seen = cache.read(slots[:, i], at + i)
                y = F.scaled_dot_product_attention(q[:, i].reshape(b * n, h, 1, d // h), keys, values,
                                                   attn_mask=seen)
                outs.append(y.view(b, n, d))
            return self.out(torch.stack(outs, 1))
        # scatter into identity space [B, slots, T], attend per identity, gather back
        s = self.slots
        index = slots.permute(0, 2, 1)                                                # [B, N, T]

        def grid(z):
            g = z.new_zeros(b, s, t, h, d // h)
            g.scatter_(1, index[..., None, None].expand(-1, -1, -1, h, d // h), z.permute(0, 2, 1, 3, 4))
            return g.permute(0, 1, 3, 2, 4).reshape(b * s, h, t, d // h)
        present = torch.zeros(b, s, t, dtype=torch.bool, device=x.device)
        present.scatter_(1, index, True)
        causal = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        mask = (causal & present.view(b * s, 1, 1, t)) | torch.eye(t, dtype=torch.bool, device=x.device)
        y = F.scaled_dot_product_attention(grid(q), grid(k), grid(v), attn_mask=mask)  # [B*S, h, T, hd]
        y = y.view(b, s, h, t, d // h).permute(0, 1, 3, 2, 4).reshape(b, s, t, d)
        out = y.gather(1, index[..., None].expand(-1, -1, -1, d))                     # [B, N, T, d]
        return self.out(out.transpose(1, 2))


class LayeredBlock(nn.Module):
    """Spatial attention within a frame, temporal attention per identity, MLP."""

    def __init__(self, dim: int, heads: int, slots: int, space_rope: bool = False):
        super().__init__()
        self.norm_s, self.norm_t, self.norm_m = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.spatial, self.temporal = SpatialAttention(dim, heads, space_rope), RoutedTemporal(dim, heads, slots)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x, slots, cache=None, at=0, place=None, placed=None):
        b, t, n, d = x.shape
        place = None if place is None else place.reshape(b * t, n, 2)
        x = x + self.spatial(self.norm_s(x).reshape(b * t, n, d), place, placed).view(b, t, n, d)
        x = x + self.temporal(self.norm_t(x), slots, cache, at)
        return x + self.mlp(self.norm_m(x))


def mean_ce(logits: torch.Tensor, target: torch.Tensor, where: torch.Tensor) -> torch.Tensor:
    if not where.any():
        return logits.sum() * 0
    return F.cross_entropy(logits[where].float(), target[where].long())


def choose(logits: torch.Tensor, temperature: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """[..., C] logits -> [...] a class: sampled at `temperature`, the most likely at 0 (Gumbel-max)."""
    if temperature <= 0:
        return logits.argmax(-1)
    u = torch.rand(logits.shape, generator=generator, device=logits.device).clamp(1e-9, 1 - 1e-9)
    return (logits.float() / temperature - (-u.log()).log()).argmax(-1)


class LayeredBase(nn.Module):
    """What the layered models share. A model adds its sprites (SPRITE_TOKENS tokens after the cells):
    KEYS (the frame fields it reads and writes), Dreamer (its dream decoder), PARTS (its loss names) and the
    hooks sprite_kind, sprite_extra, sprite_place, sprite_index, sprite_loss."""
    SPRITE_TOKENS: int
    KEYS: tuple
    PARTS: tuple
    Dreamer: type

    def __init__(self, dim: int = 96, layers: int = 8, heads: int = 4, frames: int = 64, colour_dim: int = 16,
                 recompute: bool = False, registers: int = 0, space_rope: bool = False, fixed_camera: bool = False):
        super().__init__()
        self.frames, self.dim, self.space_rope, self.fixed_camera = frames, dim, space_rope, fixed_camera
        self.recompute = recompute     # training keeps only each block's input and recomputes the rest: memory for time
        self.sprite_at, self.camera_at = CELLS, CELLS + self.SPRITE_TOKENS
        self.frame_at = self.camera_at + (0 if fixed_camera else BANDS)
        if fixed_camera:
            self.PARTS = tuple(p for p in type(self).PARTS if p != "camera")
        self.border_at = self.frame_at + FRAME_TOKENS
        self.border_end = self.register_at = self.border_at + BORDER_TOKENS
        self.tokens = self.register_at + registers
        self.slots = CELL_SLOTS + self.tokens - CELLS
        if registers:
            self.register = nn.Parameter(torch.randn(registers, dim) * 0.02)
        self.colour = nn.Embedding(LAYER_COLOURS + 1, colour_dim)               # + UNKNOWN (inputs only)
        self.cell_patch = nn.Conv2d(colour_dim, dim, CELL, stride=CELL)          # cells and border tokens
        self.mask_token = nn.Parameter(torch.randn(dim) * 0.02)
        self.kind = nn.Parameter(torch.randn(5, dim) * 0.02)                     # cell, sprite, camera, frame, border
        e = lambda n: nn.Embedding(n, dim)
        self.band, self.view, self.canvas_col, self.canvas_row, self.fine = e(BANDS), e(VIEW), e(CANVAS_COLS), \
            e(CANVAS_ROWS), e(CELL)
        if not fixed_camera:
            self.camera_band, self.camera_x, self.camera_y = e(BANDS), e(CAMERA_X), e(CAMERA_Y)
        self.backdrop, self.border_pos = e(LAYER_COLOURS), e(BORDER_TOKENS)
        self.action = nn.Linear(9, dim)
        self.blocks = nn.ModuleList(LayeredBlock(dim, heads, self.slots, space_rope) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)
        self.cell_pixels = nn.Linear(dim, CELL * CELL * colour_dim)               # tied to the colour table
        self.colour_bias = nn.Parameter(torch.zeros(LAYER_COLOURS))
        self.backdrop_head = nn.Linear(dim, LAYER_COLOURS)
        if not fixed_camera:
            self.camera_action = nn.Linear(9, dim)
            self.camera_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(),
                                             nn.Linear(dim, 2 * MOVE_CLASSES + CAMERA_X + CAMERA_Y))

    # ---- the sprites: each model's own -------------------------------------------------------------
    def sprite_kind(self) -> tuple[nn.Conv2d, int]:
        """-> (the sprites' patch conv, the leading dims of a frame's sprite pixels before the patch)."""
        raise NotImplementedError

    def sprite_index(self, layers: dict) -> torch.Tensor:
        """Frames -> the sprites' pixels as colour indices [..., SPRITE_TOKENS, P, P]."""
        raise NotImplementedError

    def sprite_extra(self, layers: dict) -> torch.Tensor | float:
        """Frames -> what the sprites' content adds beyond pixels [B, T, SPRITE_TOKENS, dim] (hidden when masked)."""
        return 0.0

    def sprite_place(self, layers: dict) -> torch.Tensor:
        """Frames -> the sprite tokens' identity and place [B, T, SPRITE_TOKENS, dim] (never hidden)."""
        raise NotImplementedError

    def sprite_screen(self, device) -> torch.Tensor | None:
        """The sprite tokens' places on screen [SPRITE_TOKENS, 2] (y, x in 16-pixel units) for space_rope, or
        None when their place is content (hidden when masked)."""
        return None

    def sprite_head(self) -> nn.Linear:
        """The sprites' pixel head."""
        return self.sprite_pixels

    def sprite_loss(self, s: torch.Tensor, s_mask: torch.Tensor, layers: dict, seen: dict, chunk: int) -> dict:
        """The sprite tokens' features [B, T, SPRITE_TOKENS, dim] and mask -> {name: mean loss}."""
        raise NotImplementedError

    # ---- inputs -------------------------------------------------------------------------------------
    def colour_table(self) -> torch.Tensor:
        """[LAYER_COLOURS, colour_dim]: the output colours' vectors (tied, scaled as dynamics.py's)."""
        return self.colour.weight[:LAYER_COLOURS] * self.colour.embedding_dim ** -0.5

    def _pixels(self, pixels: torch.Tensor, conv: nn.Conv2d) -> torch.Tensor:
        """[M, P, P] colour indices, or [M, P, P, colour_dim] soft pixels -> [M, dim]."""
        if not pixels.is_floating_point():
            pixels = self.colour(pixels.long())
        return conv(pixels.permute(0, 3, 1, 2).to(conv.weight.dtype)).flatten(1)

    def real_soft(self, frame: dict, key: str) -> torch.Tensor:
        """Real frames' pixels as soft pixels, their colours' embeddings [..., P, P, colour_dim], for inputs
        that mix real and generated frames. key: cells, sprites or border (border as patches)."""
        if key == "cells":
            index = torch.where(frame["cell_known"].bool(), frame["cells"].long(), UNKNOWN)
        elif key == "sprites":
            index = self.sprite_index(frame)
        else:
            index = patches(frame["border"].long())
        return self.colour(index)

    def pixel_tokens(self, layers: dict, soft: dict | None = None) -> dict:
        """Layered frames (each key [B, T, ...]) -> each kind's pixel content as tokens: cells [B, T, CELLS,
        dim], sprites [B, T, SPRITE_TOKENS, dim], border [B, T, BORDER_TOKENS, dim]. soft: per kind, either
        soft pixels for every frame (cells [B, T, BANDS, VIEW, 16, 16, colour_dim], sprites and border as
        their patches, colour_dim last: a dream's frames), or tokens [B, T, n, dim] used in the frames where
        soft["where"] [B, T] is set (a rollout's own frames, kept as the drawer's tokens as
        models/dynamics.py's rollouts keep theirs)."""
        b, t = layers["camera"].shape[:2]
        soft = soft or {}
        sprite_conv, sprite_lead = self.sprite_kind()
        hard = {"cells": lambda: torch.where(layers["cell_known"].bool(), layers["cells"].long(), UNKNOWN),
                "sprites": lambda: self.sprite_index(layers), "border": lambda: patches(layers["border"].long())}
        shapes = {"cells": (self.cell_patch, CELLS, 3), "sprites": (sprite_conv, self.SPRITE_TOKENS, sprite_lead),
                  "border": (self.cell_patch, BORDER_TOKENS, 2)}
        out = {}
        for key, (conv, n, lead) in shapes.items():
            given = soft.get(key)
            if given is not None and "where" not in soft:
                out[key] = self._pixels(given.flatten(0, lead), conv).view(b, t, n, -1)
                continue
            x = self._pixels(hard[key]().flatten(0, lead), conv).view(b, t, n, -1)
            if given is not None:
                x = torch.where(soft["where"][..., None, None], given.to(x.dtype), x)
            out[key] = x
        return out

    def content(self, layers: dict, soft: dict | None = None) -> torch.Tensor:
        """Layered frames (each key [B, T, ...], data/nes_layers.py) -> what each token holds, before any is
        hidden [B, T, border_end, dim]: the pixel tokens (soft: pixel_tokens), the backdrop colour; the camera
        tokens hold nothing (the camera is their place)."""
        x = self.pixel_tokens(layers, soft)
        x_frame = self.backdrop(layers["backdrop"].long())[:, :, None]
        cameras = self.frame_at - self.camera_at
        return torch.cat((x["cells"], x["sprites"] + self.sprite_extra(layers),
                          torch.zeros_like(x["cells"][:, :, :cameras]), x_frame, x["border"]), 2)

    def place(self, layers: dict) -> torch.Tensor:
        """Layered frames -> each token's identity, place and kind [B, T, border_end, dim] (never hidden)."""
        camera = layers["camera"].long()
        b, t = camera.shape[:2]
        col = (layers["cell_col"].long()[..., None] + torch.arange(VIEW, device=camera.device)) % CANVAS_COLS
        row = ((camera[..., 1] % CAMERA_Y) // CELL)[..., None].expand(-1, -1, -1, VIEW)
        place_cells = (self.band.weight[:, None] + self.view.weight[None] + self.canvas_col(col) +
                       self.canvas_row(row) + self.fine(camera[..., 0] % CELL)[..., None, :]).flatten(2, 3)
        place_camera = (place_cells[:, :, :0] if self.fixed_camera else self.camera_band.weight +
                        self.camera_x(camera[..., 0] % CAMERA_X) + self.camera_y(camera[..., 1] % CAMERA_Y))
        place_frame = place_cells.new_zeros(b, t, FRAME_TOKENS, self.dim)
        place_border = self.border_pos.weight.expand(b, t, -1, -1)
        place = torch.cat((place_cells, self.sprite_place(layers).expand(b, t, -1, -1), place_camera, place_frame,
                           place_border), 2)
        kinds = torch.repeat_interleave(self.kind, torch.tensor([CELLS, self.SPRITE_TOKENS, self.frame_at -
                                                                 self.camera_at, FRAME_TOKENS, BORDER_TOKENS],
                                                                device=camera.device), 0)
        return place + kinds

    def embed(self, layers: dict, actions: torch.Tensor, mask: torch.Tensor, soft: dict | None = None):
        """Layered frames, incoming actions [B, T], mask [B, T, tokens] (content hidden) -> tokens [B, T,
        tokens, dim]: content (or the mask token), place and kind, the buttons, then the registers. soft: soft
        pixels or tokens in place of some frames' pixels (pixel_tokens)."""
        return self.join(self.content(layers, soft), self.place(layers), actions, mask)

    def join(self, content: torch.Tensor, place: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor):
        """content and place [B, T, border_end, dim], actions [B, T], mask [B, T, tokens] -> the tokens."""
        content = torch.where(mask[..., :self.border_end, None], self.mask_token.to(content.dtype), content)
        action = self.action(button_bits(actions))[:, :, None]
        x = content + place.to(content.dtype) + action
        if self.tokens > self.register_at:                                   # registers: never hidden
            x = torch.cat((x, (self.register + action).to(x.dtype)), 2)
        return x

    def screen(self, layers: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Layered frames -> every token's place on screen [B, T, tokens, 2] (y, x in 16-pixel units: a cell's
        x is its view column less the fine scroll; the border bands sit above and below the picture, NES
        lines -8 and 232 as in model frames) and which tokens have one [tokens] (space_rope)."""
        camera = layers["camera"].long()
        b, t = camera.shape[:2]
        device = camera.device
        fine = (camera[..., 0] % CELL).float() / CELL                                  # [B, T, BANDS]
        band = torch.arange(BANDS, device=device).float()
        view = torch.arange(VIEW, device=device).float()
        cells = torch.stack((band[:, None].expand(-1, VIEW).expand(b, t, -1, -1),
                             view[None] - fine[..., None]), -1).flatten(2, 3)            # [B, T, CELLS, 2]
        column = torch.arange(COLUMNS, device=device).float()
        border = torch.stack((torch.cat((torch.full((COLUMNS,), -0.5, device=device),
                                         torch.full((COLUMNS,), 14.5, device=device))), column.repeat(2)), -1)
        sprites = self.sprite_screen(device)
        others = self.border_at - self.camera_at
        place = torch.cat((cells,
                           (sprites if sprites is not None else
                            torch.zeros(self.SPRITE_TOKENS, 2, device=device)).expand(b, t, -1, -1),
                           torch.zeros(b, t, others, 2, device=device), border.expand(b, t, -1, -1),
                           torch.zeros(b, t, self.tokens - self.border_end, 2, device=device)), 2)
        placed = torch.zeros(self.tokens, dtype=torch.bool, device=device)
        placed[:CELLS] = True
        placed[self.sprite_at:self.camera_at] = sprites is not None
        placed[self.border_at:self.border_end] = True
        return place, placed

    def forward(self, layers: dict, actions: torch.Tensor, mask: torch.Tensor, cache: list | None = None,
                at: int = 0, soft: dict | None = None) -> torch.Tensor:
        """-> features [B, T, tokens, dim]. With a cache (new_cache()), the frames are window frames at..at+T-1
        and the earlier frames come from it."""
        x = self.embed(layers, actions, mask, soft)
        slots = slots_of(layers, self.tokens)
        place, placed = self.screen(layers) if self.space_rope else (None, None)
        remember = self.recompute and self.training and torch.is_grad_enabled() and cache is None
        for i, block in enumerate(self.blocks):
            x = (checkpoint(block, x, slots, None, at, place, placed, use_reentrant=False) if remember else
                 block(x, slots, None if cache is None else cache[i], at, place, placed))
        return self.norm(x)

    def new_cache(self) -> list[SlotCache]:
        return [SlotCache(self.frames, self.slots) for _ in self.blocks]

    def mask(self, batch: int, frames: int, device, generator=None, cut: torch.Tensor | None = None) -> torch.Tensor:
        """dynamics.py's training mask over the frame's tokens, with the camera never hidden (it is an input)."""
        mask = training_mask(batch, frames, self.tokens, device, generator, cut=cut)
        mask[:, :, self.camera_at:self.frame_at] = False
        mask[:, :, self.register_at:] = False                                # registers have no content
        return mask

    # ---- outputs ------------------------------------------------------------------------------------
    def camera_logits(self, features: torch.Tensor, next_actions: torch.Tensor):
        """Camera tokens' features [..., BANDS, dim] and the next frame's incoming actions [...] -> logits of
        the next frame's camera per band: moves x, y [..., BANDS, MOVE_CLASSES] and places x, y."""
        h = self.camera_head(features + self.camera_action(button_bits(next_actions))[..., None, :])
        mx, my, px, py = h.split((MOVE_CLASSES, MOVE_CLASSES, CAMERA_X, CAMERA_Y), -1)
        return mx, my, px, py

    def pixel_logits(self, features: torch.Tensor, sprite: bool = False) -> torch.Tensor:
        """[..., dim] -> [..., P*P, LAYER_COLOURS]: cells and border, or (sprite) the sprites' pixels."""
        head = self.sprite_head() if sprite else self.cell_pixels
        pixels = head(features).unflatten(-1, (-1, self.colour.embedding_dim))
        return pixel_logits(pixels, self.colour_table(), self.colour_bias)

    def embed_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """Colour probabilities [..., LAYER_COLOURS] -> soft pixels [..., colour_dim] (dynamics.py's soft frames)."""
        return probs.to(self.colour.weight.dtype) @ self.colour.weight[:LAYER_COLOURS]

    def loss(self, features: torch.Tensor, layers: dict, actions: torch.Tensor, mask: torch.Tensor,
             chunk: int = 4096, seen: dict | None = None) -> dict:
        """The losses of a window -> {name: mean loss}; "total" has the graph. The pixels of masked cells
        (whose pixels are all known) and border tokens ("cells"), the model's sprite losses, masked frame
        tokens' backdrop, and every frame's camera from the frame before. The targets are `layers` (real);
        seen: the frames as the model was given them, when some are its own (a rollout): movement is scored
        from where the history put a thing to where it really is, so the target leads back to the real game."""
        seen = layers if seen is None else seen
        out = {}
        cells_known = layers["cell_known"].flatten(2, 3).flatten(-2).all(-1)             # [B, T, CELLS]
        cell_where = mask[:, :, :CELLS] & cells_known
        border = patches(layers["border"].long()).flatten(-2)                          # [B, T, 32, 256]
        cells = layers["cells"].long().flatten(2, 3).flatten(-2)                        # [B, T, CELLS, 256]
        bw = mask[:, :, self.border_at:self.border_end]
        h = torch.cat((features[:, :, :CELLS][cell_where], features[:, :, self.border_at:self.border_end][bw]))
        y = torch.cat((cells[cell_where], border[bw]))
        out["cells"], _ = HeadCrossEntropy.apply(h, self.cell_pixels.weight, self.cell_pixels.bias,
                                                 self.colour_table(), self.colour_bias, y, chunk)
        sprites = slice(self.sprite_at, self.camera_at)
        out.update(self.sprite_loss(features[:, :, sprites], mask[:, :, sprites], layers, seen, chunk))
        out["backdrop"] = mean_ce(self.backdrop_head(features[:, :, self.frame_at]), layers["backdrop"],
                                  mask[:, :, self.frame_at])
        if not self.fixed_camera:
            camera, seen_camera = layers["camera"].long(), seen["camera"].long()
            mx, my, px, py = self.camera_logits(features[:, :-1, self.camera_at:self.frame_at], actions[:, 1:])
            tx = move_class(camera[:, 1:, :, 0], seen_camera[:, :-1, :, 0], CAMERA_X)
            ty = move_class(camera[:, 1:, :, 1], seen_camera[:, :-1, :, 1], CAMERA_Y)
            every = torch.ones_like(tx, dtype=torch.bool)
            out["camera"] = mean_ce(mx, tx, every) + mean_ce(my, ty, every) + \
                mean_ce(px, camera[:, 1:, :, 0] % CAMERA_X, tx == JUMP) + \
                mean_ce(py, camera[:, 1:, :, 1] % CAMERA_Y, ty == JUMP)
        out["total"] = sum(out.values())
        return out


class LayeredDreamer:
    """A layered model as a game engine: one frame per action for each of B games at once, from real
    layered frames.

    layers: each key [B, T0, ...] (data/nes_layers.py; the model's KEYS are kept), actions [B, T0] incoming.
    Each step: the camera from the last frame's camera features and the new buttons (a category, committed),
    then the frame's content, hidden, decided by the model's own decide() (its sprites; their pixels stay
    probabilities), then a commit pass that writes the frame into the cache with its pixels soft (each
    pixel's colour probabilities, as models/dynamics.py's dreams). The frame returned is each pixel's most
    likely colour. When the window is full, the last `keep` frames are encoded again at positions
    0..keep-1. temperature: the categories' sampling temperature (0: the most likely). steps: passes over
    which a model with sprite decisions decides them, the surest first (MaskGIT's order).
    """

    def __init__(self, model: LayeredBase, layers: dict, actions: torch.Tensor, keep: int | None = None,
                 temperature: float = 1.0, generator: torch.Generator | None = None, steps: int = 1):
        self.model, self.temperature, self.generator, self.steps = model, temperature, generator, steps
        self.keep = model.frames * 3 // 4 if keep is None else keep
        self.device = actions.device
        layers = {k: v for k, v in layers.items() if k in model.KEYS}
        self.history = [({k: v[:, i] for k, v in layers.items()}, None) for i in range(actions.shape[1])]
        self.actions = list(actions.long().unbind(1))
        self._encode()

    def _batch(self, frames: list) -> tuple[dict, dict | None]:
        """[(frame, soft pixels or None)] -> layers [B, T, ...] and soft [B, T, ...] (None: all real)."""
        layers = {k: torch.stack([f[k] for f, _ in frames], 1) for k in frames[0][0]}
        if all(s is None for _, s in frames):
            return layers, None
        soft = {key: torch.stack([s[key] if s is not None else self.model.real_soft(f, key) for f, s in frames], 1)
                for key in ("cells", "sprites", "border")}
        return layers, soft

    @torch.no_grad()
    def _encode(self) -> None:
        self.history, self.actions = self.history[-self.keep:], self.actions[-self.keep:]
        self.cache = self.model.new_cache()
        layers, soft = self._batch(self.history)
        actions = torch.stack(self.actions, 1)
        mask = torch.zeros(*actions.shape, self.model.tokens, dtype=torch.bool, device=self.device)
        features = self.model(layers, actions, mask, self.cache, 0, soft)
        self.camera_features = features[:, -1, self.model.camera_at:self.model.frame_at]

    def decide(self, frame: dict, last: dict, act: torch.Tensor, at: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The model's own: the frame's content hidden, its sprites decided into `frame` -> (the last pass's
        features [B, tokens, dim], the sprites' soft pixels [B, SPRITE_TOKENS, P, P, colour_dim], their
        probabilities [B, SPRITE_TOKENS, P*P, LAYER_COLOURS])."""
        raise NotImplementedError

    @torch.no_grad()
    def step(self, actions: torch.Tensor) -> dict:
        """The frames `actions` [B] produce -> the layered frames [B, ...] (committed: each pixel its most
        likely colour), with "probs": cells [B, BANDS, VIEW, 256, LAYER_COLOURS], sprites [B, SPRITE_TOKENS,
        P*P, LAYER_COLOURS] and border [B, BORDER_TOKENS, 256, LAYER_COLOURS]."""
        m, temp, g = self.model, self.temperature, self.generator
        if len(self.history) == m.frames:
            self._encode()
        at = len(self.history)
        last = self.history[-1][0]
        act = actions.long().view(-1, 1)
        b = act.shape[0]
        # the camera, from the last frame and these buttons (a fixed camera stays)
        before = last["camera"].long()
        if m.fixed_camera:
            camera = before
        else:
            mx, my, px, py = m.camera_logits(self.camera_features, act[:, 0])
            cx, cy = choose(mx, temp, g), choose(my, temp, g)
            x = torch.where(cx == JUMP, choose(px, temp, g), (before[..., 0] + cx - MOVE) % CAMERA_X)
            y = torch.where(cy == JUMP, choose(py, temp, g), (before[..., 1] + cy - MOVE) % CAMERA_Y)
            camera = torch.stack((x, y), -1)
        frame = {k: torch.zeros_like(v) for k, v in last.items()}
        frame["camera"] = camera.to(last["camera"].dtype)
        frame["cell_col"] = (camera[..., 0] // CELL).to(last["cell_col"].dtype)
        frame["cell_row"] = ((camera[..., 1] % CAMERA_Y) // CELL).to(last["cell_row"].dtype)
        frame["cell_known"] = torch.ones_like(last["cell_known"])
        f, sprite_soft, sprite_probs = self.decide(frame, last, act, at)
        cell_probs = m.pixel_logits(f[:, :CELLS]).float().softmax(-1)                  # [B, CELLS, 256, C]
        border_probs = m.pixel_logits(f[:, m.border_at:m.border_end]).float().softmax(-1)   # [B, 32, 256, C]
        frame["backdrop"] = m.backdrop_head(f[:, m.frame_at]).argmax(-1).to(last["backdrop"].dtype)
        frame["cells"] = cell_probs.argmax(-1).view(b, BANDS, VIEW, CELL, CELL).to(last["cells"].dtype)
        frame["border"] = unpatch(border_probs.argmax(-1).view(b, BORDER_TOKENS, CELL, CELL)).to(last["border"].dtype)
        # commit: the frame into the cache, its pixels soft
        soft = {"cells": m.embed_probs(cell_probs).view(b, BANDS, VIEW, CELL, CELL, -1), "sprites": sprite_soft,
                "border": m.embed_probs(border_probs).view(b, BORDER_TOKENS, CELL, CELL, -1)}
        visible = torch.zeros(b, 1, m.tokens, dtype=torch.bool, device=self.device)
        features = m({k: v[:, None] for k, v in frame.items()}, act, visible, self.cache, at,
                     {k: v[:, None] for k, v in soft.items()})
        self.camera_features = features[:, 0, m.camera_at:m.frame_at]
        self.history.append((frame, soft))
        self.actions.append(act[:, 0])
        return {**frame, "probs": {"cells": cell_probs.view(b, BANDS, VIEW, CELL * CELL, -1),
                                   "sprites": sprite_probs, "border": border_probs}}


OPTIONS = ("registers", "space_rope", "share_pixels", "fixed_camera")     # a run's architecture options (off when absent)


def build(config: dict, recompute: bool = False) -> LayeredBase:
    """A layered model from a run's saved args (config["model"]: "pixels", the default, or "slots"; the
    OPTIONS it has); recompute for training (activation checkpointing)."""
    from token_world.models.layered_pixels import PixelLayers
    from token_world.models.layered_slots import SlotLayers
    kind = {"pixels": PixelLayers, "slots": SlotLayers}[config.get("model") or "pixels"]
    options = {k: config[k] for k in OPTIONS if config.get(k)}
    return kind(config["dim"], config["layers"], config["heads"], config["frames"], config["colour_dim"], recompute,
                **options)


@torch.no_grad()
def rollout(drawer: LayeredBase, layers: dict, actions: torch.Tensor, start: int, boundary: int,
            temperature: float = 1.0, generator: torch.Generator | None = None, steps: int = 1) -> tuple[dict, dict]:
    """A rollout on layered windows (each key [B, T, ...], incoming actions [B, T]): the drawer's own frames
    start..boundary-1, dreamed one after another from the real frames before `start` with the real buttons
    (its Dreamer) -> (the frames, each of its KEYS [B, k, ...], committed; their pixels as the drawer's
    tokens: cells [B, k, CELLS, dim], sprites [B, k, SPRITE_TOKENS, dim], border [B, k, BORDER_TOKENS, dim])."""
    dreamer = drawer.Dreamer(drawer, {k: v[:, :start] for k, v in layers.items()}, actions[:, :start],
                             temperature=temperature, generator=generator, steps=steps)
    frames, tokens = [], []
    for at in range(start, boundary):
        frame = {k: v for k, v in dreamer.step(actions[:, at]).items() if k != "probs"}
        soft = dreamer.history[-1][1]
        made = drawer.pixel_tokens({k: v[:, None] for k, v in frame.items()}, {k: v[:, None] for k, v in soft.items()})
        frames.append(frame)
        tokens.append({k: v[:, 0] for k, v in made.items()})
    return ({k: torch.stack([f[k] for f in frames], 1) for k in frames[0]},
            {k: torch.stack([z[k] for z in tokens], 1) for k in tokens[0]})
