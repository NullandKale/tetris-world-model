"""A critic that judges a turn with its button (2026-10-16; docs/guides/turn_direction.md).

No pixel model turns pieces the way A vs B says: per-pixel losses train a model that ignores the button toward
a hedge (cross-entropy 50 / 50, L2 the average, L1 each cell its own way), and the hedge costs almost nothing,
since the cells that tell the two turns apart are a few of a frame's pixels. This critic is conditional and
matching-aware (Reed et al. 2016, GAN-CLS): it sees the frame before a turn and the turned frame, around the
piece, with the turn button. Real: the real frame with the real button. Fake: the model's frame with the real
button, and the real frame with the button swapped A <-> B. A blend looks unlike either real turn, and a clean
turn the wrong way looks wrong for that button: the only thing that tells the real pair from the swapped one is
the direction.

Only J, L and T turns (I, S and Z have two orientations, so A and B draw the same picture; O does not turn), the
frames hidden whole when predicted (as a dream draws them: with part of the frame visible the turn can be read
off it), and the model's frame given to the critic as one colour per pixel (Gumbel-softmax, straight through),
as the real frame is, so the critic cannot win on softness alone (the likely cause of the 2026-10-01 critic's
grain: it judged soft frames against hard ones).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

from token_world.data.tetris_bot import A, B, ORIENTATIONS, TYPE_OF
from token_world.data.tetris_motion import UNKNOWN

CROP = 3                                   # tokens a side around the turning piece (48 x 48 pixels at patch 16)
TURN_TYPES = ("J", "L", "T")               # the pieces whose turn shows a direction
TURNS_AT = torch.tensor([o in ORIENTATIONS and TYPE_OF[o] in TURN_TYPES for o in range(256)])


def turn_crops(incoming: torch.Tensor, motion: torch.Tensor, piece: torch.Tensor, grid: int, turned: torch.Tensor
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The frames a J, L or T piece turns in, and where: incoming [B, T] (frame k's incoming button), motion [B,
    T, 32, 32] the real motion codes, piece [B, T] each frame's RAM orientation byte (0x42; picture k shows RAM
    k - 1), grid the token grid's side, turned [CODES] the turn codes (dynamics.TURNED) -> (b, f, top, left,
    button) [K] each: window, frame, the crop's top-left token (CROP x CROP tokens around the turned cells, in
    the picture's token rows) and the turn button (0 A, 1 B: frame f - 1's incoming, the press shows a frame
    late)."""
    codes = motion.long()
    is_turn = turned.to(codes.device)[codes.clamp(max=len(turned) - 1)] & (codes != UNKNOWN)     # [B, T, 32, 32]
    any_turn = is_turn.flatten(2).any(-1)
    any_turn[:, :2] = False
    press = torch.zeros_like(incoming)
    press[:, 1:] = incoming[:, :-1].clamp(min=0) & (A | B)                       # frame f - 1's incoming
    one = (press == A) | (press == B)
    before = torch.zeros_like(piece)
    before[:, 1:] = piece[:, :-1]                                               # the RAM the picture shows
    jlt = TURNS_AT.to(piece.device)[before.long()]
    b, f = (any_turn & one & jlt).nonzero(as_tuple=True)
    cells = is_turn[b, f].float()                                               # [K, 32, 32]
    rows = torch.arange(cells.shape[1], device=cells.device, dtype=torch.float)
    count = cells.sum((1, 2)).clamp_min(1)
    cy = (cells.sum(2) * rows).sum(1) / count                                   # the turned cells' centre, in cells
    cx = (cells.sum(1) * rows).sum(1) / count
    per = cells.shape[1] // grid                                                # cells a token
    top = ((cy + 0.5) / per).long() - CROP // 2
    left = ((cx + 0.5) / per).long() - CROP // 2
    top = top.clamp(1, grid - 1 - CROP)                                         # the picture's rows, not the border's
    left = left.clamp(0, grid - CROP)
    button = (press[b, f] == B).long()
    return b, f, top, left, button


def crop_tokens(top: torch.Tensor, left: torch.Tensor, grid: int) -> torch.Tensor:
    """[K] top-left tokens -> [K, CROP * CROP] token ids, row by row."""
    d = torch.arange(CROP, device=top.device)
    return ((top[:, None, None] + d[None, :, None]) * grid + left[:, None, None] + d[None, None, :]).flatten(1)


def crop_pixels(index: torch.Tensor, top: torch.Tensor, left: torch.Tensor, patch: int) -> torch.Tensor:
    """index [K, 256, 256] -> [K, CROP * patch, CROP * patch], the crop's pixels."""
    size = CROP * patch
    d = torch.arange(size, device=index.device)
    ys = (top * patch)[:, None] + d[None]                                       # [K, size]
    xs = (left * patch)[:, None] + d[None]
    k = torch.arange(len(index), device=index.device)[:, None, None]
    return index[k, ys[:, :, None], xs[:, None, :]]


def tokens_to_image(pixels: torch.Tensor, patch: int) -> torch.Tensor:
    """[K, CROP * CROP, patch * patch, C] per token -> [K, C, CROP * patch, CROP * patch]."""
    k, _, _, c = pixels.shape
    x = pixels.view(k, CROP, CROP, patch, patch, c).permute(0, 5, 1, 3, 2, 4)
    return x.reshape(k, c, CROP * patch, CROP * patch)


def hard_sample(logits: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """[..., C] logits -> a one-hot colour per pixel drawn from them (Gumbel-max), its gradient the softmax's
    (straight through)."""
    u = torch.rand(logits.shape, device=logits.device, generator=generator).clamp_(1e-9, 1 - 1e-9)
    noisy = logits.float() - torch.log(-torch.log(u))
    soft = noisy.softmax(-1)
    hard = F.one_hot(noisy.argmax(-1), logits.shape[-1]).to(soft.dtype)
    return hard - soft.detach() + soft


class TurnCritic(nn.Module):
    """(frame before, frame) one-hot colours [K, C, H, W] each and the button [K] (0 A, 1 B) -> a score [K]:
    high for a real turn with its own button (hinge). Spectral-normalised convolutions, a projection on the
    button (Miyato and Koyama 2018), float32."""

    def __init__(self, colours: int, width: int = 32):
        super().__init__()
        self.colour = spectral_norm(nn.Conv2d(2 * colours, width, 1))
        self.convs = nn.ModuleList(spectral_norm(nn.Conv2d(a, b, 4, 2, 1))
                                   for a, b in ((width, 2 * width), (2 * width, 4 * width), (4 * width, 4 * width)))
        self.out = spectral_norm(nn.Linear(4 * width, 1))
        self.button = spectral_norm(nn.Embedding(2, 4 * width))

    def forward(self, before: torch.Tensor, frame: torch.Tensor, button: torch.Tensor) -> torch.Tensor:
        with torch.autocast(before.device.type, enabled=False):
            h = F.leaky_relu(self.colour(torch.cat([before.float(), frame.float()], 1)), 0.2)
            for conv in self.convs:
                h = F.leaky_relu(conv(h), 0.2)
            h = h.sum((2, 3))
            return self.out(h)[:, 0] + (self.button(button) * h).sum(1)


def critic_losses(critic: TurnCritic, before: torch.Tensor, real: torch.Tensor, fake: torch.Tensor,
                  button: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """-> (the critic's hinge loss, with the model's frame detached; the model's loss, -score of its frame with the
    real button; the share of turns whose real frame scores higher with its own button than with the swapped
    one: whether the critic itself tells the direction)."""
    real_score = critic(before, real, button)
    swapped = critic(before, real, 1 - button)
    fake_score = critic(before, fake.detach(), button)
    d = F.relu(1 - real_score).mean() + 0.5 * (F.relu(1 + fake_score).mean() + F.relu(1 + swapped).mean())
    g = -critic(before, fake, button).mean()
    return d, g, (real_score > swapped).float().mean().detach()


__all__ = ["CROP", "TurnCritic", "critic_losses", "crop_pixels", "crop_tokens", "hard_sample", "tokens_to_image",
           "turn_crops"]
