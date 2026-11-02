"""The inverse-dynamics model (IDM) for Tetris: a few pictures around a press -> which effect it had.

Trained on real play (scripts/idm_check.py, test 9, 2026-10-30), scored on every frame of a seeded stream
(scripts/idm_breakdown_check.py). The press at t moves the RAM at t + 1, which the picture at t + 2 shows (a
picture shows the RAM from before its own frame), so the IDM reads the pictures t - 1 .. t - 1 + frames - 1: 5
(t - 1 .. t + 3, the first IDMs, idm_v2: the judge) or 4 (t - 1 .. t + 2, ending on the picture the effect first
shows in: the loss of the post-trains, which reads a predicted picture last). Colours embedded, plus where each
picture differs from the one before (a press's effect is a few changed cells in a still frame: a plain stack of
embeddings did not even fit 400 samples), through normalised convolutions and a max over the frame.

soft(): the same model on colour distributions [..., COLOURS] instead of indices (one-hot gives exactly forward's
answer), so a world model's predicted picture can be read with gradients (straight-through: the most likely
colours forward, the probabilities backward). Training-only: nothing of the IDM ships.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from token_world.data.tetris_bot import IDS_OF, PIECE, TYPE_OF, X, Y
from token_world.diagnostics.divergence import falling

EFFECTS = ("none", "turn +1", "turn -1", "turn (2-way piece)", "left", "right", "drop")
CROP = (slice(40, 224), slice(88, 232))          # the well, the next-piece box and the counters around it
COLOURS = 59                                      # the palette indices a model frame uses (and one spare)
EMBED = 8


def effect(r0: np.ndarray, r1: np.ndarray) -> int:
    """The press's effect between RAM frames r0 and r1, an index into EFFECTS (from the RAM alone)."""
    if not (falling(r0) and falling(r1)):
        return 0
    o0, o1 = int(r0[PIECE]), int(r1[PIECE])
    if TYPE_OF[o0] != TYPE_OF[o1]:
        return 0
    if o0 != o1:
        ids = IDS_OF[TYPE_OF[o0]]
        if len(ids) == 2:
            return 3
        d = (ids.index(o1) - ids.index(o0)) % 4
        return 1 if d == 1 else 2 if d == 3 else 0
    dx, dy = int(r1[X]) - int(r0[X]), int(r1[Y]) - int(r0[Y])
    if dx == -1:
        return 4
    if dx == 1:
        return 5
    if dy == 1:
        return 6
    return 0


def effects(ram: np.ndarray) -> np.ndarray:
    """A window's RAM [T, 2048] -> each press's effect [T] (effect t: the press at t, between ram t and t + 1;
    the last has no next frame: 0)."""
    out = np.zeros(len(ram), np.int8)
    for t in range(len(ram) - 1):
        out[t] = effect(ram[t], ram[t + 1])
    return out


class IDM(nn.Module):
    def __init__(self, width: int = 64, frames: int = 5):
        super().__init__()
        self.frames = frames
        self.embed = nn.Embedding(COLOURS, EMBED)

        def block(i, o, k, s):
            return [nn.Conv2d(i, o, k, s, k // 2 - (s == 2 and k % 2 == 0)), nn.GroupNorm(8, o), nn.GELU()]
        self.net = nn.Sequential(
            *block(EMBED * frames + frames - 1, width, 4, 2), *block(width, width, 3, 1),
            *block(width, 2 * width, 4, 2), *block(2 * width, 2 * width, 3, 1),
            *block(2 * width, 4 * width, 4, 2), nn.AdaptiveMaxPool2d(1), nn.Flatten(),
            nn.Linear(4 * width, len(EFFECTS)))

    @property
    def span(self) -> tuple[int, int]:
        """The pictures it reads for the press at t: t + span[0] .. t + span[1] - 1."""
        return -1, self.frames - 1

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames [B, frames, h, w] palette indices -> logits [B, len(EFFECTS)]."""
        b, f, h, w = frames.shape
        x = frames.long().clamp(max=COLOURS - 1)
        changed = (x[:, 1:] != x[:, :-1]).to(self.embed.weight.dtype)                  # [B, f - 1, h, w]
        emb = self.embed(x).permute(0, 1, 4, 2, 3).reshape(b, f * EMBED, h, w)
        return self.net(torch.cat([emb, changed], 1))

    def soft(self, probs: torch.Tensor) -> torch.Tensor:
        """probs [B, frames, h, w, C <= COLOURS] colour distributions -> logits: the expected embedding, and a
        pixel's chance of differing from the picture before; one-hot probs give forward's logits."""
        b, f, h, w, c = probs.shape
        if c < COLOURS:
            probs = torch.cat([probs, probs.new_zeros(b, f, h, w, COLOURS - c)], -1)
        weight = self.embed.weight.to(probs.dtype)
        emb = (probs @ weight).permute(0, 1, 4, 2, 3).reshape(b, f * EMBED, h, w)
        changed = 1 - (probs[:, 1:] * probs[:, :-1]).sum(-1)                           # [B, f - 1, h, w]
        return self.net(torch.cat([emb, changed.to(emb.dtype)], 1))


def load_idm(path: Path, device) -> IDM:
    """A saved IDM (its state dict), frozen; how many pictures it reads from its first layer's width."""
    state = torch.load(path, map_location=device)
    frames = (state["net.0.weight"].shape[1] + 1) // (EMBED + 1)
    idm = IDM(width=state["net.0.weight"].shape[0], frames=frames).to(device)
    idm.load_state_dict(state)
    return idm.eval().requires_grad_(False)
