"""Does a dream look like real play? Generic scores, for any game, that do not compare with the real future.

Frames are exact palette indices, so real play has a countable vocabulary of 16 x 16 patches, and of
patch changes (a patch at a place, then the patch there one frame later). A coherent world, whatever the
game, is built from patches and changes that occur in real play; a blend, a ghost, speckle, half a shape
or a shape that morphs makes ones that never do, while a different but legal outcome (another random
piece, a piece somewhere else) does not. PatchBank holds both vocabularies as 64-bit hashes, built from
live frames (no stored eval set). score() reads a dream's most likely colours and how sure it is:

- unseen_patch: the share of the dream's tokens that changed since its first frame whose exact content
  never occurs in the bank (the static rest is copied, so it is left out);
- unseen_change: the share of its frame-to-frame token changes never seen in the bank;
- change_px: the share of pixels changing from frame to frame (against the real game's: frozen dreams
  change too little, flickering ones too much);
- unsure_px: the share of pixels whose most likely colour is under SURE likely (hedging).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

PATCH = 16
SURE = 0.9
# a random odd 64-bit weight per pixel of a token, as int64 (arithmetic wraps modulo 2^64)
_WEIGHTS = torch.from_numpy((np.random.default_rng(20261005).integers(1, 2 ** 63, size=PATCH * PATCH, dtype=np.uint64)
                             | np.uint64(1)).view(np.int64))
_MIX = 0x9E3779B97F4A7C15 - 2 ** 64


def patch_hashes(frames) -> torch.Tensor:
    """frames [..., H, W] palette indices (array or tensor, on any device) -> [..., N] int64: each 16 x 16
    token's exact content, hashed (a random linear hash over its pixels, modulo 2^64)."""
    frames = torch.as_tensor(frames)
    *lead, h, w = frames.shape
    tokens = frames.reshape(*lead, h // PATCH, PATCH, w // PATCH, PATCH).transpose(-3, -2)
    tokens = tokens.reshape(*lead, -1, PATCH * PATCH).long()
    return (tokens * _WEIGHTS.to(tokens.device)).sum(-1)


def change_hashes(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    """Token hashes of two frames, [..., N] each -> the hash of each (before, after) pair."""
    return before * _MIX + after


class PatchBank:
    """The patches and patch changes of real play, as sorted unique hashes."""

    def __init__(self, patches: np.ndarray | None = None, changes: np.ndarray | None = None):
        self.patches = np.zeros(0, np.int64) if patches is None else patches
        self.changes = np.zeros(0, np.int64) if changes is None else changes

    @torch.no_grad()
    def add(self, frames) -> None:
        """Consecutive real frames [T, H, W] (array or tensor, hashed where they are): their patches and
        their changes."""
        h = patch_hashes(frames)
        moved = h[1:] != h[:-1]
        self.patches = np.union1d(self.patches, h.unique().cpu().numpy())
        self.changes = np.union1d(self.changes, change_hashes(h[:-1], h[1:])[moved].unique().cpu().numpy())

    def save(self, path: Path) -> None:
        np.savez_compressed(path, patches=self.patches, changes=self.changes)

    @classmethod
    def load(cls, path: Path) -> "PatchBank":
        d = np.load(path)
        return cls(np.sort(d["patches"].view(np.int64)), np.sort(d["changes"].view(np.int64)))   # sorted as int64


def _unseen(values: np.ndarray, bank: np.ndarray) -> np.ndarray:
    at = np.searchsorted(bank, values).clip(0, max(len(bank) - 1, 0))
    return bank[at] != values if len(bank) else np.ones(values.shape, bool)


def score(bank: PatchBank, start: np.ndarray, frames: np.ndarray, sure: np.ndarray | None = None) -> dict[str, float]:
    """start [H, W]: the last real frame before the dream; frames [T, H, W]: the dream's most likely
    colours; sure [T, H, W]: their probabilities (None: a committed dream, sure everywhere)."""
    h = patch_hashes(np.concatenate([start[None], frames]))
    active = h[1:] != h[:1]                                  # tokens no longer as the dream started
    moved = h[1:] != h[:-1]
    unseen_patch = _unseen(h[1:][active].numpy(), bank.patches)
    unseen_change = _unseen(change_hashes(h[:-1], h[1:])[moved].numpy(), bank.changes)
    seq = np.concatenate([start[None], frames])
    out = {"unseen_patch": float(unseen_patch.mean()) if unseen_patch.size else 0.0,
           "unseen_change": float(unseen_change.mean()) if unseen_change.size else 0.0,
           "change_px": float((seq[1:] != seq[:-1]).mean()),
           "unsure_px": float((sure < SURE).mean()) if sure is not None else 0.0}
    return out
