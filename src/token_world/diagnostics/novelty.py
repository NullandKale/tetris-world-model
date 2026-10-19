"""How new the training data is (2026-10-10): spawn boards in training, 16 x 16 patches in the data report.

Novelty takes the windows a step trains on (live_batches.LiveBatches hands it each one) and reads each emulated
frame once (windows overlap: a frame is read with the first window that brings it, by its tick): the share of
the frames' piece spawns on a settled board never seen before in the run, and how many distinct boards (a game's
own reader: Tetris's world_nes_tetris.spawn_boards, from RAM). The run's coverage of the game's random space.

Frames and patches are not tracked in training (until 2026-10-10 they were): an exact set of every frame's
picture grew without bound and slowed the VAE run's main process to a third of its speed by 20k steps, and what
they said was settled by then. About three quarters of the frames repeat a picture exactly, because the game
stands still between drops and on still screens (not reuse: every frame is new from the live stream), and the
distinct patches flatten near 6,600 by about 16k steps of the VAE's stream: the game's whole set of patches.
scripts/data_report.py counts the patches (picture_hashes) for a stream setting.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from token_world.data.model_frames import GAME_ROWS

PATCH = 16
_K = np.random.default_rng(20261010).integers(1, 2**63, 32, dtype=np.uint64) | np.uint64(1)


def mix(words: np.ndarray) -> np.ndarray:
    """[N, W] uint64 (W <= 32) -> a 64-bit hash per row."""
    with np.errstate(over="ignore"):
        h = (words * _K[:words.shape[1]]).sum(axis=1, dtype=np.uint64)
        h ^= h >> np.uint64(29)
        h *= np.uint64(0xBF58476D1CE4E5B9)
        h ^= h >> np.uint64(32)
    return h


def picture_hashes(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Model frames [N, 256, 256] uint8 -> (each frame's game-picture hash [N], each patch's hash [N, P])."""
    game = np.ascontiguousarray(frames[:, GAME_ROWS])
    n, h, w = game.shape
    tiles = game.reshape(n, h // PATCH, PATCH, w // PATCH, PATCH).transpose(0, 1, 3, 2, 4)
    words = np.ascontiguousarray(tiles).reshape(-1, PATCH * PATCH).view(np.uint64)        # [N * P, 32]
    patches = mix(words).reshape(n, -1)
    frame = np.stack([mix(patches[:, i:i + 32]) for i in range(0, patches.shape[1], 32)], 1)
    return mix(frame), patches


class HashSet:
    """64-bit hashes in one sorted array; each add inserts its new ones in place, a copy of the array: for the
    sets that stay small (a run's boards: about one new per 100 frames trained; the game's patches)."""

    def __init__(self):
        self.sorted = np.zeros(0, np.uint64)

    def __len__(self) -> int:
        return len(self.sorted)

    def add(self, hashes: np.ndarray) -> np.ndarray:
        """Adds `hashes` -> for each one, whether it was new (a repeat within `hashes` counts once as new)."""
        unique, first = np.unique(hashes, return_index=True)
        at = np.searchsorted(self.sorted, unique)
        known = (self.sorted[at.clip(max=len(self.sorted) - 1)] == unique if len(self.sorted)
                 else np.zeros(len(unique), bool))
        self.sorted = np.insert(self.sorted, at[~known], unique[~known])
        new = np.zeros(len(hashes), bool)
        new[first[~known]] = True
        return new


class Novelty:
    """board(ram [T, 2048], new: int) -> the board keys (bytes) at the piece spawns among the last `new` frames."""

    def __init__(self, board):
        self.board = board
        self.boards = HashSet()
        self.last_tick: dict[int, int] = defaultdict(lambda: -1)
        self.counts = defaultdict(int)

    def add(self, part: dict) -> None:
        tick, worker = int(part["tick"]), int(part["worker_id"])
        new = min(len(part["ram"]), tick - self.last_tick[worker])        # frames this window brings first
        self.last_tick[worker] = max(self.last_tick[worker], tick)
        if new <= 0:
            return
        keys = self.board(np.asarray(part["ram"]), new)
        if keys:
            hashes = mix(np.frombuffer(b"".join(k.ljust(32, b"\0") for k in keys), np.uint64).reshape(len(keys), 4))
            self.counts["spawns"] += len(keys)
            self.counts["new_boards"] += int(self.boards.add(hashes).sum())

    def take(self) -> dict[str, float]:
        """The share since the last call, and the run's distinct boards."""
        c, out = self.counts, {}
        if c["spawns"]:
            out.update({"novel_boards": c["new_boards"] / c["spawns"], "distinct_boards": len(self.boards)})
        self.counts = defaultdict(int)
        return out


NOVELTY_FIELDS = ["novel_boards", "distinct_boards"]
