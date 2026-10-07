"""Line clears: does a model clear the full rows and drop the stack, fed the real buttons?

Clears are found live (no stored eval set): the bot plays real games (data/real_tetris.py); when the play state
turns from checking rows to clearing them (RAM, data/tetris_events.py), the context ends LEAD frames earlier,
while the last piece still falls, and FUTURE real frames follow with the bot's real buttons. Any model
dreams them from the context (diagnostics/worlds.py World.follow; scripts/event_checks.py). The scenario checks (tetris_scenarios.py) score the first
frames of a clear; this follows it to the end: the animation, the rows gone, the stack dropped, the next piece.

Scored on the well's cell grid (WELL: 20 rows x 10 columns of 8 x 8 cells) at the last frame:
- full_rows: the dream's full rows (the real game has none left after its clear: a dream that never clears
  keeps them);
- cells_wrong: the share of the well's cells the dream fills where the real one is empty, or the reverse (the
  stack dropped by the wrong rows, or not at all);
- mass: the dream's filled cells over the real game's (above 1: rows kept; below 1: the stack erased).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from token_world.data.real_tetris import CONTEXT, RealTetris
from token_world.data.tetris_events import CHECK_ROWS, CLEAR_NAMES, CLEARING, STATE, lines_count

LEAD, FUTURE = 10, 60
WELL = (slice(56, 216), slice(96, 176))      # the well: cells at x = 96 + 8 col, y = 56 + 8 row
CELL, CELL_FILLED = 8, 24                    # a cell holds a block with this many filled pixels (a block: 7 x 7)
FIELDS = ("full_rows", "cells_wrong", "mass")


@dataclass
class LineClear:
    context: np.ndarray           # [CONTEXT, 256, 256] uint8 model frames, ending LEAD frames before the clear
    actions: np.ndarray           # [CONTEXT] the button byte that produced each
    layers: dict | None           # the context's layered frames, each key [CONTEXT, ...] (data/nes_layers.py)
    future: np.ndarray            # [FUTURE, 256, 256] the real frames after
    future_actions: np.ndarray    # [FUTURE] the bot's real buttons
    cleared: int                  # rows cleared: 1 single .. 4 tetris
    level: int

    @property
    def name(self) -> str:
        return CLEAR_NAMES[self.cleared]


def clears(seed: int, layered: bool = True) -> Iterator[LineClear]:
    """Endless live line clears, at most 3 from a game (so they come from many boards)."""
    while True:
        real = RealTetris(seed, layered)
        seed += 1
        frames, actions, layers = (deque(maxlen=CONTEXT + LEAD) for _ in range(3))
        previous, found = None, 0
        for _ in range(60_000):
            frame, action = real.step(None)
            ram = real.observation.ram
            frames.append(frame.copy())
            actions.append(action)
            if layered:
                layers.append(real.layers[-1])
            if previous == CHECK_ROWS and ram[STATE] == CLEARING and len(frames) == CONTEXT + LEAD:
                before, level = int(lines_count(ram)), int(ram[0x44])
                future, acts, counts = list(frames)[CONTEXT:], list(actions)[CONTEXT:], []
                while len(future) < FUTURE:
                    f, a = real.step(None)
                    future.append(f.copy())
                    acts.append(a)
                    counts.append(int(lines_count(real.observation.ram)))
                cleared = max(counts) - before
                context = list(layers)[:CONTEXT] if layered else None
                if cleared in CLEAR_NAMES:
                    yield LineClear(np.stack(list(frames)[:CONTEXT]), np.array(list(actions)[:CONTEXT], np.int64),
                                    {k: np.stack([f[k] for f in context]) for k in context[0]} if layered else None,
                                    np.stack(future), np.array(acts, np.int64), cleared, level)
                    found += 1
                frames.clear(), actions.clear(), layers.clear()
                if found == 3:
                    break
            previous = ram[STATE]


def well_cells(frame: np.ndarray, empty: int) -> np.ndarray:
    """A model frame [256, 256] -> its well's blocks [20, 10] bool."""
    filled = frame[WELL] != empty
    return filled.reshape(20, CELL, 10, CELL).sum((1, 3)) >= CELL_FILLED


def score(clear: LineClear, dreamed: np.ndarray) -> dict[str, float]:
    """FIELDS at the last frame (module docstring)."""
    empty = int(np.bincount(clear.future[-1][WELL].ravel()).argmax())
    real, mine = well_cells(clear.future[-1], empty), well_cells(dreamed[-1], empty)
    return {"full_rows": float(mine.all(1).sum()), "cells_wrong": float((real != mine).mean()),
            "mass": float(mine.sum() / max(real.sum(), 1))}


def sheet(clear: LineClear, dreams: dict[str, np.ndarray], palette: np.ndarray, every: int = 4) -> np.ndarray:
    """The well every `every` frames: the real game on top, a row per model's dream -> RGB [rows * 160, n * 84, 3]."""
    rows = []
    for frames in [clear.future, *dreams.values()]:
        tiles = [np.pad(palette[frames[t][WELL]], ((0, 0), (0, 4), (0, 0))) for t in range(0, FUTURE, every)]
        rows.append(np.concatenate(tiles, 1))
    return np.concatenate(rows, 0).astype(np.uint8)
