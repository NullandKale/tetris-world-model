"""Each Super Mario Bros. frame's motion, from the pictures: for every 8 x 8 cell of the model frame, the pixel
offset its picture came from in the frame before (2026-10-18).

The SMB counterpart of tetris_motion.py, the same codes per [32, 32] cell grid, for the motion in and out of the
world model (VideoJAM's motion, which is optical flow there). Tetris moves by whole cells and turns pieces, so
its motion comes from the RAM. SMB moves by pixels: the screen scrolls by 1-3 a frame, and Mario, enemies and
fireballs move a few pixels either way. So here a cell's move is found in the pictures alone, exactly: the
smallest offset (dy, dx), |dy| <= DY and |dx| <= DX pixels, at which the frame before's pixels are the cell's,
pixel for pixel.

- STILL (0): the cell's pixels are as they were (sky stays sky while the screen scrolls: still is what it looks
  like);
- moved (1 .. MOVES): its pixels are the frame before's, (dy, dx) pixels away (code_of / offset_of);
- NEW (MOVES + 1): it changed and no offset gives its pixels, or it is one colour all over (any offset would
  match: a vacated cell, not a move);
- UNKNOWN (255): a window's first frame (the frame before is outside it); a model is neither given nor scored
  on it.

The border (token rows 0 and 15) is drawn state, not picture: always STILL.
"""
from __future__ import annotations

import numpy as np

from token_world.data.model_frames import BAND

CELL, GRID = 8, 32
DY, DX = 6, 4                          # Mario falls up to about 4-5 px a frame; scrolling and walking 1-4 across
OFFSETS = [(dy, dx) for dy in range(-DY, DY + 1) for dx in range(-DX, DX + 1) if (dy, dx) != (0, 0)]
SEARCH = sorted(OFFSETS, key=lambda o: (abs(o[0]) + abs(o[1]), abs(o[0]), o))     # nearest first
STILL, MOVES = 0, len(OFFSETS)
NEW = MOVES + 1
CODES = MOVES + 2
UNKNOWN = 255
_CODE = {o: i + 1 for i, o in enumerate(OFFSETS)}
_OUTSIDE = 255                         # no model colour index: pixels from outside the frame never match
GATHER = 1 << 16                       # pixels compared per gather in frame_motion
_SEARCH_DY = np.array([o[0] for o in SEARCH])
_SEARCH_DX = np.array([o[1] for o in SEARCH])
_SEARCH_CODE = np.array([_CODE[o] for o in SEARCH], np.uint8)
_PIXELS = np.arange(CELL)


def code_of(dy: int, dx: int) -> int:
    return STILL if (dy, dx) == (0, 0) else _CODE[(dy, dx)]


def offset_of(code: int) -> tuple[int, int] | None:
    return (0, 0) if code == STILL else None if code == NEW else OFFSETS[code - 1]


def _blocks(frame: np.ndarray) -> np.ndarray:
    """A model frame [256, 256] -> its cells [32, 32, 8, 8]."""
    return frame.reshape(GRID, CELL, GRID, CELL).transpose(0, 2, 1, 3)


def frame_motion(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Two consecutive model frames [256, 256] uint8 -> the second frame's motion codes [32, 32] uint8.

    Only the changed cells are searched, gathered from the padded frame before at every offset of a chunk
    of SEARCH at once (nearest first, so the first match is the smallest offset), the chunk as large as
    GATHER pixels allow: a cell nothing matches (NEW) costs its own 116 offsets, not the whole frame's."""
    b1 = _blocks(after)
    changed = (_blocks(before) != b1).any((2, 3))
    codes = np.where(changed, NEW, STILL).astype(np.uint8)
    flat = (b1 == b1[:, :, :1, :1]).all((2, 3))
    todo = changed & ~flat
    todo[:BAND // CELL] = todo[-(BAND // CELL):] = False
    cells = np.argwhere(todo)
    if len(cells):
        padded = np.pad(before, ((DY, DY), (DX, DX)), constant_values=_OUTSIDE)
        target = b1[cells[:, 0], cells[:, 1]]                                   # [K, 8, 8]
        top, left = cells[:, 0] * CELL + DY, cells[:, 1] * CELL + DX            # each cell in the padded frame
        left_k = np.arange(len(cells))
        i = 0
        while i < len(SEARCH) and len(left_k):
            n = max(1, min(len(SEARCH) - i, GATHER // (CELL * CELL * len(left_k))))
            dy, dx = _SEARCH_DY[i:i + n], _SEARCH_DX[i:i + n]
            rows = top[left_k, None, None] - dy[None, :, None] + _PIXELS                # [k, n, 8]
            cols = left[left_k, None, None] - dx[None, :, None] + _PIXELS
            match = (padded[rows[..., :, None], cols[..., None, :]] == target[left_k, None]).all((2, 3))   # [k, n]
            hit = match.any(1)
            codes[cells[left_k[hit], 0], cells[left_k[hit], 1]] = _SEARCH_CODE[i + match[hit].argmax(1)]
            left_k = left_k[~hit]
            i += n
    codes[:BAND // CELL] = codes[-(BAND // CELL):] = STILL
    return codes


def window_motion(x: np.ndarray) -> np.ndarray:
    """A window's model frames [T, 256, 256] -> motion codes [T, 32, 32] uint8, the first frame UNKNOWN."""
    out = np.full((len(x), GRID, GRID), UNKNOWN, np.uint8)
    for t in range(1, len(x)):
        out[t] = frame_motion(x[t - 1], x[t])
    return out
