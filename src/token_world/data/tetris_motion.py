"""Each frame's motion, from the game's RAM: for every 8 x 8 cell of the model frame, where its picture came from
in the frame before (2026-10-15).

What a game engine's G-buffer would render as motion vectors, made from the RAM instead (it is training data
only: the model is to predict it, never to be given it). Optical flow cannot do Tetris: a turn is a flip, in one
frame, so which old cell became which new one is only known from the piece's orientation. Per frame, a [32, 32]
code per cell (the model frame's 256 x 256 in 8 x 8 cells):

- STILL (0): the cell's pixels are as they were;
- moved (1 .. MOVES): its pixels are the frame before's cell (dy, dx) cells away, dy in -4..4 (a tetris drops the
  stack four rows), dx in -2..2 (code_of / offset_of);
- NEW (MOVES + 1): it changed and came from nowhere this frame: a spawn, a vacated cell, a cleared row's blink,
  the digits and the next piece, menus;
- UNKNOWN (255): no motion can be made (a window's first two frames: the RAM before them is outside it); a
  model is neither given nor scored on it.

Moves stand only where the pictures agree, pixel for pixel (a moved cell whose pixels are not its source's is
NEW):

- the falling piece, from the RAM (0x40 / 0x41 its column / row, 0x42 its orientation): every cell of it, the
  ones that overlap where it was too; a turn's cells correspond through the rotation that maps the old
  orientation's cells onto the new one's, about the piece's centre. The picture of frame k shows the RAM captured
  with frame k - 1 (the render lag), so frame k's piece moves are from RAM k - 2 to RAM k - 1: every one of 840
  moves (27 turns) on live windows matched its picture there, 26% at a frame either side (2026-10-15);
- the stack dropping after a line clear, from the pictures: the RAM's board drops at once when the clear begins,
  but the picture shows the clear's animation for about 16 frames and draws the drop in one frame while the play
  state is 5-8 (counters and spawn), so then each changed row of the well takes the drop of n rows (1-4) that its
  pixels match (15 of 15 clears on live windows, 100% of the changed cells, 2026-10-15);
- the top-out curtain, from the pictures the same way: while the play state is 10 it covers one more row of
  the well every 4 frames, top to bottom, each new row the one above it in the frame before, so it slides
  down a row (its first row, with nothing above it in the well, is NEW).

The border (token rows 0 and 15) is drawn state, not picture: always STILL.
"""
from __future__ import annotations

import numpy as np

from token_world.data.model_frames import BAND
from token_world.data.tetris_bot import ORIENTATIONS, PIECE, TYPE_OF, X, Y
from token_world.data.tetris_events import FALLING, IN_GAME, MODE, STATE, TOP_OUT

CELL, GRID = 8, 32                     # cells of 8 x 8 pixels, 32 x 32 of them in the model frame
WELL_ROW, WELL_COL = 7, 12             # the well's top-left cell in the model frame (pixels 56, 96)
AFTER_CLEAR = range(5, 9)              # play states that update the counters and spawn: the drop is drawn then
DY, DX = 4, 2                          # the largest move, in cells, down / up and sideways
OFFSETS = [(dy, dx) for dy in range(-DY, DY + 1) for dx in range(-DX, DX + 1) if (dy, dx) != (0, 0)]
STILL, MOVES = 0, len(OFFSETS)
NEW = MOVES + 1
CODES = MOVES + 2                      # every code: still, the moves, new
UNKNOWN = 255
_CODE = {o: i + 1 for i, o in enumerate(OFFSETS)}


def code_of(dy: int, dx: int) -> int:
    """A move of (dy, dx) cells -> its code (STILL for none)."""
    return STILL if (dy, dx) == (0, 0) else _CODE[(dy, dx)]


def offset_of(code: int) -> tuple[int, int] | None:
    """A code -> its move (dy, dx); (0, 0) for STILL, None for NEW."""
    return (0, 0) if code == STILL else None if code == NEW else OFFSETS[code - 1]


def _rotations(offsets):
    """(row, column) offsets -> the same turned a quarter clockwise, a quarter anticlockwise and a half (screen
    rows grow downward: clockwise takes right to down)."""
    yield [(dx, -dy) for dy, dx in offsets]
    yield [(-dx, dy) for dy, dx in offsets]
    yield [(-dy, -dx) for dy, dx in offsets]


def piece_moves(before: np.ndarray, after: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]] | None:
    """Two consecutive frames' RAM -> the falling piece's cells as (well cell after, well cell it came from), when
    the same piece falls in both; None otherwise (no piece, a spawn, a lock)."""
    for ram in (before, after):
        if ram[MODE] != IN_GAME or ram[STATE] != FALLING or int(ram[PIECE]) not in ORIENTATIONS:
            return None
    o0, o1 = int(before[PIECE]), int(after[PIECE])
    if TYPE_OF[o0] != TYPE_OF[o1]:
        return None
    old, new = list(ORIENTATIONS[o0]), list(ORIENTATIONS[o1])
    source = dict(zip(new, old)) if o0 == o1 else None
    if source is None:
        for turned in _rotations(old):           # which old offset each new one is, turned
            if sorted(turned) == sorted(new):
                source = dict(zip(turned, old))
                break
    if source is None:                           # not a pure rotation in the game's table: no correspondence
        return None
    y0, x0, y1, x1 = int(before[Y]), int(before[X]), int(after[Y]), int(after[X])
    return [((y1 + r[0], x1 + r[1]), (y0 + source[r][0], x0 + source[r][1])) for r in new]


def _blocks(frame: np.ndarray) -> np.ndarray:
    """A model frame [256, 256] -> its cells [32, 32, 8, 8]."""
    return frame.reshape(GRID, CELL, GRID, CELL).transpose(0, 2, 1, 3)


def frame_motion(before: np.ndarray, after: np.ndarray, ram_before: np.ndarray | None,
                 ram_after: np.ndarray | None) -> tuple[np.ndarray, int, int]:
    """Two consecutive model frames [256, 256] and the RAM each shows (or None) -> the second frame's motion
    codes [32, 32] uint8, and the moves the RAM gave and how many of them the pictures confirmed."""
    b0, b1 = _blocks(before), _blocks(after)
    changed = (b0 != b1).any((2, 3))
    codes = np.where(changed, NEW, STILL).astype(np.uint8)
    codes[:BAND // CELL] = codes[-(BAND // CELL):] = STILL
    given = confirmed = 0
    if ram_after is not None and ram_after[MODE] == IN_GAME and (ram_after[STATE] in AFTER_CLEAR
                                                              or ram_after[STATE] == TOP_OUT):
        _drop(codes, b0, b1, changed)
    if ram_before is not None and ram_after is not None:
        for (r1, c1), (r0, c0) in piece_moves(ram_before, ram_after) or []:
            dy, dx = r1 - r0, c1 - c0
            if (dy, dx) == (0, 0) or abs(dy) > DY or abs(dx) > DX:
                continue
            if not (0 <= r1 < 20 and 0 <= c1 < 10 and 0 <= r0 < 20 and 0 <= c0 < 10):
                continue                         # above the well (hidden), or off it
            at, came = (WELL_ROW + r1, WELL_COL + c1), (WELL_ROW + r0, WELL_COL + c0)
            given += 1
            if (b1[at] == b0[came]).all():       # the piece's every cell moves, overlapping ones too
                codes[at] = code_of(dy, dx)
                confirmed += 1
    return codes, given, confirmed


def _drop(codes: np.ndarray, b0: np.ndarray, b1: np.ndarray, changed: np.ndarray) -> None:
    """The well after a line clear, or under the top-out curtain -> each changed row's drop, in place: the fewest rows n (1 .. DY) such that the
    whole row is the frame before's row n up (rows between two cleared ones drop less than the rows above both),
    given to the row's blocks that changed; a row no drop matches, and the cells the drop leaves empty, stay NEW
    (as the cells a piece leaves)."""
    rows, cols = slice(WELL_ROW, WELL_ROW + 20), slice(WELL_COL, WELL_COL + 10)
    well0, well1, moved = b0[rows, cols], b1[rows, cols], changed[rows, cols]
    values, counts = np.unique(well1, return_counts=True)
    blocks = (well1 != values[counts.argmax()]).any((2, 3))      # not the well's background
    for r in np.flatnonzero(moved.any(1)):
        for n in range(1, min(DY, r) + 1):
            if (well1[r] == well0[r - n]).all():
                codes[rows, cols][r, moved[r] & blocks[r]] = code_of(n, 0)
                break


def stream_motion(frames: np.ndarray, ram: np.ndarray) -> tuple[np.ndarray, int, int]:
    """A window's model frames [T, 256, 256] and RAM [T, 2048] -> each frame's motion codes [T, 32, 32] uint8
    (frames 0 and 1 UNKNOWN: the frame or the RAM before them is outside the window), and the moves the RAM gave
    and the pictures confirmed over the window."""
    t = len(frames)
    out = np.full((t, GRID, GRID), UNKNOWN, np.uint8)
    given = confirmed = 0
    for k in range(2, t):
        out[k], g, c = frame_motion(frames[k - 1], frames[k], ram[k - 2], ram[k - 1])
        given, confirmed = given + g, confirmed + c
    return out, given, confirmed
