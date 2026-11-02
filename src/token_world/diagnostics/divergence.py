"""Where does a button dream first go wrong, and why? (test 2 of output/test_plan_2026-10-30.md)

A dream is followed against the real game on the well's cell grid (line_clears.well_cells, the dream's most
likely colours), frame by frame, with the real presses fed in. The first frame whose cells differ is its first
divergence; the first one the dream does not come back from within LASTING frames is its lasting divergence.
Each is labelled from the real RAM around that frame (the picture at t shows the RAM of frame t - 1,
tetris_events.frame_events) by matching the dream's cells against what the real piece could have done instead:

- the real piece's moves: every orientation of its type, a column either side, up to two rows lower, on the
  real stack. A match names what the dream did differently: a turn not drawn, drawn the wrong way, or made
  with no press; a shift not drawn, the wrong way, or made on its own; a drop late or early (gravity, or soft
  drop when Down is held);
- the frame before (the dream did not change when the real game did): a spawn, lock or clear drawn late;
- anything else: a spawn of another piece, the stack changed wrongly (lock, clear), or the piece broke (cells
  gained or lost with no event to explain it).

The fair line-clear tests (clear_scores) start dreams a few frames before a real row fills, with the real
presses: does the dream fill the same rows, clear them, and end on the real board? clear_given_full asks of
any dream: once a row is full, does it clear?
"""
from __future__ import annotations

import numpy as np

from token_world.data.tetris_bot import DOWN, IDS_OF, ORIENTATIONS, PIECE, TYPE_OF, X, Y, settled_board
from token_world.data.tetris_events import FALLING, STATE

LASTING = 16                # a divergence the dream does not come back from within this many frames lasts
CLEAR_WITHIN = 40           # frames a full row may stand before it must be gone (the real animation: ~18-22)
PIECE_EVENTS = ("turn", "shift", "drop")


def piece_grid(orientation: int, x: int, y: int) -> np.ndarray | None:
    """A piece's cells on the 20 x 10 well [20, 10] bool (cells above the well are not drawn), or None when a
    cell is off the well's sides or bottom."""
    grid = np.zeros((20, 10), bool)
    for dy, dx in ORIENTATIONS[orientation]:
        r, c = y + dy, x + dx
        if not 0 <= c < 10 or r >= 20:
            return None
        if r >= 0:
            grid[r, c] = True
    return grid


def ram_grid(ram: np.ndarray) -> np.ndarray:
    """The well as the RAM has it [20, 10] bool: the settled stack, plus the falling piece while one falls."""
    grid = settled_board(ram).copy()
    if ram[STATE] == FALLING and int(ram[PIECE]) in ORIENTATIONS:
        piece = piece_grid(int(ram[PIECE]), int(ram[X]), int(ram[Y]))
        if piece is not None:
            grid |= piece
    return grid


def falling(ram: np.ndarray) -> bool:
    return bool(ram[STATE] == FALLING and int(ram[PIECE]) in ORIENTATIONS)


def real_event(r0: np.ndarray, r1: np.ndarray, press: int) -> str:
    """What the real game did between RAM frames r0 and r1 (press: the button byte that took r0 to r1): turn,
    shift, drop, spawn, stack (a lock or a clear: the settled board changed, or the play state left falling),
    or none."""
    if falling(r0) and falling(r1):
        o0, o1 = int(r0[PIECE]), int(r1[PIECE])
        if TYPE_OF[o0] != TYPE_OF[o1]:
            return "spawn"
        if o0 != o1:
            return "turn"
        if int(r0[X]) != int(r1[X]):
            return "shift"
        if int(r0[Y]) != int(r1[Y]):
            return "drop"
        return "none"
    if not falling(r0) and falling(r1):
        return "spawn"
    if (settled_board(r0) != settled_board(r1)).any() or falling(r0) != falling(r1):
        return "stack"
    return "none"


def _turn(o_from: int, o_to: int) -> int:
    """+1 / -1: o_to is the next / previous orientation of o_from's type; 0 the same; 2 otherwise (a half turn,
    or either way on a 2-orientation piece)."""
    ids = IDS_OF[TYPE_OF[o_from]]
    d = (ids.index(o_to) - ids.index(o_from)) % len(ids)
    if d == 0:
        return 0
    if len(ids) == 2:
        return 2
    return 1 if d == 1 else -1 if d == len(ids) - 1 else 2


def dream_piece(dreamed: np.ndarray, r1: np.ndarray) -> tuple[int, int, int] | None:
    """The real piece's type placed where the dream has it: (orientation, x, y) whose cells on r1's settled
    board are exactly the dream's cells, searched near r1's piece (columns +-2, rows -1..+3), or None."""
    if not falling(r1):
        return None
    board = settled_board(r1)
    o1, x1, y1 = int(r1[PIECE]), int(r1[X]), int(r1[Y])
    for o in IDS_OF[TYPE_OF[o1]]:
        for x in range(x1 - 2, x1 + 3):
            for y in range(y1 - 1, y1 + 4):
                grid = piece_grid(o, x, y)
                if grid is not None and not (grid & board).any() and ((grid | board) == dreamed).all():
                    return o, x, y
    return None


def any_piece(dreamed: np.ndarray, board: np.ndarray) -> tuple[int, int, int] | None:
    """Some whole tetromino, any kind, orientation and place, that with the settled board makes exactly the
    dream's cells: (orientation, x, y), or None (the dream's cells are no board plus one piece)."""
    extra = dreamed & ~board
    if (board & ~dreamed).any() or not 1 <= extra.sum() <= 4:
        return None
    rows, cols = np.nonzero(extra)
    for o in ORIENTATIONS:
        for x in range(cols.min() - 2, cols.max() + 3):
            for y in range(rows.min() - 1, rows.max() + 3):
                grid = piece_grid(o, x, y)
                if grid is not None and (grid == extra).all():
                    return o, x, y
    return None


def label(dreamed: np.ndarray, before: np.ndarray, r0: np.ndarray, r1: np.ndarray, press: int) -> str:
    """The cause of a divergence: the dream's cells [20, 10] where the real picture differs; the cells of the
    frame before (dream and real agreed there); the RAM the two pictures show (r0, then r1); the button byte
    that took r0 to r1."""
    event = real_event(r0, r1, press)
    if (dreamed == before).all() and event != "none":
        return f"{event}: not drawn (late or missed)"
    placed = dream_piece(dreamed, r1) if falling(r0) and falling(r1) and event in ("none", *PIECE_EVENTS) else None
    if placed is not None:
        o, x, y = placed
        o0, x0, y0 = int(r0[PIECE]), int(r0[X]), int(r0[Y])
        o1, x1, y1 = int(r1[PIECE]), int(r1[X]), int(r1[Y])
        wrong = []
        if o != o1:
            if o1 != o0 and o == o0:
                wrong.append("turn not drawn")
            elif o1 != o0 and _turn(o0, o) == -_turn(o0, o1) and _turn(o0, o1) in (1, -1):
                wrong.append("turn the wrong way")
            elif o1 == o0:
                wrong.append("turn with no press" if not press & 3 else "turn drawn (none in RAM)")
            else:
                wrong.append("turn garbled")
        if x != x1:
            if x1 != x0 and x == x0:
                wrong.append("shift not drawn")
            elif x1 != x0 and x == 2 * x0 - x1:
                wrong.append("shift the wrong way")
            else:
                wrong.append("shift on its own")
        if y != y1:
            how = "soft drop" if press & DOWN else "gravity"
            wrong.append(f"{how} late" if y < y1 else f"{how} early")
        return " + ".join(wrong) if wrong else "piece right, stack wrong"
    if event == "spawn":
        return "spawn: another piece or place"
    if event == "stack":
        return "stack: lock or clear drawn wrong"
    other = any_piece(dreamed, settled_board(r1)) if falling(r1) else None
    if other is not None:                       # a whole piece, not the real one's moves: another kind seen late
        same = TYPE_OF[other[0]] == TYPE_OF[int(r1[PIECE])]
        return "piece elsewhere (same kind)" if same else "spawn: another piece (seen late)"
    gained = int(dreamed.sum()) - int(before.sum())
    return f"piece broke ({'cells gained' if gained > 0 else 'cells lost' if gained < 0 else 'cells moved'}" \
           f", real: {event})"


def divergences(dreamed: np.ndarray, real: np.ndarray) -> tuple[int | None, int | None]:
    """A dream's cells and the real ones [T, 20, 10] -> (its first divergence, its first lasting divergence: a
    differing frame with no agreeing one in the LASTING frames from it), frame indices or None."""
    differ = (dreamed != real).any((1, 2))
    first = int(np.argmax(differ)) if differ.any() else None
    lasting = None
    for t in np.nonzero(differ)[0]:
        if differ[t:t + LASTING].all():
            lasting = int(t)
            break
    return first, lasting


def explain(t: int, dreamed: np.ndarray, real: np.ndarray, ram: np.ndarray, presses: np.ndarray,
            start_ram: np.ndarray, start_cells: np.ndarray, start_press: int) -> str:
    """The label of the divergence at frame t of a dream (cells [T, 20, 10]) against the real cells and RAM
    [T, 2048] (ram[t]: after the frame-t step; the picture at t shows ram[t - 1]) and presses [T] (the byte
    that produced each frame: presses[t] took ram[t - 1] to ram[t]). The context's end: start_ram [2, 2048],
    its last two RAM frames; start_cells, its last picture's cells; start_press, its last frame's byte. The
    frame before t is the real one (they agreed there)."""
    full = np.concatenate([start_ram, ram])                     # full[k] is ram[k - 2]
    r0, r1 = full[t], full[t + 1]                               # ram[t - 2], ram[t - 1]: pictures t - 1 and t
    before = real[t - 1] if t > 0 else start_cells
    press = int(presses[t - 1]) if t > 0 else start_press     # the byte that took ram[t - 2] to ram[t - 1]
    return label(dreamed[t], before, r0, r1, press)


def full_rows(cells: np.ndarray) -> np.ndarray:
    """[T, 20, 10] -> [T] the number of full rows."""
    return cells.all(-1).sum(-1)


def clear_given_full(cells: np.ndarray) -> tuple[int, int]:
    """A dream's (or the real game's) cells [T, 20, 10] -> (rows that became full with CLEAR_WITHIN frames left
    to watch, how many of them cleared: within CLEAR_WITHIN frames no row is full and the cell count fell)."""
    full = full_rows(cells)
    count = cells.sum((1, 2))
    filled = cleared = 0
    for t in range(1, len(cells) - CLEAR_WITHIN):
        if full[t] > full[t - 1]:
            filled += 1
            after = range(t + 1, t + 1 + CLEAR_WITHIN)
            if any(full[u] == 0 and count[u] < count[t] for u in after):
                cleared += 1
    return filled, cleared


def clear_scores(dreamed: np.ndarray, real: np.ndarray, fill: int) -> dict[str, float]:
    """A dream started before a real row fills (at frame `fill` of the real cells [T, 20, 10]):
    - filled: the dream has the real full rows at `fill` (same rows full);
    - exact_at_fill: its whole well equals the real one there;
    - cleared: within CLEAR_WITHIN frames of its first full row near `fill` (from 4 frames before to 8 after),
      no row is full and it has fewer cells;
    - exact_end: its well equals the real one at the last frame (after the real clear);
    - stuck: a full row still stands at the last frame."""
    rows_real, rows_mine = real[fill].all(-1), dreamed[fill].all(-1)
    full = full_rows(dreamed)
    count = dreamed.sum((1, 2))
    near = [u for u in range(max(fill - 4, 0), min(fill + 9, len(dreamed))) if full[u] > 0]
    cleared = 0.0
    if near:
        u = near[0]
        cleared = float(any(full[v] == 0 and count[v] < count[u] for v in range(u + 1, min(u + 1 + CLEAR_WITHIN,
                                                                                           len(dreamed)))))
    return {"filled": float((rows_real == rows_mine).all() and rows_real.any()),
            "exact_at_fill": float((real[fill] == dreamed[fill]).all()),
            "cleared": cleared, "exact_end": float((real[-1] == dreamed[-1]).all()),
            "stuck": float(full[-1] > 0)}
