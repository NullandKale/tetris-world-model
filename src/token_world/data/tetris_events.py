"""Tetris events read from console RAM.

events_at names what happens between two RAM frames (a piece falling,
shifting or rotating, spawns, line clears, level ups, top-outs, new games).
The scenario checks (diagnostics/tetris_scenarios.py) score the model on
them.

Measured on World NES (docs/guides/dynamics.md): play states at 0x48 are 1
falling, 2 lock, 3 check rows, 4 line-clear animation (~18 frames), then 5-8
update counters and spawn; 10 top-out. Game mode 0xC0 is 4 in a game; level
0x44; lines BCD at 0x50-0x51.
"""
from __future__ import annotations

import numpy as np

from token_world.data.tetris_bot import DOWN, LEFT, ORIENTATIONS, PIECE, RIGHT, TYPE_OF, X, Y

STATE, LEVEL, MODE, LINES = 0x48, 0x44, 0xC0, 0x50
FALLING, CHECK_ROWS, CLEARING, TOP_OUT, IN_GAME = 1, 3, 4, 10, 4
CLEAR_NAMES = {1: "line clear: single", 2: "line clear: double", 3: "line clear: triple", 4: "line clear: tetris"}
ALONE = 16                          # "falling alone": no buttons for this many frames while a piece falls
LOOKAHEAD = 40                      # RAM frames past an event that detection may read (line counts)

def lines_count(ram: np.ndarray) -> np.ndarray:
    """[..., 2048] -> the BCD lines counter."""
    lo, hi = ram[..., LINES].astype(int), ram[..., LINES + 1].astype(int)
    return (lo >> 4) * 10 + (lo & 15) + 100 * (hi & 15)


def events_at(ram: np.ndarray, actions: np.ndarray, e: int) -> list[str]:
    """Events that happen between RAM frames e - 1 and e.

    ram [L, 2048] and actions [L - 1] (actions[t] took frame t to t + 1) of one
    continuous stream. Line clears read up to LOOKAHEAD frames ahead for the
    number of lines.
    """
    prev, cur = ram[e - 1], ram[e]
    out = []
    if prev[MODE] != IN_GAME and cur[MODE] == IN_GAME:
        out.append("game start")
    if cur[MODE] != IN_GAME:
        return out
    if prev[STATE] == FALLING and cur[STATE] == FALLING and prev[PIECE] in ORIENTATIONS and cur[PIECE] in ORIENTATIONS:
        dx, dy = int(cur[X]) - int(prev[X]), int(cur[Y]) - int(prev[Y])
        pressed = int(actions[e - 1])
        if prev[PIECE] == cur[PIECE]:
            if dy == 1 and dx == 0:
                if pressed == 0:
                    out.append("gravity (no buttons)")
                elif pressed & DOWN:
                    out.append("soft drop")
            elif dy == 0 and abs(dx) == 1 and pressed & (LEFT | RIGHT):
                out.append("shift")
        elif TYPE_OF[prev[PIECE]] == TYPE_OF[cur[PIECE]] and dx == 0 and dy == 0:
            out.append("rotate")
        ahead = ram[e:e + ALONE]
        if (e >= 2 and actions[e - 2] != 0 and len(ahead) == ALONE and not np.any(actions[e - 1:e - 1 + ALONE])
                and (ahead[:, STATE] == FALLING).all() and (ahead[:, PIECE] == cur[PIECE]).all()):
            out.append("falling alone")                     # the player lets go; gravity alone moves the piece
    if prev[STATE] != FALLING and cur[STATE] == FALLING and cur[PIECE] in ORIENTATIONS:
        out.append("spawn")
    if prev[STATE] == CHECK_ROWS and cur[STATE] == CLEARING:
        counts = lines_count(ram[e:e + LOOKAHEAD])
        cleared = int(counts.max() - counts[0])
        if cleared in CLEAR_NAMES:
            out.append(CLEAR_NAMES[cleared])
    if prev[MODE] == IN_GAME and int(cur[LEVEL]) == int(prev[LEVEL]) + 1:
        out.append("level up")
    if prev[STATE] != TOP_OUT and cur[STATE] == TOP_OUT:
        out.append("top-out")
    return out
