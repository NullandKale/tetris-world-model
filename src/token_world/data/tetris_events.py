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
EVENTS = ("gravity (no buttons)", "falling alone", "soft drop", "shift", "rotate", "spawn", *CLEAR_NAMES.values(),
          "level up", "top-out", "game start")       # what events_at names; bit i of frame_events is EVENTS[i]
SETTLE = 90                         # frames after an event to look for its first visible change
NEXT_BOX = (slice(120, 152), slice(188, 228))      # the next-piece box (its redraw is random, never an event)
LOOKAHEAD = 40                      # RAM frames past an event that detection may read (line counts)


def event_slug(name: str) -> str:
    """An EVENTS name as a column name's part ("line clear: single" -> "line_clear_single"; the trainer's live_
    columns, the run viewer's live panels)."""
    return "".join(c if c.isalnum() else "_" for c in name).strip("_").replace("__", "_")


def lines_count(ram: np.ndarray) -> np.ndarray:
    """[..., 2048] -> the BCD lines counter."""
    lo, hi = ram[..., LINES].astype(int), ram[..., LINES + 1].astype(int)
    return (lo >> 4) * 10 + (lo & 15) + 100 * (hi & 15)


def changes(frames: np.ndarray) -> np.ndarray:
    """[T, 256, 256] model frames -> [T, 256, 256] bool: each pixel that differs from the frame before, outside
    the tick border and the next-piece box (frame 0: none)."""
    from token_world.data.model_frames import BAND
    out = np.zeros(frames.shape, bool)
    out[1:] = frames[1:] != frames[:-1]
    out[:, :BAND] = out[:, -BAND:] = False
    out[:, NEXT_BOX[0], NEXT_BOX[1]] = False
    return out


def frame_events(frames: np.ndarray, ram: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """One window's frames [T, 256, 256], RAM [T, 2048] and stream actions [T] (actions[t] took frame t to
    t + 1) -> [T] int64: bit i set on the picture where an EVENTS[i] event first shows (the first picture
    from e + 1 on that changes, within SETTLE; an event between RAM frames e - 1 and e: the picture shows the
    RAM of the frame before), as the scenario checks align them (diagnostics/tetris_scenarios.py). For the
    training data's own event scores (train_dynamics_ui.py live_event_scores); events whose picture falls past
    the window are not labelled."""
    t = len(frames)
    changed = changes(frames).reshape(t, -1).any(1)
    out = np.zeros(t, np.int64)
    for e in range(1, t - 1):
        names = events_at(ram, actions[:-1], e)
        if not names:
            continue
        shown = [f for f in range(e + 1, min(e + 1 + SETTLE, t)) if changed[f]]
        if shown:
            for name in names:
                out[shown[0]] |= 1 << EVENTS.index(name)
    return out


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
