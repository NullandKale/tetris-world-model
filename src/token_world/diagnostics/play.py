"""Pieces under play: do a model's blocks stay blocks, and clear only full rows, while real buttons move them?

The long dreams press no buttons and the line-clear check scores only real clears, so neither sees a dream
that erases cells or clears a row that is not full while someone plays (the browser page, 2026-10-07: "line
clears are overzealous, they happen when the line is not full, and the tetrominos morph"). Here the bot plays
live games (data/real_tetris.py); from a context in the middle of a game, FUTURE real frames follow with the
bot's real buttons (no top-out in them), and any model dreams them (diagnostics/worlds.py World.follow;
scripts/event_checks.py).

Scored on the well's cell grid (line_clears.well_cells), frame to frame, so a dream need not follow the real
game's exact pieces; per 100 frames:
- lost: cells that vanish while no row was full in the last SETTLE frames (a false clear, or a piece losing
  blocks). In the real game cells leave the well only in a clear's animation, after a full row.
- morph: frames whose filled-cell count changes by anything but 0 or +4 (a spawn), outside a clear (a piece
  gaining or losing blocks).
- full: full rows that never clear (frames with a full row, past SETTLE frames since it filled).
- size: pieces born too big or too small: across each stretch with the top rows filled or a row full (a spawn,
  a clear), the cell count changing by anything but four per piece less ten per cleared row.
Frames with anything in the top TOP rows, before or after, are left out of lost and morph: a piece spawns
there partly above the well and comes into view a row at a time (a vertical I over four rows), so the real
count changes too. Top-outs (from five or more full rows, the curtain, until the well is empty again) are left
out of full and size: the curtain fills every row, and the board empties for the menus and a new game.
And over the real piece's fall (from the window's start until the next piece shows, the frames its blocks are
all below the top TOP rows), as a share of those frames:
- shape: the dream's piece (its cells outside the stack: the cells filled in every context frame, so the
  falling piece is never stack) is the real piece's tetromino, in any rotation. A window where a piece locked
  during the context has no shape score (NaN). A piece that keeps 4 blocks but turns into another tetromino counts against it
  ("the tetrominos morph"); where it is does not matter.
The real frames score 0 on lost, morph and full and 1 on shape (score(play, play.future); tests/test_play.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from token_world.data.real_tetris import CONTEXT, RealTetris
from token_world.data.world_nes_tetris import GAME_OVER, PLAY_STATE
from token_world.diagnostics.line_clears import WELL, well_cells

FUTURE = 240
SETTLE = 30                   # frames after a full row in which cells may leave (the clear's animation and drop)
SKIP = 400                    # game frames between windows, so they come from different boards
PRESSED = 0.05                # a window's frames with a button down, at least (the bot has idle stretches)
STACK = 12                    # the well's filled cells at the context's end, at least (a board, not a new game)
TOP = 4                       # the well's top rows, where pieces spawn and come into view (an I is 4 tall)
FIELDS = ("lost", "morph", "full", "size", "shape")


@dataclass
class Play:
    context: np.ndarray           # [CONTEXT, 256, 256] uint8 model frames, mid-game
    actions: np.ndarray           # [CONTEXT] the button byte that produced each
    layers: dict | None           # the context's layered frames, each key [CONTEXT, ...] (data/nes_layers.py)
    future: np.ndarray            # [FUTURE, 256, 256] the real frames after
    future_actions: np.ndarray    # [FUTURE] the bot's real buttons


def plays(seed: int, layered: bool = True) -> Iterator[Play]:
    """Endless live stretches of play (buttons pressed, a stack in the well), at most 3 from a game, none
    reaching a top-out."""
    while True:
        real = RealTetris(seed, layered)
        seed += 1
        found, wait = 0, SKIP
        for _ in range(60_000):
            real.step(None)
            wait -= real.playing()
            if wait > 0 or len(real.frames) < CONTEXT or not real.playing():
                continue
            context, actions = np.stack(real.frames), np.array(real.actions, np.int64)
            layers = {k: np.stack([f[k] for f in real.layers]) for k in real.layers[0]} if layered else None
            future, acts, ended = [], [], False
            for _ in range(FUTURE):
                frame, action = real.step(None)
                future.append(frame)
                acts.append(action)
                ended |= real.observation.ram[PLAY_STATE] == GAME_OVER or not real.playing()
            if ended:
                break                                            # the game is ending: a fresh one
            empty = int(np.bincount(context[-1][WELL].ravel()).argmax())
            if np.mean(np.array(acts) != 0) < PRESSED or well_cells(context[-1], empty).sum() < STACK:
                wait = SKIP // 4
                continue
            yield Play(context, actions, layers, np.stack(future), np.array(acts, np.int64))
            found, wait = found + 1, SKIP
            if found == 3:
                break


def score(play: Play, dreamed: np.ndarray) -> dict[str, float]:
    """FIELDS over the dream (module docstring), its first frame against the context's last."""
    empty = int(np.bincount(play.context[-1][WELL].ravel()).argmax())
    cells = np.stack([well_cells(f, empty) for f in [play.context[-1], *dreamed]])        # [1 + T, 20, 10]
    return {**integrity(cells), "shape": _shape(play, cells[1:], empty)}


def integrity(cells: np.ndarray) -> dict[str, float]:
    """lost, morph and full per 100 frames (module docstring) from a dream's well cells [1 + T, 20, 10]: the
    last real frame's, then each dreamed frame's. Needs nothing of the real game after the start, so it scores
    a dream of any length (diagnostics/long_dream.py)."""
    count = cells.sum((1, 2)).astype(int)
    full = cells.all(2).any(1)                                                             # a full row
    since, run = np.zeros(len(cells), int), np.zeros(len(cells), int)     # frames since one; frames it has stood
    for t in range(len(cells)):
        since[t] = 0 if full[t] else (since[t - 1] + 1 if t else SETTLE + 1)
        run[t] = run[t - 1] + 1 if full[t] and t else int(full[t])
    change = np.diff(count)                                                                # frame t-1 -> t
    top = cells[:, :TOP].any((1, 2))
    quiet = (since[:-1] > SETTLE) & ~top[:-1] & ~top[1:]  # no full row in the SETTLE frames before, the top empty
    lost = np.maximum(-change, 0)[quiet].sum()
    morph = (quiet & (change != 0) & (change != 4)).sum()
    per = 100 / (len(cells) - 1)
    over = _over(cells.all(2).sum(1), count)
    stood = (run[1:] > SETTLE) & ~top[1:] & ~over[1:]     # not a topped-out board (the curtain fills every row)
    return {"lost": float(lost * per), "morph": float(morph * per), "full": float(stood.sum() * per),
            "size": float(_sizes(count, (since > SETTLE) & ~top, over) * per)}


def _over(full_rows: np.ndarray, count: np.ndarray) -> np.ndarray:
    """[T] the frames of a top-out: from the first with five or more full rows (the curtain: a clear is at most
    four) until the well is empty again (the menus, a new game)."""
    out = np.zeros(len(count), bool)
    on = False
    for t in range(len(count)):
        on = (on and count[t] > 0) or full_rows[t] >= 5
        out[t] = on
    return out


def _sizes(count: np.ndarray, settled: np.ndarray, over: np.ndarray) -> int:
    """size: across each stretch with the top rows filled or a row full (a spawn, a clear), the cell count must
    change by four per piece and minus ten per cleared row (4a - 10b, a piece or two, up to four rows); the
    changes it does not -> how many (pieces born too big or too small, which morph cannot see: they are
    already wrong when they leave the top rows). Stretches with a top-out in them (over) are left out."""
    allowed = {4 * a - 10 * b for a in range(3) for b in range(5)}
    frames = np.nonzero(settled)[0]
    gaps = np.nonzero(np.diff(frames) > 1)[0]                 # the last settled frame before each gap
    return sum(int(count[frames[g + 1]] - count[frames[g]]) not in allowed
               for g in gaps if not over[frames[g]:frames[g + 1] + 1].any())


def _shape(play: Play, dreamed: np.ndarray, empty: int) -> float:
    """shape (module docstring): dreamed cells [T, 20, 10] against the real piece's fall."""
    stack = np.stack([well_cells(f, empty) for f in play.context]).all(0)
    real = np.stack([well_cells(f, empty) for f in play.future]) & ~stack
    piece = None
    hits = []
    for t in range(len(real)):
        if real[t].sum() != 4:
            if piece is not None:
                break                                          # the next piece, or a lock that changed the stack
            continue
        if real[t, :TOP].any():
            continue
        piece = piece or {_trimmed(np.rot90(real[t], k)) for k in range(4)}
        hits.append(_trimmed(dreamed[t] & ~stack) in piece)
    return float(np.mean(hits)) if hits else float("nan")


def _trimmed(mask: np.ndarray) -> bytes | None:
    """A cell mask cut to its bounding box, as a key (where it is does not matter)."""
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return None
    box = np.ascontiguousarray(mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1])
    return bytes([*box.shape]) + np.packbits(box).tobytes()


def sheet(play: Play, dreams: dict[str, np.ndarray], palette: np.ndarray, every: int = 12) -> np.ndarray:
    """The well every `every` frames: the real game on top, a row per model's dream -> RGB [rows * 160, n * 84, 3]."""
    rows = []
    for frames in [play.future, *dreams.values()]:
        tiles = [np.pad(palette[frames[t][WELL]], ((0, 0), (0, 4), (0, 0))) for t in range(0, FUTURE, every)]
        rows.append(np.concatenate(tiles, 1))
    return np.concatenate(rows, 0).astype(np.uint8)
