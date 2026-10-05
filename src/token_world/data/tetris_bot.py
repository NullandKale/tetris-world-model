"""A RAM-driven Tetris player for the live data streams: perfect / random / idle brains.

Replaces llm-thing19's vision-based PerfectPolicy, which located the falling
piece from pixels (the RAM addresses for it "hadn't been found") and cleared
under 2 lines per 1,000 frames at level 0. Everything here comes from RAM:

- 0x40 / 0x41: the falling piece's column / row (its rotation centre);
- 0x42: its orientation, one of 19 (ORIENTATIONS); 0xBF: the next piece;
- 0x400: the settled 20 x 10 board, 0xEF = empty.

On each new piece the perfect brain searches every orientation and column of
the current piece, plus the next piece in its best reply, and scores boards
with the classic weights (aggregate height, lines, holes, bumpiness). It then
taps A until the orientation byte matches (rotation registers on press edges),
taps Left/Right until the column matches, and holds Down. The brain switcher
keeps llm-thing19's in-game mix: perfect 60% (2,400-6,000 frames), random 25%
and idle 15% (150-500 frames); game lengths, giving up and well-building are
described in BrainSwitcher.
"""
from __future__ import annotations

import numpy as np

A, B, SELECT, START, UP, DOWN, LEFT, RIGHT = 1, 2, 4, 8, 16, 32, 64, 128
X, Y, PIECE, NEXT, BOARD = 0x40, 0x41, 0x42, 0xBF, 0x400
EMPTY = 0xEF

# (row, column) offsets of the 4 cells from (Y, X), per orientation id (NES Tetris orientationTable).
ORIENTATIONS = {
    0x00: ((-1, 0), (0, -1), (0, 0), (0, 1)),     # T up
    0x01: ((-1, 0), (0, 0), (0, 1), (1, 0)),      # T right
    0x02: ((0, -1), (0, 0), (0, 1), (1, 0)),      # T down (spawn)
    0x03: ((-1, 0), (0, -1), (0, 0), (1, 0)),     # T left
    0x04: ((-1, 0), (0, 0), (1, -1), (1, 0)),     # J left
    0x05: ((-1, -1), (0, -1), (0, 0), (0, 1)),    # J up
    0x06: ((-1, 0), (-1, 1), (0, 0), (1, 0)),     # J right
    0x07: ((0, -1), (0, 0), (0, 1), (1, 1)),      # J down (spawn)
    0x08: ((0, -1), (0, 0), (1, 0), (1, 1)),      # Z horizontal (spawn)
    0x09: ((-1, 1), (0, 0), (0, 1), (1, 0)),      # Z vertical
    0x0A: ((0, -1), (0, 0), (1, -1), (1, 0)),     # O
    0x0B: ((0, 0), (0, 1), (1, -1), (1, 0)),      # S horizontal (spawn)
    0x0C: ((-1, 0), (0, 0), (0, 1), (1, 1)),      # S vertical
    0x0D: ((-1, 0), (0, 0), (1, 0), (1, 1)),      # L right
    0x0E: ((0, -1), (0, 0), (0, 1), (1, -1)),     # L down (spawn)
    0x0F: ((-1, -1), (-1, 0), (0, 0), (1, 0)),    # L left
    0x10: ((-1, 1), (0, -1), (0, 0), (0, 1)),     # L up
    0x11: ((-2, 0), (-1, 0), (0, 0), (1, 0)),     # I vertical
    0x12: ((0, -2), (0, -1), (0, 0), (0, 1)),     # I horizontal (spawn)
}
TYPE_OF = {i: t for t, ids in {"T": (0, 1, 2, 3), "J": (4, 5, 6, 7), "Z": (8, 9), "O": (10,), "S": (11, 12),
                               "L": (13, 14, 15, 16), "I": (17, 18)}.items() for i in ids}
IDS_OF = {t: tuple(i for i in ORIENTATIONS if TYPE_OF[i] == t) for t in set(TYPE_OF.values())}


def settled_board(ram: np.ndarray) -> np.ndarray:
    """[20, 10] bool: the settled stack (the falling piece is not included)."""
    return ram[BOARD:BOARD + 200].reshape(20, 10) != EMPTY


def cells(orientation: int, x: int, y: int) -> list[tuple[int, int]]:
    return [(y + dy, x + dx) for dy, dx in ORIENTATIONS[orientation]]


def _moves(piece: str):
    """Every (orientation, column) of a piece that keeps it on the board: cell row offsets [K, 4],
    absolute cell columns [K, 4], and the moves themselves."""
    dys, columns, moves = [], [], []
    for orientation in IDS_OF[piece]:
        dy = np.array([r for r, _ in ORIENTATIONS[orientation]])
        dx = np.array([c for _, c in ORIENTATIONS[orientation]])
        for x in range(-dx.min(), 10 - dx.max()):
            dys.append(dy)
            columns.append(x + dx)
            moves.append((orientation, x))
    return np.array(dys), np.array(columns), moves


MOVES = {piece: _moves(piece) for piece in IDS_OF}


def _tops(boards: np.ndarray) -> np.ndarray:
    """[..., 10] row of each column's highest filled cell, 20 for an empty column."""
    return np.where(boards.any(-2), boards.argmax(-2), 20)


def place_all(boards: np.ndarray, piece: str):
    """Drop every move of a piece straight down onto each board.

    boards [B, 20, 10] -> (boards [B, K, 20, 10] after locking and clearing,
    lines cleared [B, K], ok [B, K]; False where the piece locks above the top).
    Dropped from above, a piece rests on the column tops alone: its row is the
    largest y that keeps every cell above its column's top.
    """
    dy, columns, _ = MOVES[piece]
    rows = (_tops(boards)[:, columns] - 1 - dy).min(-1)[..., None] + dy          # [B, K, 4]
    ok = rows.min(-1) >= 0
    b, k = ok.shape
    new = np.repeat(boards[:, None], k, 1)
    new[np.arange(b)[:, None, None], np.arange(k)[None, :, None], np.maximum(rows, 0), columns] = True
    full = new.all(-1)
    new[full] = False
    order = np.argsort(~full, -1, kind="stable")                                 # cleared rows move to the top
    return np.take_along_axis(new, order[..., None], -2), full.sum(-1), ok


WELL = 9                                            # the column a well-building brain keeps open


def scores(boards: np.ndarray, lines: np.ndarray, well: bool = False) -> np.ndarray:
    """Classic Tetris heuristic: aggregate height, lines cleared, holes, bumpiness.

    well: build for tetrises instead. Cells in column WELL cost 2 each, a
    4-line clear earns 8 and a 3-line clear 3, and 1-2 line clears cost 1 per
    line, so the stack grows on the other columns until an I piece fills the
    well. Bumpiness ignores the well's edge.
    """
    heights = 20 - _tops(boards)
    total = heights.sum(-1)
    holes = total - boards.sum((-2, -1))                                         # empty cells under the tops
    if not well:
        bumpiness = np.abs(np.diff(heights, axis=-1)).sum(-1)
        return -0.510066 * total + 0.760666 * lines - 0.35663 * holes - 0.184483 * bumpiness
    bumpiness = np.abs(np.diff(heights[..., :WELL], axis=-1)).sum(-1)
    bonus = np.where(lines >= 4, 8.0, np.where(lines == 3, 3.0, -1.0 * lines))
    return (-0.510066 * total - 0.35663 * holes - 0.184483 * bumpiness
            - 2.0 * boards[..., WELL].sum(-1) + bonus)


def best_placement(board: np.ndarray, piece: str, next_piece: str | None, well: bool = False):
    """(orientation, column) of the best placement, looking one piece ahead, or None if nothing fits."""
    first, lines, ok = (a[0] for a in place_all(board[None], piece))
    score = scores(first, lines, well)
    if next_piece is not None:
        second, more, reply_ok = place_all(first, next_piece)
        replies = np.where(reply_ok, scores(second, lines[:, None] + more, well), -np.inf)
        score = np.where(reply_ok.any(1), replies.max(1), score - 100.0)
    if not ok.any():
        return None
    return MOVES[piece][2][int(np.where(ok, score, -np.inf).argmax())]


PATIENT_FRAMES = (30, 180)                          # no-button wait of a patient piece, then soft drop
SPIN_HEIGHT = 8                                     # a spinning piece needs a stack at most this many rows high
SPIN_RATE = 1 / 8                                   # chance per waiting frame that a spinning piece rotates


class PerfectBrain:
    """Plans each new piece from RAM, then rotates, shifts and soft-drops it into place.

    well: build for tetrises (scores). patience: the chance that a piece, once
    lined up, is left to fall by gravity for PATIENT_FRAMES frames before it is
    soft-dropped (drawn per piece), so the data shows pieces falling on their
    own at every level without slow levels taking ~860 frames a piece. spin: the
    chance that a patient piece over a low stack (SPIN_HEIGHT rows or fewer)
    spins while it waits, A or B at random (SPIN_RATE a frame), so the data shows
    rotations in every orientation as a player makes them, not only the few on
    the way to a planned one; when the wait ends it turns back and drops. Call
    reset() when it takes over mid-game: it replans only when a new piece
    spawns, so a stale target from an earlier piece would otherwise steer the
    falling one.
    """

    def __init__(self, well: bool = False, patience: float = 0.0, spin: float = 0.0):
        self.well, self.patience, self.spin = well, patience, spin
        self.reset()

    def reset(self) -> None:
        self.target = None
        self.last_y = None
        self.last_action = 0
        self.wait = 0
        self.waiting = self.spinning = False

    def choose(self, ram: np.ndarray, orientation: int, rng: np.random.Generator):
        return best_placement(settled_board(ram), TYPE_OF[orientation], TYPE_OF.get(int(ram[NEXT])), self.well)

    def __call__(self, ram: np.ndarray, rng: np.random.Generator) -> int:
        orientation, x, y = int(ram[PIECE]), int(ram[X]), int(ram[Y])
        if orientation not in ORIENTATIONS:
            self.last_action = 0
            return 0
        if self.target is None or self.last_y is None or y < self.last_y:     # a new piece spawned
            self.target = self.choose(ram, orientation, rng)
            self.wait = int(rng.integers(*PATIENT_FRAMES)) if rng.random() < self.patience else 0
            self.waiting = False
            board = settled_board(ram)
            height = 20 - int(board.any(1).argmax()) if board.any() else 0
            self.spinning = (self.spin > 0 and self.wait > 0 and height <= SPIN_HEIGHT
                             and rng.random() < self.spin)
            self.last_y = y
            self.last_action = 0
            return 0                                         # soft drop only re-arms after Down is released
        self.last_y = y
        action = DOWN
        if self.target is not None:
            goal, column = self.target
            if not self.waiting and orientation == goal and x == column and self.wait > 0:
                self.waiting = True                              # lined up, patient: let gravity work
            if self.waiting:
                self.wait -= 1
                self.waiting = self.wait > 0
                action = int(rng.choice((A, B))) if self.spinning and rng.random() < SPIN_RATE else 0
            elif orientation != goal:
                action = A
            elif x < column:
                action = RIGHT
            elif x > column:
                action = LEFT
        if action != DOWN and self.last_action & action:                     # release: moves act on press edges
            action = 0
        self.last_action = action
        return action


ROTATE_HOLD = (2, 7)                                # frames a holding brain keeps A or B down per turn


class HoldingBrain(PerfectBrain):
    """Plans each piece like the perfect brain, then plays like a patient person who holds buttons: A or B
    held a few frames per turn (a turn registers on the press; released a frame between turns), Left or
    Right held until the piece reaches its column (the game's autorepeat moves it: a cell on the press,
    then one every 6 frames after 16), and never Down: every piece falls by gravity all the way, lands,
    locks and is followed by a spawn untouched. The other brains tap and soft-drop, so held shifts (0.45
    per 1,000 frames), landings by gravity (1.9) and untouched falls over 180 frames (none) were rare
    in training (event census 2026-10-02, docs/guides/dynamics.md)."""

    def reset(self) -> None:
        super().reset()
        self.turn = 0                                   # frames left holding the current turn's button

    def __call__(self, ram: np.ndarray, rng: np.random.Generator) -> int:
        orientation, x, y = int(ram[PIECE]), int(ram[X]), int(ram[Y])
        if orientation not in ORIENTATIONS:
            self.last_action = self.turn = 0
            return 0
        if self.target is None or self.last_y is None or y < self.last_y:     # a new piece spawned
            self.target = self.choose(ram, orientation, rng)
            self.last_y, self.last_action, self.turn = y, 0, 0
            return 0
        self.last_y = y
        action = 0
        if self.target is not None:
            goal, column = self.target
            if orientation != goal or self.turn > 0:
                if self.turn > 0:                       # keep holding this turn's button
                    self.turn -= 1
                    action = self.last_action & (A | B)
                elif not self.last_action & (A | B):    # press for the next turn (after a release)
                    action = int(rng.choice((A, B)))
                    self.turn = int(rng.integers(*ROTATE_HOLD)) - 1
            elif x < column:
                action = RIGHT
            elif x > column:
                action = LEFT
        self.last_action = action
        return action


class RecklessBrain(PerfectBrain):
    """Careless fast play: drives each piece like the perfect brain, but to a random legal
    placement, soft-dropping it. Tops out in a few hundred frames at any level."""

    def choose(self, ram: np.ndarray, orientation: int, rng: np.random.Generator):
        piece = TYPE_OF[orientation]
        ok = place_all(settled_board(ram)[None], piece)[2][0]
        legal = np.flatnonzero(ok)
        return MOVES[piece][2][int(rng.choice(legal))] if len(legal) else None


class RandomBrain:
    """Chaotic play: a random Tetris button held for 4-20 frames at a time."""

    CHOICES = (0, LEFT, RIGHT, DOWN, A, B)

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.action, self.left = 0, 0

    def __call__(self, ram, rng):
        if self.left <= 0:
            self.action = int(rng.choice(self.CHOICES))
            self.left = int(rng.integers(4, 21))
        self.left -= 1
        return self.action


class IdleBrain:
    """Never presses anything: pieces fall straight down and stack up."""

    def reset(self) -> None:
        pass

    def __call__(self, ram, rng) -> int:
        return 0


class BrainSwitcher:
    """One game's player, long or short, then giving up.

    Rare events (top-out, curtain, menus, game start) come once per game and
    everything else scales with frames, so most games are short while about
    half the frames come from long ones (LONG_SHARE of games; the session
    decides before the menu, since it sets the start level):
    - long games (LONG frames): the perfect brain only, from start levels
      0-9, so they climb through every palette with a level-up every 10 lines
      (NES Tetris delays the first level-up to 100 lines from start levels
      10-19) and reach levels 20+; idle, random or holding stretches at those
      speeds would top the game out early (holding: autorepeat waits 16 frames,
      and from level 15 a piece falls the well in ~40);
    - short games (SHORT frames): perfect / holding / random / idle at random intervals,
      from any start level 0-19: the variety and the sloppy boards.
    When its length runs out the player gives up, reckless (random placements,
    soft-dropped) or random, until the stack tops out. The perfect brain leaves
    PATIENCE of its pieces to fall by gravity, and SPIN of those spin while they
    fall if the stack is low. In WELL_SHARE of games
    the perfect brain builds for tetrises. Every brain resets when it takes over.
    """

    INTERVALS = {"perfect": (2400, 6000), "holding": (2400, 6000), "random": (150, 500), "idle": (150, 500),
                 "reckless": (150, 500)}
    WEIGHTS = {"perfect": 0.45, "holding": 0.25, "random": 0.20, "idle": 0.10}
    LONG_WEIGHTS = {"perfect": 1.0}
    GIVE_UP = {"reckless": 0.75, "random": 0.25}
    SHORT, LONG, LONG_SHARE = (1500, 3500), (20_000, 45_000), 0.08
    SHORT_START_LEVELS, LONG_START_LEVELS = 20, 10
    WELL_SHARE = 0.5
    PATIENCE = 0.3
    SPIN = 0.5

    def __init__(self, rng: np.random.Generator, long: bool):
        self.brains = {"perfect": PerfectBrain(well=rng.random() < self.WELL_SHARE, patience=self.PATIENCE,
                                               spin=self.SPIN),
                       "holding": HoldingBrain(well=rng.random() < self.WELL_SHARE),
                       "random": RandomBrain(),
                       "idle": IdleBrain(), "reckless": RecklessBrain()}
        self.mix = self.LONG_WEIGHTS if long else self.WEIGHTS
        lo, hi = self.LONG if long else self.SHORT
        self.game_left = int(rng.integers(lo, hi + 1))
        self._switch(rng)

    def _switch(self, rng: np.random.Generator) -> None:
        names = list(self.mix)
        probs = np.array([self.mix[n] for n in names])
        self.current = str(rng.choice(names, p=probs / probs.sum()))
        self.brains[self.current].reset()
        lo, hi = self.INTERVALS[self.current]
        self.left = int(rng.integers(lo, hi + 1))

    def __call__(self, ram: np.ndarray, rng: np.random.Generator) -> int:
        self.game_left -= 1
        if self.game_left == 0:                              # give up: play badly until the stack tops out
            self.mix = self.GIVE_UP
            self._switch(rng)
        self.left -= 1
        if self.left <= 0:
            self._switch(rng)
        return self.brains[self.current](ram, rng)
