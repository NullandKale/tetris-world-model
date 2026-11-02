"""A RAM-driven Tetris player for the live data streams: perfect / holding / person / random / idle brains.

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
mixes the brains within a game (BrainSwitcher.WEIGHTS: perfect, holding and person
2,400-6,000 frames, random and idle 150-500); game lengths, giving up and
well-building are described in BrainSwitcher.
"""
from __future__ import annotations

import numpy as np

A, B, SELECT, START, UP, DOWN, LEFT, RIGHT = 1, 2, 4, 8, 16, 32, 64, 128
X, Y, PIECE, NEXT, BOARD, LEVEL = 0x40, 0x41, 0x42, 0xBF, 0x400, 0x44
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


def fits(board: np.ndarray, orientation: int, x: int, y: int) -> bool:
    """The piece at (orientation, x, y) is on the board (rows above the top count as free) and on no settled cell."""
    return all(0 <= c <= 9 and r <= 19 and (r < 0 or not board[r, c]) for r, c in cells(orientation, x, y))


def rest_row(board: np.ndarray, orientation: int, x: int, y: int) -> int:
    """The row the piece at (orientation, x, y) comes to rest on, falling straight down from there."""
    while fits(board, orientation, x, y + 1):
        y += 1
    return y


def top_row(board: np.ndarray, orientation: int, x: int) -> int:
    """The row a piece dropped from above comes to rest on: on the column tops (place_all), not under an overhang."""
    dy = np.array([r for r, _ in ORIENTATIONS[orientation]])
    dx = np.array([c for _, c in ORIENTATIONS[orientation]])
    return int((_tops(board[None])[0][x + dx] - 1 - dy).min())


def tucks(board: np.ndarray, piece: str):
    """Every placement under an overhang one column over: the piece dropped down column `start` to its rest,
    then shifted one column into a gap the column tops hide, and dropped from there. -> [(orientation, start,
    column, rest row, the board after locking and clearing, lines cleared)]."""
    out = []
    on_board = set(MOVES[piece][2])
    for orientation, start in MOVES[piece][2]:
        low = top_row(board, orientation, start)
        if min(r for r, _ in cells(orientation, start, low)) < 0:
            continue                                     # would lock above the top
        for column in (start - 1, start + 1):
            if ((orientation, column) not in on_board or low <= top_row(board, orientation, column)
                    or not fits(board, orientation, column, low)):
                continue
            rest = rest_row(board, orientation, column, low)
            new = board.copy()
            for r, c in cells(orientation, column, rest):
                new[r, c] = True
            full = new.all(1)
            new = np.concatenate([np.zeros((full.sum(), 10), bool), new[~full]])
            out.append((orientation, start, column, rest, new, int(full.sum())))
    return out


def best_placement(board: np.ndarray, piece: str, next_piece: str | None, well: bool = False,
                   tuck: bool = False):
    """(orientation, column, start) of the best placement, looking one piece ahead, or None if nothing fits.
    start is None for a straight drop; with tuck, a placement under an overhang (tucks) may win, reached by
    dropping down column start and shifting into the gap at the bottom."""
    first, lines, ok = (a[0] for a in place_all(board[None], piece))
    moves = [(o, c, None) for o, c in MOVES[piece][2]]
    if tuck:
        found = tucks(board, piece)
        if found:
            first = np.concatenate([first, np.stack([t[4] for t in found])])
            lines = np.concatenate([lines, [t[5] for t in found]])
            ok = np.concatenate([ok, np.ones(len(found), bool)])
            moves += [(o, c, start) for o, start, c, _, _, _ in found]
    score = scores(first, lines, well)
    if next_piece is not None:
        second, more, reply_ok = place_all(first, next_piece)
        replies = np.where(reply_ok, scores(second, lines[:, None] + more, well), -np.inf)
        score = np.where(reply_ok.any(1), replies.max(1), score - 100.0)
    if not ok.any():
        return None
    return moves[int(np.where(ok, score, -np.inf).argmax())]


def turn_button(orientation: int, goal: int, rng: np.random.Generator) -> int:
    """The button that turns the piece toward `goal` the short way: A takes it to the next id in IDS_OF (the
    game's order), B to the one before; a half turn, or a piece with two orientations, either at random. Turning
    with A alone made 81% of the data's turns A and tied the direction to the press itself, and no model learned
    which way B turns (docs/guides/recipe_attempts.md, 2026-10-14)."""
    ids = IDS_OF[TYPE_OF[orientation]]
    step = (ids.index(goal) - ids.index(orientation)) % len(ids)
    if len(ids) == 4 and step != 2:
        return A if step == 1 else B
    return int(rng.choice((A, B)))


FALL_FRAMES = (48, 43, 38, 33, 28, 23, 18, 13, 8, 6, 5, 5, 5, 4, 4, 4, 3, 3, 3) + (2,) * 10 + (1,)  # per row, by level
HOLD_MAX = 12                                       # the longest a perfect press is held (autorepeat starts at 16)
DOWN_TURNS = 0.4                                    # chance that a held-press piece keeps Down held through its turns
MAX_SHIFTS = 5                                      # the most shifts a placement needs (spawn column 5 to 0 or 9)
NUDGE = 0.3                                         # chance a held-press piece presses into a wall or the stack
NUDGES = (1, 3)                                     # how many times it does, once lined up beside one
LATE = 0.3                                          # chance a held-press piece leaves its last shift until it lands
LATE_ROWS = (1, 4)                                  # rows above its landing that the late shift is pressed at
TUCK_FALL = 3                                       # tucks only where a row takes at least this many frames


def hold_cap(level: int) -> int:
    """The longest a perfect brain may hold a press, or release between presses, at `level`: half a row's
    fall, 1 (a tap, released a frame) from level 16."""
    return max(1, min(HOLD_MAX, FALL_FRAMES[min(level, len(FALL_FRAMES) - 1)] // 2))


PATIENT_FRAMES = (30, 180)                          # no-button wait of a patient piece, then soft drop
SPIN_HEIGHT = 8                                     # a spinning piece needs a stack at most this many rows high
SPIN_RATE = 1 / 8                                   # chance per waiting frame that a spinning piece rotates


class PerfectBrain:
    """Plans each new piece from RAM, then rotates (the short way: turn_button), shifts and soft-drops it
    into place.

    well: build for tetrises (scores). patience: the chance that a piece, once
    lined up, is left to fall by gravity for PATIENT_FRAMES frames before it is
    soft-dropped (drawn per piece), so the data shows pieces falling on their
    own at every level without slow levels taking ~860 frames a piece. spin: the
    chance that a patient piece over a low stack (SPIN_HEIGHT rows or fewer)
    spins while it waits, A or B at random (SPIN_RATE a frame), so the data shows
    rotations in every orientation as a player makes them, not only the few on
    the way to a planned one; when the wait ends it turns back and drops. hold:
    each turn and shift press is held 1 to h frames instead of tapped for one,
    and released 1 to g frames before the next, h and g drawn per piece from 1
    to hold_cap(level), so the same moves come with the press timing a person
    uses (2026-11-01: every brain tapped 1 frame, 1 apart, and dreams broke on a
    person's 2-3 frame presses 2-5 apart); a turn acts on the press and a shift
    repeats only after 16 frames held, so the moves are unchanged. With hold,
    DOWN_TURNS of the pieces also hold Down from the first turn through the
    last (Down with A or B: dreams broke 25-47x on it, and no brain did it),
    then let go before shifting, since the game ignores shifts under Down; the
    soft drop while turning is capped (down_frames) so the piece still has the
    rows to shift into place before it lands. Also with hold, as a person
    plays (2026-11-01: no planned brain pressed into a wall or the stack, or
    moved a piece near its landing): NUDGE of the pieces, once lined up beside a
    wall or a stack that blocks the shift on every row down to the landing,
    press into it NUDGES times (the piece does not move); LATE of the pieces
    with a shift to make hold their last one, soft-dropping, until LATE_ROWS
    rows above where they land; and the search takes tucks (a slide under an
    overhang at the bottom) when they score best, where a row falls in
    TUCK_FALL frames or more: dropped down the start column, Down let go above
    the overhang, shifted under it on the first row it fits. Call
    reset() when it takes over mid-game: it replans only when a new piece
    spawns, so a stale target from an earlier piece would otherwise steer the
    falling one.
    """

    def __init__(self, well: bool = False, patience: float = 0.0, spin: float = 0.0, hold: bool = False):
        self.well, self.patience, self.spin, self.hold = well, patience, spin, hold
        self.reset()

    def reset(self) -> None:
        self.target = None
        self.last_y = None
        self.last_action = 0
        self.wait = 0
        self.waiting = self.spinning = False
        self.hold_max, self.held = 1, 0                 # this piece's longest hold; frames left on this press
        self.gap_max, self.gap = 1, 0                   # this piece's longest release; frames left of it
        self.down_frames, self.downing = 0, False       # soft-drop frames allowed while turning; Down held now
        self.start = None                               # a tuck's column, until the piece slides out of it
        self.nudges = self.late = 0                     # presses into a wall or the stack; the late shift's rows

    def choose(self, ram: np.ndarray, orientation: int, rng: np.random.Generator):
        fall = FALL_FRAMES[min(int(ram[LEVEL]), len(FALL_FRAMES) - 1)]
        move = best_placement(settled_board(ram), TYPE_OF[orientation], TYPE_OF.get(int(ram[NEXT])), self.well,
                              tuck=self.hold and fall >= TUCK_FALL)
        if move is None:
            return None
        self.start = move[2]
        return move[:2]

    def __call__(self, ram: np.ndarray, rng: np.random.Generator) -> int:
        orientation, x, y = int(ram[PIECE]), int(ram[X]), int(ram[Y])
        if orientation not in ORIENTATIONS:
            self.last_action = 0
            return 0
        stale = self.target is not None and TYPE_OF[self.target[0]] != TYPE_OF[orientation]
        if self.target is None or self.last_y is None or y < self.last_y or stale:     # a new piece spawned
            self.start = None
            self.target = self.choose(ram, orientation, rng)
            patient = rng.random() < self.patience and self.start is None    # a tuck goes down at once
            self.wait = int(rng.integers(*PATIENT_FRAMES)) if patient else 0
            self.nudges = int(rng.integers(*NUDGES)) if self.hold and rng.random() < NUDGE else 0
            self.late = (int(rng.integers(*LATE_ROWS)) if self.hold and self.start is None and not self.wait
                         and rng.random() < LATE else 0)
            cap = hold_cap(int(ram[LEVEL])) if self.hold else 1
            self.hold_max, self.gap_max = int(rng.integers(1, cap + 1)), int(rng.integers(1, cap + 1))
            self.held = self.gap = 0
            self.waiting = self.downing = False
            board = settled_board(ram)
            height = 20 - int(board.any(1).argmax()) if board.any() else 0
            fall = FALL_FRAMES[min(int(ram[LEVEL]), len(FALL_FRAMES) - 1)]
            shifting = -(-MAX_SHIFTS * (self.hold_max + self.gap_max) // fall) + 1    # rows gravity takes meanwhile
            room = 20 - height - y - 4 - shifting                       # rows left to soft-drop while turning
            self.down_frames = 2 * room if self.hold and room > 0 and rng.random() < DOWN_TURNS else 0
            self.spinning = (self.spin > 0 and self.wait > 0 and height <= SPIN_HEIGHT
                             and rng.random() < self.spin)
            self.last_y = y
            self.last_action = 0
            return 0                                         # soft drop only re-arms after Down is released
        self.last_y = y
        if self.held > 0:                                    # keep holding this press
            self.held -= 1
            return self._down_while_turning(self.last_action & ~DOWN, orientation)
        if self.gap > 0:                                     # released between presses
            self.gap -= 1
            return self._down_while_turning(0, orientation)
        action = DOWN
        if self.target is not None:
            goal, column = self.target
            lined = orientation == goal and x == column and self.start is None
            if not self.waiting and lined and self.wait > 0:
                self.waiting = True                              # lined up, patient: let gravity work
            nudge = self._nudge(ram, orientation, x, y, rng) if lined and self.nudges else 0
            if nudge:                                            # into a wall or the stack: nothing moves
                action = nudge
                self.nudges -= 1
            elif self.waiting:
                self.wait -= 1
                self.waiting = self.wait > 0
                action = int(rng.choice((A, B))) if self.spinning and rng.random() < SPIN_RATE else 0
            elif orientation != goal:
                action = turn_button(orientation, goal, rng)
                self.downing = self.downing or self.down_frames > 0
            elif self.start is not None and x == self.start:     # a tuck: down the start column, then under
                board = settled_board(ram)
                edge = top_row(board, goal, column)
                if y > edge and fits(board, goal, column, y):
                    action = RIGHT if column > x else LEFT
                    self.start = None
                elif y >= edge - 1:
                    action = 0                                   # gravity alone past the overhang's edge
            elif x != (column if self.start is None else self.start):
                aim = column if self.start is None else self.start
                action = RIGHT if x < aim else LEFT
                if self.late and abs(column - x) == 1:           # the last shift, left until it lands
                    board = settled_board(ram)
                    if y < min(top_row(board, goal, x), top_row(board, goal, column)) - self.late:
                        action = DOWN
                    else:
                        self.late = 0
        if action & (A | B | LEFT | RIGHT):                  # a press: held, then released (moves act on edges)
            self.held = int(rng.integers(1, self.hold_max + 1)) - 1
            self.gap = int(rng.integers(1, self.gap_max + 1))
            return self._down_while_turning(action, orientation)
        self.downing = False
        self.last_action = action
        return action

    def _down_while_turning(self, action: int, orientation: int) -> int:
        """`action` with Down added while this piece holds Down through its turns: from the first turn press to
        the end of the last turn's release, within down_frames, never with a shift."""
        if self.downing and (orientation != self.target[0] or action & (A | B) or not action)                 and not action & (LEFT | RIGHT) and self.down_frames > 0:
            self.down_frames -= 1
            action |= DOWN
        else:
            self.downing = False
        self.last_action = action
        return action

    def _nudge(self, ram: np.ndarray, orientation: int, x: int, y: int, rng: np.random.Generator) -> int:
        """Left or Right into a wall or the stack beside the piece, or 0: a side blocked on every row from here
        to where the piece lands, so the press (and the game's charged autorepeat while it is held) moves
        nothing."""
        board = settled_board(ram)
        rows = range(y, rest_row(board, orientation, x, y) + 1)
        blocked = [button for button, dx in ((LEFT, -1), (RIGHT, 1))
                   if not any(fits(board, orientation, x + dx, r) for r in rows)]
        return int(rng.choice(blocked)) if blocked else 0


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


PERSON_HOLD = (1, 7)                                # the longest a person's press is held, drawn per piece
PERSON_GAP = (1, 9)                                 # the longest gap between a person's presses, drawn per piece
PERSON_REACT = (0, 16)                              # frames a person waits after a spawn
PERSON_DOWN = {"gravity": 0.2, "drop": 0.35, "held": 0.3, "tapped": 0.15}  # how a piece uses Down
PERSON_TOGETHER = 0.4                               # chance that a piece's turns and shifts overlap
PERSON_KEEP_DOWN = 0.3                              # chance that Down held at a lock stays held into the spawn
PERSON_STALL = 8                                    # frames Down is held without the piece falling, then re-pressed
PERSON_TRIES = (1, 4)                               # shifts a person tries under Down before letting go of it


class PersonBrain(PerfectBrain):
    """Plans each piece like the perfect brain, then presses like a person, varied piece to piece so the data
    covers how people press, not one way of pressing (2026-11-01: dreams broke 25-47x when Down was held
    through turns and shifts, which no brain did, and 3x on a person's 2-3 frame taps 2-5 frames apart;
    the other brains tap 1 frame, 1 apart, and never press two buttons together; output/tests_2026-11-01).

    Each piece draws its own style: a reaction of PERSON_REACT frames after the spawn (Down kept held through
    it PERSON_KEEP_DOWN of the time if it was held at the lock); each press held 1 to h frames and released
    1 to g frames, h and g drawn from PERSON_HOLD and PERSON_GAP; Down (PERSON_DOWN) never (the piece falls
    and locks by gravity), once lined up, held from the first press to the lock (Down with each turn and
    shift), or tapped on and off at the press timing throughout; and PERSON_TOGETHER of the time the turns
    and shifts run at once on their own timers (A with Left, B with Right, overlapping by any offset),
    otherwise turns first, then shifts. Presses stay under the game's 16-frame autorepeat delay (held shifts
    are the holding brain's). The game ignores Left and Right while Down is held (2026-11-01, real game: 0 of
    1,168 Down + shift frames moved the piece), so as a person would, after PERSON_TRIES shifts pressed under
    Down that did not move the piece it lets go of Down until the piece is lined up; and soft drop arms only on
    a Down press made without Left or Right (and not on one held into the spawn), so Down held PERSON_STALL
    frames without the piece falling is let go for a frame and pressed again."""

    def reset(self) -> None:
        super().reset()
        self.react, self.hold_max, self.gap_max = 0, 1, 1
        self.down, self.together, self.keep_down = "drop", False, False
        self.pressing = False                           # a turn or shift has been pressed for this piece
        self.shift_x = None                             # the column a shift was pressed from under Down
        self.still = 0                                  # frames since the piece last fell a row
        self.down_off = False                           # let go of Down after shifts it blocked
        self.tries = 1                                  # blocked shifts before letting go of Down
        self.channels = {name: [0, 0, 0] for name in ("turn", "shift", "down")}   # button, frames held, gap

    def _channel(self, name: str, wanted: int, rng: np.random.Generator) -> int:
        """One button's timer: hold the press, then release for a gap, then press `wanted` (0: nothing)."""
        state = self.channels[name]
        if state[1] > 0:                                # keep holding the press
            state[1] -= 1
            if state[1] == 0:
                state[2] = int(rng.integers(1, self.gap_max + 1))
            return state[0]
        if state[2] > 0:                                # released between presses
            state[2] -= 1
            return 0
        if not wanted:
            return 0
        state[0], state[1] = wanted, int(rng.integers(1, self.hold_max + 1)) - 1
        if state[1] == 0:
            state[2] = int(rng.integers(1, self.gap_max + 1))
        return wanted

    def __call__(self, ram: np.ndarray, rng: np.random.Generator) -> int:
        orientation, x, y = int(ram[PIECE]), int(ram[X]), int(ram[Y])
        if orientation not in ORIENTATIONS:
            held_down = bool(self.last_action & DOWN)
            self.reset()
            self.last_action = DOWN if held_down else 0  # remembered for keep_down at the spawn
            return 0
        stale = self.target is not None and TYPE_OF[self.target[0]] != TYPE_OF[orientation]
        if self.target is None or self.last_y is None or y < self.last_y or stale:     # a new piece spawned
            keep = bool(self.last_action & DOWN) and rng.random() < PERSON_KEEP_DOWN
            self.reset()
            self.target = self.choose(ram, orientation, rng)
            self.react = int(rng.integers(*PERSON_REACT))
            self.hold_max = int(rng.integers(*PERSON_HOLD))
            self.gap_max = int(rng.integers(*PERSON_GAP))
            self.down = str(rng.choice(list(PERSON_DOWN), p=list(PERSON_DOWN.values())))
            self.together = rng.random() < PERSON_TOGETHER
            self.tries = int(rng.integers(*PERSON_TRIES))
            self.keep_down = keep
            self.last_y = y
            self.last_action = DOWN if keep else 0
            return self.last_action
        self.still = self.still + 1 if y == self.last_y else 0
        self.last_y = y
        if self.react > 0:
            self.react -= 1
            self.last_action = DOWN if self.keep_down else 0
            return self.last_action
        if self.shift_x is not None:                    # did the shift pressed under Down move the piece?
            if x == self.shift_x:
                self.tries -= 1
                self.down_off = self.down_off or self.tries <= 0
            self.shift_x = None
        turn = shift = 0
        if self.target is not None:
            goal, column = self.target
            if orientation != goal:
                turn = turn_button(orientation, goal, rng)
            if x != column:
                shift = RIGHT if x < column else LEFT
        if not self.together and turn:
            shift = 0                                   # one after the other: turns first
        action = self._channel("turn", turn, rng) | self._channel("shift", shift, rng)
        self.pressing = self.pressing or bool(action)
        busy = bool(turn or shift or self.channels["turn"][1] or self.channels["shift"][1])
        if not busy:
            self.down_off = False
        if self.down_off:
            pass
        elif self.down == "drop" and not busy:
            action |= DOWN
        elif self.down == "held" and (self.pressing or not busy):
            action |= DOWN
        elif self.down == "tapped":
            action |= self._channel("down", DOWN, rng)
        if action & DOWN and self.last_action & DOWN and self.still >= PERSON_STALL:
            action &= ~DOWN                             # Down held, the piece not dropping: press it again
        if action & DOWN and not self.last_action & DOWN:
            self.still = 0                              # a fresh press: give it time to drop
        if action & DOWN and action & (LEFT | RIGHT) and not self.last_action & (LEFT | RIGHT):
            self.shift_x = x
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
    """Chaotic play: a random Tetris button held for 4-20 frames at a time, COMBO of the time with a second
    one (Down with a shift or turn, A with B, Left with Right ...: no other brain presses most pairs)."""

    CHOICES = (0, LEFT, RIGHT, DOWN, A, B)
    COMBO = 0.3

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.action, self.left = 0, 0

    def __call__(self, ram, rng):
        if self.left <= 0:
            self.action = int(rng.choice(self.CHOICES))
            if self.action and rng.random() < self.COMBO:
                self.action |= int(rng.choice([c for c in self.CHOICES if c and c != self.action]))
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
    - short games (SHORT frames): perfect / holding / person / random / idle at random intervals,
      from any start level 0-19: the variety and the sloppy boards.
    When its length runs out the player gives up, reckless (random placements,
    soft-dropped) or random, until the stack tops out. The perfect brain leaves
    PATIENCE of its pieces to fall by gravity, and SPIN of those spin while they
    fall if the stack is low. In WELL_SHARE of games
    the perfect brain builds for tetrises. Every brain resets when it takes over.
    """

    INTERVALS = {"perfect": (2400, 6000), "holding": (2400, 6000), "person": (2400, 6000), "random": (150, 500),
                 "idle": (150, 500), "reckless": (150, 500)}
    WEIGHTS = {"perfect": 0.35, "holding": 0.20, "person": 0.15, "random": 0.20, "idle": 0.10}
    LONG_WEIGHTS = {"perfect": 1.0}
    GIVE_UP = {"reckless": 0.75, "random": 0.25}
    SHORT, LONG, LONG_SHARE = (1500, 3500), (20_000, 45_000), 0.08
    SHORT_START_LEVELS, LONG_START_LEVELS = 20, 10
    WELL_SHARE = 0.5
    PATIENCE = 0.3
    SPIN = 0.5

    def __init__(self, rng: np.random.Generator, long: bool):
        self.brains = {"perfect": PerfectBrain(well=rng.random() < self.WELL_SHARE, patience=self.PATIENCE,
                                               spin=self.SPIN, hold=True),
                       "holding": HoldingBrain(well=rng.random() < self.WELL_SHARE),
                       "person": PersonBrain(well=rng.random() < self.WELL_SHARE),
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
