"""One motion vocabulary for a run on several games (2026-10-22: Tetris and SMB in one model, the user: "from both
tetris and mario using the border colors as the signal to the model which rom we are playing").

Each game's motion codes (data/tetris_motion.py: 8-pixel cell moves; data/smb_motion.py: pixel moves) share one
layout: STILL 0, the game's moves 1 .. MOVES, NEW MOVES + 1, UNKNOWN 255. Their moves mean different things, so
the run's vocabulary keeps both, side by side, and the shared codes once:

    STILL 0 | Tetris's moves 1 .. 44 | SMB's moves 45 .. 160 | NEW 161          (UNKNOWN 255 as before)

NEW stays the last code (the model and its scores read "moved" as 0 < code < codes - 1). A one-game run keeps its
game's own codes (vocabulary(("smb",)) is smb_motion.CODES).
"""
from __future__ import annotations

import numpy as np
import torch

from token_world.data import smb_motion, tetris_motion

GAME_CODES = {"tetris": tetris_motion.CODES, "smb": smb_motion.CODES}
_FIRST = {"tetris": 1, "smb": tetris_motion.MOVES + 1}         # each game's first move in the shared vocabulary
SHARED = 1 + tetris_motion.MOVES + smb_motion.MOVES + 1


def vocabulary(games) -> int:
    """The run's motion vocabulary: one game's own codes, or SHARED for both."""
    games = tuple(games)
    if len(games) == 1:
        return GAME_CODES[games[0]]
    if set(games) != {"tetris", "smb"}:
        raise ValueError(f"no shared motion vocabulary for {games}")
    return SHARED


def table(game: str, codes: int) -> np.ndarray:
    """uint8 [256]: the game's code -> its code in a vocabulary of `codes` (the game's own: unchanged)."""
    lut = np.arange(256, dtype=np.uint8)
    if codes == GAME_CODES[game]:
        return lut
    if codes != SHARED:
        raise ValueError(f"{game}'s motion codes do not map into a vocabulary of {codes}")
    moves = GAME_CODES[game] - 2
    lut[1:moves + 1] = np.arange(_FIRST[game], _FIRST[game] + moves)
    lut[moves + 1] = SHARED - 1                                  # NEW
    return lut


def to_vocabulary(motion: torch.Tensor, game: str, codes: int) -> torch.Tensor:
    """A game's motion codes (uint8, any shape) in a vocabulary of `codes`."""
    if codes == GAME_CODES[game]:
        return motion
    return torch.from_numpy(table(game, codes)).to(motion.device)[motion.long()]


def from_vocabulary(motion: torch.Tensor, game: str, codes: int) -> torch.Tensor:
    """Codes in a vocabulary of `codes` -> the game's own (another game's moves: UNKNOWN), for its pictures."""
    if codes == GAME_CODES[game]:
        return motion
    inverse = np.full(256, 255, np.uint8)
    lut = table(game, codes)
    inverse[lut[:GAME_CODES[game]]] = np.arange(GAME_CODES[game], dtype=np.uint8)
    return torch.from_numpy(inverse).to(motion.device)[motion.long()]
