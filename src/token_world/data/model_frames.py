"""NES palette-plane frames -> the model's uint8 index frames, with the border.

World NES captures each frame as PPU palette indices (0-63) plus an emphasis
plane. The model's frames are 256 x 256 in the shared 58-entry index space
(nes_palette.py), 16 x 16 tokens of 16 x 16 pixels, laid out to fit them:

- token rows 1-14 (rows 16-239) hold NES lines 8-231, the picture a TV shows,
  mapped by colour to indices 0-54 (NES line y at row y + 8). The 8 overscan
  lines above and below are dropped: in Tetris they change only with the
  screen (legal screen, menus, top-out), never in play;
- token rows 0 and 15 are the border, 128 cells of 8 x 8 pixels, four per
  token, each a solid border shade (the game's three, indices 55-57 as
  `slots` gives them) showing one base-3 digit, least significant first:
  cells 0-11 (top tokens 0-2) the frame counter, then 8 cells (two tokens)
  per state byte, 6 digits and 2 unused: hidden RAM a game draws every frame
  (Tetris: the fall timer and the autorepeat counter), so it is in every
  frame, not only in the frames where it changes something on screen. The
  RAM is the one captured with the frame, which reflects the buttons up to
  the one that produced the frame, never the next. Other cells cycle the
  three shades in a fixed pattern.

The counter and state are imperative: in every frame and in the loss at full
weight.
"""
from __future__ import annotations

import numpy as np
import torch

from token_world.data.nes_palette import NES_COLOURS, NES_MASTER_PALETTE, palette_indices

BORDER_VERSION = "tokens-v2"                    # runs record it; frames of another layout are refused
SIZE, BAND, CELL = 256, 16, 8                   # frame side, border rows top and bottom, digit cell side
OVERSCAN = 8                                    # NES lines dropped above and below the picture
GAME_ROWS = slice(BAND, SIZE - BAND)
_CELLS = 2 * (BAND // CELL) * (SIZE // CELL)    # 128: four per border token
_TICK_DIGITS, _STATE_DIGITS, _STATE_CELLS = 12, 6, 8
MAX_STATE = (_CELLS - _TICK_DIGITS) // _STATE_CELLS

# PPU palette index -> model colour index (0-54).
NES_TO_INDEX = np.array([NES_COLOURS.index(c) for c in NES_MASTER_PALETTE], dtype=np.uint8)


def border_slots(palette: torch.Tensor) -> np.ndarray:
    """The index each of the game's three border shades takes in `palette` (frame_palette layout)."""
    shades = palette[len(NES_COLOURS):].float().div(127.5).sub(1)[:, :, None, None]
    return palette_indices(shades, palette).flatten().numpy().astype(np.uint8)


def border_codes(ticks: np.ndarray, state: np.ndarray | None = None) -> np.ndarray:
    """[T] frame counters and [T, K] state bytes -> [T, 2 * BAND, SIZE] shade codes 0-2 (the top band's
    rows, then the bottom band's)."""
    ticks = np.maximum(np.asarray(ticks, dtype=np.int64), 0)
    t = len(ticks)
    cells = np.broadcast_to(np.arange(_CELLS) % 3, (t, _CELLS)).copy()
    cells[:, :_TICK_DIGITS] = (ticks[:, None] // 3 ** np.arange(_TICK_DIGITS)) % 3
    if state is not None:
        state = np.asarray(state, dtype=np.int64).reshape(t, -1)
        if state.shape[1] > MAX_STATE:
            raise ValueError(f"at most {MAX_STATE} state bytes fit the border")
        for k in range(state.shape[1]):
            at = _TICK_DIGITS + k * _STATE_CELLS
            cells[:, at:at + _STATE_DIGITS] = (state[:, k:k + 1] // 3 ** np.arange(_STATE_DIGITS)) % 3
    # cell c: token c // 4 (0-15 top band, 16-31 bottom), quarter c % 4 (row-major within the token)
    grid = cells.reshape(t, 2, SIZE // 16, 2, 2).transpose(0, 1, 3, 2, 4).reshape(t, 2, 2, SIZE // CELL)
    return grid.repeat(CELL, axis=2).repeat(CELL, axis=3).reshape(t, 2 * BAND, SIZE)


def model_index_frames(frames: np.ndarray, emphasis: np.ndarray, ticks: np.ndarray,
                       slots: np.ndarray, state: np.ndarray | None = None) -> np.ndarray:
    """frames/emphasis [T, 240, 256] uint8 palette planes, ticks [T], state [T, K] bytes or None
    -> [T, 256, 256] uint8 model indices.

    Raises if any pixel has an emphasis bit set: emphasised colours are outside
    the index space.
    """
    if frames.shape[1:] != (240, SIZE) or emphasis.shape != frames.shape:
        raise ValueError("expected [T, 240, 256] palette and emphasis planes")
    if emphasis.any():
        raise ValueError("emphasis bits set: colours outside the model's index space")
    out = np.empty((len(frames), SIZE, SIZE), dtype=np.uint8)
    out[:, GAME_ROWS] = NES_TO_INDEX[frames[:, OVERSCAN:240 - OVERSCAN]]
    bands = slots[border_codes(ticks, state)]
    out[:, :BAND], out[:, SIZE - BAND:] = bands[:, :BAND], bands[:, BAND:]
    return out
