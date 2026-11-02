"""Motion drawn on a frame (data/tetris_motion.py or, for SMB, data/smb_motion.py codes): an arrow from each moved
cell's source to it, coloured by direction as optical flow's colour wheel, and NEW cells outlined, on the frame
dimmed (2026-10-15). Tetris's moves are in cells, SMB's in pixels (arrows SMB_ARROW times longer, to be seen).
The training previews (diagnostics/run_outputs.py save_previews) and scripts/motion_check.py draw with it."""
from __future__ import annotations

import colorsys

import numpy as np
from PIL import Image, ImageDraw

import token_world.data.smb_motion as SMB
import token_world.data.tetris_motion as TETRIS

SMB_ARROW = 3                          # SMB's arrows: pixels drawn per pixel moved


def scheme(game: str):
    """The game's motion codes: its module, and the pixels an arrow is drawn per step of a move (at zoom 1)."""
    return (SMB, SMB_ARROW) if game == "smb" else (TETRIS, TETRIS.CELL)


def direction_colour(dy: int, dx: int) -> tuple[int, int, int]:
    """A move's direction as a hue (optical flow's colour wheel): down green-yellow, left cyan, right red..."""
    r, g, b = colorsys.hsv_to_rgb((np.arctan2(dy, dx) / (2 * np.pi)) % 1.0, 1.0, 1.0)
    return int(255 * r), int(255 * g), int(255 * b)


def motion_overlay(rgb: np.ndarray, codes: np.ndarray, zoom: int = 1, top: int = 0, left: int = 0,
                   dim: float = 0.45, game: str = "tetris") -> np.ndarray:
    """A frame's colours [H, W, 3] uint8 (cells of 8 * zoom pixels, its top-left at model cell (top, left)) and
    the motion codes of the whole frame [32, 32] -> the frame dimmed with the motion drawn on it [H, W, 3]."""
    M, step = scheme(game)
    img = Image.fromarray((rgb * dim).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    cell = M.CELL * zoom
    rows, cols = rgb.shape[0] // cell, rgb.shape[1] // cell
    width = max(1, zoom)
    for r in range(top, min(top + rows, M.GRID)):
        for c in range(left, min(left + cols, M.GRID)):
            v = int(codes[r, c])
            x, y = (c - left) * cell, (r - top) * cell
            if v == M.NEW:
                draw.rectangle((x, y, x + cell - 1, y + cell - 1), outline=(255, 255, 255), width=width)
            elif v not in (M.STILL, M.UNKNOWN):
                dy, dx = M.offset_of(v)
                end = (x + cell // 2, y + cell // 2)
                start = (end[0] - dx * step * zoom, end[1] - dy * step * zoom)
                colour = direction_colour(dy, dx)
                draw.line((start, end), fill=colour, width=width + zoom // 2)
                dot = 1 + zoom
                draw.ellipse((end[0] - dot, end[1] - dot, end[0] + dot, end[1] + dot), fill=colour)
    return np.asarray(img)
