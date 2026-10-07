"""NES frames as the picture processor's layers: a background fixed to the world, the sprites, a camera.

World NES (llm-thing23) captures, per frame, the background and the sprites as the PPU drew them, which
OAM slot drew each sprite pixel, and each scanline's scroll (regions ppu_background, ppu_sprites,
ppu_sprite_index, ppu_scroll), with oam and palette_ram. Here they become a layered frame in the model's
colours (nes_palette.py: final NES colours, so no palette RAM or ROM data is in it), which composes back
to the frame exactly (compose). See docs/guides/layered_tokens.md.

The picture is NES lines 0-239 in BANDS bands of 16 lines (a band is a token row; SMB's status bar ends at
line 32, a band boundary). Per frame:

- cells [BANDS, VIEW, 16, 16]: the background cells in view, fixed to the world grid. The background is a
  512 x 480 canvas (the PPU's four nametables, CANVAS_ROWS x CANVAS_COLS cells); band r shows canvas row
  cell_row[r] from canvas column cell_col[r] on, VIEW cells (17: a band scrolled to a fine offset spans
  17). Its pixels come from a canvas each stream keeps of every background pixel seen so far
  (cell_known: seen at least once; a cell scrolling into view arrives in part). Where the background is
  transparent a cell holds TRANSPARENT, and the frame's backdrop colour (backdrop) shows there.
- camera [BANDS, 2]: the band's scroll (x 0..511, y 0..479) at its first line.
- the sprites, two ways (models/layered_pixels.py reads the first, models/layered_slots.py the second):
  - sprite_layer [LINES, WIDTH]: the sprite pixels that show, in screen space (TRANSPARENT elsewhere: no
    sprite, or a sprite behind an opaque background pixel). The frame is this over the background.
  - sprites [SPRITES, 8, 8] the pixels each OAM slot drew (TRANSPARENT where none: hidden by a lower slot or
    the PPU's 8-per-line limit), sprite_xy [SPRITES, 2] its screen x, y (OAM y + 1), sprite_flags bit 0 on
    screen, bit 1 behind the background.
- border [2 * BAND, 256]: the tick border's two bands, as in model frames (model_frames.py).
"""
from __future__ import annotations

import numpy as np

from token_world.data.model_frames import BAND, NES_TO_INDEX, OVERSCAN, SIZE, border_codes
from token_world.data.nes_palette import BORDER_SLOTS, NES_COLOURS

REGIONS = ("oam", "palette_ram", "ppu_background", "ppu_sprites", "ppu_sprite_index", "ppu_scroll")
LINES, WIDTH, CELL = 240, 256, 16
BANDS = LINES // CELL                       # 15 token rows of the picture
VIEW = WIDTH // CELL + 1                    # 17 cells per band: 16, plus one when scrolled to a fine offset
CANVAS_ROWS, CANVAS_COLS = 30, 32           # the nametables as cells: 480 x 512 pixels
SPRITES, SPRITE = 64, 8
TRANSPARENT = len(NES_COLOURS) + BORDER_SLOTS          # 58: nothing drawn in that layer (one more colour)
LAYER_COLOURS = TRANSPARENT + 1
ON_SCREEN, BEHIND = 1, 2


class LayerCanvas:
    """One stream's background canvas (what it has seen of the world) and its frames' layers."""

    def __init__(self):
        self.canvas = np.full((CANVAS_ROWS * CELL, CANVAS_COLS * CELL), TRANSPARENT, np.uint8)
        self.known = np.zeros(self.canvas.shape, bool)

    def push(self, regions: dict, tick: int, slots: np.ndarray, state: np.ndarray | None = None) -> dict:
        """One observation's regions (REGIONS, one array each), its frame counter, the game's border shades
        (model_frames.border_slots) and the RAM state drawn in the border -> its layered frame."""
        palette = np.asarray(regions["palette_ram"], np.uint8)
        colours = NES_TO_INDEX[palette & 0x3F]                          # palette RAM entry -> model colour
        bg = np.asarray(regions["ppu_background"], np.uint8).reshape(LINES, WIDTH)
        sp = np.asarray(regions["ppu_sprites"], np.uint8).reshape(LINES, WIDTH)
        owner = np.asarray(regions["ppu_sprite_index"], np.uint8).reshape(LINES, WIDTH)
        oam = np.asarray(regions["oam"], np.uint8).reshape(SPRITES, 4).astype(np.int64)
        scroll = np.asarray(regions["ppu_scroll"], np.uint8).view("<u2").reshape(LINES, 2).astype(np.int64)

        ys = scroll[:, 1] % (CANVAS_ROWS * CELL)
        xs = (scroll[:, 0][:, None] + np.arange(WIDTH)) % (CANVAS_COLS * CELL)
        self.canvas[ys[:, None], xs] = np.where(bg > 0, colours[bg], TRANSPARENT)
        self.known[ys[:, None], xs] = True

        camera = scroll[::CELL]                                          # [BANDS, 2], each band's first line
        cell_row = (camera[:, 1] % (CANVAS_ROWS * CELL)) // CELL
        cell_col = camera[:, 0] // CELL
        rows = (cell_row[:, None, None] * CELL + np.arange(CELL)[None, None, :])          # [BANDS, 1, 16]
        cols = ((cell_col[:, None] + np.arange(VIEW)) % CANVAS_COLS)[:, :, None] * CELL + np.arange(CELL)
        cells = self.canvas[rows[:, :, :, None], cols[:, :, None, :]]                       # [BANDS, VIEW, 16, 16]
        known = self.known[rows[:, :, :, None], cols[:, :, None, :]]

        # the PPU's rule: the front sprite pixel shows unless it is behind an opaque background pixel
        shows = ((sp & 3) > 0) & (((sp & 0x20) == 0) | (bg == 0))
        sprite_layer = np.where(shows, colours[(sp & 0x1F) | 0x10], TRANSPARENT).astype(np.uint8)
        sprite_colours = np.where(sp & 0x1F, colours[(sp & 0x1F) | np.where(sp & 3, 0x10, 0)], TRANSPARENT)
        sprites = np.full((SPRITES, SPRITE, SPRITE), TRANSPARENT, np.uint8)
        xy = np.zeros((SPRITES, 2), np.uint8)
        flags = np.zeros(SPRITES, np.uint8)
        for slot in np.nonzero(oam[:, 0] < LINES - 1)[0]:
            x, y = int(oam[slot, 3]), int(oam[slot, 0]) + 1
            mine = owner[y:y + SPRITE, x:x + SPRITE] == slot + 1
            block = sprites[slot, :mine.shape[0], :mine.shape[1]]
            block[mine] = sprite_colours[y:y + SPRITE, x:x + SPRITE][mine]
            xy[slot] = x, y
            flags[slot] = ON_SCREEN | (BEHIND if oam[slot, 2] & 0x20 else 0)
        border = slots[border_codes(np.array([tick]), None if state is None else np.asarray(state)[None])[0]]
        return {"cells": cells, "cell_known": known, "cell_row": cell_row.astype(np.int16),
                "cell_col": cell_col.astype(np.int16), "camera": camera.astype(np.int16),
                "sprite_layer": sprite_layer, "sprites": sprites, "sprite_xy": xy, "sprite_flags": flags,
                "border": border.astype(np.uint8),
                "backdrop": np.uint8(colours[0])}


def compose(frame: dict) -> np.ndarray:
    """A layered frame -> the picture [LINES, WIDTH] in model colours (NES lines 0-239): its sprite picture
    over the background, or, for a frame with sprite slots only (models/layered_slots.py's dreams), the slots
    drawn by the PPU's rule (lower slots in front of higher ones; a slot behind the background shows only
    where the background is transparent)."""
    cells, camera = frame["cells"], frame["camera"].astype(np.int64)
    out = np.empty((LINES, WIDTH), np.uint8)
    opaque = np.empty((LINES, WIDTH), bool)
    for r in range(BANDS):
        strip = cells[r].transpose(1, 0, 2).reshape(CELL, VIEW * CELL)                     # [16, 272]
        fine = int(camera[r, 0]) % CELL
        band = strip[:, fine:fine + WIDTH]
        out[r * CELL:(r + 1) * CELL] = np.where(band != TRANSPARENT, band, frame["backdrop"])
        opaque[r * CELL:(r + 1) * CELL] = band != TRANSPARENT
    if "sprite_layer" in frame:
        sprites = frame["sprite_layer"]
        return np.where(sprites != TRANSPARENT, sprites, out).astype(np.uint8)
    drawn = np.zeros((LINES, WIDTH), bool)
    for slot in range(SPRITES):
        if not frame["sprite_flags"][slot] & ON_SCREEN:
            continue
        x, y = (int(v) for v in frame["sprite_xy"][slot])
        h, w = min(SPRITE, LINES - y), min(SPRITE, WIDTH - x)
        if h <= 0 or w <= 0:
            continue
        pix = frame["sprites"][slot, :h, :w]
        put = (pix != TRANSPARENT) & ~drawn[y:y + h, x:x + w]
        show = put & ~opaque[y:y + h, x:x + w] if frame["sprite_flags"][slot] & BEHIND else put
        out[y:y + h, x:x + w][show] = pix[show]
        drawn[y:y + h, x:x + w] |= put
    return out


def model_frame(picture: np.ndarray, border: np.ndarray) -> np.ndarray:
    """A composed picture [LINES, WIDTH] and its border [2 * BAND, 256] -> today's 256 x 256 model frame
    (model_frames.py: the border bands, NES lines 8-231 between them), for every score made for those."""
    out = np.empty((SIZE, SIZE), np.uint8)
    out[BAND:SIZE - BAND] = picture[OVERSCAN:LINES - OVERSCAN]
    out[:BAND], out[SIZE - BAND:] = border[:BAND], border[BAND:]
    return out
