"""The NES colour set, so a decoder can emit exact palette indices instead of RGB.

Every NES frame is drawn from the PPU's 64-entry master palette; nes-py renders
it with the table below (read from nes_py/_native's palette, 0xRRGGBB). The
letterbox border adds the game's three identity shades, which take three
fixed slots after the NES colours. A frame pixel outside this set means the
pipeline changed, so the lookup refuses it rather than guessing a nearest
colour. Measured: Tetris, Super Mario Bros., Zelda and Contra (14-22 colours
each over 6,000 frames) produced no pixel outside the master palette.
"""
from __future__ import annotations

import torch

NES_MASTER_PALETTE = (
    0x7C7C7C, 0x0000FC, 0x0000BC, 0x4428BC, 0x940084, 0xA80020, 0xA81000, 0x881400,
    0x503000, 0x007800, 0x006800, 0x005800, 0x004058, 0x000000, 0x000000, 0x000000,
    0xBCBCBC, 0x0078F8, 0x0058F8, 0x6844FC, 0xD800CC, 0xE40058, 0xF83800, 0xE45C10,
    0xAC7C00, 0x00B800, 0x00A800, 0x00A844, 0x008888, 0x000000, 0x000000, 0x000000,
    0xF8F8F8, 0x3CBCFC, 0x6888FC, 0x9878F8, 0xF878F8, 0xF85898, 0xF87858, 0xFCA044,
    0xF8B800, 0xB8F818, 0x58D854, 0x58F898, 0x00E8D8, 0x787878, 0x000000, 0x000000,
    0xFCFCFC, 0xA4E4FC, 0xB8B8F8, 0xD8B8F8, 0xF8B8F8, 0xF8A4C0, 0xF0D0B0, 0xFCE0A8,
    0xF8D878, 0xD8F878, 0xB8F8B8, 0xB8F8D8, 0x00FCFC, 0xF8D8F8, 0x000000, 0x000000,
)


NES_COLOURS = sorted(set(NES_MASTER_PALETTE))  # 55 distinct; indices 0-54 in every game
BORDER_SLOTS = 3                                # indices 55-57: the game's border shades


def frame_palette(border_colors) -> torch.Tensor:
    """The fixed palette layout as uint8 [58, 3]: NES colours, then border slots.

    Indices 0-54 are the same NES colours for every game. Indices 55-57 hold
    the game's three border shades in the order it paints them, so a decoder
    names "border shade 1", not a particular game's colour, and one index
    space serves every game. border_colors: (r, g, b) triples in [0, 1],
    rounded exactly as a painted frame is when mapped back to 8 bits. A shade
    that equals an NES colour resolves to the NES index.
    """
    if len(border_colors) != BORDER_SLOTS:
        raise ValueError(f"expected {BORDER_SLOTS} border shades")
    keys = list(NES_COLOURS)
    for r, g, b in border_colors:
        r, g, b = (round(float(c) * 255) for c in (r, g, b))
        keys.append(r << 16 | g << 8 | b)
    return torch.tensor([[k >> 16, (k >> 8) & 255, k & 255] for k in keys], dtype=torch.uint8)


_LOOKUP: dict[tuple, torch.Tensor] = {}


def _lookup(palette: torch.Tensor, device) -> torch.Tensor:
    """24-bit colour -> palette index + 1 (0 = not in the palette), cached per palette and device."""
    key = (tuple(palette.flatten().tolist()), str(device))
    if key not in _LOOKUP:
        table = torch.zeros(1 << 24, dtype=torch.uint8)
        # Built on the CPU in reverse order, so a colour listed twice resolves to its lower index.
        for index in range(len(palette) - 1, -1, -1):
            r, g, b = (int(c) for c in palette[index])
            table[r << 16 | g << 8 | b] = index + 1
        _LOOKUP[key] = table.to(device)
    return _LOOKUP[key]


def palette_indices(x: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    """[N, 3, H, W] frames in [-1, 1] -> [N, H, W] long indices into palette [K, 3] uint8."""
    q = x.float().add(1).mul(127.5).round().long()
    if q.min() < 0 or q.max() > 255:
        raise ValueError("frame values outside [-1, 1]")
    keys = q[:, 0] << 16 | q[:, 1] << 8 | q[:, 2]
    found = _lookup(palette, x.device)[keys]
    if not found.all():
        missing = found == 0
        bad = keys[missing][0].item()
        raise ValueError(f"frame colour #{bad:06X} is not in the palette "
                         f"({int(missing.sum())} pixels)")
    return found.long() - 1


# Tetris's letterbox border shades (llm-thing19's game_border, where the frames were first bordered), in
# the order the border paints them; the model's frames and every checkpoint's palette use these.
TETRIS_BORDER_COLORS = ((0.987, 0.857, 0.935), (0.98, 0.78, 0.9), (0.637, 0.507, 0.585))


def tetris_palette() -> torch.Tensor:
    """uint8 [58, 3]: the NES master palette plus Tetris's letterbox border shades."""
    return frame_palette(TETRIS_BORDER_COLORS)
