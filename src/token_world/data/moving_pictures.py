"""Moving pictures: synthetic two-layer windows that can only be predicted by copying (2026-10-21).

The SMB models draw the background sharp and Mario as a blob, even one step from a real history, given the real
motion, and when Mario is unchanged from the frame before (docs/guides/recipe_attempts.md, 2026-10-21): the
background is a few thousand patches the model recalls, Mario's patches are each nearly new (any offset in a
patch, any pose), and the model never learned to carry pixels over from the frame before. These windows make
recall useless and copying the only way to lower the loss. Two sources:

levels (harvested from the bot's live game):
- the background layer: a real SMB background (the PPU's), its playfield sliding at a constant speed for the whole
  window (np.roll; up to SPEED px a frame, a fraction, so the moves go 1, 2, 1, 2 as real scrolling does), the HUD
  rows still;
- the sprite layer: 1 to MAX_SPRITES cut-outs of real sprite pixels (boxes of the live game's sprite layer: Mario,
  enemies, items, parts of them), each flipped at random, moving at its own constant velocity and bouncing off
  the playfield's edges, later ones in front;
- every window's colours permuted (the game's 55; TRANSPARENT and the border's three shades kept).

rom (the user's, 2026-10-21: SMB's own tiles, no levels or emulator): the ROM's CHR, 256 sprite and 256
background tiles of 8 x 8 (rom_tiles), in random palettes of the game's colours, every tile and offset evenly:
- the background layer: a canvas of 2 x 2 background metatiles (blocks of the pattern table as drawn: bricks,
  blocks, pipes, ground, letters) at random places, the rest the backdrop, a palette per metatile from four;
  the view slides over it at a constant speed; the HUD rows a still line of the font's letters and digits;
- the sprite layer: 1 to MAX_SPRITES metasprites, blocks of 1 x 1 to 3 x 4 tiles of the sprite table as drawn
  (Mario's poses, enemies, items), each in one of four palettes, flipped at random, moving at its own velocity
  (some with the background, as enemies standing on it) and bouncing off the playfield's edges.

In both the border's frame counter counts on as in the real stream; the buttons are random and explain nothing.
The windows have the real stream's layout (world_nes_smb.WorldNesSmbStreams with layers and motion): x [frames,
2, 256, 256] uint8, action [frames], motion [frames, 32, 32] (smb_motion.frame_motion of the composed pictures,
the first UNKNOWN), tick, in_game_restarts, worker_id, toss_names. The trainer's --copy-pretrain [levels|rom] uses
them in place of the real SMB stream; relaunching the run without it switches to real play.
"""
from __future__ import annotations

from collections import deque
from functools import lru_cache

import numpy as np
import torch

from token_world.data.model_frames import BAND, SIZE, border_codes, border_slots, model_index_frames
from token_world.data.nes_layers import TRANSPARENT, TWO_LAYER_REGIONS, compose_layers, two_layers
from token_world.data.smb_motion import GRID, UNKNOWN, frame_motion
from token_world.data.world_nes_smb import SmbGame, smb_palette, smb_rom
from token_world.data.world_nes_windows import window_stride, worker_seed

PLAY = slice(48, SIZE - BAND)          # model rows of the playfield (below the HUD, above the border)
HUD = slice(24, 48)                    # model rows of SMB's HUD
GAME_COLOURS = 55                      # the NES colours a game draws (the border's shades are 55-57)
SPEED = 3.5                            # the fastest slide or sprite, px a frame
MAX_SPRITES = 8
CUT = (8, 33)                          # a harvested sprite cut-out's side, px
HARVEST = 4                            # live game frames played for each levels window (fresh sources; 16 cost
#                                        140 ms a window, the trainer waiting on data, 2026-10-21)
BANK = (48, 512)                       # backgrounds and sprite cut-outs kept
BUTTONS = (0, 0x80, 0x82, 0x81, 0x83, 0x40, 0x42, 0x01, 0x20)     # none, right, run, jump, ..., left, A, down
SPRITE_BLOCKS = ((1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (2, 4), (3, 2), (3, 4))   # metasprite (columns, rows)
SOURCES = ("levels", "rom")


class MovingPictures(torch.utils.data.IterableDataset):
    """Endless synthetic windows (above): source levels, one bot game per DataLoader worker feeding the sources;
    source rom, the ROM's tiles alone."""

    def __init__(self, frames: int = 64, seed: int = 17, repo: str | None = None, stride: int | None = None,
                 source: str = "levels"):
        if source not in SOURCES:
            raise ValueError(f"source {source!r}: one of {SOURCES}")
        self.frames, self.seed, self.repo, self.source = int(frames), int(seed), repo, source
        self.new_frames = window_stride(self.frames, stride)

    def __iter__(self):
        worker_id, seed = worker_seed(self.seed)
        rng = np.random.default_rng(seed)
        slots = border_slots(smb_palette())
        tick = int(rng.integers(0, 100_000))
        meta = {"in_game_restarts": 0, "worker_id": worker_id, "toss_judged": 1,
                "toss_names": {f"synthetic: moving pictures ({self.source})"}}
        if self.source == "rom":
            tiles = rom_tiles(str(smb_rom(self.repo)))
            while True:
                window = rom_window(tiles, self.frames, tick, slots, rng)
                tick += self.frames
                yield {**window, **meta}
        game = SmbGame(seed, rom=smb_rom(self.repo), sprites=True)
        backgrounds, cuts = deque(maxlen=BANK[0]), deque(maxlen=BANK[1])
        try:
            while True:
                for _ in range(HARVEST):
                    o = game.observation
                    x = model_index_frames(o.frames[None], o.emphasis[None], np.array([game.tick]), slots)[0]
                    layers = two_layers(x, {k: o.regions[k] for k in TWO_LAYER_REGIONS})
                    if rng.random() < 0.25:
                        backgrounds.append(layers[0].copy())
                    cuts.extend(sprite_cuts(layers[1], rng))
                    game.step()
                if not backgrounds or not cuts:
                    continue
                window = moving_window(backgrounds[int(rng.integers(len(backgrounds)))], list(cuts), self.frames,
                                       tick, slots, rng)
                tick += self.frames
                yield {**window, **meta}
        finally:
            game.close()


@lru_cache(maxsize=2)
def rom_tiles(rom: str) -> tuple[np.ndarray, np.ndarray]:
    """An NROM ROM's CHR (iNES) -> its two pattern tables [256, 8, 8] uint8 of 2-bit pixels (0-3): SMB's sprites
    (table 0) and background (table 1)."""
    data = open(rom, "rb").read()
    prg, chr_banks, trainer = data[4], data[5], bool(data[6] & 4)
    start = 16 + (512 if trainer else 0) + prg * 16384
    planes = np.frombuffer(data[start:start + chr_banks * 8192], np.uint8).reshape(-1, 16)
    bits = np.unpackbits(planes, axis=1).reshape(-1, 2, 8, 8)
    tiles = bits[:, 0] | (bits[:, 1] << 1)
    return tiles[:256], tiles[256:512]


def _palettes(rng: np.random.Generator) -> np.ndarray:
    """Four palettes of three distinct game colours [4, 4] (index 0, the backdrop or transparent, set by the
    caller)."""
    out = np.zeros((4, 4), np.uint8)
    for p in range(4):
        out[p, 1:] = rng.choice(GAME_COLOURS, 3, replace=False)
    return out


def _block(table: np.ndarray, columns: int, rows: int, rng: np.random.Generator) -> np.ndarray:
    """A block of the pattern table as drawn (16 tiles a row): its 2-bit pixels [8 rows, 8 columns]."""
    r0, c0 = int(rng.integers(0, 17 - rows)), int(rng.integers(0, 17 - columns))
    index = (np.arange(r0, r0 + rows)[:, None] * 16 + np.arange(c0, c0 + columns)[None, :])
    return table[index].transpose(0, 2, 1, 3).reshape(rows * 8, columns * 8)


def rom_window(tiles: tuple[np.ndarray, np.ndarray], frames: int, tick: int, slots: np.ndarray,
               rng: np.random.Generator) -> dict:
    """One synthetic window from the ROM's tiles (source rom, the module's doc)."""
    sprite_table, back_table = tiles
    back_pal, sprite_pal = _palettes(rng), _palettes(rng)
    backdrop = int(rng.integers(GAME_COLOURS))
    back_pal[:, 0] = backdrop
    still = rng.random() < 0.2
    vx_bg = 0.0 if still else rng.uniform(-SPEED, SPEED)
    vy_bg = 0.0 if still or rng.random() < 0.8 else rng.uniform(-1, 1)
    play_h = PLAY.stop - PLAY.start
    pad_x, pad_y = int(np.ceil(SPEED * frames)) + 16, int(np.ceil(abs(vy_bg) * frames)) + 16
    canvas_h, canvas_w = (play_h + 2 * pad_y + 15) // 16 * 16, (SIZE + 2 * pad_x + 15) // 16 * 16
    # the canvas: 16 x 16 metatiles (2 x 2 background tiles as drawn), each with a palette, or the backdrop
    fill = rng.uniform(0.15, 0.7)
    canvas = np.full((canvas_h, canvas_w), backdrop, np.uint8)
    for y in range(0, canvas_h, 16):
        for x in range(0, canvas_w, 16):
            if rng.random() < fill:
                canvas[y:y + 16, x:x + 16] = back_pal[int(rng.integers(4))][_block(back_table, 2, 2, rng)]
    oy, ox = pad_y + int(rng.integers(-8, 9)), pad_x + int(rng.integers(-8, 9))
    hud = np.full((HUD.stop - HUD.start, SIZE), backdrop, np.uint8)
    font_pal = back_pal[int(rng.integers(4))]
    for x in range(0, SIZE, 8):
        if rng.random() < 0.5:
            letter = back_table[int(rng.integers(0, 36))]                   # the font: 0-9, A-Z
            hud[8:16, x:x + 8] = font_pal[letter]
    sprites = []
    for _ in range(int(rng.integers(1, MAX_SPRITES + 1))):
        columns, rows = SPRITE_BLOCKS[int(rng.integers(len(SPRITE_BLOCKS)))]
        block = _block(sprite_table, columns, rows, rng)
        if (block != 0).sum() < 12:
            continue
        pal = sprite_pal[int(rng.integers(4))]
        cut = np.where(block != 0, pal[block], TRANSPARENT).astype(np.uint8)
        cut = cut[:, ::-1] if rng.random() < 0.5 else cut
        cut = cut[::-1] if rng.random() < 0.1 else cut
        h, w = cut.shape
        if rng.random() < 0.3:                                              # standing on the scenery
            vy, vx = vy_bg, vx_bg
        else:
            vy = 0.0 if rng.random() < 0.4 else rng.uniform(-3, 3)
            vx = 0.0 if rng.random() < 0.2 else rng.uniform(-SPEED, SPEED)
        sprites.append([cut, rng.uniform(PLAY.start, PLAY.stop - h), rng.uniform(0, SIZE - w), vy, vx])
    backs = np.empty((frames, SIZE, SIZE), np.uint8)
    layers = np.full((frames, SIZE, SIZE), TRANSPARENT, np.uint8)
    for t in range(frames):
        backs[t, BAND:SIZE - BAND] = backdrop
        backs[t, HUD] = hud
        y, x = oy - int(round(vy_bg * t)), ox - int(round(vx_bg * t))     # the view moves against the scenery
        backs[t, PLAY] = canvas[y:y + play_h, x:x + SIZE]
        _draw_sprites(layers[t], sprites)
    return _finish(backs, layers, tick, slots, rng)


def _draw_sprites(layer: np.ndarray, sprites: list) -> None:
    """Draw each sprite [cut, y, x, vy, vx] into the sprite layer (later ones in front), then move it, bouncing
    off the playfield's edges."""
    for s in sprites:
        cut, y, x = s[0], int(round(s[1])), int(round(s[2]))
        h, w = cut.shape
        region = layer[y:y + h, x:x + w]
        opaque = cut != TRANSPARENT
        region[opaque] = cut[opaque]
        s[1] += s[3]
        s[2] += s[4]
        if not PLAY.start <= s[1] <= PLAY.stop - h:
            s[3] = -s[3]
            s[1] = float(np.clip(s[1], PLAY.start, PLAY.stop - h))
        if not 0 <= s[2] <= SIZE - w:
            s[4] = -s[4]
            s[2] = float(np.clip(s[2], 0, SIZE - w))


def _finish(backs: np.ndarray, sprite_layers: np.ndarray, tick: int, slots: np.ndarray,
            rng: np.random.Generator) -> dict:
    """Backgrounds and sprite layers [frames, 256, 256] -> a window: the border's counter drawn on the
    background (the sprite layer TRANSPARENT there), the composed pictures' motion, random buttons."""
    frames = len(backs)
    ticks = tick + np.arange(frames)
    bands = slots[border_codes(ticks)]
    x = np.empty((frames, 2, SIZE, SIZE), np.uint8)
    x[:, 0], x[:, 1] = backs, sprite_layers
    x[:, 0, :BAND], x[:, 0, SIZE - BAND:] = bands[:, :BAND], bands[:, BAND:]
    x[:, 1, :BAND], x[:, 1, SIZE - BAND:] = TRANSPARENT, TRANSPARENT
    pictures = compose_layers(x)
    motion = np.full((frames, GRID, GRID), UNKNOWN, np.uint8)
    for t in range(1, frames):
        motion[t] = frame_motion(pictures[t - 1], pictures[t])
    actions = np.repeat(rng.choice(BUTTONS, frames), rng.integers(1, 31, frames))[:frames]
    return {"x": torch.from_numpy(x), "action": torch.from_numpy(actions.astype(np.int64)),
            "motion": torch.from_numpy(motion), "tick": int(ticks[-1])}


def sprite_cuts(sprite_layer: np.ndarray, rng: np.random.Generator, most: int = 4) -> list[np.ndarray]:
    """A frame's sprite layer [256, 256] -> up to `most` cut-outs: boxes of CUT px around random sprite pixels
    of the playfield, TRANSPARENT where no sprite drew, each with at least 12 sprite pixels."""
    ys, xs = np.nonzero(sprite_layer[PLAY] != TRANSPARENT)
    out = []
    for i in rng.integers(0, len(ys), min(most, len(ys))) if len(ys) else []:
        h, w = rng.integers(*CUT, 2)
        y0 = int(np.clip(ys[i] + PLAY.start - rng.integers(0, h), PLAY.start, PLAY.stop - h))
        x0 = int(np.clip(xs[i] - rng.integers(0, w), 0, SIZE - w))
        cut = sprite_layer[y0:y0 + h, x0:x0 + w].copy()
        if (cut != TRANSPARENT).sum() >= 12:
            out.append(cut)
    return out


def moving_window(background: np.ndarray, cuts: list[np.ndarray], frames: int, tick: int, slots: np.ndarray,
                  rng: np.random.Generator) -> dict:
    """One synthetic window from a harvested background layer [256, 256] and sprite cut-outs (source levels)."""
    perm = np.arange(256, dtype=np.uint8)
    perm[:GAME_COLOURS] = rng.permutation(GAME_COLOURS)
    still = rng.random() < 0.2
    vy_bg = 0.0 if still or rng.random() < 0.8 else rng.uniform(-2, 2)
    vx_bg = 0.0 if still else rng.uniform(-SPEED, SPEED)
    sprites = []
    for _ in range(int(rng.integers(1, MAX_SPRITES + 1))):
        cut = cuts[int(rng.integers(len(cuts)))]
        cut = cut[:, ::-1] if rng.random() < 0.5 else cut
        h, w = cut.shape
        y = rng.uniform(PLAY.start, PLAY.stop - h)
        x = rng.uniform(0, SIZE - w)
        vy = 0.0 if rng.random() < 0.4 else rng.uniform(-3, 3)
        vx = 0.0 if rng.random() < 0.2 else rng.uniform(-SPEED, SPEED)
        sprites.append([cut, y, x, vy, vx])
    backs = np.empty((frames, SIZE, SIZE), np.uint8)
    layers = np.full((frames, SIZE, SIZE), TRANSPARENT, np.uint8)
    for t in range(frames):
        bg = background.copy()
        bg[PLAY] = np.roll(background[PLAY], (int(round(vy_bg * t)), int(round(vx_bg * t))), (0, 1))
        bg[BAND:SIZE - BAND] = perm[bg[BAND:SIZE - BAND]]
        backs[t] = bg
        _draw_sprites(layers[t], sprites)
        layers[t] = np.where(layers[t] != TRANSPARENT, perm[layers[t]], TRANSPARENT)
    return _finish(backs, layers, tick, slots, rng)
