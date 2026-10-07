"""Layered NES frames (data/nes_layers.py): the canvas keeps the world, cells follow the camera, the sprite
picture holds the sprite pixels that show, and the layers compose back to the picture by the PPU's rule."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from token_world.data.model_frames import NES_TO_INDEX
from token_world.data.nes_layers import (BANDS, BEHIND, CELL, ON_SCREEN, REGIONS, SPRITES, TRANSPARENT, VIEW,
                                         LayerCanvas, compose, model_frame)

SLOTS = np.array([55, 56, 57], np.uint8)


def regions(bg, sp=None, owner=None, scroll_x=0, scroll_y=0, oam=None, palette=None):
    """Fake PPU regions: background [240, 256] (4 * palette + colour), sprites and owners [240, 256]."""
    scroll = np.zeros((240, 2), "<u2")
    scroll[:, 0] = scroll_x if np.ndim(scroll_x) == 0 else np.asarray(scroll_x)
    scroll[:, 1] = (scroll_y + np.arange(240)) % 480
    return {"palette_ram": palette if palette is not None else np.arange(32, dtype=np.uint8) + 0x10,
            "ppu_background": bg.astype(np.uint8).ravel(),
            "ppu_sprites": (np.zeros((240, 256)) if sp is None else sp).astype(np.uint8).ravel(),
            "ppu_sprite_index": (np.zeros((240, 256)) if owner is None else owner).astype(np.uint8).ravel(),
            "ppu_scroll": scroll.view(np.uint8).ravel(),
            "oam": (np.full((64, 4), 0xF8) if oam is None else oam).astype(np.uint8).ravel()}


def expected(bg, sp, palette):
    """The PPU's composition, in model colours."""
    colour = np.where(bg > 0, palette[bg], palette[0])
    s = sp & 0x1F
    show = (s > 0) & ((bg == 0) | ((sp & 0x20) == 0))
    colour = np.where(show, palette[s | np.where(s & 3, 0x10, 0)], colour)
    return NES_TO_INDEX[colour & 0x3F]


class LayerTests(unittest.TestCase):
    def test_a_still_background_composes_and_fills_its_cells(self):
        rng = np.random.default_rng(0)
        bg = rng.integers(0, 16, (240, 256))
        f = LayerCanvas().push(regions(bg), 5, SLOTS)
        self.assertEqual(f["cells"].shape, (BANDS, VIEW, CELL, CELL))
        palette = np.arange(32, dtype=np.uint8) + 0x10
        self.assertTrue((compose(f) == expected(bg, np.zeros_like(bg), palette)).all())
        self.assertTrue(f["cell_known"][:, :16].all())
        self.assertFalse(f["cell_known"][:, 16].any())                  # the 17th cell is not in view yet

    def test_cells_stay_with_the_world_as_the_camera_moves(self):
        world = np.random.default_rng(1).integers(1, 16, (240, 512))
        canvas = LayerCanvas()
        first = canvas.push(regions(world[:, :256]), 0, SLOTS)
        moved = canvas.push(regions(world[:, 21:277], scroll_x=21), 1, SLOTS)
        self.assertEqual(int(moved["cell_col"][3]), 1)                  # the view starts at the second cell
        self.assertTrue((moved["cells"][:, 0] == first["cells"][:, 1]).all())   # the same world cell
        palette = np.arange(32, dtype=np.uint8) + 0x10
        self.assertTrue((compose(moved) == expected(world[:, 21:277], np.zeros((240, 256), int), palette)).all())

    def test_a_split_band_keeps_its_own_camera(self):
        world = np.random.default_rng(2).integers(1, 16, (240, 512))
        x = np.where(np.arange(240) < 32, 0, 40)                       # a status bar, then the level scrolled
        bg = np.where(np.arange(240)[:, None] < 32, world[:, :256], world[:, 40:296])
        f = LayerCanvas().push(regions(bg, scroll_x=x), 0, SLOTS)
        self.assertEqual(f["camera"][:2, 0].tolist(), [0, 0])
        self.assertEqual(f["camera"][2:, 0].tolist(), [40] * (BANDS - 2))
        palette = np.arange(32, dtype=np.uint8) + 0x10
        self.assertTrue((compose(f) == expected(bg, np.zeros_like(bg), palette)).all())

    def test_the_sprite_picture_holds_the_sprite_pixels_that_show(self):
        bg = np.zeros((240, 256), int)
        bg[100:108, 60:68] = 5                                          # an opaque patch for the behind sprite
        bg[100:104, 60:68] = 0                                          # its top half transparent
        sp = np.zeros_like(bg)
        sp[100:108, 60:68] = 0x20 | 0x05                                # behind the background
        sp[50:58, 30:38] = 0x06                                         # in front
        f = LayerCanvas().push(regions(bg, sp), 0, SLOTS)
        layer = f["sprite_layer"]
        self.assertEqual(layer.shape, (240, 256))
        self.assertTrue((layer[50:58, 30:38] != TRANSPARENT).all())
        self.assertTrue((layer[100:104, 60:68] != TRANSPARENT).all())   # behind, over transparent background
        self.assertTrue((layer[104:108, 60:68] == TRANSPARENT).all())   # behind an opaque one: hidden
        self.assertEqual(int((layer != TRANSPARENT).sum()), 64 + 32)
        palette = np.arange(32, dtype=np.uint8) + 0x10
        self.assertTrue((compose(f) == expected(bg, sp, palette)).all())

    def test_sprites_keep_their_slots_and_both_forms_compose_alike(self):
        bg = np.zeros((240, 256), int)
        bg[100:108, 60:68] = 5                                          # an opaque patch for the behind sprite
        sp, owner = np.zeros_like(bg), np.zeros_like(bg)
        oam = np.full((64, 4), 0xF8)
        oam[3] = (99, 1, 0x20, 60)                                      # slot 3, behind the background
        oam[7] = (49, 2, 0, 30)                                         # slot 7, in front
        sp[100:108, 60:68], owner[100:108, 60:68] = 0x20 | 0x05, 4
        sp[50:58, 30:38], owner[50:58, 30:38] = 0x06, 8
        f = LayerCanvas().push(regions(bg, sp, owner, oam=oam), 0, SLOTS)
        self.assertEqual(f["sprites"].shape, (SPRITES, 8, 8))
        self.assertEqual(f["sprite_flags"][3], ON_SCREEN | BEHIND)
        self.assertEqual(f["sprite_flags"][7], ON_SCREEN)
        self.assertEqual(f["sprite_xy"][7].tolist(), [30, 50])
        self.assertTrue((f["sprites"][0] == TRANSPARENT).all())        # an empty slot
        palette = np.arange(32, dtype=np.uint8) + 0x10
        slots_only = {k: v for k, v in f.items() if k != "sprite_layer"}
        self.assertTrue((compose(f) == expected(bg, sp, palette)).all())
        self.assertTrue((compose(slots_only) == compose(f)).all())    # the slots drawn by the PPU's rule

    def test_the_model_frame_keeps_todays_layout(self):
        bg = np.random.default_rng(3).integers(1, 16, (240, 256))
        f = LayerCanvas().push(regions(bg), 12, SLOTS)
        frame = model_frame(compose(f), f["border"])
        self.assertEqual(frame.shape, (256, 256))
        self.assertTrue((frame[16:240] == compose(f)[8:232]).all())
        self.assertTrue(set(np.unique(frame[:16])) <= set(SLOTS.tolist()))


class LiveLayerTests(unittest.TestCase):
    """Real frames: needs World NES with the layer regions and the Tetris ROM."""

    def test_tetris_frames_compose_exactly(self):
        try:
            from world_nes import Capture, Console
            from token_world.data.rom import tetris_rom
            console = Console(tetris_rom(), capture=Capture(frames="palette", ram=True, regions=REGIONS))
        except Exception as error:                                       # noqa: BLE001
            self.skipTest(f"World NES with the layer regions and the ROM are needed: {error}")
        from token_world.data.world_nes_tetris import TetrisSession
        buf, obs, bot, canvas = console.allocate_observation(), console.observe(), TetrisSession(3), LayerCanvas()
        exact = 0
        for _ in range(1500):
            console.advance_frame_into(buf, buttons=(bot.act(obs.frames, obs.ram), 0))
            obs = buf.record
            f = canvas.push({k: obs.regions[k] for k in REGIONS}, int(obs.frame_id), SLOTS)
            real = NES_TO_INDEX[np.asarray(obs.frames).reshape(240, 256) & 0x3F]
            slots_only = {k: v for k, v in f.items() if k != "sprite_layer"}
            exact += bool((compose(f) == real).all() and (compose(slots_only) == real).all())
        self.assertGreaterEqual(exact, 1490)


if __name__ == "__main__":
    unittest.main()
