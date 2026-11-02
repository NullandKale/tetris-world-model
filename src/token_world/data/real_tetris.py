"""The real game in World NES, one frame per step, as model frames: what the play app and the long-dream
trials run the world model against."""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch

from token_world.data.model_frames import blank_bands, border_slots, model_index_frames
from token_world.data.nes_layers import REGIONS, TWO_LAYER_REGIONS, LayerCanvas, two_layers
from token_world.data.rom import tetris_rom
from token_world.data.nes_palette import tetris_palette
from token_world.data.world_nes_tetris import BORDER_STATE, TetrisSession

CONTEXT = 48                                   # real frames a dream starts from
MODE, IN_GAME, STATE, FALLING = 0xC0, 4, 0x48, 1


class RealTetris:
    """The real game in World NES, one frame per step, with the last CONTEXT model frames and the
    controller byte that produced each. The bot plays unless step() is given a button byte."""

    def __init__(self, seed: int, layered: bool = False, two: bool = False, blank: bool = False):
        """layered: also keep the last CONTEXT frames' layers (data/nes_layers.py; needs World NES with the
        layer regions); two: their two layers [2, 256, 256] (nes_layers.two_layers, the --two-layers models');
        blank: frames in the BLANK_BORDER layout (not with layered)."""
        if blank and layered:
            raise ValueError("blank: the layered frames draw their own border")
        self.blank = blank
        from world_nes import Capture, Console
        regions = REGIONS if layered else TWO_LAYER_REGIONS if two else ()
        self.console = Console(tetris_rom(), capture=Capture(frames="palette", ram=True, regions=regions))
        self.canvas = LayerCanvas() if layered else None
        self.layers: deque[dict] = deque(maxlen=CONTEXT)
        self.two = two
        self.twos: deque[np.ndarray] = deque(maxlen=CONTEXT)
        self.buffer = self.console.allocate_observation()
        self.observation = self.console.observe()
        self.slots = border_slots(tetris_palette())
        self.seed = seed
        self.bot = TetrisSession(seed)
        self.frames: deque[np.ndarray] = deque(maxlen=CONTEXT)
        self.actions: deque[int] = deque(maxlen=CONTEXT)

    def new_bot(self) -> None:
        """A fresh bot: it finds its way from whatever screen the game is on (menus or play)."""
        self.seed += 1
        self.bot = TetrisSession(self.seed)

    def step(self, action: int | None) -> tuple[np.ndarray, int]:
        """Advance one frame with `action` (None: the bot chooses) -> (model frame [256, 256], action)."""
        if action is None:
            action = self.bot.act(self.observation.frames, self.observation.ram)
        self.console.advance_frame_into(self.buffer, buttons=(int(action), 0))
        self.observation = self.buffer.record
        o = self.observation
        frame = model_index_frames(o.frames[None], o.emphasis[None], np.array([o.frame_id], np.int64), self.slots,
                                   o.ram[None][:, BORDER_STATE])[0]
        if self.blank:
            blank_bands(frame, self.slots)
        self.frames.append(frame)
        self.actions.append(int(action))
        if self.two:
            self.twos.append(two_layers(frame, {k: o.regions[k] for k in TWO_LAYER_REGIONS}))
        if self.canvas is not None:
            self.layers.append(self.canvas.push({k: o.regions[k] for k in REGIONS}, int(o.frame_id), self.slots,
                                                o.ram[list(BORDER_STATE)]))
        return frame, int(action)

    def playing(self) -> bool:
        return bool(self.observation.ram[MODE] == IN_GAME and self.observation.ram[STATE] == FALLING)

    def context(self, device="cuda") -> tuple[torch.Tensor, torch.Tensor]:
        """The last CONTEXT frames [T, 256, 256] and the buttons that produced them [T]."""
        return torch.from_numpy(np.stack(self.frames)).to(device), torch.tensor(list(self.actions), device=device)


START = 3                        # frames a dream starts from, as the browser's (prefill takes all but the last)
LEVEL = 0x44                     # the game's level byte
SETTLE = 8                       # frames into the game the start ends: the camera has moved to the playfield


def game_start(seed: int = 0) -> RealTetris:
    """A live game at the browser's start (scripts/export_onnx.py ships its last START frames' layers): SETTLE
    frames into a level-0 game, after its black frames (the bot plays the menus; seeds from `seed` are tried
    until it picks level 0), the camera the same over the last START frames. Ending on the playfield: from the
    black frames alone the model dreams the menu back first; and past the playfield's camera switch (the menus'
    nametable to the playfield's, 2 frames in), as the dreams hold the start's camera."""
    for seed in range(seed, seed + 200):
        game = RealTetris(seed, layered=True, two=True)        # the two layers too: a two-layer model's start
        for _ in range(20_000):
            frame, _ = game.step(None)
            if game.observation.ram[MODE] == IN_GAME and (frame[16:240] == frame[16, 0]).all():
                break                                    # the first black frame of the game
        else:
            continue
        while True:                                      # to the last one
            frame, _ = game.step(None)
            if not (frame[16:240] == frame[16, 0]).all():
                break
        for _ in range(SETTLE):                          # the playfield's camera (its nametable) comes 2 frames in
            game.step(None)
        if game.observation.ram[LEVEL] != 0:             # (set by now: Start held with A adds 10 levels)
            continue
        frames = list(game.layers)[-START:]
        if any((f["camera"] != frames[-1]["camera"]).any() for f in frames):
            continue
        return game
    raise RuntimeError("no level-0 game start found")


def start_context(game: RealTetris) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray | None]:
    """A game's last START frames: model frames [START, 256, 256], controller bytes [START], layers (each key
    [START, ...]) and, if the game keeps them, their two layers [START, 2, 256, 256], what a dream starts from. A
    --two-layers model starts from the two layers: given the pictures alone it reads them as background, the
    falling piece as settled stack, and the dream's pieces never fall (2026-10-30)."""
    layers = list(game.layers)[-START:]
    return (np.stack(list(game.frames)[-START:]), np.array(list(game.actions)[-START:], np.int64),
            {k: np.stack([f[k] for f in layers]) for k in layers[0]},
            np.stack(list(game.twos)[-START:]) if game.two else None)

