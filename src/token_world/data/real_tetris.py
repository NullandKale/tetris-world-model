"""The real game in World NES, one frame per step, as model frames: what the play app and the long-dream
trials run the world model against."""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch

from token_world.data.model_frames import border_slots, model_index_frames
from token_world.data.nes_layers import REGIONS, LayerCanvas
from token_world.data.rom import tetris_rom
from token_world.data.nes_palette import tetris_palette
from token_world.data.world_nes_tetris import BORDER_STATE, TetrisSession

CONTEXT = 48                                   # real frames a dream starts from
MODE, IN_GAME, STATE, FALLING = 0xC0, 4, 0x48, 1


class RealTetris:
    """The real game in World NES, one frame per step, with the last CONTEXT model frames and the
    controller byte that produced each. The bot plays unless step() is given a button byte."""

    def __init__(self, seed: int, layered: bool = False):
        """layered: also keep the last CONTEXT frames' layers (data/nes_layers.py; needs World NES with the
        layer regions)."""
        from world_nes import Capture, Console
        self.console = Console(tetris_rom(),
                               capture=Capture(frames="palette", ram=True, regions=REGIONS if layered else ()))
        self.canvas = LayerCanvas() if layered else None
        self.layers: deque[dict] = deque(maxlen=CONTEXT)
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
        self.frames.append(frame)
        self.actions.append(int(action))
        if self.canvas is not None:
            self.layers.append(self.canvas.push({k: o.regions[k] for k in REGIONS}, int(o.frame_id), self.slots,
                                                o.ram[list(BORDER_STATE)]))
        return frame, int(action)

    def playing(self) -> bool:
        return bool(self.observation.ram[MODE] == IN_GAME and self.observation.ram[STATE] == FALLING)

    def context(self, device="cuda") -> tuple[torch.Tensor, torch.Tensor]:
        """The last CONTEXT frames [T, 256, 256] and the buttons that produced them [T]."""
        return torch.from_numpy(np.stack(self.frames)).to(device), torch.tensor(list(self.actions), device=device)
