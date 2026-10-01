"""Long dreams with no buttons: does the world model keep a falling piece intact and let it fall?

A trial is a real game paused at a fresh piece: its last CONTEXT real frames and the buttons that made
them, then FUTURE real frames in which nobody presses anything (gravity alone). A model dreams the same
FUTURE frames from the context. Scored on the playfield: pixels wrong at HORIZONS, and the block mass at
the last frame against the real one (1.0 = as many filled pixels; pieces that melt or grow move it).
Trials are live games (no stored set). Used by scripts/long_dream_check.py, and by training every
LONG_DREAM_EVERY steps (scripts/train_dynamics_ui.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import IterableDataset, get_worker_info

from token_world.data.real_tetris import CONTEXT, RealTetris
from token_world.models.dynamics import Decoding, Dreamer

FUTURE = 128
PLAYFIELD = (slice(52, 212), slice(92, 172))
CROP = (slice(44, 212), slice(88, 176))
HORIZONS = (16, 32, 64, 128)
BANDS = ((0, 4), (5, 9), (10, 14), (15, 19))
LEVEL, PIECE_Y = 0x44, 0x41


@dataclass
class Trial:
    context: np.ndarray           # [CONTEXT, 256, 256] uint8 model frames
    actions: np.ndarray           # [CONTEXT] the button byte that produced each
    future: np.ndarray            # [FUTURE, 256, 256] the real frames with no buttons
    level: int


def trials(seed: int, per_band: int | None = None) -> Iterator[Trial]:
    """Real games paused at a fresh piece, then FUTURE frames with no buttons. per_band: that many from
    each level band in BANDS, then stop; None: endless, whatever levels the bot's games reach."""
    found = {band: 0 for band in BANDS}
    while per_band is None or any(n < per_band for n in found.values()):
        real = RealTetris(seed)
        seed += 1
        for played in range(20_000):
            real.step(None)
            ram = real.observation.ram
            if played > 700 and real.playing() and int(ram[PIECE_Y]) <= 1 and len(real.frames) == CONTEXT:
                level = int(ram[LEVEL])
                band = next((b for b in BANDS if b[0] <= level <= b[1]), None)
                if per_band is not None and (band is None or found[band] >= per_band):
                    break
                if band is not None:
                    found[band] += 1
                context, actions = np.stack(real.frames), np.array(real.actions, np.int64)
                future = np.stack([real.step(0)[0] for _ in range(FUTURE)])
                yield Trial(context, actions, future, level)
                break


class TrialStream(IterableDataset):
    """Endless live trials, one console per DataLoader worker (training keeps one worker filling it)."""

    def __init__(self, seed: int):
        self.seed = seed

    def __iter__(self):
        info = get_worker_info()
        yield from trials(self.seed * 1000 + (info.id if info else 0))


@torch.no_grad()
def dream(model, trial: Trial, decoding: Decoding, keep: int = 48) -> np.ndarray:
    """The model's FUTURE frames [FUTURE, 256, 256] from the trial's context, nobody pressing anything."""
    device = next(model.parameters()).device
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        dreamer = Dreamer(model, torch.from_numpy(trial.context).to(device),
                          torch.from_numpy(trial.actions).to(device), decoding, keep)
        return np.stack([dreamer.step(0).cpu().numpy() for _ in range(FUTURE)])


def score(trial: Trial, dreamed: np.ndarray) -> dict[str, float]:
    """Playfield pixels wrong at each horizon (wrong_16 ...) and block mass at the last frame (mass)."""
    out = {f"wrong_{h}": float((dreamed[h - 1][PLAYFIELD] != trial.future[h - 1][PLAYFIELD]).mean())
           for h in HORIZONS}
    real = trial.future[-1][PLAYFIELD]
    background = np.bincount(real.ravel()).argmax()                       # the empty well's colour
    out["mass"] = float((dreamed[-1][PLAYFIELD] != background).sum() / max((real != background).sum(), 1))
    return out


def sheet(trial: Trial, dreams: dict[str, np.ndarray], palette: np.ndarray, every: int = 8) -> Image.Image:
    """Rows: the real game, then each dream; columns: the playfield every `every` frames, scaled 2x."""
    h, w = trial.future[0][CROP].shape
    rows = {"real": trial.future} | dreams
    columns = range(every - 1, FUTURE, every)
    out = Image.new("RGB", (220 + len(columns) * (w * 2 + 4), 20 + len(rows) * (h * 2 + 6)), "#181820")
    draw = ImageDraw.Draw(out)
    draw.text((4, 4), f"level {trial.level}, no buttons for {FUTURE} frames; playfield every {every} frames",
              fill="white")
    for r, (name, frames) in enumerate(rows.items()):
        y = 20 + r * (h * 2 + 6)
        draw.text((4, y + h), name, fill="white")
        for c, f in enumerate(columns):
            out.paste(Image.fromarray(palette[frames[f][CROP]]).resize((w * 2, h * 2), Image.NEAREST),
                      (220 + c * (w * 2 + 4), y))
    return out


def animation(trial: Trial, dreams: dict[str, np.ndarray], palette: np.ndarray, every: int = 2) -> list[Image.Image]:
    """GIF frames: the real playfield next to each dream's, every `every` frames, scaled 2x, labelled."""
    h, w = trial.future[0][CROP].shape
    panels = {"real": trial.future} | dreams
    width = len(panels) * (w * 2 + 8)
    frames = []
    for f in range(0, FUTURE, every):
        image = Image.new("RGB", (width, h * 2 + 34), "#181820")
        draw = ImageDraw.Draw(image)
        for i, (name, seq) in enumerate(panels.items()):
            x = i * (w * 2 + 8)
            draw.text((x + 2, 2), name, fill="white")
            image.paste(Image.fromarray(palette[seq[f][CROP]]).resize((w * 2, h * 2), Image.NEAREST), (x, 16))
        draw.text((2, h * 2 + 20), f"level {trial.level}, frame +{f + 1}, no buttons", fill="#a8a8b8")
        frames.append(image)
    return frames
