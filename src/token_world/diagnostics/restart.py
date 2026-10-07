"""Game end -> restart: does a model dream the top-out curtain, the menus and a new game, fed the real buttons?

Restarts are found live (no stored eval set): the bot plays real games (data/real_tetris.py) until one tops
out; the last CONTEXT real frames are the context, then FUTURE real frames follow with the bot's real buttons
(Start taps, the level menu, a new game). Any model dreams them, fed the same buttons (diagnostics/worlds.py
World.follow; scripts/event_checks.py), and is scored on whole-screen pixels wrong (each pixel's most likely
colour) at milestones:

- curtain: 4 frames before the level menu shows (the curtain closing on the dead board);
- menu: 16 frames into the level menu;
- play / play64: 8 and 63 frames into the new game;
- copy_menu / copy_play: the same for copying the top-out frame, the floor a model has to beat.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from token_world.data.real_tetris import CONTEXT, RealTetris
from token_world.data.world_nes_tetris import GAME_MODE, GAME_OVER, LEVEL_MENU, PLAY_STATE

FUTURE = 320
MILESTONES = ("curtain", "menu", "play", "play64")
FIELDS = (*MILESTONES, "copy_menu", "copy_play")


@dataclass
class Restart:
    context: np.ndarray           # [CONTEXT, 256, 256] uint8 model frames up to the top-out
    actions: np.ndarray           # [CONTEXT] the button byte that produced each
    layers: dict | None           # the context's layered frames, each key [CONTEXT, ...] (data/nes_layers.py)
    future: np.ndarray            # [FUTURE, 256, 256] the real frames after
    future_actions: np.ndarray    # [FUTURE] the bot's real buttons
    menu: int                     # future frame the level menu first shows
    play: int                     # future frame the new game's first piece falls


def restarts(seed: int, layered: bool = True) -> Iterator[Restart]:
    """Endless live restarts, each from a fresh game: the bot plays to a top-out, then on into a new game."""
    while True:
        real = RealTetris(seed, layered)
        seed += 1
        for t in range(60_000):
            real.step(None)
            if t > 500 and real.observation.ram[PLAY_STATE] == GAME_OVER and len(real.frames) == CONTEXT:
                context, actions = np.stack(real.frames), np.array(real.actions, np.int64)
                layers = {k: np.stack([f[k] for f in real.layers]) for k in real.layers[0]} if layered else None
                future, acts, menu, play = [], [], None, None
                for h in range(FUTURE):
                    frame, action = real.step(None)
                    future.append(frame)
                    acts.append(action)
                    if menu is None and real.observation.ram[GAME_MODE] == LEVEL_MENU:
                        menu = h
                    if menu is not None and play is None and real.playing():
                        play = h
                if play is not None and play + 64 <= FUTURE and menu >= 4:
                    yield Restart(context, actions, layers, np.stack(future), np.array(acts, np.int64), menu, play)
                break


def score(restart: Restart, dreamed: np.ndarray) -> dict[str, float]:
    """Whole-screen pixels wrong at each milestone, and the same for copying the top-out frame."""
    real, menu, play = restart.future, restart.menu, restart.play
    wrong = (dreamed != real).mean((1, 2))
    copy = (real != restart.context[-1]).mean((1, 2))
    return {"curtain": float(wrong[menu - 4]), "menu": float(wrong[menu + 16]), "play": float(wrong[play + 8]),
            "play64": float(wrong[play + 63]), "copy_menu": float(copy[menu + 16]), "copy_play": float(copy[play + 8])}


def marks(restart: Restart) -> list[int]:
    """The future frames a sheet shows: early curtain, the milestones, and around the new game's start."""
    menu, play = restart.menu, restart.play
    return [0, min(75, menu - 5), menu - 4, menu + 2, menu + 16, play - 4, play + 8, play + 32, play + 63]


def sheet(restart: Restart, dreams: dict[str, np.ndarray], palette: np.ndarray) -> np.ndarray:
    """The real frames at marks(restart) on top, one row per model's dream below -> RGB [rows * 256, 9 * 256, 3]."""
    at = marks(restart)
    rows = [restart.future, *dreams.values()]
    return np.concatenate([np.concatenate([palette[r[m]] for m in at], 1) for r in rows], 0).astype(np.uint8)
