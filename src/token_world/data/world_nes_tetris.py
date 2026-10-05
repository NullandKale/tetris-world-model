"""Live NES Tetris histories from World NES (../llm-thing23), played by the RAM-driven bot in tetris_bot.py.

One World NES console per DataLoader worker, in external mode
(world_nes_windows.py): each frame the worker's TetrisSession reads the
observation (palette plane and RAM) and chooses the controller byte. The
session is the same player as the nes-py stream it replaces
(tetris_stream.ContinuousTetrisSession), except that the brains (perfect /
random / idle, tetris_bot.BrainSwitcher) read RAM instead of pixels. During a
game the switcher plays; after a top-out
it taps Start through the curtain to the level menu, picks a sampled start
level with the menu script, and hands control back once the first piece
spawns. Boot goes through the same menu path. The emulator is never reset.

Workers yield uint8 model index frames (model_frames.py), about 12x less
data than the float RGB frames before.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from torch.utils.data import IterableDataset

from token_world.data.model_frames import border_slots
from token_world.data.rom import tetris_rom
from token_world.data.tetris_bot import A, DOWN, LEFT, RIGHT, START, UP, BrainSwitcher
from token_world.data.world_nes_windows import window_stride, world_nes_windows, worker_seed

BORDER_STATE = (0x45, 0x46)     # drawn in every frame's border: the fall timer and the autorepeat counter

GAME_MODE, LEVEL_MENU, IN_GAME = 0xC0, 3, 4          # RAM game mode: level menu, in a game
PLAY_STATE, ACTIVE, GAME_OVER = 0x48, 1, 10          # RAM play state: piece falling, top-out curtain
PIECE = 0x42                                         # current piece id; 0 until a game's first piece spawns
MENU_TAP_GAP = 3                                     # level-menu cursor moves on press edges only
CURTAIN_FRAMES = 150                                 # the top-out curtain animates ~144 frames, then waits
FREEZE_FRAMES = 1200                                 # identical game image outside a restart -> error
RESTART_LIMIT, BOOT_LIMIT = 2400, 3600               # frames allowed to reach a playable game


class TetrisSession:
    """Chooses one controller byte per observation, for one continuous stream."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.policy = None
        self.restart_phase: str | None = "seek_menu"  # boot reaches the first game like any restart
        self.restart_ticks = self.restart_total_ticks = 0
        self.restart_limit = BOOT_LIMIT
        self.restart_script: deque[int] = deque()
        self.restart_count = -1                     # completed in-game restarts; the boot start is not one
        self.game_over_frames = self.same_frames = 0
        self.images = np.zeros((2, 240, 256), np.uint8)     # this and the previous palette plane
        self.observed = 0

    @property
    def in_game_restarts(self) -> int:
        return max(self.restart_count, 0)

    def _tap(self, button: int, taps: int) -> list[int]:
        return [button, *[0] * MENU_TAP_GAP] * taps

    def _start_level(self) -> int:
        """Decides the next game's kind (BrainSwitcher: long or short) and its start level."""
        self.long_game = bool(self.rng.random() < BrainSwitcher.LONG_SHARE)
        levels = BrainSwitcher.LONG_START_LEVELS if self.long_game else BrainSwitcher.SHORT_START_LEVELS
        return int(self.rng.integers(0, levels))

    def _menu_script(self) -> deque[int]:
        """Level-menu buttons for a sampled start level: UP then LEFT x4 reaches level 0 from any
        cursor; RIGHT/DOWN step +1/+5; holding A while pressing Start adds 10."""
        level = self._start_level()
        script = [0] * 10
        script += self._tap(UP, 1) + self._tap(LEFT, 4)
        script += self._tap(RIGHT, level % 5) + self._tap(DOWN, (level % 10) // 5)
        if level >= 10:
            script.extend((A, A))
        script.append(A | START if level >= 10 else START)
        return deque(script)

    def _resume_play(self, ram: np.ndarray) -> int:
        self.restart_phase = None
        self.policy = BrainSwitcher(self.rng, self.long_game)
        self.restart_count += 1
        self.restart_total_ticks = 0
        self.restart_limit = RESTART_LIMIT
        return int(self.policy(ram, self.rng))

    def _record(self, same: bool, ram: np.ndarray) -> None:
        """Bookkeeping for an observation produced by the previous action."""
        self.same_frames = self.same_frames + 1 if same else 0
        if self.restart_phase is not None:
            self.restart_ticks += 1
            self.restart_total_ticks += 1
            if self.restart_total_ticks > self.restart_limit:
                raise RuntimeError("Tetris did not reach a playable game")
        self.game_over_frames = self.game_over_frames + 1 if ram[PLAY_STATE] == GAME_OVER else 0
        if self.restart_phase is None and self.game_over_frames > CURTAIN_FRAMES:
            self.restart_phase = "seek_menu"
            self.restart_ticks = self.restart_total_ticks = 0
        if self.same_frames > FREEZE_FRAMES and self.restart_phase is None:
            raise RuntimeError("Tetris picture froze outside a restart")

    def act(self, palette: np.ndarray, ram: np.ndarray) -> int:
        """palette [240, 256] PPU indices and ram [2048] of the current observation -> button byte."""
        current, previous = self.images[self.observed % 2], self.images[(self.observed + 1) % 2]
        np.copyto(current, palette)
        if self.observed:
            self._record(np.array_equal(current, previous), ram)
        self.observed += 1
        if self.restart_phase is None:
            return int(self.policy(ram, self.rng))
        playing = ram[PLAY_STATE] == ACTIVE and ram[PIECE] != 0 and ram[GAME_MODE] == IN_GAME
        if self.restart_phase == "seek_menu":
            if ram[GAME_MODE] == LEVEL_MENU:
                self.restart_script = self._menu_script()
                self.restart_phase = "menu_script"          # falls through to play its first button
            elif playing:
                return self._resume_play(ram)
            else:
                return START if self.restart_ticks % 4 == 0 else 0
        if self.restart_phase == "menu_script":
            action = self.restart_script.popleft()
            if not self.restart_script:
                self.restart_phase = "await_piece"
                self.restart_ticks = 0
            return action
        if playing:                                         # await_piece
            return self._resume_play(ram)
        if self.restart_ticks > 900:
            self.restart_phase = "seek_menu"
            self.restart_ticks = 0
        return 0


class WorldNesTetrisStreams(IterableDataset):
    """Endless windows of `frames` consecutive observations from one World NES console per worker.

    Each window adds `stride` new observations (default frames - 1: consecutive
    windows share one). A smaller stride uses every emulated frame in more
    windows, at different positions: the emulator is 83% of a worker's CPU, so
    this is how the stream keeps up with a fast model. Every window is kept.
    Yields x
    [frames, 256, 256] uint8 model indices; action [frames]
    long, where action[t] took frame t to t + 1 and the last entry repeats the
    one before it as padding; ram [frames, 2048] uint8, each frame's console RAM
    (for event detection, not training); tick = the last frame's counter.
    """

    def __init__(self, frames: int = 64, seed: int = 17, repo: str | None = None, stride: int | None = None):
        self.frames, self.seed, self.repo = int(frames), int(seed), repo
        self.new_frames = window_stride(self.frames, stride)

    def __iter__(self):
        from token_world.data.nes_palette import tetris_palette
        worker_id, seed = worker_seed(self.seed)
        rom = tetris_rom(self.repo)
        session = TetrisSession(seed)
        yield from world_nes_windows(rom, session, border_slots(tetris_palette()), BORDER_STATE,
                                     self.frames, self.new_frames, worker_id)
