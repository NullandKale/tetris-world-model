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

The training streams can play ScreensSession (2026-10-10): the same bot, plus the screens a player sees
around play: pauses, the game's soft reset (A+B+Select+Start: the legal screen, the title, the menus) and
short waits (at most STILL frames) on the game-over screen, the title and the menus. Their rates are a Screens
setting per stream: WORLD_MODEL for the world models, VAE for the VAE (screens() labels each frame's screen
from RAM; scripts/data_report.py measures what a setting does to the screens, the events and the coverage).
A stream can also toss windows by what is in them (Toss, WindowToss): VAE_TOSS for the VAE, which needs the
rare pictures (clears, level ups, the curtain, the screens) more often than the game shows them; the world
models keep every window, so events come at the game's own rates.
A stream can also give some workers' windows a counterfactual twin (TurnBranches, 2026-10-16): the same
window replayed from the emulator's saved state at one of its turn presses, with the other turn pressed or
none, so a model has windows that differ by the button alone.
The default, no Screens, is TetrisSession alone: what the play app, the long-dream trials, the scenario checks
and the browser's start game run (real_tetris.py), the same games as before.

Workers yield uint8 model index frames (model_frames.py), about 12x less
data than the float RGB frames before.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset

from token_world.data.model_frames import border_slots
from token_world.data.rom import tetris_rom
from token_world.data.tetris_bot import A, B, BOARD, DOWN, EMPTY, LEFT, RIGHT, SELECT, START, UP, BrainSwitcher
from token_world.data.tetris_events import ALONE, LOOKAHEAD, events_at, frame_events
from token_world.data.tetris_motion import stream_motion
from token_world.data.world_nes_windows import window_stride, world_nes_windows, worker_seed

BORDER_STATE = (0x45, 0x46)     # drawn in every frame's border: the fall timer and the autorepeat counter

GAME_MODE, LEVEL_MENU, IN_GAME = 0xC0, 3, 4          # RAM game mode: level menu, in a game
LEGAL, TITLE, GAME_TYPE, DEMO = 0, 1, 2, 5           # the other game modes (6: one frame into the demo)
PAUSE_FLAG, PAUSED = 0xA7, 7                         # 7 while paused, 2 in play (found 2026-10-10)
PLAY_STATE, ACTIVE, GAME_OVER = 0x48, 1, 10          # RAM play state: piece falling, top-out curtain
PIECE = 0x42                                         # current piece id; 0 until a game's first piece spawns
MENU_TAP_GAP = 3                                     # level-menu cursor moves on press edges only
CURTAIN_FRAMES = 150                                 # the top-out curtain animates ~144 frames, then waits
FREEZE_FRAMES = 1200                                 # identical game image outside a restart -> error
RESTART_LIMIT, BOOT_LIMIT = 2400, 3600               # frames allowed to reach a playable game
SOFT_RESET = A | B | SELECT | START                  # the game's own reset: back to the legal screen
SCREENS = ("play", "paused", "game over", "legal", "title", "demo", "menu", "other")


def screens(ram: np.ndarray) -> np.ndarray:
    """RAM [..., 2048] -> each frame's screen, as an index into SCREENS."""
    mode, state = ram[..., GAME_MODE], ram[..., PLAY_STATE]
    out = np.full(mode.shape, SCREENS.index("other"), np.int64)
    out[mode == IN_GAME] = SCREENS.index("play")
    out[(mode == IN_GAME) & (ram[..., PAUSE_FLAG] == PAUSED)] = SCREENS.index("paused")
    out[(mode == IN_GAME) & (state == GAME_OVER)] = SCREENS.index("game over")
    out[mode == LEGAL] = SCREENS.index("legal")
    out[mode == TITLE] = SCREENS.index("title")
    out[(mode == DEMO) | (mode == DEMO + 1)] = SCREENS.index("demo")
    out[(mode == GAME_TYPE) | (mode == LEVEL_MENU)] = SCREENS.index("menu")
    return out


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
        return self._decide(ram)

    def _decide(self, ram: np.ndarray) -> int:
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


STILL = 48                                           # the longest any added wait lasts: level 0's gravity, a
                                                     # row every 48 frames, so no wait outlasts a still piece


@dataclass(frozen=True)
class Screens:
    """How often the screens around play come up in a stream (ScreensSession). Every wait is at most STILL frames.

    pause_every: frames of play between pauses, on average; pause: a pause's length.
    end_reset: the share of games that end with a soft reset (the legal screen, the title, the menus) instead
    of Start; reset_every: frames of play between soft resets mid-game, on average, in short games only (a
    long game is never cut: those climb the levels and their palettes), or None for none.
    wait: frames on the game-over screen after the curtain, and on the title and the menus before Start.
    """
    pause_every: int
    pause: tuple[int, int]
    end_reset: float
    reset_every: int | None
    wait: tuple[int, int] = (0, STILL)

    def __post_init__(self):
        if max(self.pause[1], self.wait[1]) > STILL:
            raise ValueError(f"no wait may pass {STILL} frames")


# The world model's stream: the screens come up now and then, so their transitions are learned, while the
# events keep their rates (scripts/data_report.py compares them with PLAIN). The VAE's stream: more of them, as
# it needs every screen's look, not the game's flow.
WORLD_MODEL = Screens(pause_every=6000, pause=(4, STILL), end_reset=0.15, reset_every=30_000)
VAE = Screens(pause_every=1500, pause=(4, STILL), end_reset=0.5, reset_every=None)   # no mid-game resets:
# they cut short games before their top-out (data report 2026-10-10: top-outs -40%, singles +16%)


class ScreensSession(TetrisSession):
    """TetrisSession's bot, plus the screens around play at the rates of `screens` (Screens). Its own random
    draws (`events`), so the bot's choices are still those of the seed.

    - pauses: in play, left with Start again; a Start the game did not take (PAUSE_FLAG unchanged) is given
      up, so a pause never goes unseen;
    - soft resets (SOFT_RESET): the legal screen (the game holds it ~260 frames, whatever is pressed), the title
      and the menus; at the end of a game, and mid-game in short games;
    - waits, each at most STILL frames: on the game-over screen after the curtain, then on the title and the
      menus before Start is tapped. (The title's demo needs ~1,050 idle frames, so it never comes up.)
    """

    LIMIT = RESTART_LIMIT + 4 * STILL                # frames allowed to reach play: the restart, plus the waits

    def __init__(self, seed: int, screens: Screens):
        super().__init__(seed)
        self.screens = screens
        self.events = np.random.default_rng([seed, 1])
        self.restart_limit = BOOT_LIMIT + 4 * STILL
        self.holding = 0                            # frames left holding SOFT_RESET
        self.pause: int | None = None               # this pause's length, while paused
        self.pause_ticks = 0
        self.end_left: int | None = None            # game-over wait left; reset afterwards?
        self.end_reset = False
        self.screen: int | None = None              # the screen being waited on, and the wait left
        self.wait = 0

    def _draw(self, low_high: tuple[int, int]) -> int:
        return int(self.events.integers(low_high[0], low_high[1] + 1))

    def _menu_script(self) -> deque[int]:
        script = super()._menu_script()
        script.extendleft([0] * self._draw(self.screens.wait))
        return script

    def _soft_reset(self) -> int:
        self.holding = 1                            # held for two frames
        self.restart_phase = "seek_menu"
        self.restart_ticks = self.restart_total_ticks = 0
        self.restart_limit = self.LIMIT
        self.pause, self.end_left, self.screen = None, None, None
        return SOFT_RESET

    def _resume_play(self, ram: np.ndarray) -> int:
        action = super()._resume_play(ram)
        self.restart_limit, self.screen = self.LIMIT, None
        return action

    def _paused(self, ram: np.ndarray) -> int:
        self.pause_ticks += 1
        if self.pause_ticks == 4 and ram[PAUSE_FLAG] != PAUSED:       # the Start was not taken: play on
            self.pause = None
            return super()._decide(ram)
        if self.pause_ticks < self.pause:
            return 0
        if ram[PAUSE_FLAG] == PAUSED:               # leave: Start again, every 4 frames until it is taken
            return START if (self.pause_ticks - self.pause) % 4 == 0 else 0
        self.pause = None
        return super()._decide(ram)

    def _decide(self, ram: np.ndarray) -> int:
        s = self.screens
        if self.holding:
            self.holding -= 1
            return SOFT_RESET
        if self.restart_phase is None:
            if self.pause is not None:
                return self._paused(ram)
            if ram[PLAY_STATE] == ACTIVE and self.events.random() < 1 / s.pause_every:
                self.pause, self.pause_ticks = self._draw(s.pause), 0
                return START
            if s.reset_every and not self.long_game and self.events.random() < 1 / s.reset_every:
                return self._soft_reset()
            return super()._decide(ram)
        mode = int(ram[GAME_MODE])
        if self.restart_phase == "seek_menu" and mode == IN_GAME and ram[PLAY_STATE] == GAME_OVER:
            if self.end_left is None:               # the curtain is down: wait, then Start or a reset
                self.end_left = self._draw(s.wait)
                self.end_reset = bool(self.events.random() < s.end_reset)
            if self.end_left > 0:
                self.end_left -= 1
                return 0
            if self.end_reset:
                self.end_reset = False
                return self._soft_reset()
        else:
            self.end_left = None
        if self.restart_phase == "seek_menu" and mode in (TITLE, GAME_TYPE):
            if mode != self.screen:                 # a new screen: wait on it before tapping Start
                self.screen, self.wait = mode, self._draw(s.wait)
            if self.wait > 0:
                self.wait -= 1
                return 0
        return super()._decide(ram)


@dataclass(frozen=True)
class Toss:
    """Which windows a stream keeps (WindowToss): each with the chance `keep` gives the rarest thing in it (an
    event of tetris_events.events_at, in the window or in the `aftermath` frames before it, so a clear's
    animation, counters and the curtain after a top-out count too; or a screen of SCREENS other than play shown
    in it), else `plain`."""
    keep: dict
    plain: float
    aftermath: int = 32


# The VAE's (2026-10-10; the world models' recipe dropped tossing 2026-10-05: it kept too few windows a second
# for them, and they learn the game's own event rates). The old world-model chances, plus every screen; the
# long, still legal and game-over screens at 0.3 (at 1.0 they were 21% of the VAE's frames, nearly the same
# picture), plain play at 0.05 (at 0.1 rare events gained only ~3x). Measured: scripts/data_report.py.
VAE_TOSS = Toss(keep={"line clear: double": 1.0, "line clear: triple": 1.0, "line clear: tetris": 1.0,
                      "level up": 1.0, "top-out": 1.0, "game start": 1.0,
                      "paused": 1.0, "title": 1.0, "menu": 1.0, "other": 1.0,
                      "game over": 0.3, "legal": 0.3,
                      "line clear: single": 0.4, "falling alone": 0.4},
                plain=0.05)


def spawn_boards(ram: np.ndarray, new: int) -> list[bytes]:
    """RAM [T, 2048] -> the settled board (200 cells, packed bits) at each piece spawn among the last `new`
    frames (diagnostics/novelty.py: how many boards a run has seen)."""
    keys = []
    for t in range(max(1, len(ram) - new), len(ram)):
        if ram[t, GAME_MODE] == IN_GAME and ram[t, PLAY_STATE] == ACTIVE and ram[t - 1, PLAY_STATE] != ACTIVE:
            keys.append(np.packbits(ram[t, BOARD:BOARD + 200] != EMPTY).tobytes())
    return keys


KEEP_ALL = Toss(keep={}, plain=1.0)              # keeps every window, labelled: the world models' streams

# An event-weighted world-model stream (2026-10-11, the latent "fast learning" run, scripts/train_latent_ui.py
# --toss events): every window with a line clear, level up, top-out, game start or a screen other than play kept,
# plain play (falling alone included) at half: rare events about twice as often. It departs from KEEP_ALL's
# reason (a world model learns the game's own event rates), so a run on it dreams events too often; it is for
# learning speed. The stream gives ~220 world-model windows a second (16-frame stride), a run needs ~45.
EVENTS_TOSS = Toss(keep={"line clear: single": 1.0, "line clear: double": 1.0, "line clear: triple": 1.0,
                         "line clear: tetris": 1.0, "level up": 1.0, "top-out": 1.0, "game start": 1.0,
                         "paused": 1.0, "title": 1.0, "menu": 1.0, "other": 1.0, "game over": 1.0, "legal": 1.0},
                   plain=0.5)

# What the training windows are tracked by (live_batches.LiveBatches; the run window's Data tab).
DATA_NAMES = ("line clear: single", "line clear: double", "line clear: triple", "line clear: tetris", "level up",
              "top-out", "game start", "falling alone", "paused", "game over", "legal", "title", "menu")


class WindowToss:
    """Keeps or tosses one stream's windows by what is in them (Toss). A window is judged once the frames its
    events read ahead (LOOKAHEAD) have arrived, so push() hands back the window from a few pushes before, kept,
    or None. A kept window carries what was found in it ("toss_names") and how many windows were judged since
    the last kept one, itself included ("toss_judged"); the frame counter then jumps by whole strides. judged:
    only each window's last `judged` frames are judged (a long window of history before the frames a model
    trains on; None: all of it). marked: every window is handed back, with "keep" saying what the toss chose
    (and "toss_judged" 1): for a trainer that must see every frame, as one encoding its streams continuously."""

    def __init__(self, toss: Toss, rng: np.random.Generator, stride: int, judged: int | None = None,
                 marked: bool = False):
        self.toss, self.rng, self.stride, self.span, self.marked = toss, rng, stride, judged, marked
        self.pending: deque = deque()              # (window, first frame, frames) awaiting their look-ahead
        self.ram: np.ndarray | None = None         # the stream's recent RAM, from frame `base`
        self.actions: np.ndarray | None = None     # actions[t] took ram[t] to ram[t + 1]
        self.base = 0
        self.found: dict[int, list[str]] = {}      # frame -> events between it and the frame before
        self.judged = 0

    def push(self, window: dict, ram: np.ndarray, actions: np.ndarray) -> dict | None:
        """window, its RAM [T, 2048] and actions [T - 1]; each window starts `stride` frames after the last."""
        frames = len(ram)
        if self.ram is None:
            self.ram, self.actions, start = ram, actions, 0
        else:
            start = self.base + len(self.ram) - (frames - self.stride)
            self.ram = np.concatenate([self.ram, ram[frames - self.stride:]])
            self.actions = np.concatenate([self.actions, actions[frames - 1 - self.stride:]])
        self.pending.append((window, start, frames))
        end = self.base + len(self.ram)
        oldest, first, length = self.pending[0]
        if first + length + max(LOOKAHEAD, ALONE) > end:
            return None
        self.pending.popleft()
        span = min(self.span or length, length)
        names = self._events(first + length - span, span)
        shown = screens(self.ram[first + length - span - self.base:first + length - self.base])
        names |= {SCREENS[k] for k in np.unique(shown)} - {"play"}
        self.judged += 1
        chance = max([self.toss.keep.get(n, 0.0) for n in names] + [self.toss.plain])
        drop = (self.pending[0][1] if self.pending else end) - self.toss.aftermath - 2 - self.base
        if drop > 0:                               # what no window still pending can read
            self.ram, self.actions, self.base = self.ram[drop:], self.actions[drop:], self.base + drop
            self.found = {e: v for e, v in self.found.items() if e >= self.base}
        keep = self.rng.random() < chance
        if self.marked:
            self.judged = 0
            return {**oldest, "toss_names": sorted(names), "toss_judged": 1, "keep": keep}
        if not keep:
            return None
        kept = {**oldest, "toss_names": sorted(names), "toss_judged": self.judged}
        self.judged = 0
        return kept

    def _events(self, first: int, length: int) -> set[str]:
        names: set[str] = set()
        for e in range(max(first - self.toss.aftermath, self.base + 1), first + length):
            if e not in self.found:
                self.found[e] = events_at(self.ram, self.actions, e - self.base)
            names.update(self.found[e])
        return names


BRANCH_FROM = 32           # a branch's press is at window position 32 or later (a long history before it, and a
BRANCH_AFTER = 4           # replay of at most 31 frames) and 4 frames before the end or earlier (its turn shows)
BRANCH_KINDS = ("swapped", "released")


class TurnBranches:
    """A window's counterfactual twin (world_nes_windows branch): one of its A or B press edges in play (a turn;
    position BRANCH_FROM .. frames - 1 - BRANCH_AFTER), and the window replayed from the emulator's state just
    before it with that press "swapped" (the other turn button, held as long as the real one was) or
    "released" (no turn), every other button as played. The frames up to the press are the window's own, so the
    two differ by the button alone: a model that ignores A vs B cannot fit both (the real turn's frames
    and motion against the swapped one's). A window without such a press gets the last twin made, so every
    window of a branching worker has one once the first is made (None before). The bot never sees a branch.
    -> {"x", "ram", "action" (as the window's), "at": the press's position p (action p is the press, so frames
    p + 1 on differ), "kind"}."""

    def __init__(self, rng: np.random.Generator, frames: int):
        self.rng, self.frames = rng, frames
        self.last: dict | None = None

    def presses(self, ram: np.ndarray, actions: np.ndarray) -> list[int]:
        """ram [frames, 2048], actions [frames - 1] (action p takes observation p to p + 1) -> the positions of
        the turn press edges a branch can start at."""
        found = []
        for p in range(BRANCH_FROM, self.frames - 1 - BRANCH_AFTER):
            turn = int(actions[p]) & (A | B)
            if turn not in (A, B) or int(actions[p - 1]) & turn:
                continue
            if ram[p, GAME_MODE] == IN_GAME and ram[p, PLAY_STATE] == ACTIVE and ram[p, PIECE] != 0:
                found.append(p)
        return found

    def __call__(self, window: dict, replay) -> dict | None:
        ram, actions = window["ram"].numpy(), window["action"].numpy()[:-1]
        found = self.presses(ram, actions)
        if not found:
            return self.last
        p = int(self.rng.choice(found))
        kind = BRANCH_KINDS[int(self.rng.integers(len(BRANCH_KINDS)))]
        turn = int(actions[p]) & (A | B)
        played = actions[p:].copy()
        held = 0
        while held < len(played) and int(played[held]) & turn:
            held += 1
        instead = (A | B) ^ turn if kind == "swapped" else 0
        played[:held] = (played[:held] & ~turn) | instead
        x, new_ram = replay(p, played)
        new_actions = np.concatenate([actions[:p], played])
        self.last = {"x": torch.cat([window["x"][:p + 1], torch.from_numpy(x)]),
                     "ram": torch.cat([window["ram"][:p + 1], torch.from_numpy(new_ram)]),
                     "action": torch.from_numpy(np.append(new_actions, new_actions[-1])), "at": p, "kind": kind}
        return self.last


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
    (for event detection, not training); tick = the last frame's counter. layered: the frames' layers
    (data/nes_layers.py) under "layers" instead of x. screens: the screens around play (Screens: WORLD_MODEL
    for the world models' training, VAE for the VAE's); None plays the plain bot, as the checks' fixed trials do.
    toss: which windows to keep (Toss: VAE_TOSS for the VAE's training); None keeps every window. judged: the
    toss judges only each window's last `judged` frames (None: all); marked: every window is yielded, with "keep"
    saying what the toss chose (WindowToss). sprites: also each frame's sprite mask. motion: also each frame's
    motion codes under "motion" [frames, 32, 32] uint8 (data/tetris_motion.py: made from the window's RAM).
    Every window also has "piece" [frames], each frame's RAM orientation byte (0x42), and "events" [frames],
    the events each picture first shows (tetris_events.frame_events). branches: workers 0 .. branches - 1 give every window a counterfactual twin under "branch" (TurnBranches;
    with its motion too), and hold their windows back until their first twin is made.
    """

    def __init__(self, frames: int = 64, seed: int = 17, repo: str | None = None, stride: int | None = None,
                 layered: bool = False, screens: Screens | None = None, toss: Toss | None = None,
                 sprites: bool = False, judged: int | None = None, marked: bool = False, motion: bool = False,
                 branches: int = 0):
        self.frames, self.seed, self.repo, self.layered, self.motion = int(frames), int(seed), repo, layered, motion
        self.branches = int(branches)
        self.sprites, self.judged, self.marked = sprites, judged, marked
        self.new_frames = window_stride(self.frames, stride)
        self.screens, self.toss = screens, toss

    def __iter__(self):
        from token_world.data.nes_palette import tetris_palette
        worker_id, seed = worker_seed(self.seed)
        rom = tetris_rom(self.repo)
        session = TetrisSession(seed) if self.screens is None else ScreensSession(seed, self.screens)
        toss = (WindowToss(self.toss, np.random.default_rng([seed, 2]), self.new_frames, self.judged, self.marked)
                if self.toss is not None else None)
        branch = (TurnBranches(np.random.default_rng([seed, 3]), self.frames) if worker_id < self.branches
                  else None)
        windows = world_nes_windows(rom, session, border_slots(tetris_palette()), BORDER_STATE,
                                    self.frames, self.new_frames, worker_id, toss=toss, layered=self.layered,
                                    sprites=self.sprites, branch=branch)
        for window in windows:
            if branch is not None and "branch" not in window:
                continue                                # no twin made yet
            for w in (window, window.get("branch")):    # each frame's piece byte (the turn critic's J / L / T)
                if w is not None and "piece" not in w:    # and the events each picture first shows
                    w["piece"] = w["ram"][:, PIECE].clone()
                    w["events"] = torch.from_numpy(frame_events(w["x"].numpy(), w["ram"].numpy(),
                                                                w["action"].numpy()))
            if self.motion:
                for w in (window, window.get("branch")):
                    if w is not None and "motion" not in w:   # a reused twin keeps the motion it was given
                        w["motion"] = torch.from_numpy(stream_motion(w["x"].numpy(), w["ram"].numpy())[0])
            yield window
