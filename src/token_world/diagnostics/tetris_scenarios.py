"""Tetris scenarios: roll the world model out across specific game events and score what the event changes.

Events are found live in console RAM (no stored eval set): the bot plays in
World NES streams, and each detected event yields a 64-frame window whose
first 48 frames are context and whose first visible change after the event
appears at generated frame +LEAD. Several models are scored on the same
instances, any model (diagnostics/worlds.py): the streams are layered
(data/nes_layers.py) and their frames composed from the layers, exactly, so the
pixel model and the layered models see the same windows. (Aligning on the screen, not the RAM frame, matters: the top-out
curtain starts well over 16 frames after the play state says game over, and
the line-clear animation steps every few frames.)

Measured on World NES (docs/guides/dynamics.md): the picture shows the RAM of
the frame before it (an event between RAM frames e - 1 and e appears in
picture e + 1), falling-piece cells sit at x = 92 + 8 col, y = 52 + 8 row in
model frames, and the NEXT box redraw (a random piece, unlearnable) at
y 128-143, x 196-219. The events and their RAM come from
data/tetris_events.py.

Scores per instance and horizon h (generated frame +h against the real one), as expectations over the
pixel model's soft frames, and counts of the layered models' committed ones:
- event wrong: of the pixels that really changed since the last context frame
  (tick border and NEXT box excluded), the share the model expects wrong;
- false change: of the playfield pixels that really stayed the same, the
  share the model expects changed;
- exact: at the event's first visible frame (+LEAD), every event pixel is
  more likely right than wrong.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from token_world.data.model_frames import BAND
from token_world.data.nes_layers import TRANSPARENT, compose, model_frame
from token_world.data.tetris_events import CLEAR_NAMES, LEVEL, LOOKAHEAD, events_at
from token_world.data.world_nes_tetris import WorldNesTetrisStreams
from token_world.models.dynamics import incoming_actions

CONTEXT, FRAMES, LEAD = 48, 64, 2
HORIZONS = (LEAD, 8, 16)
PLAYFIELD = (slice(52, 212), slice(92, 172))
NEXT_BOX = (slice(120, 152), slice(188, 228))
SCENARIOS = ("gravity (no buttons)", "falling alone", "soft drop", "shift", "rotate", "spawn",
             *CLEAR_NAMES.values(), "level up", "top-out", "game start")
SETTLE = 90                         # frames after an event to look for its first visible change


@dataclass
class Instance:
    scenario: str
    frames: np.ndarray            # [64, 256, 256] uint8 model frames
    actions: np.ndarray           # [64] stream actions: actions[t] took frame t to t + 1; the last is padding
    stream: int                   # which live stream (DataLoader worker)
    game: int                     # that stream's game number (its in-game restarts so far)
    level: int                    # the game level at the event
    frame: int                    # the stream's frame index of the event's first visible change
    layers: dict | None = None    # the window's layered frames, each key [64, ...] (data/nes_layers.py)


class Timeline:
    """One stream's recent frames, actions and RAM, joined across its overlapping windows.

    Absolute frame indices count from the stream's first frame; only the last
    KEEP frames are held.
    """

    KEEP = 256

    def __init__(self, stream: int = 0):
        self.stream = stream
        self.frames: list[np.ndarray] = []
        self.actions: list[int] = []       # actions[i] took frames[i] to frames[i + 1]
        self.ram: list[np.ndarray] = []
        self.layers: list[dict] = []       # each frame's layers (layered streams)
        self.start = 0                     # absolute index of frames[0]
        self.scanned = 1                   # next absolute RAM index to scan
        self.game = 0

    def add(self, x: np.ndarray, action: np.ndarray, ram: np.ndarray, game: int = 0,
            layers: dict | None = None) -> None:
        """One window: x [64, ...], action [64] (last is padding), ram [64, 2048], the stream's
        game number, and its layered frames (each key [64, ...]) or None. Consecutive windows share
        a frame: this window's frame 0 is the previous window's last frame."""
        first = 0 if not self.frames else 1
        self.frames += list(x[first:])
        self.ram += list(ram[first:])
        if layers is not None:
            self.layers += [{k: v[t] for k, v in layers.items()} for t in range(first, len(x))]
        self.actions += [int(a) for a in action[:-1]]
        self.game = game
        drop = len(self.frames) - self.KEEP
        if drop > 0:
            del self.frames[:drop], self.ram[:drop], self.actions[:drop], self.layers[:drop]
            self.start += drop

    @property
    def end(self) -> int:
        return self.start + len(self.frames)

    def first_change(self, e: int) -> int | None:
        """Absolute picture index of the first visible change at or after picture e + 1 (the
        picture of RAM frame e), outside the tick border and the NEXT box; None within SETTLE."""
        keep = np.ones((256, 256), bool)
        keep[:BAND], keep[256 - BAND:] = False, False
        keep[NEXT_BOX] = False
        for f in range(e + 1, min(e + 1 + SETTLE, self.end)):
            if ((self.frames[f - self.start] != self.frames[f - 1 - self.start]) & keep).any():
                return f
        return None

    def instances(self) -> list[Instance]:
        """Scan newly complete RAM frames for events and cut their windows.

        An event between RAM e - 1 and e first shows in some picture f >= e + 1
        (first_change); the context ends LEAD frames before f, so the window is
        pictures f - LEAD - 47 .. f - LEAD + 16.
        """
        found = []
        ram = np.stack(self.ram)
        acts = np.array(self.actions)
        last = self.end - 1 - max(LOOKAHEAD, SETTLE + FRAMES - CONTEXT)   # lookahead, settling and the future
        for e in range(max(self.scanned, self.start + 1), last + 1):
            names = events_at(ram, acts, e - self.start)
            if not names:
                continue
            f = self.first_change(e)
            if f is None:
                continue
            first = f - LEAD - (CONTEXT - 1)
            if first < self.start:
                continue
            i = first - self.start
            window_actions = np.array(self.actions[i:i + FRAMES - 1] + [self.actions[i + FRAMES - 2]])
            frames = np.stack(self.frames[i:i + FRAMES])
            level = int(ram[e - self.start, LEVEL])
            layers = ({k: np.stack([self.layers[j][k] for j in range(i, i + FRAMES)]) for k in self.layers[i]}
                      if self.layers else None)
            found += [Instance(name, frames, window_actions, self.stream, self.game, level, f, layers)
                      for name in names]
        self.scanned = max(self.scanned, last + 1)
        return found


def live_instances(seed: int, workers: int = 8, repo: str | None = None):
    """Endless scenario instances from fresh live layered streams, one per DataLoader worker: each window's
    model frames composed from its layers (exact), so every model sees the same frames."""
    loader = DataLoader(WorldNesTetrisStreams(FRAMES, seed, repo, layered=True), batch_size=None,
                        num_workers=workers, persistent_workers=False, prefetch_factor=2)
    timelines: dict[int, Timeline] = {}
    for window in loader:
        stream = int(window["worker_id"])
        layers = {k: np.asarray(v) for k, v in window["layers"].items()}
        x = np.stack([model_frame(compose({k: v[t] for k, v in layers.items()}), layers["border"][t])
                      for t in range(len(layers["camera"]))])
        x = np.where(x == TRANSPARENT, 0, x).astype(np.uint8)
        timeline = timelines.setdefault(stream, Timeline(stream))
        timeline.add(x, np.asarray(window["action"]), np.asarray(window["ram"]), int(window["in_game_restarts"]),
                     layers)
        yield from timeline.instances()


def spread(instances, per_scenario: int, seconds: float, streams: int, clock=time.monotonic):
    """Accept at most per_scenario instances of each scenario, spaced evenly in time: per
    stream, at most one instance of a scenario every seconds * streams / per_scenario, so
    even frequent events are drawn from the whole collection period (many games, levels and
    board heights), not the first minutes. Stops when every scenario is full or after `seconds`."""
    start = clock()
    gap = seconds * streams / per_scenario
    taken: dict[str, int] = defaultdict(int)
    last: dict[tuple[str, int], float] = {}
    for instance in instances:
        now = clock()
        key = (instance.scenario, instance.stream)
        if taken[instance.scenario] < per_scenario and now - last.get(key, -gap) >= gap:
            taken[instance.scenario] += 1
            last[key] = now
            yield instance
        if now - start > seconds or all(taken[name] >= per_scenario for name in SCENARIOS):
            return


def score(real: np.ndarray, right: np.ndarray, same: np.ndarray) -> dict[str, float]:
    """real [64, 256, 256] window; for the generated frames CONTEXT..63, right [16, 256, 256] each pixel's
    probability of the real colour and same [16, 256, 256] its probability of the last context frame's colour
    (diagnostics/worlds.py rollout: a soft model's own, 0 or 1 for committed ones) -> metrics."""
    last = real[CONTEXT - 1]
    keep = np.ones((256, 256), bool)
    keep[:BAND], keep[256 - BAND:] = False, False
    keep[NEXT_BOX] = False
    out = {}
    for h in HORIZONS:
        ref = real[CONTEXT - 1 + h]
        event = (ref != last) & keep
        wrong = 1 - right[h - 1]
        static = ~(ref != last)[PLAYFIELD]
        out[f"event_wrong_{h}"] = float(wrong[event].mean()) if event.any() else float("nan")
        out[f"false_change_{h}"] = float((1 - same[h - 1][PLAYFIELD])[static].sum() / max(static.sum(), 1))
        if h == LEAD:                                    # every event pixel more likely right than wrong
            out["exact"] = float((wrong[event] < 0.5).all()) if event.any() else float("nan")
    return out


def evaluate(world, instances: list[Instance], batch: int = 16) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Roll a World (diagnostics/worlds.py) out on each instance -> (per-instance scores, its frames' colours [n,
    16, 256, 256, 3] uint8, their pixels' probability of being wrong [n, 16, 256, 256] float16)."""
    frames = np.stack([g.frames for g in instances])
    actions = incoming_actions(torch.from_numpy(np.stack([g.actions for g in instances]))).numpy()
    layers = ({k: np.stack([g.layers[k] for g in instances]) for k in instances[0].layers}
              if world.layered else None)
    made, right, same = world.rollout(frames, actions, CONTEXT, layers, batch=batch)
    scores = [score(g.frames, r.astype(np.float32), s.astype(np.float32)) for g, r, s in zip(instances, right, same)]
    return scores, world.palette[made], (1 - right.astype(np.float32)).astype(np.float16)
