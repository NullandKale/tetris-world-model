"""Windows of consecutive model frames from one World NES console, for any game's session.

Each game's stream (world_nes_tetris.py, world_nes_contra.py) is a DataLoader
dataset with one console per worker, in external mode: every frame the game's
session reads the observation (palette plane and RAM) and chooses the
controller byte. This module is the part they share: the console, the frame
IDs checked consecutive, the conversion to model index frames
(model_frames.py) and the windows.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator, Protocol

import numpy as np
import torch
from torch.utils.data import get_worker_info

from token_world.data.model_frames import model_index_frames
from token_world.data.nes_layers import REGIONS, SPRITE_REGIONS, LayerCanvas, sprite_mask


class Session(Protocol):
    """A game's player: one button byte per observation, and its count of in-game restarts."""

    in_game_restarts: int

    def act(self, palette: np.ndarray, ram: np.ndarray) -> int: ...


def window_stride(frames: int, stride: int | None) -> int:
    """New observations per window: `stride`, default frames - 1 (consecutive windows share one)."""
    if frames < 2:
        raise ValueError("frames must be at least 2")
    new_frames = int(stride or frames - 1)
    if not 1 <= new_frames <= frames - 1:
        raise ValueError("stride must be 1..frames - 1")
    return new_frames


def worker_seed(seed: int) -> tuple[int, int]:
    """-> (DataLoader worker id, this worker's own seed)."""
    info = get_worker_info()
    worker_id = info.id if info else 0
    return worker_id, int(np.random.SeedSequence([seed, worker_id]).generate_state(1)[0])


Replay = Callable[[int, np.ndarray], tuple[np.ndarray, np.ndarray]]


def world_nes_windows(rom: Path, session: Session, slots: np.ndarray, state: tuple[int, ...] | None,
                      frames: int, new_frames: int, worker_id: int, toss=None, layered: bool = False,
                      sprites: bool = False, branch: Callable[[dict, Replay], dict | None] | None = None
                      ) -> Iterator[dict]:
    """Endless windows of `frames` consecutive observations, `new_frames` new ones each.

    slots: the game's three border shades (model_frames.border_slots); state:
    the RAM addresses drawn in every frame's border, or None. toss: an object
    whose push(window, ram, actions) returns a window to keep or None; the
    frame counter then jumps by whole strides.
    Yields x [frames, 256, 256] uint8 model indices; action [frames] long,
    where action[t] took frame t to t + 1 and the last entry repeats the one
    before it as padding; ram [frames, 2048] uint8, each frame's console RAM
    (for event detection, not training); tick = the last frame's counter.
    layered: the window is the frames' layers instead (data/nes_layers.py, each key stacked [frames, ...])
    under "layers", and no x (compose them for the picture).
    sprites: also "sprites" [frames, 256, 256] bool, the pixels the PPU drew as sprites (nes_layers.sprite_mask; a
    training signal, never a model input).
    branch: called with each window (before any toss) and a replay(start, actions) that plays `actions` [n] (button
    bytes) from the window's observation `start` on a second console loaded with that observation's state
    -> (x [n, 256, 256] model indices, ram [n, 2048]) of the n observations after it: what the game would
    have shown had those buttons been pressed. Its result, if not None, goes in the window under "branch"
    (pictures only: not with layered or sprites). The pool then keeps every observation's state.
    """
    if branch is not None and (layered or sprites):
        raise ValueError("branches are made of pictures alone: not with layered or sprites")
    from world_nes import Capture, Console, RolloutPool, StreamSpec
    pool = RolloutPool([StreamSpec(rom)], workers=1, transitions=new_frames, mode="external",
                       prefetch_batches=0, capture=Capture(frames="palette", ram=True,
                                                           regions=REGIONS if layered else
                                                           SPRITE_REGIONS if sprites else (),
                                                           checkpoints="frame" if branch else "none"))
    replayer = Console(rom, capture=Capture(frames="palette", ram=True)) if branch else None
    canvas = LayerCanvas() if layered else None
    buffer = pool.allocate_batch()
    addresses = list(state) if state else None

    def policy(observation):
        return np.array([[session.act(observation.frames[0], observation.ram[0]), 0]], dtype=np.uint8)

    segment = None
    x = ram = ids = actions = masks = None          # the last `frames` observations and their transitions
    saved: list[bytes] = []                         # with branch: their consoles' states

    def replay(start: int, buttons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        replayer.load_state(saved[start])
        played = [replayer.advance_frame((int(a), 0)) for a in buttons]
        after = ids[start] + 1 + np.arange(len(buttons))
        new_ram = np.stack([o.ram for o in played])
        return (model_index_frames(np.stack([o.frames for o in played]), np.stack([o.emphasis for o in played]),
                                   after, slots, None if addresses is None else new_ram[:, addresses]), new_ram)
    try:
        while True:
            pool.collect_into(policy, buffer, callback_views=True)
            record = buffer.record
            new_ids = record.frame_id[0].astype(np.int64)
            segments = record.segment_id[0]
            segment = int(segments[0]) if segment is None else segment
            if (segments != segment).any():
                raise RuntimeError("World NES segment changed inside a stream")
            if (np.diff(new_ids) != 1).any() or (ids is not None and new_ids[0] != ids[-1]):
                raise RuntimeError("World NES frame IDs are not consecutive")
            states = None if addresses is None else record.ram[0][:, addresses]
            if layered:                             # the boundary observation was pushed with the last batch
                start = 0 if ids is None else 1
                new_x = [canvas.push({k: record.regions[k][0][i] for k in REGIONS}, int(new_ids[i]), slots,
                                     None if states is None else states[i]) for i in range(start, len(new_ids))]
                if ids is not None:
                    new_x.insert(0, None)           # the held boundary's place
            else:
                new_x = model_index_frames(record.frames[0], record.emphasis[0], new_ids, slots, states)
            new_masks = (np.stack([sprite_mask({k: record.regions[k][0][i] for k in SPRITE_REGIONS})
                                   for i in range(len(new_ids))]) if sprites else None)
            new_actions = record.actions[0, :, 0].astype(np.int64)
            new_ram = record.ram[0].copy()
            if branch is not None:
                states = [record.checkpoints.get(0, i) for i in range(len(new_ids))]
                saved = (saved + states[1:])[-frames:] if saved else states
            if ids is None:                         # the first observation is also the boundary
                x, ram, ids, actions, masks = new_x, new_ram, new_ids, new_actions, new_masks
            else:                                   # the boundary observation is already held
                x = (x + new_x[1:])[-frames:] if layered else np.concatenate([x, new_x[1:]])[-frames:]
                ram = np.concatenate([ram, new_ram[1:]])[-frames:]
                ids = np.concatenate([ids, new_ids[1:]])[-frames:]
                if sprites:
                    masks = np.concatenate([masks, new_masks[1:]])[-frames:]
                actions = np.concatenate([actions, new_actions])[-(frames - 1):]
            if len(ids) < frames:
                continue
            pictures = ({"layers": {k: torch.from_numpy(np.stack([f[k] for f in x])) for k in x[0]}} if layered
                        else {"x": torch.from_numpy(x)})
            if sprites:
                pictures["sprites"] = torch.from_numpy(masks)
            window = {**pictures, "action": torch.from_numpy(np.append(actions, actions[-1])),
                      "ram": torch.from_numpy(ram), "tick": int(ids[-1]),
                      "in_game_restarts": session.in_game_restarts, "worker_id": worker_id}
            if branch is not None:                  # made now, while the states are this window's (a toss
                made = branch(window, replay)       # may hand the window back later)
                if made is not None:
                    window["branch"] = made
            if toss is not None:
                window = toss.push(window, ram, actions)
            if window is not None:
                yield window
    finally:
        pool.close()
