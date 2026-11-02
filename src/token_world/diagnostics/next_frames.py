"""Next frames (2026-10-22): one step, then a short rollout, on SMB windows from every level.

PER_LEVEL windows of FRAMES frames from each of the 32 levels (2026-10-23; before, 128 windows of 8 fresh
training-stream games, two-thirds world 1: fresh games start at 1-1): a fresh game of the training bot started at
the level (SmbGame.start_level), its windows made exactly as the training stream makes them
(WorldNesSmbStreams.windows), every SPREAD-th one, made again from the seed each time, never stored: every
checkpoint sees the same windows, and each is scored with its level too. A model is given each window's first frames, their real motion codes and the
real buttons, and draws the last DRAWN one after another as a dream does (Dreamer, keeping its whole window but
one: 63 frames behind every frame it draws; its own frames go back in). Frame 1 is the one-step score, frames 4
and 8 the autoregressive one. Each drawn frame (its most likely colours, what play shows) against the real one:

- wrong: game pixels (the border band left out) wrong;
- changed: pixels the real frame changed since the last real frame the model was given, wrong (what has to be
  predicted, not copied; a copy gets all of them wrong); static: the rest, wrong (smear, drift);
- sprites / mario: real sprite-layer pixels (Mario's: in his box from the RAM) wrong in the picture;
- mario_sure: Mario's pixels not drawn right with probability SURE or more (1 - a sharp Mario's share);
- border: the border band's pixels wrong (the tick and the RAM bytes' digits);
- copy_wrong: the game pixels a copy of the last real frame gets wrong (how much there was to predict).

Rates pool pixels over the windows (sum wrong / sum counted). The user's measure: "check how many of the tokens
in the first frame are wrong and how many in the 4th and 8th generated frames are wrong, this will show us how
good the model is at predicting 1 frame and then give us an auto recursive score" (2026-10-22: 10k-50k of s12,
Mario 43% -> 35% wrong one step out, the background 0.1-0.2%; the trainer's test every 2,000 steps since).
"""
from __future__ import annotations

import gc
import threading
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from token_world.data.model_frames import GAME_ROWS
from token_world.data.motion_vocab import to_vocabulary
from token_world.data.nes_layers import TRANSPARENT
from token_world.models.dynamics import Dreamer, compose_frames, incoming_actions

FRAMES, DRAWN = 72, 8
PER_LEVEL = 4                  # windows from each level (128 in all)
WORKERS = 8                    # emulator workers making them (the windows do not depend on it)
SPREAD = 4                     # every 4th window of a level's game (71 frames apart): ~280 frames between
SURE = 0.9
SHOWN_AT = (1, 4, 8)
SHOWN = 4                      # windows pictured each test, a random set
BATCH = 8
KINDS = ("wrong", "changed", "static", "sprites", "mario", "mario_sure", "border", "copy_wrong")


def background_priority(_worker_id: int) -> None:
    """Emulator workers below normal priority, so a trainer gets the CPU first."""
    import sys
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x4000)


class LevelWindows(torch.utils.data.IterableDataset):
    """Each level's PER_LEVEL windows, the levels shared out over the DataLoader's workers: a fresh game of the
    training bot (seed * 1000 + the level's index) from the level's start, every SPREAD-th of its windows."""

    def __init__(self, seed: int):
        self.seed = seed

    def __iter__(self):
        from torch.utils.data import get_worker_info
        from token_world.data.world_nes_smb import LEVELS, SmbGame, WorldNesSmbStreams
        info = get_worker_info()
        stream = WorldNesSmbStreams(FRAMES, self.seed, stride=FRAMES - 1, motion=True, layers=True)
        for k, (world, level) in enumerate(LEVELS):
            if info is not None and k % info.num_workers != info.id:
                continue
            game = SmbGame(self.seed * 1000 + k, sprites=True)
            game.start_level(world, level)
            taken = 0
            try:
                for n, w in enumerate(stream.windows(game)):
                    if n % SPREAD:
                        continue
                    yield {"x": w["x"], "action": w["action"], "motion": w["motion"],
                           "ram": w["ram"][FRAMES - DRAWN:], "level": k, "stage": f"{world + 1}-{level + 1}"}
                    taken += 1
                    if taken == PER_LEVEL:
                        break
            finally:
                game.close()


def windows(seed: int, log=None) -> list[dict]:
    """Every level's PER_LEVEL windows (two layers, motion, buttons, the drawn frames' RAM, its level), in level
    order."""
    loader = DataLoader(LevelWindows(seed), batch_size=None, num_workers=WORKERS, worker_init_fn=background_priority)
    out = []
    started = time.time()
    for w in loader:
        out.append({k: v.clone() if torch.is_tensor(v) else v for k, v in w.items()})
        out[-1]["ram"] = out[-1]["ram"].numpy()
        if log is not None and len(out) % 32 == 0:
            log(f"next-frames windows {len(out)} ({time.time() - started:.0f} s)")
    del loader
    return sorted(out, key=lambda w: w["level"])


class Windows:
    """The windows made on a background thread (a trainer's: its tests wait for them only if they are not done)."""

    def __init__(self, seed: int):
        self.done = threading.Event()
        self.data: list[dict] = []
        threading.Thread(target=self._fill, args=(seed,), daemon=True, name="next-frames-windows").start()

    def _fill(self, seed: int) -> None:
        try:
            self.data = windows(seed)
        finally:
            self.done.set()

    def take(self) -> list[dict]:
        self.done.wait()
        if not self.data:
            raise RuntimeError("the next-frames windows could not be made")
        return self.data


def mario_mask(sprites: np.ndarray, ram: np.ndarray) -> np.ndarray:
    """Mario's real sprite pixels: the sprite layer within his box from the RAM."""
    from token_world.diagnostics.smb_long_dream import mario_box
    m = np.zeros_like(sprites)
    x0, y0, x1, y1 = mario_box(ram)
    m[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = True
    return m & sprites


class Tally:
    """Pooled wrong / counted pixels for each kind at each drawn frame."""

    def __init__(self):
        self.wrong = {k: np.zeros(DRAWN) for k in KINDS}
        self.count = {k: np.zeros(DRAWN) for k in KINDS}

    def add(self, h: int, kind: str, wrong: torch.Tensor, counted: torch.Tensor) -> None:
        self.wrong[kind][h] += (wrong & counted).sum().item()
        self.count[kind][h] += counted.sum().item()

    def rates(self) -> dict[str, np.ndarray]:
        with np.errstate(invalid="ignore", divide="ignore"):
            return {k: self.wrong[k] / self.count[k] for k in KINDS}


@torch.no_grad()
def check(model, data: list[dict], shown: list[int] = (), batch: int = BATCH) -> tuple[dict, list, dict]:
    """A model over every window -> its rates by kind [DRAWN], the `shown` windows' (real, drawn) frames
    [DRAWN, 256, 256] each, and each level's rates ({stage: rates})."""
    device = next(model.parameters()).device
    tally, pictures = Tally(), {}
    levels: dict[str, Tally] = {}
    game = torch.zeros(256, 256, dtype=torch.bool)
    game[GAME_ROWS] = True
    history = data[0]["x"].shape[0] - DRAWN
    for lo in range(0, len(data), batch):
        group = data[lo:lo + batch]
        x = torch.stack([w["x"] for w in group])                                   # [B, T, 2, 256, 256]
        incoming = incoming_actions(torch.stack([w["action"] for w in group]))
        motion = (to_vocabulary(torch.stack([w["motion"] for w in group])[:, :history], "smb", model.motion_codes)
                  .to(device) if model.motion else None)                   # SMB's codes in the model's vocabulary
        real = compose_frames(x)                                                    # [B, T, 256, 256]
        before = real[:, history - 1]
        drawn = []
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            dreamer = Dreamer(model, x[:, :history].to(device), incoming[:, :history].to(device), motion=motion)
            for h in range(DRAWN):
                probs = dreamer.step(incoming[:, history + h].to(device)).float()
                ref = real[:, history + h]
                colours = probs.argmax(-1)
                p_real = probs.gather(-1, ref.to(device).long()[..., None])[..., 0].cpu()
                colours = colours.cpu()
                drawn.append(colours.byte())
                miss = colours != ref.long()
                g = game.expand_as(miss)
                changed = (ref != before) & g
                sprites = (x[:, history + h, 1] != TRANSPARENT) & g
                mario = torch.from_numpy(np.stack([mario_mask(sprites[i].numpy(), w["ram"][h])
                                                   for i, w in enumerate(group)]))
                parts = {"wrong": (miss, g), "changed": (miss, changed), "static": (miss, g & ~changed),
                         "sprites": (miss, sprites), "mario": (miss, mario), "mario_sure": (p_real < SURE, mario),
                         "border": (miss, ~g), "copy_wrong": (ref != before, g)}
                for kind, (wrong, counted) in parts.items():
                    tally.add(h, kind, wrong, counted)
                    for i, w in enumerate(group):       # and each window's level
                        levels.setdefault(w["stage"], Tally()).add(h, kind, wrong[i], counted[i])
            del dreamer
        gc.collect()                         # a Dreamer's cycles hold its cache: free it now, not at some GC
        if device.type == "cuda":            # (without, 5 checkpoints filled the 24 GB card, 2026-10-22)
            torch.cuda.empty_cache()
        drawn = torch.stack(drawn, 1)
        for i in range(len(group)):
            if lo + i in shown:
                pictures[lo + i] = (real[i, history:].numpy(), drawn[i].numpy())
    return tally.rates(), [pictures[i] for i in shown], {k: t.rates() for k, t in levels.items()}


def strip(shown: list, palette: np.ndarray) -> np.ndarray:
    """Rows of windows: real frames 1/4/8, then drawn 1/4/8, as RGB [H, W, 3] uint8."""
    rows = []
    for real, drawn in shown:
        cells = [real[h - 1] for h in SHOWN_AT] + [drawn[h - 1] for h in SHOWN_AT]
        rows.append(np.concatenate([np.pad(palette[c], ((2, 2), (2, 2), (0, 0)), constant_values=40)
                                    for c in cells], 1))
    return np.concatenate(rows, 0)


def rows(step: int, rates: dict[str, np.ndarray]) -> list[dict]:
    """next_frames.csv's rows for one test: one per drawn frame."""
    return [{"step": step, "frame": h + 1, **{k: float(rates[k][h]) for k in KINDS}} for h in range(DRAWN)]


def level_rows(step: int, levels: dict[str, dict[str, np.ndarray]]) -> list[dict]:
    """next_frames_levels.csv's rows for one test: one per level and frame of SHOWN_AT."""
    return [{"step": step, "stage": stage, "frame": h, **{k: float(r[k][h - 1]) for k in KINDS}}
            for stage, r in levels.items() for h in SHOWN_AT]


def line(rates: dict[str, np.ndarray]) -> str:
    """One test's rates at SHOWN_AT, for a log."""
    at = lambda k, f="{:.3f}": "/".join(f.format(rates[k][h - 1]) for h in SHOWN_AT)
    return (f"frame 1/4/8: Mario wrong {at('mario')}, Mario sharp {'/'.join(f'{1 - rates['mario_sure'][h - 1]:.3f}' for h in SHOWN_AT)}, "
            f"sprites wrong {at('sprites')}, changed wrong {at('changed')}, static wrong {at('static', '{:.4f}')}, "
            f"border wrong {at('border')}")
