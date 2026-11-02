"""Does the model's Mario follow the buttons? A counterfactual test (2026-10-25).

The user, playing s6 in the play app: "its just not learned the mechanics at all". From one real moment in each of
the 32 levels (a fresh game of the training bot from the level's start, SKIP frames in, Mario on screen and in
play), the real game is forked and played on H frames five ways (VARIANTS: the bot's own presses, Right held,
Left held, Right + A held, nothing), and the model dreams the same five button sequences from the same CONTEXT
real frames (two layers or pictures, buttons, motion codes, as play starts a dream). Each frame ahead h, on the
union of the five real Mario boxes (from the RAM): m(s, s') = the share of those pixels where the dream with
buttons s draws the colour the real game shows with buttons s'.

- follow(s) = m(s, s) - the mean of m(s, s') over the other s': how much better the dream with s matches the game
  with s than the games with other buttons. 0: the buttons make no difference to the dream; > 0: it follows them;
- spread = the mean share of those pixels where two variants' real games differ: how much the buttons change the
  real game by frame h (follow cannot exceed it);
- kept = m(s, s): the dream's Mario area right against its own real game;
- held = the mean follow of the four held variants over the spread: the share of the buttons' effect the dreams
  carry (the headline; s6 stairs 0.40 / 0.32 / 0.23 at frames 8 / 16 / 32 on the first version, whose bot had no
  fumbles).

The first result (s6 stairs, s12 cooled): every dream matches its own buttons' game best (the model reads the
buttons), but held presses the bot rarely makes are followed far worse than the bot's own. Used by
scripts/smb_button_check.py (a window, any checkpoints) and the trainer (every --long-dream-every steps).
"""
from __future__ import annotations

import gc
import threading
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from token_world.data.tetris_bot import A, LEFT, RIGHT                       # the NES's bits

CONTEXT, H = 48, 32
SKIP = (150, 900)                  # frames the bot plays into a level before the moment (seeded)
VARIANTS = {"bot": None, "right": RIGHT, "left": LEFT, "right+jump": RIGHT | A, "nothing": 0}
HELD = tuple(v for v in VARIANTS if v != "bot")
SHOWN_AT = (4, 8, 16, 32)
MARGIN = 4
WORKERS = 8


class Moments(IterableDataset):
    """One moment a level: the CONTEXT frames before it and each variant's H real frames after it."""

    def __init__(self, seed: int):
        self.seed = seed

    def __iter__(self):
        from token_world.data.model_frames import border_slots
        from token_world.data.nes_layers import TWO_LAYER_REGIONS, two_layers
        from token_world.data.smb_motion import GRID, UNKNOWN, frame_motion
        from token_world.data.smb_state import SmbState
        from token_world.data.world_nes_smb import LEVELS, SmbGame, smb_palette
        from token_world.diagnostics.smb_long_dream import START_MARIO, mario_box, mario_pixels, _observe
        info = get_worker_info()
        slots = border_slots(smb_palette())
        for k, (world, level) in enumerate(LEVELS):
            if info is not None and k % info.num_workers != info.id:
                continue
            rng = np.random.default_rng([self.seed, k])
            game = SmbGame(self.seed * 1000 + k, sprites=True)
            game.start_level(world, level)
            history: list[tuple] = []
            produced_by = 0
            skip = int(rng.integers(*SKIP))
            try:
                for played in range(6000):
                    o = game.observation
                    x, mask, ram = _observe(o, game.tick, slots)
                    motion = frame_motion(history[-1][0], x) if history else np.full((GRID, GRID), UNKNOWN, np.uint8)
                    history = (history + [(x, two_layers(x, {r: o.regions[r] for r in TWO_LAYER_REGIONS}),
                                           produced_by, motion)])[-CONTEXT:]
                    if (played >= skip and len(history) == CONTEXT and SmbState.read(ram).playable
                            and mario_pixels(mask, ram) >= START_MARIO):
                        break
                    produced_by, jumped = game.step()
                    if jumped:
                        history = []
                else:
                    continue
                tick = game.tick
                futures = {}
                for name, held in VARIANTS.items():
                    if name == "bot":                     # the bot's own presses: on a fork of the console its
                        continue                          # session would diverge, so they come from playing on
                    branch = game.console.fork()
                    try:
                        buffer = branch.allocate_observation()
                        frames, boxes = [], []
                        for h in range(H):
                            branch.advance_frame_into(buffer, buttons=(held, 0))
                            x, _, r = _observe(buffer.record, tick + h + 1, slots)
                            frames.append(x)
                            boxes.append(mario_box(r))
                        futures[name] = (np.stack(frames), np.array(boxes), np.full(H, held, np.int64))
                    finally:
                        branch.close()
                frames, boxes, bot_buttons = [], [], []
                for h in range(H):
                    action, jumped = game.step()
                    if jumped:
                        break
                    x, _, r = _observe(game.observation, game.tick, slots)
                    frames.append(x)
                    boxes.append(mario_box(r))
                    bot_buttons.append(action)
                if len(frames) < H:
                    continue
                futures["bot"] = (np.stack(frames), np.array(boxes), np.array(bot_buttons, np.int64))
                pictures, layers, actions, motion = zip(*history)
                yield {"stage": f"{world + 1}-{level + 1}", "pictures": np.stack(pictures), "layers": np.stack(layers),
                       "actions": np.array(actions, np.int64), "motion": np.stack(motion),
                       "futures": {n: futures[n] for n in VARIANTS}}
            finally:
                game.close()


def numpy_of(value):
    """A DataLoader's moment (numpy arrays come back as tensors, tuples as lists) -> numpy arrays again."""
    if torch.is_tensor(value):
        return value.numpy()
    if isinstance(value, dict):
        return {k: numpy_of(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [numpy_of(v) for v in value]
    return value


def moments(seed: int, log=None) -> list[dict]:
    """Every level's moment, in level order (WORKERS emulators; a level whose moment cannot be found in time is
    left out)."""
    from token_world.diagnostics.next_frames import background_priority
    out, started = [], time.time()
    for m in DataLoader(Moments(seed), batch_size=None, num_workers=WORKERS, worker_init_fn=background_priority):
        out.append(numpy_of(m))
        if log is not None and len(out) % 8 == 0:
            log(f"button moments {len(out)} ({time.time() - started:.0f} s)")
    return sorted(out, key=lambda m: [int(v) for v in m["stage"].split("-")])


class MomentsLater:
    """The moments made on a background thread (a trainer's: its tests wait for them only if they are not done);
    an error in making them is raised by take()."""

    def __init__(self, seed: int):
        self.done = threading.Event()
        self.data: list[dict] = []
        self.error: BaseException | None = None
        threading.Thread(target=self._fill, args=(seed,), daemon=True, name="button-moments").start()

    def _fill(self, seed: int) -> None:
        try:
            self.data = moments(seed)
        except BaseException as e:
            self.error = e
        finally:
            self.done.set()

    def take(self) -> list[dict]:
        self.done.wait()
        if self.error is not None:
            raise RuntimeError("the button moments could not be made") from self.error
        return self.data


def score(model, moment: dict) -> dict:
    """One moment -> follow and kept [variants, H], spread [H], the matrix m [H, dream, real], and the dreams' and
    real futures' colours [variants, H, 256, 256]."""
    from token_world.data.motion_vocab import to_vocabulary
    from token_world.models.dynamics import Dreamer
    device = next(model.parameters()).device
    names = list(VARIANTS)
    n = len(names)
    frames = moment["layers"] if model.two_layers else moment["pictures"]
    frames = torch.from_numpy(frames).to(device)[None].expand(n, *frames.shape).contiguous()
    actions = torch.from_numpy(moment["actions"]).to(device)[None].expand(n, -1).contiguous()
    motion = (to_vocabulary(torch.from_numpy(moment["motion"]), "smb", model.motion_codes).to(device)[None]
              .expand(n, -1, -1, -1).contiguous() if model.motion else None)
    buttons = torch.from_numpy(np.stack([moment["futures"][v][2] for v in names])).to(device)
    dreams = []
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        dreamer = Dreamer(model, frames, actions, motion=motion)
        for h in range(H):
            dreams.append(dreamer.step(buttons[:, h]).argmax(-1).byte().cpu().numpy())
    del dreamer
    dreams = np.stack(dreams, 1)                                                  # [n, H, 256, 256]
    real = np.stack([moment["futures"][v][0] for v in names])                     # [n, H, 256, 256]
    boxes = np.stack([moment["futures"][v][1] for v in names])                    # [n, H, 4]
    follow, kept, spread = np.zeros((n, H)), np.zeros((n, H)), np.zeros(H)
    matrix = np.full((H, n, n), np.nan)
    for h in range(H):
        area = np.zeros((256, 256), bool)
        for x0, y0, x1, y1 in boxes[:, h]:
            area[max(y0 - MARGIN, 48):max(min(y1 + MARGIN, 240), 0), max(x0 - MARGIN, 0):max(x1 + MARGIN, 0)] = True
        if not area.any():
            follow[:, h] = kept[:, h] = spread[h] = np.nan
            continue
        m = matrix[h] = np.array([[(dreams[s, h][area] == real[t, h][area]).mean() for t in range(n)]
                                  for s in range(n)])
        kept[:, h] = m.diagonal()
        follow[:, h] = m.diagonal() - (m.sum(1) - m.diagonal()) / (n - 1)
        spread[h] = np.mean([(real[s, h][area] != real[t, h][area]).mean() for s in range(n) for t in range(n) if s < t])
    return {"follow": follow, "kept": kept, "spread": spread, "matrix": matrix, "dreams": dreams, "real": real}


def check(model, data: list[dict], shown: int | None = 0) -> dict:
    """A model over every moment -> {"follow"/"kept": {variant: [H]}, "spread": [H], "held": [H] (the held
    variants' mean follow over the spread), "matrix": [H, dream, real], "picture": the `shown` moment's (stage,
    real, dreams) or None}."""
    follow, kept, spread, matrix = [], [], [], []
    picture = None
    for i, moment in enumerate(data):
        s = score(model, moment)
        follow.append(s["follow"])
        kept.append(s["kept"])
        spread.append(s["spread"])
        matrix.append(s["matrix"])
        if i == shown:
            picture = (moment["stage"], s["real"], s["dreams"])
        del s
        gc.collect()                                  # the dreamer's caches: else VRAM fills and thrashes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    mean = lambda a: np.nanmean(np.stack(a), 0)
    f, k, sp = mean(follow), mean(kept), mean(spread)
    names = list(VARIANTS)
    held = np.mean([f[names.index(v)] for v in HELD], 0) / np.maximum(sp, 1e-9)
    return {"follow": {v: f[i] for i, v in enumerate(names)}, "kept": {v: k[i] for i, v in enumerate(names)},
            "spread": sp, "held": held, "matrix": mean(matrix), "picture": picture}


def rows(step: int, result: dict) -> list[dict]:
    """buttons.csv's rows for one test: one per frame ahead."""
    return [{"step": step, "frame": h + 1, "held": float(result["held"][h]), "spread": float(result["spread"][h]),
             **{f"follow_{v}": float(result["follow"][v][h]) for v in VARIANTS},
             **{f"kept_{v}": float(result["kept"][v][h]) for v in VARIANTS}} for h in range(H)]


def line(result: dict) -> str:
    """One line: the held share and each variant's follow at SHOWN_AT."""
    at = lambda a: "/".join(f"{a[h - 1]:.2f}" for h in SHOWN_AT)
    return (f"held share of the buttons' effect at {'/'.join(map(str, SHOWN_AT))}: {at(result['held'])}; follow "
            + "; ".join(f"{v} {'/'.join(f'{result['follow'][v][h - 1]:+.3f}' for h in SHOWN_AT)}" for v in VARIANTS))


def strip(picture: tuple, palette: np.ndarray) -> np.ndarray:
    """One moment's real futures (left block) and dreams (right block): a row per variant, a column per SHOWN_AT
    frame -> an RGB image."""
    _, real, dreams = picture
    blocks = []
    for block in (real, dreams):
        blocks.append(np.concatenate([np.concatenate([np.pad(palette[block[v, h - 1]], ((2, 2), (2, 2), (0, 0)),
                                                             constant_values=40) for h in SHOWN_AT], 1)
                                      for v in range(len(VARIANTS))], 0))
    gap = np.full((blocks[0].shape[0], 12, 3), 255, np.uint8)
    return np.concatenate([blocks[0], gap, blocks[1]], 1).astype(np.uint8)
