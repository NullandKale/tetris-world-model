"""Train the world models (models/dynamics.py) on live NES games (GAMES), in a visible UI.

Two sizes train side by side on the same batches (MODELS: small ~2M and large
~7.6M parameters, both with the tied pixel head): one set of emulator workers
feeds one batch per step, and each model takes its own step on it, so the
CPU-bound data path is shared and the two curves compare window for window.
Each has its own run folder, optimizer, averaged weights and metrics.

A Genie-style spatiotemporal MaskGIT transformer over windows of 64 exact
palette frames, conditioned on the controller bytes. Every step has the same
number of histories from each game (GAMES); each game's frames map through
its own palette into the shared index space (NES colours 0-54, the game's
three border shades 55-57), so one model and one vocabulary cover both.
Loss: cross-entropy over the pixels of masked tokens, every pixel at the same
weight, the letterbox tick border included. The game, border and changed-pixel
parts are reported, not weighted. The model's own frames are part of every
step (models/dynamics.py): half the windows have their context corrupted at a
level per frame (wrong but real content, as GameNGen and Diffusion Forcing
corrupt theirs), the other half carry 1-32 frames it generated one after
another from the real past; the training pass is told each frame's level and
scores the real pixels.

Windows are tossed by the events in them (data/tetris_events.py): every
window with a rare event is kept, few ordinary ones, so level ups, top-outs
and big line clears come round several times more often per step.
--keep-all-windows trains on the game as it is played instead: tossing makes
events look more frequent than they are, and dreams may then run ahead. Alongside
the trained weights the checkpoint keeps an exponential moving average of
them (EMA_DECAY), which play and the checks load.

Previews roll the model out on the current batch: the first 48 frames of each
window are context, the last 16 are generated one after another with the real
actions, and are scored per game against the real frames; four windows, from
the most changing to the quieter ones, are saved as animated GIFs
(previews/). Every --long-dream-every steps (1,000) each model's averaged weights dream
live no-button trials 128 frames ahead (diagnostics/long_dream.py), appended
to long_dream.csv with an animated GIF. Fresh live frames only; no eval set.
The window (ui/run_viewer.py) reads it all back from the run folders. See
docs/guides/dynamics.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import queue
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from token_world.data.model_frames import BAND, BORDER_VERSION, GAME_ROWS
from token_world.data.nes_palette import tetris_palette
from token_world.data.world_nes_tetris import WorldNesTetrisStreams
from token_world.diagnostics.long_dream import TrialStream, animation, dream, score
from token_world.diagnostics.long_dream import sheet as dream_sheet
from token_world.models.dynamics import (HEAD, SIZE, Decoding, Dynamics, build, corrupt_context, deepen, incoming_actions, masked_loss,
                                         patch_pixels, rollout, training_mask)
from token_world.data.live_batches import LiveBatches

# Tetris only: the small, fast model on the simpler game first. GameBatches supports
# "contra" too (data/world_nes_contra.py; earlier runs and results in docs/guides/dynamics.md).
# Adding it also needs its own --output-root: run folders are named world_model_tetris_<name>.
GAMES = ("tetris",)
# recompute: keep only the patch and block inputs for backward and recompute the rest (same model and
# gradients, less memory, one more forward): the large model needs it, the small one is 7% faster without.
MODELS = {"small": {"dim": 96, "layers": 8, "heads": 4, "recompute": False},   # ~2.0M parameters
          # ~3.2M: the small model grown deeper (--grow-from, models/dynamics.py deepen), its pixel head
          # frozen: the trunk learns more dynamics against the head the small model trained
          "deep": {"dim": 96, "layers": 16, "heads": 4, "recompute": True, "frozen_head": True},   # 24 GB: recompute
          "large": {"dim": 192, "layers": 10, "heads": 6, "recompute": True}}  # ~7.6M parameters
HORIZONS = (1, 2, 4, 8, 16)
WINDOW_STRIDE = 16                                # a window every 16 frames: each frame in four windows
STOP_FILE = ROOT / "output" / "stop_training"      # create it to stop cleanly, like Stop Training
EMA_DECAY = 0.999                                 # averaged weights: about the last 1,000 steps
ROLLOUT_FRAMES = 32                               # longest rollout generated inside a training step
# How training rollouts generate: one MaskGIT step, as in play (lowest long-dream error at step 21.7k).
ROLLOUT_DECODING = Decoding(steps=1)
KINDS = ("wrong", "changed", "wrong_changed", "wrong_static", "border_wrong", "border_copy")
LONG_DREAM_EVERY = 1000                           # steps between long-dream tests
LONG_DREAM_TRIALS = 6                             # live trials per test, the same for every model
LONG_DREAM_VARIANTS = {"level 2": Decoding(steps=1), "level 0": Decoding(steps=1, level=0)}
PREVIEW_GIFS = 4                                  # preview windows saved as animated GIFs
FIELDS = ["step", "seconds", "lr", "train_loss", "train_loss_game", "train_loss_border",
          "train_loss_changed", "masked_fraction", "corrupted_fraction",
          "rollout_frames", "rollout_wrong",
          "data_wait_ms_mean", "data_wait_ms_max",
          "stream_tick_min", "stream_tick_max", "in_game_restarts",
          *(f"{g}_h{h}_{kind}" for g in GAMES for h in HORIZONS for kind in KINDS)]


def background_priority(_worker_id: int) -> None:
    """DataLoader workers run below normal priority: the emulators fill every core, and the
    trainer's process must still get the CPU the moment it has GPU work to launch."""
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x4000)   # BELOW_NORMAL_PRIORITY_CLASS
    else:
        os.nice(5)


class GameBatches:
    """Each step's batch: workers_per_game live histories of every game, in GAMES order.

    Streams come from World NES (../llm-thing23) and arrive as uint8 model index
    frames in one layout (data/model_frames.py). Tetris windows are tossed by
    their events; Contra has no event detector, so every Contra window is kept.
    """

    def __init__(self, frames: int, seed: int, args):
        unknown = set(GAMES) - {"tetris", "contra"}
        if unknown:
            raise ValueError(f"no World NES stream for {sorted(unknown)}")
        self.palettes, streams = {}, {}
        if "tetris" in GAMES:
            self.palettes["tetris"] = tetris_palette()
            streams["tetris"] = WorldNesTetrisStreams(frames, seed, args.tetris_repo,
                                                      toss=not args.keep_all_windows, stride=WINDOW_STRIDE)
        if "contra" in GAMES:                     # imported only when used: Contra can't break a Tetris run
            from token_world.data.world_nes_contra import WorldNesContraStreams, contra_palette
            self.palettes["contra"] = contra_palette()
            streams["contra"] = WorldNesContraStreams(frames, seed + 1, stride=WINDOW_STRIDE)
        self.batches = {g: LiveBatches(DataLoader(streams[g], batch_size=None,
                                                  num_workers=args.workers_per_game, pin_memory=True,
                                                  persistent_workers=True, worker_init_fn=background_priority,
                                                  prefetch_factor=args.prefetch_factor),
                                       args.workers_per_game, streams[g].new_frames) for g in GAMES}

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        """-> index [B, T, 256, 256] uint8, stream actions [B, T] (action t takes frame t to t + 1)."""
        index, actions = [], []
        for game in GAMES:
            x, a = self.batches[game].next()
            index.append(x)
            actions.append(a)
        return torch.cat(index), torch.cat(actions)

    def take_stats(self) -> dict[str, float]:
        """Stream counters over all games; data waits add up, since games are fetched in turn."""
        s = [b.take_stats() for b in self.batches.values()]
        return {"data_wait_ms_mean": sum(v["data_wait_ms_mean"] for v in s),
                "data_wait_ms_max": sum(v["data_wait_ms_max"] for v in s),
                "stream_tick_min": min(v["stream_tick_min"] for v in s),
                "stream_tick_max": max(v["stream_tick_max"] for v in s),
                "in_game_restarts": sum(v["in_game_restarts"] for v in s)}


def border_pixels(model: Dynamics, device) -> torch.Tensor:
    """[N, P*P] bool: which pixels of each token lie in the letterbox border."""
    rows = torch.arange(SIZE, device=device)[:, None].expand(SIZE, SIZE)
    return patch_pixels((rows < BAND) | (rows >= SIZE - BAND), model.patch)


def train_step(model, forward, index, actions, max_rollout: int = ROLLOUT_FRAMES):
    """One MaskGIT step on windows index [B, T, 256, 256] -> losses (total has the graph).

    Every window's context is corrupted, at a level per frame the model is told, and the loss is always
    on the real pixels:
    - corrupted context (the first half): each frame draws a level and that share of its tokens is
      replaced by wrong but real content (corrupt_context);
    - rollouts (the second half): frames s..s+k-1 (k log-uniform in 1..max_rollout, one draw per step:
      long rollouts regularly, short ones mostly, so the average cost stays near 9 frames) are
      generated one after another from the real frames before s, exactly as in a rollout, and
      every frame before s + k stays visible, so the frames after them are predicted from a real
      start followed by the model's own frames, which carry ROLLOUT_DECODING.level.
    """
    b, t = index.shape[:2]
    n = model.grid ** 2
    device = index.device
    incoming = incoming_actions(actions)
    target = patch_pixels(index, model.patch)                             # [B, T, N, P*P]
    own = b - b // 2                                                      # windows [0, own): corrupted
    corrupted, own_level, changed_tokens = corrupt_context(index[:own], model.patch)
    k = int(np.exp(np.random.uniform(0, np.log(min(max_rollout, t - 2) + 1))))  # log-uniform 1..max
    s = int(torch.randint(1, t - k, ()))                                  # s >= 1 and s + k <= t - 1
    generated = rollout(model, index[own:, :s + k], incoming[own:, :s + k], s, ROLLOUT_DECODING, compiled=True)
    inputs = index.clone()
    inputs[:own] = corrupted
    inputs[own:, s:s + k] = generated
    level = torch.zeros(b, t, dtype=torch.long, device=device)
    level[:own] = own_level
    level[own:, s:s + k] = ROLLOUT_DECODING.level
    cut = torch.zeros(b, dtype=torch.long, device=device)
    cut[own:] = s + k
    mask = training_mask(b, t, n, device, cut=cut)
    features = forward(inputs, incoming, mask, level)
    total, per_pixel, where = masked_loss(model, features, target, mask)
    per_pixel = per_pixel.detach()
    border = border_pixels(model, index.device)[where[2]]
    changed = (target[:, 1:] != target[:, :-1])[where[0], where[1] - 1, where[2]]   # frame 0 is never masked
    mean = lambda v, m: (v * m).sum() / m.sum().clamp_min(1)
    return {"total": total, "game": mean(per_pixel, ~border), "border": mean(per_pixel, border),
            "changed": mean(per_pixel, changed & ~border), "masked": mask[:, 1:].float().mean(),
            "corrupted": (changed_tokens & ~mask[:own])[:, 1:].float().mean(),   # visible tokens made wrong
            "rollout_frames": torch.tensor(float(k)),
            "rollout_wrong": (generated != index[own:, s:s + k]).float().mean()}


def preview(model, index, actions, palettes, out: Path, args) -> dict[str, float]:
    """Per game: generated vs real frames at each horizon, contact sheets and horizon_curve.csv.

    The last max(HORIZONS) frames of each window are generated one after another from the rest.
    """
    start = index.shape[1] - max(HORIZONS)
    model.eval()
    generated = rollout(model, index, incoming_actions(actions), start, Decoding(steps=args.decode_steps))
    model.train()
    real, last_context = index[:, start:], index[:, start - 1]
    game_rows = GAME_ROWS
    per = args.workers_per_game
    m = {}
    for g, game in enumerate(GAMES):
        rows = slice(g * per, (g + 1) * per)
        for h in HORIZONS:
            gen, ref, before = generated[rows, h - 1], real[rows, h - 1], last_context[rows]
            wrong = gen != ref
            changed = (ref != before)[:, game_rows]
            wg = wrong[:, game_rows]
            rate = lambda a, b: (a & b).sum().item() / max(b.sum().item(), 1)
            m.update({f"{game}_h{h}_wrong": wg.float().mean().item(),
                      f"{game}_h{h}_changed": changed.float().mean().item(),
                      f"{game}_h{h}_wrong_changed": rate(wg, changed),
                      f"{game}_h{h}_wrong_static": rate(wg, ~changed),
                      f"{game}_h{h}_border_wrong": torch.cat((wrong[:, :BAND], wrong[:, -BAND:]),
                                                             1).float().mean().item(),
                      f"{game}_h{h}_border_copy": torch.cat(((ref != before)[:, :BAND], (ref != before)[:, -BAND:]),
                                                            1).float().mean().item()})   # copying frame 47
        save_sheet(real[rows], generated[rows], last_context[rows], palettes[game], out / f"rollout_{game}.png")
        save_previews(real[rows], generated[rows], last_context[rows], palettes[game], out / "previews", game)
    temporary = out / "horizon_curve.csv.tmp"
    with temporary.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["game", "horizon", *KINDS])
        for game in GAMES:
            for h in HORIZONS:
                writer.writerow([game, h, *(m[f"{game}_h{h}_{kind}"] for kind in KINDS)])
    replace(temporary, out / "horizon_curve.csv")
    return m


def replace(temporary: Path, path: Path) -> None:
    """temporary -> path, retried for a few seconds: on Windows a viewer reading `path` blocks it briefly."""
    for _ in range(50):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            time.sleep(0.1)
    temporary.replace(path)


def save_gif(frames: list, path: Path, ms: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.gif")
    frames[0].save(temporary, save_all=True, append_images=frames[1:], duration=ms, loop=0)
    replace(temporary, path)


def save_previews(real, generated, last_context, palette, folder: Path, game: str) -> None:
    """PREVIEW_GIFS windows, from the most changing to the quieter ones, as animated GIFs: the last
    context frame, then each generated frame, as REAL | GENERATED | WRONG (magenta)."""
    change = (real[:, -1] != last_context).float().mean((1, 2))
    order = change.argsort(descending=True).tolist()
    picks = list(dict.fromkeys(order[i * len(order) // PREVIEW_GIFS] for i in range(PREVIEW_GIFS)))
    palette = palette.cpu()
    rgb = lambda frame: palette[frame.long().cpu()].numpy()
    for slot, w in enumerate(picks):
        frames = []
        for t in range(-1, real.shape[1]):
            ref = last_context[w] if t < 0 else real[w, t]
            gen = last_context[w] if t < 0 else generated[w, t]
            miss = (rgb(ref) * 0.35).astype(np.uint8)
            miss[(gen != ref).cpu().numpy()] = (255, 0, 200)
            image = Image.new("RGB", (3 * SIZE + 16, SIZE + 18), "#181820")
            draw = ImageDraw.Draw(image)
            for c, (title, panel) in enumerate((("REAL", rgb(ref)), ("GENERATED", rgb(gen)), ("WRONG", miss))):
                image.paste(Image.fromarray(panel, "RGB"), (c * (SIZE + 8), 18))
                draw.text((c * (SIZE + 8) + 2, 3), f"{title} {'context' if t < 0 else f'+{t + 1}'}", fill="white")
            frames.append(image)
        save_gif(frames, folder / f"{game}_{slot}.gif", 160)


def save_sheet(real, generated, last_context, palette, path: Path) -> None:
    """Columns: last context frame, then each horizon. Rows: real, generated, wrong pixels."""
    show = int((real[:, -1] != last_context).float().mean((1, 2)).argmax())   # the most-changing history
    palette = palette.cpu()
    rgb = lambda frame: palette[frame.long().cpu()].numpy()
    columns = [("context t", last_context[show], last_context[show])]
    columns += [(f"+{h}", real[show, h - 1], generated[show, h - 1]) for h in HORIZONS]
    label = 20
    sheet = Image.new("RGB", (len(columns) * SIZE, 3 * (SIZE + label)), "#181820")
    draw = ImageDraw.Draw(sheet)
    for c, (name, ref, gen) in enumerate(columns):
        wrong = (gen != ref).cpu().numpy()
        miss = (rgb(ref) * 0.35).astype(np.uint8)
        miss[wrong] = (255, 0, 200)
        for r, (title, image) in enumerate((("REAL", rgb(ref)), ("GENERATED", rgb(gen)), ("WRONG", miss))):
            y = r * (SIZE + label)
            draw.text((c * SIZE + 4, y + 4), f"{title} {name}", fill="white")
            sheet.paste(Image.fromarray(image, "RGB"), (c * SIZE, y + label))
    temporary = path.with_name(path.stem + ".tmp.png")
    sheet.save(temporary)
    replace(temporary, path)


def open_metrics(path: Path):
    if path.exists():
        with path.open(newline="") as f:
            header = next(csv.reader(f), [])
        if header != FIELDS:                            # renamed after closing it: Windows refuses an open file
            path.rename(path.with_name(f"metrics_schema_{int(time.time())}.csv"))
    append = path.exists()
    f = path.open("a" if append else "w", newline="")
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    if not append:
        writer.writeheader()
    return f, writer


class Member:
    """One model on the shared batches: weights, compiled forward, optimizer, averaged weights, run folder."""

    def __init__(self, name: str, args):
        self.name, self.out = name, args.outputs[name]
        self.out.mkdir(parents=True, exist_ok=True)
        self.config = {**MODELS[name], "patch": args.patch, "frames": args.frames,
                       "colour_dim": args.colour_dim, "border": BORDER_VERSION}
        self.model = build(self.config, self.config["recompute"]).cuda()
        self.path = self.out / "model_latest.pt"
        saved = torch.load(self.path, map_location="cuda", weights_only=False) if self.path.is_file() else None
        if saved is None and args.grow_from:              # a new run grown from a trained shallower one
            source = torch.load(args.grow_from, map_location="cuda", weights_only=False)
            deepen(self.model, source["ema"])
            print(f"{name}: grown from {args.grow_from} (step {source['step']})", flush=True)
        if self.config.get("frozen_head"):
            for n, p in self.model.named_parameters():
                p.requires_grad_(n not in HEAD)
        self.forward = torch.compile(self.model)          # one compiled graph per model (about a minute each)
        trained = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        decay = [p for n, p in trained if p.ndim >= 2 and "pos" not in n and "colour" not in n and "level" not in n]
        rest = [p for n, p in trained if not any(p is q for q in decay)]
        self.optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                                            {"params": rest, "weight_decay": 0.0}],
                                           lr=args.lr, betas=(0.9, 0.95), fused=True)
        self.step = 0
        if saved is not None:                             # resume this run
            self.model.load_state_dict(saved["model"])
            self.optimizer.load_state_dict(saved["optimizer"])
            self.step = int(saved["step"])
        self.live = list(self.model.state_dict().values())
        self.ema = ([saved["ema"][k].to("cuda", torch.float32) for k in self.model.state_dict()]
                    if saved is not None else [v.detach().clone() for v in self.live])
        self.start_step = self.saved_step = self.step
        self.parameters = sum(p.numel() for p in self.model.parameters())
        self.losses = None
        self.averaged = None                              # an eval copy holding the averaged weights

    def archive(self, folder: Path) -> None:
        """Keep this step's checkpoint as folder/<run>/model_step<step>.pt, next to its long-dream test, so
        a run that gets good and then worse can go back. A missing drive is reported, never fatal."""
        target = folder / self.out.name / f"model_step{self.step}.pt"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".pt.tmp")
            shutil.copyfile(self.path, temporary)
            temporary.replace(target)
        except OSError as exc:
            print(f"{self.name}: could not archive step {self.step} to {target}: {exc}", flush=True)

    def averaged_model(self) -> Dynamics:
        """The averaged weights (EMA) in an eval copy of the model, as play and the checks load them."""
        if self.averaged is None:
            self.averaged = build(self.config).cuda().eval()
        self.averaged.load_state_dict(dict(zip(self.model.state_dict(), self.ema)))
        return self.averaged

    def long_dream_test(self, trials: list, palette: np.ndarray) -> dict[str, dict[str, float]]:
        """Dream the trials with each LONG_DREAM_VARIANTS; append long_dream.csv, write long_dream.gif and
        .png (the first trial, real next to every variant) -> mean scores per variant."""
        model = self.averaged_model()
        rows, first = [], {}
        for name, decoding in LONG_DREAM_VARIANTS.items():
            for i, trial in enumerate(trials):
                frames = dream(model, trial, decoding)
                rows.append({"step": self.step, "variant": name, "level": trial.level, **score(trial, frames)})
                if i == 0:
                    first[name] = frames
        path = self.out / "long_dream.csv"
        new = not path.exists()
        with path.open("a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            if new:
                writer.writeheader()
            writer.writerows(rows)
        save_gif(animation(trials[0], first, palette), self.out / "long_dream.gif", 60)
        temporary = self.out / "long_dream.tmp.png"
        dream_sheet(trials[0], first, palette).save(temporary)
        replace(temporary, self.out / "long_dream.png")
        return {name: {k: float(np.mean([r[k] for r in rows if r["variant"] == name]))
                       for k in rows[0] if k not in ("step", "variant", "level")} for name in LONG_DREAM_VARIANTS}

    def train_step(self, index, actions, args) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate(self.step, args)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            self.losses = train_step(self.model, self.forward, index, actions)
        self.losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        with torch.no_grad():
            torch._foreach_lerp_(self.ema, self.live, 1 - EMA_DECAY)
        self.step += 1

    def checkpoint(self, index, actions, batches, stats, started, writer, args) -> dict:
        losses, step = self.losses, self.step
        if not torch.isfinite(losses["total"]):          # checked here, not every step: it waits for the GPU
            raise RuntimeError(f"{self.name}: nonfinite loss at step {step}; last checkpoint is {self.saved_step}")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m = preview(self.model, index, actions, batches.palettes, self.out, args)
        m.update(stats)
        m.update({"seconds": time.monotonic() - started, "lr": self.optimizer.param_groups[0]["lr"],
                  "train_loss": losses["total"].item(), "train_loss_game": losses["game"].item(),
                  "train_loss_border": losses["border"].item(),
                  "train_loss_changed": losses["changed"].item(),
                  "masked_fraction": losses["masked"].item(),
                  "corrupted_fraction": losses["corrupted"].item(),
                  "rollout_frames": losses["rollout_frames"].item(),
                  "rollout_wrong": losses["rollout_wrong"].item()})
        writer.writerow({"step": step, **m})
        temporary = self.out / "model_latest.pt.tmp"
        torch.save({"step": step, "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                    "ema": dict(zip(self.model.state_dict(), self.ema)),
                    "args": {**vars(args), **self.config}, "palettes": batches.palettes, "metrics": m}, temporary)
        replace(temporary, self.path)
        self.saved_step = step
        print(f"{self.name} step={step} loss={m['train_loss']:.4f} (changed {m['train_loss_changed']:.3f}) "
              "changed game pixels wrong h1/4/16 "
              + " ".join(f"{g} " + "/".join(f"{m[f'{g}_h{h}_wrong_changed']:.3f}" for h in (1, 4, 16))
                         for g in GAMES)
              + f" steps/s={(step - self.start_step) / max(m['seconds'], 1):.2f}", flush=True)
        return m


def learning_rate(step: int, args) -> float:
    """Linear warmup, then constant; with --cooldown START LENGTH, linear to zero from START to
    START + LENGTH, where training ends (the warmup-stable-decay schedule). START is fixed on the
    command line, so a relaunch after a crash continues the same schedule."""
    lr = args.lr * min(1.0, (step + 1) / args.warmup)
    if args.cooldown:
        start, length = args.cooldown
        lr *= min(1.0, max(0.0, (start + length - step) / length))
    return lr


class LongDreams:
    """Live long-dream trials for training: one low-priority DataLoader worker plays real games to a fresh
    piece and records the no-button future (diagnostics/long_dream.py); a thread keeps a few ready, so a
    test waits for games only at the start."""

    def __init__(self, seed: int):
        self.ready: queue.Queue = queue.Queue(maxsize=2 * LONG_DREAM_TRIALS)
        loader = DataLoader(TrialStream(seed), batch_size=None, num_workers=1, persistent_workers=True,
                            worker_init_fn=background_priority, prefetch_factor=2)
        threading.Thread(target=self._fill, args=(iter(loader),), daemon=True, name="long-dream-trials").start()

    def _fill(self, trials) -> None:
        for trial in trials:
            self.ready.put(trial)

    def take(self, n: int) -> list:
        return [self.ready.get() for _ in range(n)]


def train(args, events: queue.Queue, stop: threading.Event) -> None:
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        torch.manual_seed(args.seed)
        torch.set_num_threads(2)                        # the work is on the GPU; leave the cores to the workers
        logging.getLogger("torch.utils._sympy.interp").setLevel(logging.ERROR)
        members = [Member(name, args) for name in args.models]
        steps = [m.step for m in members]
        torch.manual_seed(args.seed + sum(steps))
        stream_seed = int(np.random.SeedSequence([args.seed, *steps]).generate_state(1)[0])
        batches = GameBatches(args.frames, stream_seed, args)
        histories = args.workers_per_game * len(GAMES)
        for m in members:
            (m.out / "run_config.json").write_text(json.dumps({
                **vars(args), **m.config, "model": m.name, "start_step": m.start_step,
                "stream_seed": stream_seed, "parameters": m.parameters, "trained_with": list(args.models),
                "games": GAMES, "palettes": {g: p.tolist() for g, p in batches.palettes.items()},
                "source": "fresh continuous World NES frames (../llm-thing23), RAM-driven Tetris bot "
                          "(data/tetris_bot.py), windows tossed by events (data/tetris_events.py); exact "
                          "palette pixels; no archive or eval set; the same batches as every model in "
                          "trained_with",
            }, default=str, indent=2))
        events.put(("ready", {m.name: m.parameters for m in members}, histories))
        print("World models: " + ", ".join(f"{m.name} {m.parameters:,} params (resume {m.step})" for m in members)
              + f"; {histories} histories ({args.workers_per_game} per game: {', '.join(GAMES)}) x "
              f"{args.frames}-frame windows, shared", flush=True)
        files = {m.name: open_metrics(m.out / "metrics.csv") for m in members}
        long_dreams = LongDreams(stream_seed % 100_000)
        palette = batches.palettes["tetris"].cpu().numpy().astype(np.uint8)
        started = time.monotonic()
        index = actions = None

        def checkpoint(member):
            m = member.checkpoint(index, actions, batches, stats, started, files[member.name][1], args)
            files[member.name][0].flush()
            events.put(("preview", member.name, member.step, m))

        try:
            while not stop.is_set() and (args.steps <= 0 or min(m.step for m in members) < args.steps):
                if STOP_FILE.exists():                    # a clean stop without the window: checkpoint, exit 0
                    STOP_FILE.unlink()
                    print(f"{STOP_FILE} found: stopping", flush=True)
                    break
                index, actions = batches.next()
                for member in members:
                    member.train_step(index, actions, args)
                due = [m for m in members if m.step == m.start_step + 1 or m.step % args.preview_every == 0]
                if due:
                    stats = batches.take_stats()          # data waits since the last checkpoint, shared
                    for member in due:
                        checkpoint(member)
                if members[0].step % args.long_dream_every == 0:
                    trials = long_dreams.take(LONG_DREAM_TRIALS)   # the same trials for every model
                    for member in members:
                        if member.saved_step != member.step:
                            checkpoint(member)
                        member.archive(args.archive)
                        summary = member.long_dream_test(trials, palette)
                        events.put(("longdream", member.name, member.step, summary))
                        print(f"{member.name} step={member.step} long dream: " + "; ".join(
                            f"{v} wrong +16/+128 {r['wrong_16']:.2%}/{r['wrong_128']:.2%} mass {r['mass']:.2f}"
                            for v, r in summary.items()), flush=True)
                if args.seconds > 0 and time.monotonic() - started >= args.seconds:
                    break
            if index is not None:
                stats = batches.take_stats()
                for member in members:
                    if member.step != member.saved_step:
                        checkpoint(member)
        finally:
            for f, _ in files.values():
                f.close()
        events.put(("done", {m.name: m.step for m in members}))
    except BaseException as exc:
        import traceback
        traceback.print_exc()
        args.failed = True
        events.put(("error", str(exc)))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--keep-all-windows", action="store_true",
                   help="no event tossing: every window, so events come at the rate the game has them")
    p.add_argument("--grow-from", type=Path, default=None,
                   help="a shallower run's checkpoint a new deeper run starts from (deepen); ignored on resume")
    p.add_argument("--models", nargs="+", choices=list(MODELS), default=["small"],
                   help="models trained side by side on the same batches (MODELS; large is paused)")
    p.add_argument("--output-root", type=Path, default=ROOT / "output",
                   help="each model's run folder is world_model_tetris_<name> in here")
    p.add_argument("--compare", type=Path, nargs="*", default=[],
                   help="earlier runs drawn dashed beside these in the window (by step)")
    p.add_argument("--seconds", type=int, default=0, help="wall-clock limit; 0 trains until Stop Training")
    p.add_argument("--steps", type=int, default=0, help="optimizer-step limit; 0 has none")
    p.add_argument("--workers-per-game", type=int, default=16, help="live histories per game per step")
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--preview-every", type=int, default=200)
    p.add_argument("--long-dream-every", type=int, default=LONG_DREAM_EVERY,
                   help="steps between long-dream tests; each tested step's checkpoint is archived")
    p.add_argument("--archive", type=Path, default=Path("D:/token_world_checkpoints"),
                   help="checkpoints kept at every long-dream test, in <archive>/<run folder>/ (D: has the space)")
    p.add_argument("--frames", type=int, default=64, help="window length: the model's context")
    p.add_argument("--colour-dim", type=int, default=16, help="per-pixel colour embedding size")
    p.add_argument("--patch", type=int, default=16, help="pixels per transformer token side")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=1000, help="linear learning-rate warmup steps")
    p.add_argument("--cooldown", type=int, nargs=2, metavar=("START", "LENGTH"), default=None,
                   help="decay the learning rate linearly to zero from step START over LENGTH steps, then stop")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--decode-steps", type=int, default=1, help="MaskGIT steps per previewed frame")
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--tetris-repo", type=str, default=None)
    p.add_argument("--close-when-done", action="store_true",
                   help="close the window after training finishes or fails (unattended runs)")
    args = p.parse_args()
    if args.cooldown and not args.steps:
        args.steps = sum(args.cooldown)
    args.failed, args.games = False, GAMES
    args.outputs = {name: args.output_root / f"world_model_tetris_{name}" for name in args.models}
    args.shapes = {name: MODELS[name] for name in args.models}
    from token_world.ui.dynamics_app import launch
    launch(args, train)
    sys.exit(1 if args.failed else 0)


if __name__ == "__main__":
    main()
