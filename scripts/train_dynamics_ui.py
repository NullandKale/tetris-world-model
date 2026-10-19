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
weight, the letterbox border included. The game, border and changed-pixel
parts are reported, not weighted. The model is soft: its frames are colour
distributions, never argmax (models/dynamics.py).

Two stages, as HorizonDrive trains (examples/papers/horizondrive_2605.11596):
the base model on real, clean context (teacher forcing: --models base), then
scheduled rollout recovery from its weights (--models srr --grow-from ...
--rollouts ... --blend ...), where every window's history is the model's own
rollout, its last frames fading to the real ones, and the frames after it are
scored against the real ones. Every window is trained on, at the rate the game has its
events. Alongside the trained weights the checkpoint keeps an exponential
moving average of them (EMA_DECAY), which play and the checks load.

Previews roll the model out on the current batch: the first 48 frames of each
window are context, the last 16 are generated one after another with the real
actions, and are scored per game against the real frames; four windows, from
the most changing to the quieter ones, are saved as animated GIFs
(previews/). Every --long-dream-every steps (1,000) each model's averaged weights dream
live trials 1,000 frames ahead (diagnostics/long_dream.py), 32 with no buttons and
32 with the bot's buttons, appended to long_dream.csv with an animated GIF. Fresh live frames only; no eval set.
The window (ui/run_viewer.py) reads it all back from the run folders. See
docs/guides/dynamics.md.
"""
from __future__ import annotations

import argparse
import gc
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
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from token_world.data.model_frames import BAND, BORDER_VERSION
from token_world.data.nes_palette import tetris_palette
from token_world.data.tetris_bot import A, B, LEFT, RIGHT
from token_world.data.world_nes_tetris import DATA_NAMES, KEEP_ALL, WORLD_MODEL, WorldNesTetrisStreams, spawn_boards
from token_world.diagnostics.novelty import NOVELTY_FIELDS, Novelty
from token_world.diagnostics import bias
from token_world.diagnostics.coherence import PatchBank
from token_world.diagnostics.coherence import score as coherence
from token_world.diagnostics.long_dream import TrialStream, animation, dreams, score, stacked
from token_world.diagnostics.long_dream import sheet as dream_sheet
from token_world.diagnostics.run_outputs import (HORIZONS, append_long_dream, horizon_fields, horizon_outputs,
                                               load_optimizer, open_metrics, optimizer_state, replace, save_gif,
                                               write_horizon_curve)
from token_world.data.tetris_motion import UNKNOWN as MOTION_UNKNOWN
from token_world.models.dynamics import (COLOURS, SIZE, Dynamics, border_pixels, build, choice_kl, deepen, incoming_actions,
                                         stacked_source,
                                         MOVED, TURNED, masked_loss, motion_loss, motion_tokens, own_share, patch_pixels, rollout, sample_choice,
                                         training_mask)
from token_world.data.live_batches import LiveBatches, data_fields
from token_world.models.turn_critic import (TURNS_AT, TurnCritic, critic_losses, crop_pixels, crop_tokens,
                                            hard_sample, tokens_to_image, turn_crops)
from token_world.data.tetris_events import EVENTS, NEXT_BOX, event_slug

# Tetris only: the small, fast model on the simpler game first. GameBatches supports
# "contra" too (data/world_nes_contra.py; earlier runs and results in docs/guides/dynamics.md).
# Adding it also needs its own --output-root: run folders are named world_model_tetris_<name>.
GAMES = ("tetris",)
# recompute: keep only the patch and block inputs for backward and recompute the rest (same model and
# gradients, less memory, one more forward): the large model needs it, the small one is 7% faster without.
MODELS = {# HorizonDrive's two stages at the small model's size (~2.0M parameters): the base model on real
          # context only, then scheduled rollout recovery from its weights (--grow-from, --rollouts, --blend)
          "base": {"dim": 96, "layers": 8, "heads": 4, "recompute": False},
          "srr": {"dim": 96, "layers": 8, "heads": 4, "recompute": False},
          # a shallow model for fast experiments (2026-10-14): the colour embedding and the pixel head cost
          # the same at any depth, so it trains about twice as fast, not 8 / 3 times
          "s3": {"dim": 96, "layers": 3, "heads": 4, "recompute": False},
          # s3 grown a layer at a time (--grow-from the one before; deepen inserts the new block), 2026-10-14
          **{f"s{n}": {"dim": 96, "layers": n, "heads": 4, "recompute": False} for n in range(4, 13)},
          # buttons into the last block only (2026-10-15): no input add, no actnorm in blocks 1-2 (Dynamics
          # action_blocks); needs --action-norm
          "s3_late": {"dim": 96, "layers": 3, "heads": 4, "recompute": False, "action_blocks": (2,)},
          # which layer a grown one copies (2026-10-15): s3 + a halved copy of block k right after it (deepen new_at k,
          # copy); copying, placement is the choice of block (zero-start new blocks stayed unused)
          **{f"s4copy_b{k}": {"dim": 96, "layers": 4, "heads": 4, "recompute": False, "new_at": k, "copy": True}
             for k in (1, 2, 3)},
          # the base with a sampled choice per frame (4 categoricals of 8, models/dynamics.py latent), grown from
          # the base (--grow-from): random outcomes are chosen, not averaged
          "latent": {"dim": 96, "layers": 8, "heads": 4, "recompute": False, "latent": [4, 8]},
          # s3 with the turn critic (models/turn_critic.py, 2026-10-16): its weight on the model's loss; trained
          # beside an s3 on the same batches, the control
          "s3_critic": {"dim": 96, "layers": 3, "heads": 4, "recompute": False, "critic": 1.0},
          # an s3 stacked whole three times (--grow-from it; deepen stack, G_stack), 2026-10-17: plain copies, each
          # copy with its source block's optimizer state, the learning rate and step count carried on
          "s9_stack": {"dim": 96, "layers": 9, "heads": 4, "recompute": False, "stack": 3}}
WINDOW_STRIDE = 16                                # a window every 16 frames: each frame in four windows
BANK_EVERY = 16                                   # steps between additions to the coherence bank
STOP_FILE = ROOT / "output" / "stop_training"      # create it to stop cleanly, like Stop Training
EMA_DECAY = 0.999                                 # averaged weights: about the last 1,000 steps
LONG_DREAM_EVERY = 2000                           # steps between long-dream tests
LONG_DREAM_TRIALS = 32                            # trials per test: the same games at every test and in every
                                                  # run with the same --seed, so tests compare like with like
LONG_DREAM_VARIANT = "1 pass"                     # long_dream.csv's variant column: no buttons
PRESSED_VARIANT = "buttons"                       # and the bot's buttons (since 2026-10-08)
LIVE_FIELDS = [f"live_{event_slug(n)}_{k}" for n in EVENTS for k in ("n", "exact", "wrong", "false")] + [
    "live_turn_n", "live_turn_exact", "live_turn_exact_swapped", "live_turn_gap", "live_turn_wrong",
    "live_turn_wrong_swapped"]


FIELDS = ["step", "seconds", "lr", "train_loss", "train_loss_game", "train_loss_border",
          "train_loss_changed", "masked_fraction", "rollout_depth", "blend_frames", "latent_kl",
          "rollout_frames", "rollout_wrong", "rollout_ghost_px", "rollout_resynced",
          "train_loss_motion", "train_motion_moved", "train_motion_turned", "preview_motion_moved",
          "preview_motion_turned", "critic_loss", "critic_g", "critic_direction", "critic_turns",
          *LIVE_FIELDS,
          "data_wait_ms_mean", "data_wait_ms_max", "data_branches", "data_branch_swapped",
          "data_branch_released",
          "stream_tick_min", "stream_tick_max", "in_game_restarts",
          *horizon_fields(GAMES), *data_fields(DATA_NAMES)]


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
    frames in one layout (data/model_frames.py). Every window is kept.
    """

    def __init__(self, frames: int, seed: int, args, toss=KEEP_ALL, marked: bool = False):
        """toss: the Tetris stream's (world_nes_tetris: KEEP_ALL, every window; EVENTS_TOSS, the latent trainer's
        --toss events); marked: every window comes, `kept` says which the toss kept (the latent trainer's
        streaming encoder sees every frame)."""
        unknown = set(GAMES) - {"tetris", "contra"}
        if unknown:
            raise ValueError(f"no World NES stream for {sorted(unknown)}")
        self.palettes, streams = {}, {}
        if "tetris" in GAMES:
            self.palettes["tetris"] = tetris_palette()
            streams["tetris"] = WorldNesTetrisStreams(frames, seed, args.tetris_repo, stride=WINDOW_STRIDE,
                                                       screens=WORLD_MODEL, toss=toss, marked=marked,
                                                       motion=getattr(args, "motion", False),
                                                       branches=getattr(args, "branches", 0))
        if "contra" in GAMES:                     # imported only when used: Contra can't break a Tetris run
            from token_world.data.world_nes_contra import WorldNesContraStreams, contra_palette
            self.palettes["contra"] = contra_palette()
            streams["contra"] = WorldNesContraStreams(frames, seed + 1, stride=WINDOW_STRIDE)
        self.batches = {g: LiveBatches(DataLoader(streams[g], batch_size=None,
                                                  num_workers=args.workers_per_game, pin_memory=True,
                                                  persistent_workers=True, worker_init_fn=background_priority,
                                                  prefetch_factor=args.prefetch_factor),
                                       args.workers_per_game, streams[g].new_frames,
                                       novelty=Novelty(spawn_boards) if g == "tetris" else None)
                        for g in GAMES}

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        """-> index [B, T, 256, 256] uint8, stream actions [B, T] (action t takes frame t to t + 1); `kept` [B]:
        the windows the toss kept (all but in a marked stream); `branch_at` [B]: each row's branch press
        position, -1 for a real window (rows after each game's workers are counterfactual twins, --branches)."""
        index, actions = [], []
        for game in GAMES:
            x, a = self.batches[game].next()
            index.append(x)
            actions.append(a)
        self.kept = torch.cat([self.batches[g].kept for g in GAMES])
        piece = [self.batches[g].piece for g in GAMES]
        events = [self.batches[g].events for g in GAMES]
        self.events = torch.cat(events) if all(e is not None for e in events) else None   # [B, T]
        self.piece = torch.cat(piece) if all(p is not None for p in piece) else None     # [B, T]
        self.branch_at = torch.cat([self.batches[g].branch_at for g in GAMES])
        motion = [self.batches[g].motion for g in GAMES]
        self.motion = torch.cat(motion) if all(m is not None for m in motion) else None   # [B, T, 32, 32]
        return torch.cat(index), torch.cat(actions)

    def take_stats(self) -> dict[str, float]:
        """Stream counters over all games; data waits add up, since games are fetched in turn."""
        s = [b.take_stats() for b in self.batches.values()]
        labelled = {k: v for k, v in s[0].items()
                    if (k.startswith("data_") and not k.startswith("data_wait")) or k in NOVELTY_FIELDS}
        return {**labelled,                       # what the windows hold: the first game's (Tetris: labelled)
                "data_wait_ms_mean": sum(v["data_wait_ms_mean"] for v in s),
                "data_wait_ms_max": sum(v["data_wait_ms_max"] for v in s),
                "stream_tick_min": min(v["stream_tick_min"] for v in s),
                "stream_tick_max": max(v["stream_tick_max"] for v in s),
                "in_game_restarts": sum(v["in_game_restarts"] for v in s)}


DIRECTED = A | B | LEFT | RIGHT     # the presses with a direction (which way a piece turns or shifts)
LIVE_EVERY = 8                      # steps between the training batch's own event scores (live_event_scores)
LIVE_ORDER = ("game start", "top-out", "level up", "line clear: tetris", "line clear: triple",
              "line clear: double", "line clear: single", "spawn", "rotate", "shift", "soft drop",
              "falling alone", "gravity (no buttons)")     # rarest first: the event a window is scored on
PLAYFIELD = (slice(52, 212), slice(92, 172))                # the well, in model frames


def live_event_scores(model, forward, index: torch.Tensor, incoming: torch.Tensor, events: torch.Tensor,
                      motion: torch.Tensor | None = None, piece: torch.Tensor | None = None,
                      generator: torch.Generator | None = None) -> dict[str, float]:
    """The training batch's own event scores, one step ahead from the real past (2026-10-17; the user: "why are
    we not gathering these stats just on the fucking training data"): the stream is fresh every step, so every
    batch is unseen. In each window one event frame (events [B, T], tetris_events.frame_events; the rarest
    present, LIVE_ORDER, at a random frame of it) is hidden whole, as a dream draws it, everything before it
    real (no gradient). Scored as the scenario checks score (diagnostics/tetris_scenarios.py), on that frame:
    exact (every pixel that changed from the frame before more likely right than wrong), wrong (the changed
    pixels' expected wrong share), false (the well's unchanged pixels' expected changed share). The J / L /
    T turns are scored again with A and B swapped (piece [B, T], each frame's RAM orientation byte): the turn
    gap, exact with the real press less with the swapped one. -> sums: "<event> n / exact / wrong / false",
    "turn n / exact / exact swapped / wrong / wrong swapped"."""
    b, t = index.shape[:2]
    device, n, p = index.device, model.grid ** 2, model.patch
    ev = events.cpu().clone()
    ev[:, :2] = 0
    rows, frames, names = [], [], []
    for w in range(b):
        for name in LIVE_ORDER:
            at = ((ev[w] >> EVENTS.index(name)) & 1).nonzero()[:, 0]
            if len(at):
                rows.append(w)
                frames.append(int(at[torch.randint(len(at), (1,), generator=generator)]))
                names.append(name)
                break
    if not rows:
        return {}
    rw, rf = torch.tensor(rows, device=device), torch.tensor(frames, device=device)
    mask = torch.zeros(b, t, n, dtype=torch.bool, device=device)
    mask[rw, rf] = True
    given = None
    if model.motion:
        given = motion_tokens(motion, p)
        given = torch.where(mask[..., None], torch.full_like(given, MOTION_UNKNOWN), given)
    keep = torch.ones(SIZE, SIZE, dtype=torch.bool, device=device)
    keep[:BAND], keep[SIZE - BAND:] = False, False
    keep[NEXT_BOX] = False
    well = torch.zeros(SIZE, SIZE, dtype=torch.bool, device=device)
    well[PLAYFIELD] = True
    keep, well = patch_pixels(keep[None, None], p)[0, 0], patch_pixels(well[None, None], p)[0, 0]   # [N, P*P]
    real = patch_pixels(index[rw, rf][:, None], p)[:, 0].long()                   # [K, N, P*P]
    before = patch_pixels(index[rw, rf - 1][:, None], p)[:, 0].long()
    changed, static = (real != before) & keep, (real == before) & well

    def scored(acts):
        probs = model.logits(forward(index, acts, mask, motion=given)[rw, rf]).float().softmax(-1)
        wrong = 1 - probs.gather(-1, real[..., None])[..., 0]                      # [K, N, P*P]
        moved = 1 - probs.gather(-1, before[..., None])[..., 0]
        has = changed.flatten(1).any(1)
        exact = torch.where(has, ((wrong < 0.5) | ~changed).flatten(1).all(1).float(), torch.nan)
        miss = torch.where(has, (wrong * changed).flatten(1).sum(1) / changed.flatten(1).sum(1).clamp_min(1), torch.nan)
        false = (moved * static).flatten(1).sum(1) / static.flatten(1).sum(1).clamp_min(1)
        return exact.cpu(), miss.cpu(), false.cpu()
    exact, miss, false = scored(incoming)
    out = {}
    for i, name in enumerate(names):
        if torch.isnan(exact[i]):
            continue
        for key, value in (("n", 1.0), ("exact", float(exact[i])), ("wrong", float(miss[i])),
                           ("false", float(false[i]))):
            out[f"{name} {key}"] = out.get(f"{name} {key}", 0.0) + value
    turning = [i for i, name in enumerate(names) if name == "rotate" and piece is not None
               and bool(TURNS_AT[int(piece[rows[i], frames[i] - 1])]) and not torch.isnan(exact[i])]
    swapped = incoming.clone()
    for i in turning:                         # the press shows a frame late: frame f - 1's incoming
        w, f = rows[i], frames[i]
        for at in (f - 1, f):
            v = int(swapped[w, at])
            if v >= 0 and (v & (A | B)) in (A, B):
                swapped[w, at] = (v & ~(A | B)) | ((A | B) ^ (v & (A | B)))
    if turning:
        exact_s, miss_s, _ = scored(swapped)
        out["turn n"] = float(len(turning))
        out["turn exact"] = float(sum(exact[i] for i in turning))
        out["turn exact swapped"] = float(sum(exact_s[i] for i in turning))
        out["turn wrong"] = float(sum(miss[i] for i in turning))
        out["turn wrong swapped"] = float(sum(miss_s[i] for i in turning))
    return out


def live_means(sums: dict[str, float]) -> dict[str, float]:
    """live_event_scores' sums over many batches -> metrics.csv's live_ columns (means, and counts)."""
    out = {}
    for name in EVENTS:
        k = sums.get(f"{name} n", 0.0)
        out[f"live_{event_slug(name)}_n"] = k
        for key in ("exact", "wrong", "false"):
            out[f"live_{event_slug(name)}_{key}"] = sums.get(f"{name} {key}", 0.0) / k if k else float("nan")
    k = sums.get("turn n", 0.0)
    out["live_turn_n"] = k
    for key, field in (("turn exact", "live_turn_exact"), ("turn exact swapped", "live_turn_exact_swapped"),
                       ("turn wrong", "live_turn_wrong"), ("turn wrong swapped", "live_turn_wrong_swapped")):
        out[field] = sums.get(key, 0.0) / k if k else float("nan")
    out["live_turn_gap"] = out["live_turn_exact"] - out["live_turn_exact_swapped"]
    return out


def press_weights(incoming: torch.Tensor, weight: float) -> torch.Tensor | None:
    """Incoming actions [B, T] -> each frame's loss weight [B, T]: `weight` on the frames an A, B, Left or Right
    press reaches (its incoming button, or the frame before's: the press shows a frame late), 1 on the rest;
    None when weight is 1. They are 17% of the frames (2026-10-15; with every button 45%, most of them held
    soft drops). The buttons are inputs the model always sees, so the weight leaves the probabilities it trains
    to as they are (nothing to undo when decoding, unlike Dynamics.change_weight): it only gives those frames,
    every pixel of them, the loss's attention, the cells a wrong turn would fill as much as the real turn's."""
    if weight == 1.0:
        return None
    return 1.0 + (weight - 1.0) * pressed_frames(incoming).float()


def pressed_frames(incoming: torch.Tensor, branch_at: torch.Tensor | None = None) -> torch.Tensor:
    """Incoming actions [B, T] -> [B, T] the frames an A, B, Left or Right press reaches (its incoming button,
    or the frame before's: the press shows a frame late). branch_at [B] (GameBatches): a counterfactual twin's
    two frames after its branch press are counted too, so a released press's frames (no button) are
    treated as the real window's are."""
    pressed = (incoming >= 0) & (incoming & DIRECTED > 0)
    reached = pressed.clone()
    reached[:, 1:] |= pressed[:, :-1]
    if branch_at is not None:
        at = branch_at.to(incoming.device)[:, None]
        place = torch.arange(incoming.shape[1], device=incoming.device)[None]
        reached |= (at >= 0) & (place > at) & (place <= at + 2)
    return reached


def train_step(model, forward, index, actions, depth: int = 0, blend: int = 0,
               generator: torch.Generator | None = None, drawer: Dynamics | None = None, read_choice=None,
               resync: int | None = None, hidden: float = 0.0, press_weight: float = 1.0,
               motion: torch.Tensor | None = None, hide_presses: bool = False,
               branch_at: torch.Tensor | None = None, critic: TurnCritic | None = None,
               critic_weight: float = 0.0, piece: torch.Tensor | None = None):
    """One MaskGIT step on windows index [B, T, 256, 256] -> losses (total has the graph).

    depth 0, the base model (HorizonDrive's first stage): teacher forcing. Every window is real and clean;
    each frame's tokens are hidden at a rate drawn per frame and predicted from the real frames before it
    (training_mask: half the windows with every frame before a random cut fully visible).
    depth > 0, scheduled rollout recovery (its second stage, sec. 4.2): every window is a rollout. From a
    real start s, `drawer` generates k frames (k uniform in 1..depth) one after another, as in a dream, and
    they go in as its soft frames; the frames from the boundary s + k on are real and scored, every frame
    before the boundary visible. blend: the history's last w frames (w uniform in 0..blend each step)
    fade from the drawer's own to the real ones (own_share), so the frame before the first scored one runs
    from nearly real (w = blend) to fully its own (w = 0, a dream's condition); the targets are always the
    real frames. drawer: the model that draws the rollouts (HorizonDrive's cached rollouts: a frozen copy
    refreshed every --refresh steps; default the model itself). resync: after each rollout frame, its
    patches with at least this many pixels off (most likely colour not the real one) are replaced by the real
    ones before the next frame reads them (models/dynamics.py FrameDecoder): the history stays in sync with
    the real game, so the real buttons and the real targets never contradict it, and keeps only the model's
    small errors, which it learns to ignore.
    With a latent (models/dynamics.py), every frame after the first gets the choice its posterior reads from
    the real frame and the one before (sampled, straight-through; a rollout's dreamed history keeps the
    choices it was drawn with, and its frames are left out of the KL), and the prior learns to predict it from
    the previous frame's features and the frame's incoming buttons: DreamerV3's KL balancing (0.5 dynamics, 0.1 representation), in nats
    per pixel as the reconstruction. No free nats (DreamerV3 has 1 per step): Tetris averages well under
    0.1 nats of randomness a frame (a new piece about 2, most frames none), and with a free nat the choice
    took over gravity and Start, which the frames and buttons decide (docs/guides/recipe_attempts.md).
    """
    drawer = model if drawer is None else drawer
    b, t = index.shape[:2]
    if model.motion and motion is None:
        raise ValueError("a motion model trains on windows with their motion")
    n = model.grid ** 2
    device = index.device
    incoming = incoming_actions(actions)
    target = patch_pixels(index, model.patch)                             # [B, T, N, P*P]
    cut = torch.zeros(b, dtype=torch.long, device=device)                 # 0: a random clean cut, half the time
    soft = unsure = wrong = resynced = own_motion = None
    k = w = 0
    if depth:
        k = int(torch.randint(1, min(depth, t - 2) + 1, (), generator=generator))
        s = int(torch.randint(1, t - k, (), generator=generator))           # s >= 1, the boundary s + k <= t - 1
        w = int(torch.randint(0, blend + 1, (), generator=generator))
        boundary = s + k
        tokens, unsure, wrong, drawn, resynced, own_motion = rollout(
            drawer, index[:, :boundary], incoming[:, :boundary], s, compiled=True, resync=resync,
            motion=motion[:, :boundary] if model.motion else None)
        lo = max(s, boundary - w)                                           # the first frame with a real part
        if lo < boundary:
            real = model._patches(index[:, lo:boundary]).to(tokens.dtype)
            part = own_share(s, boundary, w, device)[lo - s:][None, :, None, None].to(tokens.dtype)
            tokens = torch.cat((tokens[:, :lo - s], part * tokens[:, lo - s:] + (1 - part) * real), 1)
        where = torch.zeros(b, t, dtype=torch.bool, device=device)
        where[:, s:boundary] = True
        values = torch.zeros(b, t, n, tokens.shape[-1], dtype=tokens.dtype, device=device)
        values[:, s:boundary] = tokens
        soft = (where, values)
        cut[:] = boundary
    hide = pressed_frames(incoming, branch_at) if hide_presses else None     # turns from the button alone
    crops = turning = None
    if critic is not None:                           # the J / L / T turn frames the critic judges: hidden whole
        if motion is None or piece is None:
            raise ValueError("the turn critic needs the batch's motion and piece bytes")
        crops = turn_crops(incoming, motion, piece, model.grid, TURNED)
        turning = torch.zeros(b, t, dtype=torch.bool, device=device)
        turning[crops[0], crops[1]] = True
        hide = turning if hide is None else hide | turning
    mask = training_mask(b, t, n, device, generator, cut=cut, hidden=hidden, hide=hide)
    z = posterior = None
    if model.latent:
        posterior = (read_choice or model.read_choice)(index)              # [B, T, groups, classes]
        z = sample_choice(posterior, straight_through=True)
        z = z * (torch.arange(t, device=device) > 0)[None, :, None]       # frame 0 has no choice
        if depth:                                    # the dreamed history goes with the choices that drew it
            z = torch.cat((z[:, :s], drawn.to(z.dtype), z[:, boundary:]), 1)
    given = real_motion = None
    if model.motion:                                  # VideoJAM: the motion goes in and is predicted
        real_motion = motion_tokens(motion, model.patch)                      # [B, T, N, cells]
        context = real_motion
        if depth:                                     # the dreamed history goes with the motion it was drawn
            context = torch.cat((real_motion[:, :s], motion_tokens(own_motion, model.patch).to(real_motion.dtype),
                                 real_motion[:, boundary:]), 1)                # with (real where resynced)
        given = withheld_motion(context, generator)
        scored = mask[..., None].expand_as(given)     # the masked tokens' cells: their motion withheld and scored,
        if model.motion_first:                        # but for motion_first a share of masked frames is given all
            told = torch.rand(b, t, 1, 1, device=device, generator=generator) < MOTION_FIRST_SHARE   # or part
            rate = torch.rand(b, t, 1, 1, device=device, generator=generator)                       # of it
            rate = torch.where(torch.rand(b, t, 1, 1, device=device, generator=generator) < 0.5, 1.0, rate)
            shown = told & (torch.rand(given.shape, device=device, generator=generator) < rate)
            if turning is not None:                   # a judged turn is drawn without its motion, as in a dream
                shown = shown & ~turning[:, :, None, None]
            scored = scored & ~shown
        given = torch.where(scored, torch.full_like(given, MOTION_UNKNOWN), given)
    features = forward(index, incoming, mask, soft=soft, z=z, motion=given)
    weight = model.loss_weights(target)
    frames = press_weights(incoming, press_weight)
    if frames is not None:
        frames = frames[:, :, None, None].expand_as(target)
        weight = frames if weight is None else weight * frames
    total, per_pixel, where = masked_loss(model, features, target, mask, weight=weight)
    kl = torch.tensor(float("nan"))
    if model.latent:
        dynamics_kl, representation_kl = choice_kl(posterior[:, 1:], model.prior(features[:, :-1], incoming[:, 1:]))
        read = torch.ones(t - 1, device=device)      # frames 1..T-1 whose choice the posterior read: not the
        if depth:                                    # dreamed history's (its choices are the drawer's)
            read[s - 1:boundary - 1] = 0
        balanced = 0.5 * dynamics_kl + 0.1 * representation_kl                    # [B, T - 1]
        total = total + (balanced * read).sum() / (read.sum() * b) / SIZE ** 2
        kl = (dynamics_kl.detach() * read).sum() / (read.sum() * b)
    moving = {}
    if model.motion:
        moving = motion_loss(model, features, real_motion, scored)
        total = total + moving["motion"]
    critic_out = {}
    if crops is not None and len(crops[0]):
        cb, cf, top, left, button = crops
        made = model.logits(features[cb[:, None], cf[:, None], crop_tokens(top, left, model.grid)])
        fake = tokens_to_image(hard_sample(made, generator), model.patch)              # [K, C, 48, 48]
        one_hot = lambda frames: F.one_hot(crop_pixels(frames.long(), top, left, model.patch),
                                           COLOURS).permute(0, 3, 1, 2).float()
        d, g, direction = critic_losses(critic, one_hot(index[cb, cf - 1]), one_hot(index[cb, cf]), fake, button)
        # the critic's pull on each judged pixel, in the units of the pixel loss (a mean over the masked pixels)
        judged = fake.shape[0] * fake.shape[2] * fake.shape[3]
        total = total + critic_weight * g * judged / (mask.sum() * model.patch ** 2).clamp_min(1)
        critic_out = {"critic_loss": d, "critic_g": g.detach(), "critic_direction": direction}
    per_pixel = per_pixel.detach()
    border = border_pixels(model.patch, index.device)[where[2]]
    changed = (target[:, 1:] != target[:, :-1])[where[0], where[1] - 1, where[2]]   # frame 0 is never masked
    mean = lambda v, m: (v * m).sum() / m.sum().clamp_min(1)
    nan = torch.tensor(float("nan"))
    return {"total": total, "game": mean(per_pixel, ~border), "border": mean(per_pixel, border),
            "changed": mean(per_pixel, changed & ~border), "masked": mask[:, 1:].float().mean(),
            "rollout_frames": torch.tensor(float(k)), "blend": torch.tensor(float(w)), "latent_kl": kl,
            "rollout_wrong": wrong.mean() if wrong is not None else nan,          # its frames' pixels expected wrong
            "rollout_ghost_px": unsure.mean() if unsure is not None else nan,     # unsure pixels per generated frame
            "rollout_resynced": resynced.mean() if resynced is not None else nan,  # its patches replaced by real
            "motion": moving["motion"].detach() if moving else nan,               # motion loss and the share
            "motion_moved": moving.get("motion_moved", nan),                     # of moved / turned cells
            "motion_turned": moving.get("motion_turned", nan),                   # it gets right
            "critic_loss": critic_out.get("critic_loss", nan), "critic_g": critic_out.get("critic_g", nan),
            "critic_direction": critic_out.get("critic_direction", nan),         # real beats swapped
            "critic_turns": torch.tensor(float(len(crops[0]) if crops is not None else 0))}


MOTION_FIRST_SHARE = 0.5   # motion_first: the share of masked frames given their cells' real motion, all of it half
                           # the time (the pixels learn to follow it), else a random share of the cells (the second
                           # motion step learns to draw the rest given them); given cells are not scored
MOTION_DROP = 0.2       # VideoJAM drops the motion input in 20% of steps; here per window, and as many again
                        # lose only the frames before a cut: a dream starts with none (a screenshot), then its own


def withheld_motion(motion: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Each window's motion tokens [B, T, N, cells] -> the ones the model is given: a MOTION_DROP share of windows
    none (UNKNOWN), as many none before a random cut, the rest all; the loss always scores the real ones."""
    b, t = motion.shape[:2]
    device = motion.device
    u = torch.rand(b, device=device, generator=generator)
    cut = torch.randint(1, t, (b,), device=device, generator=generator)
    hidden = (u < MOTION_DROP)[:, None] | ((u < 2 * MOTION_DROP)[:, None] & (torch.arange(t, device=device) < cut[:, None]))
    return torch.where(hidden[:, :, None, None], torch.full_like(motion, MOTION_UNKNOWN), motion)


def preview(model, index, actions, palettes, out: Path, args, motion: torch.Tensor | None = None) -> dict[str, float]:
    """Per game: generated vs real frames at each horizon, contact sheets and horizon_curve.csv.

    The last max(HORIZONS) frames of each window are generated one after another from the rest, as soft
    frames; "wrong" is each pixel's probability of anything but the real colour (expected wrong pixels).
    A motion model's context gets the batch's motion, and the motion it draws for its own frames is scored on
    the real moved and turned cells (preview_motion_moved / _turned).
    """
    b, t = index.shape[:2]
    start = t - max(HORIZONS)
    incoming = incoming_actions(actions)
    real, last_context = index[:, start:], index[:, start - 1]
    model.eval()
    decoder = model.decoder(b, index.device)
    decoder.prefill(index[:, :start], incoming[:, :start], motion[:, :start] if model.motion else None)
    wrong, rgb, drawn = [], [], []
    colours = {g: palettes[g].to(index.device, torch.float32) for g in GAMES}
    per = args.workers_per_game
    for i, at in enumerate(range(start, t)):
        made = decoder.next(incoming[:, at:at + 1], at)
        probs = made["probs"].float()                                                # [B, 256, 256, COLOURS]
        if model.motion:
            drawn.append(made["motion"])
        wrong.append(1 - probs.gather(-1, real[:, i, ..., None].long())[..., 0])
        rgb.append(torch.cat([probs[g * per:(g + 1) * per] @ colours[game] for g, game in enumerate(GAMES)])
                   .round().clamp(0, 255).byte())
    model.train()
    wrong, rgb = torch.stack(wrong, 1), torch.stack(rgb, 1)                         # [B, H, 256, 256], [.., 3]
    m = {}
    drawn = torch.stack(drawn, 1) if drawn else None                               # [B, H, 32, 32] its own motion
    for g, game in enumerate(GAMES):
        rows = slice(g * per, (g + 1) * per)
        m.update(horizon_outputs(real[rows], rgb[rows], wrong[rows], last_context[rows], palettes[game], out,
                                 game, motion[rows, start - 1:] if drawn is not None else None,
                                 drawn[rows] if drawn is not None else None))
    write_horizon_curve(m, GAMES, out)
    nan = float("nan")
    m["preview_motion_moved"] = m["preview_motion_turned"] = nan
    if model.motion:                         # the motion it drew in its own frames against the real
        real_motion, drawn = motion[:, start:].long(), drawn.long()
        code = real_motion.clamp(max=len(MOVED) - 1)
        for key, kind in (("preview_motion_moved", MOVED), ("preview_motion_turned", TURNED)):
            cells = kind.to(code.device)[code] & (real_motion != MOTION_UNKNOWN)
            if cells.any():
                m[key] = (drawn == real_motion)[cells].float().mean().item()
    return m


class Member:
    """One model on the shared batches: weights, compiled forward, optimizer, averaged weights, run folder."""

    def __init__(self, name: str, args):
        self.name, self.out = name, args.outputs[name]
        self.out.mkdir(parents=True, exist_ok=True)
        self.config = {**MODELS[name], "patch": args.patch, "frames": args.frames,
                       "colour_dim": args.colour_dim, "border": BORDER_VERSION,
                       "space_rope": args.space_rope, "qk_norm": args.qk_norm, "action_norm": args.action_norm,
                       "change_weight": args.change_weight, "motion": args.motion,
                       "motion_first": args.motion_first, "local_attention": args.local_attention}
        self.model = build(self.config, self.config["recompute"]).cuda()
        self.path = self.out / "model_latest.pt"
        saved = torch.load(self.path, map_location="cuda", weights_only=False) if self.path.is_file() else None
        source = None
        if saved is None and args.grow_from:              # a new run grown from a trained one (deepen)
            source = torch.load(args.grow_from, map_location="cuda", weights_only=False)
            stack = self.config.get("stack")              # stacked: the trained weights with their optimizer
            deepen(self.model, source["model" if stack else "ema"], self.config.get("new_at"),
                   self.config.get("copy", False), stack)
            print(f"{name}: grown from {args.grow_from} (step {source['step']})", flush=True)
        self.forward = torch.compile(self.model)          # one compiled graph per model (about a minute each)
        self.read_choice = torch.compile(self.model.read_choice) if self.model.latent else None
        trained = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        decay = [p for n, p in trained if p.ndim >= 2 and "pos" not in n and "colour" not in n
                 and "rope_freq" not in n]
        rest = [p for n, p in trained if not any(p is q for q in decay)]
        self.names = ([n for n, p in trained if any(p is q for q in decay)]
                      + [n for n, p in trained if not any(p is q for q in decay)])     # the optimizer's order
        self.optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                                            {"params": rest, "weight_decay": 0.0}],
                                           lr=args.lr, betas=(0.9, 0.95), fused=True)
        self.step = 0
        if saved is not None:                             # resume this run
            self.model.load_state_dict(saved["model"])
            load_optimizer(self.optimizer, saved["optimizer"], self.names)
            self.step = int(saved["step"])
        elif source is not None and self.config.get("stack"):   # stacked: each copy takes its source block's
            old = len(self.model.blocks) // self.config["stack"]  # AdamW state (Growing to See, 2610.16817), and
            load_optimizer(self.optimizer, source["optimizer"], self.names,   # the schedule carries on, no
                           source_of=lambda n: stacked_source(n, old))        # warmup again (Bu, 2511.04981)
            self.step = int(source["step"])
        self.live = list(self.model.state_dict().values())
        self.ema = ([saved["ema"][k].to("cuda", torch.float32) for k in self.model.state_dict()]
                    if saved is not None else [v.detach().clone() for v in self.live])
        self.start_step = self.saved_step = self.step
        self.parameters = sum(p.numel() for p in self.model.parameters())
        self.losses = None
        self.averaged = None                              # an eval copy holding the averaged weights
        self.drawer = None                                # the frozen copy that draws rollouts (--refresh)
        self.live_sums: dict[str, float] = {}             # live_event_scores' sums since the last checkpoint
        self.critic_weight = float(self.config.get("critic", 0.0))
        self.critic = self.critic_optimizer = None
        if self.critic_weight:                            # the turn critic (models/turn_critic.py), trained alongside
            self.critic = TurnCritic(COLOURS).cuda()
            self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=1e-4, betas=(0.5, 0.9),
                                                      weight_decay=0.0)
            if saved is not None and "critic" in saved:
                self.critic.load_state_dict(saved["critic"])
                self.critic_optimizer.load_state_dict(saved["critic_optimizer"])

    def archive(self, folder: Path) -> None:
        """Keep this step's checkpoint as folder/<run>/model_step<step>.pt, next to its long-dream test, so
        a run that gets good and then worse can go back. A missing drive is reported, never fatal."""
        target = folder / self.out.name / f"model_step{self.step}.pt"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".pt.tmp")
            shutil.copyfile(self.path, temporary)
            replace(temporary, target)
        except OSError as exc:
            print(f"{self.name}: could not archive step {self.step} to {target}: {exc}", flush=True)

    def averaged_model(self) -> Dynamics:
        """The averaged weights (EMA) in an eval copy of the model, as play and the checks load them."""
        if self.averaged is None:
            self.averaged = build(self.config).cuda().eval()
        self.averaged.load_state_dict(dict(zip(self.model.state_dict(), self.ema)))
        return self.averaged

    def long_dream_test(self, sets: dict[str, list], palette: np.ndarray,
                        bank: PatchBank) -> dict[str, dict[str, float]]:
        """Dream each variant's trials (no buttons, the bot's buttons); append long_dream.csv, write
        long_dream.gif and .png (each variant's first trial, real next to the dream, one above the other) ->
        {variant: mean scores}. Scored against the real game (long_dream.py) and, generically, against real
        play's patches and changes (bank, diagnostics/coherence.py)."""
        model = self.averaged_model()
        means, gifs, sheets = {}, [], []
        for variant, trials in sets.items():
            rows, likely = [], []
            for i, (trial, dreamed) in enumerate(zip(trials, dreams(model, trials, palette, pictures=1))):
                rows.append({"step": self.step, "variant": variant, "level": trial.level,
                             **score(trial, dreamed),
                             **coherence(bank, trial.context[-1], dreamed.likely, dreamed.sure)})
                likely.append(dreamed.likely)
                if i == 0:
                    gifs.append(animation(trial, {variant: dreamed}, palette))
                    sheets.append(dream_sheet(trial, {variant: dreamed}, palette))
            bias.write(self.out, self.step, variant, [np.asarray(t.future) for t in trials], likely,
                       [np.asarray(t.context) for t in trials])
            append_long_dream(self.out, rows)
            means[variant] = {k: float(np.nanmean([r[k] for r in rows]))
                              for k in rows[0] if k not in ("step", "variant", "level")}
            gc.collect()                                  # the dreams' CUDA graphs (see train)
            torch.cuda.empty_cache()
        save_gif(stacked(*gifs), self.out / "long_dream.gif", 60)
        temporary = self.out / "long_dream.tmp.png"
        stacked(*sheets).save(temporary)
        replace(temporary, self.out / "long_dream.png")
        return means

    def train_step(self, index, actions, args, motion: torch.Tensor | None = None,
                   branch_at: torch.Tensor | None = None, piece: torch.Tensor | None = None,
                   events: torch.Tensor | None = None) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate(self.step, args)
        depth = rollout_depth(self.step, args)
        if depth and (self.drawer is None or self.step % args.refresh == 0):
            if self.drawer is None:                       # rollouts drawn by weights that hold still for
                self.drawer = build(self.config).cuda().eval()   # --refresh steps (HorizonDrive's cache)
            self.drawer.load_state_dict(self.model.state_dict())  # in place: its CUDA graph stays valid
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            self.losses = train_step(self.model, self.forward, index, actions, depth, args.blend or 0,
                                     drawer=self.drawer, read_choice=self.read_choice, resync=args.resync,
                                     hidden=args.hidden, press_weight=args.press_weight, motion=motion,
                                     hide_presses=args.hide_presses, branch_at=branch_at, critic=self.critic,
                                     critic_weight=self.critic_weight, piece=piece)
        self.losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        if self.critic is not None and self.losses["critic_loss"].requires_grad:   # then the critic, on the
            self.critic_optimizer.zero_grad(set_to_none=True)                       # model's frames detached
            self.losses["critic_loss"].backward()
            self.critic_optimizer.step()
        with torch.no_grad():
            torch._foreach_lerp_(self.ema, self.live, 1 - EMA_DECAY)
        if events is not None and self.step % LIVE_EVERY == 0:   # the batch's own event scores, unseen data
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                scores = live_event_scores(self.model, self.forward, index, incoming_actions(actions), events,
                                           motion, piece)
            for k, v in scores.items():
                self.live_sums[k] = self.live_sums.get(k, 0.0) + v
        self.step += 1

    def checkpoint(self, index, actions, batches, stats, started, writer, args) -> dict:
        """batches.motion: the batch's motion codes (a motion model's context in the previews)."""
        losses, step = self.losses, self.step
        if not torch.isfinite(losses["total"]):          # checked here, not every step: it waits for the GPU
            raise RuntimeError(f"{self.name}: nonfinite loss at step {step}; last checkpoint is {self.saved_step}")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m = preview(self.model, index, actions, batches.palettes, self.out, args, batches.motion)
        m.update(stats)
        m.update({"seconds": time.monotonic() - started, "lr": self.optimizer.param_groups[0]["lr"],
                  "train_loss": losses["total"].item(), "train_loss_game": losses["game"].item(),
                  "train_loss_border": losses["border"].item(),
                  "train_loss_changed": losses["changed"].item(),
                  "masked_fraction": losses["masked"].item(),
                  "rollout_depth": rollout_depth(step - 1, args), "blend_frames": losses["blend"].item(),
                  "latent_kl": losses["latent_kl"].item(),
                  "rollout_frames": losses["rollout_frames"].item(),
                  "rollout_wrong": losses["rollout_wrong"].item(),
                  "rollout_ghost_px": losses["rollout_ghost_px"].item(),
                  "rollout_resynced": losses["rollout_resynced"].item(),
                  "train_loss_motion": losses["motion"].item(), "train_motion_moved": losses["motion_moved"].item(),
                  "train_motion_turned": losses["motion_turned"].item(),
                  "critic_loss": losses["critic_loss"].item(), "critic_g": losses["critic_g"].item(),
                  "critic_direction": losses["critic_direction"].item(),
                  "critic_turns": losses["critic_turns"].item(), **live_means(self.live_sums)})
        self.live_sums = {}
        writer.writerow({"step": step, **m})
        temporary = self.out / "model_latest.pt.tmp"
        torch.save({"step": step, "model": self.model.state_dict(), "optimizer": optimizer_state(self.optimizer, self.names),
                    "ema": dict(zip(self.model.state_dict(), self.ema)),
                    "args": {**vars(args), **self.config}, "palettes": batches.palettes, "metrics": m,
                    **({"critic": self.critic.state_dict(), "critic_optimizer": self.critic_optimizer.state_dict()}
                       if self.critic is not None else {})}, temporary)
        replace(temporary, self.path)
        self.saved_step = step
        print(f"{self.name} step={step} loss={m['train_loss']:.4f} (changed {m['train_loss_changed']:.3f}) "
              "changed game pixels wrong h1/4/16 "
              + " ".join(f"{g} " + "/".join(f"{m[f'{g}_h{h}_wrong_changed']:.3f}" for h in (1, 4, 16))
                         for g in GAMES)
              + (f" motion {m['train_loss_motion']:.4f}, right on moved / turned cells: masked "
                 f"{m['train_motion_moved']:.2f}/{m['train_motion_turned']:.2f}, previews "
                 f"{m['preview_motion_moved']:.2f}/{m['preview_motion_turned']:.2f}" if self.model.motion else "")
              + f" steps/s={(step - self.start_step) / max(m['seconds'], 1):.2f}", flush=True)
        return m


def rollout_depth(step: int, args) -> int:
    """--rollouts DEEPEST SHALLOWEST STEPS -> the deepest rollout at this step of the run (a new run,
    --grow-from the base, so its steps count from its start): linear from DEEPEST to SHALLOWEST over STEPS
    steps, then stays (HorizonDrive's boundary-decay curriculum: 10 -> 4 chunks over 8k of 10k steps, the
    deepest drift first); 0 without --rollouts (the base model: teacher forcing)."""
    if not args.rollouts:
        return 0
    deepest, shallowest, steps = args.rollouts
    return round(deepest + (shallowest - deepest) * min(1.0, step / max(steps, 1)))



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
    """Long-dream trials for training: one low-priority DataLoader worker plays real games (from the run's
    seed) to a fresh piece and records the future, with no buttons or (pressed) the bot playing on
    (diagnostics/long_dream.py); layered: with the context's layers. The first LONG_DREAM_TRIALS are kept and
    every test dreams those: a paired comparison across steps and runs."""

    def __init__(self, seed: int, layered: bool = False, pressed: bool = False):
        self.ready: queue.Queue = queue.Queue(maxsize=LONG_DREAM_TRIALS)
        loader = DataLoader(TrialStream(seed, layered, pressed), batch_size=None, num_workers=1,
                            persistent_workers=True, worker_init_fn=background_priority, prefetch_factor=2)
        threading.Thread(target=self._fill, args=(iter(loader),), daemon=True, name="long-dream-trials").start()

    def _fill(self, trials) -> None:
        for n, trial in enumerate(trials, 1):        # the kept ones only: a trial holds 1,000 future frames
            self.ready.put(trial)
            if n == LONG_DREAM_TRIALS:
                break

    def take(self, n: int) -> list:
        if not hasattr(self, "kept"):
            self.kept = [self.ready.get() for _ in range(n)]
        return self.kept


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
                          "(data/tetris_bot.py) with pauses, soft resets and short waits (world_nes_tetris.WORLD_MODEL), every window; exact palette pixels; no archive or eval set; "
                          "the same batches as every model in trained_with",
            }, default=str, indent=2))
        events.put(("ready", {m.name: m.parameters for m in members}, histories))
        print("World models: " + ", ".join(f"{m.name} {m.parameters:,} params (resume {m.step})" for m in members)
              + f"; {histories} histories ({args.workers_per_game} per game: {', '.join(GAMES)}) x "
              f"{args.frames}-frame windows, shared", flush=True)
        files = {m.name: open_metrics(m.out / "metrics.csv", FIELDS) for m in members}
        # the trials depend on --seed alone (the stream seed of a launch at step 0), so a run's tests stay the
        # same games after a relaunch, and runs with the same --seed are tested on the same games
        trial_seed = int(np.random.SeedSequence([args.seed, *[0] * len(members)]).generate_state(1)[0])
        long_dreams = {LONG_DREAM_VARIANT: LongDreams(trial_seed % 100_000),     # the same games: pressed
                       PRESSED_VARIANT: LongDreams(trial_seed % 100_000, pressed=True)}   # ones play on
        palette = batches.palettes["tetris"].cpu().numpy().astype(np.uint8)
        # real play's patches and changes for the coherence scores: every history's window every BANK_EVERY
        # steps (windows 256 frames apart, about 128k frames by the first test), hashed on the GPU; one wait
        # for the GPU per BANK_EVERY steps, not one a step; kept with the run
        bank_path = members[0].out / "patch_bank.npz"
        bank = PatchBank.load(bank_path) if bank_path.is_file() else PatchBank()
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
                if members[0].step % BANK_EVERY == 0:
                    for window in index:
                        bank.add(window)
                for member in members:
                    member.train_step(index, actions, args, batches.motion, batches.branch_at, batches.piece,
                                      batches.events)
                due = [m for m in members if m.step == m.start_step + 1 or m.step % args.preview_every == 0]
                if due:
                    stats = batches.take_stats()          # data waits since the last checkpoint, shared
                    for member in due:
                        checkpoint(member)
                if members[0].step % args.long_dream_every == 0:
                    trials = {v: d.take(LONG_DREAM_TRIALS) for v, d in long_dreams.items()}  # same for every model
                    bank.save(bank_path)
                    for member in members:
                        if member.saved_step != member.step:
                            checkpoint(member)
                        member.archive(args.archive)
                        summary = member.long_dream_test(trials, palette, bank)
                        # each dream captured its own CUDA graph (Dreamer); give their memory back, or
                        # training allocates on top of it until the GPU spills into system memory
                        # (32 trials filled the 24 GB card and slowed training ~10x, 2026-10-01)
                        gc.collect()
                        torch.cuda.empty_cache()
                        events.put(("longdream", member.name, member.step, summary))
                        print(f"{member.name} step={member.step} long dream: " + "; ".join(
                            f"{v}: wrong +16/+128 {r['wrong_16']:.2%}/{r['wrong_128']:.2%} mass {r['mass']:.2f} "
                            f"piece +32 {r['piece_32']:.2f} presence {r['presence']:.2f} activation "
                            f"{r['activation']:.2f} hit {r['piece_hit_32']:.2f} ghost "
                            f"{r['piece_ghost_32']:.2f} fall {r['fall']:.2f} spawn {r['spawn']:.2f} "
                            f"timer wrong +16/+128 {r['timer_wrong_16']:.2%}/{r['timer_wrong_128']:.2%} "
                            f"unseen patch/change {r['unseen_patch']:.2%}/{r['unseen_change']:.2%} "
                            f"change {r['change_px']:.2%} unsure {r['unsure_px']:.2%} "
                            f"per 100 of 1,000 frames: lost {r['lost_long']:.2f} morph {r['morph_long']:.2f} "
                            f"full {r['full_long']:.2f} size {r['size_long']:.2f}"
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
    p.add_argument("--rollouts", type=int, nargs=3, metavar=("DEEPEST", "SHALLOWEST", "STEPS"),
                   help="scheduled rollout recovery: every window a rollout of 1..depth of the model's own frames, "
                        "depth falling from DEEPEST to SHALLOWEST over STEPS steps (HorizonDrive); without it, "
                        "teacher forcing (the base model)")
    p.add_argument("--refresh", type=int, default=2000,
                   help="with --rollouts: the rollouts are drawn by a frozen copy of the model, refreshed with its "
                        "weights every this many steps (HorizonDrive's rollout cache, R = 2,000)")
    p.add_argument("--blend", type=int, metavar="MAX",
                   help="with --rollouts: the history's last w frames fade from the model's own to the real ones, "
                        "w drawn uniform in 0..MAX each step; the scored frames are always real "
                        "(HorizonDrive's blend, history side; models/dynamics.py own_share)")
    p.add_argument("--resync", type=int, metavar="PIXELS",
                   help="with --rollouts: after each rollout frame, every patch with at least PIXELS pixels whose "
                        "most likely colour is not the real one is replaced by the real patch before the next frame "
                        "reads it: the history stays in sync with the real game and its buttons, with only the "
                        "model's small errors (models/dynamics.py FrameDecoder)")
    p.add_argument("--hide-presses", action="store_true",
                   help="the frames an A, B, Left or Right press reaches are hidden entirely whenever predicted "
                        "(pressed_frames): a turn learned from the button, as a dream draws it")
    p.add_argument("--branches", type=int, default=0, metavar="N",
                   help="N workers' windows each come with a counterfactual twin, replayed from the emulator at "
                        "a turn press with the other turn or none (world_nes_tetris.TurnBranches); the batch "
                        "is the workers' windows plus the N twins")
    p.add_argument("--hidden", type=float, default=0.0, metavar="SHARE",
                   help="the share of masked frames hidden entirely in training, as a dream draws every frame "
                        "(models/dynamics.py training_mask): the model cannot read a change off the frame itself")
    p.add_argument("--change-weight", type=float, default=1.0, metavar="W",
                   help="the loss weight on game pixels that change from the frame before (models/dynamics.py "
                        "Dynamics.loss_weights); decoding undoes it (FrameDecoder). A resumed run must give the same")
    p.add_argument("--motion", action="store_true",
                   help="motion in and out (VideoJAM; models/dynamics.py motion_loss, data/tetris_motion.py): each "
                        "frame's motion goes in with its visible tokens and is predicted for the masked ones")
    p.add_argument("--local-attention", action="store_true",
                   help="the spatial attention gets a learned bias per token offset, from zero (Swin's relative "
                        "position bias; models/dynamics.py SpatialAttention)")
    p.add_argument("--motion-first", action="store_true",
                   help="with --motion: a generated frame's motion is drawn first and its pixels given it (two "
                        "passes); training gives a share of masked frames their real motion")
    p.add_argument("--press-weight", type=float, default=1.0, metavar="W",
                   help="the loss weight on whole frames an A, B, Left or Right press reaches (press_weights); "
                        "training only, decoding is unchanged")
    p.add_argument("--grow-from", type=Path, default=None,
                   help="a checkpoint a new run starts from (the base, for --rollouts; deepen); ignored on resume")
    p.add_argument("--models", nargs="+", choices=list(MODELS), default=["base"],
                   help="models trained side by side on the same batches (MODELS)")
    p.add_argument("--out", type=Path, default=None,
                   help="the run folder, for one model (default: world_model_tetris_<name> in --output-root)")
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
    p.add_argument("--space-rope", action="store_true",
                   help="rotary in the spatial attention by each patch's place, instead of a learned place "
                        "(models/dynamics.py; a resumed run must give the same)")
    p.add_argument("--qk-norm", action="store_true", help="QK-norm in both attentions (models/dynamics.py)")
    p.add_argument("--action-norm", action="store_true",
                   help="the buttons condition every block's norms, adaptive LayerNorm (models/dynamics.py Block)")
    p.add_argument("--colour-dim", type=int, default=16, help="per-pixel colour embedding size")
    p.add_argument("--patch", type=int, default=16, help="pixels per transformer token side")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=1000, help="linear learning-rate warmup steps")
    p.add_argument("--cooldown", type=int, nargs=2, metavar=("START", "LENGTH"), default=None,
                   help="decay the learning rate linearly to zero from step START over LENGTH steps, then stop")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--tetris-repo", type=str, default=None)
    p.add_argument("--close-when-done", action="store_true",
                   help="close the window after training finishes or fails (unattended runs)")
    args = p.parse_args()
    if args.blend and not args.rollouts:
        p.error("--blend blends rollouts: it needs --rollouts")
    if args.cooldown and not args.steps:
        args.steps = sum(args.cooldown)
    args.failed, args.games = False, GAMES
    if args.out and len(args.models) != 1:
        p.error("--out names one model's run folder")
    args.outputs = {name: args.out or args.output_root / f"world_model_tetris_{name}" for name in args.models}
    args.shapes = {name: MODELS[name] for name in args.models}
    from token_world.ui.dynamics_app import launch
    launch(args, train)
    sys.exit(1 if args.failed else 0)


if __name__ == "__main__":
    main()
