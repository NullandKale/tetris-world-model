"""Train a layered world model (models/layered.py: --model pixels or slots) on live layered Tetris
(data/nes_layers.py).

    powershell -ExecutionPolicy Bypass -File scripts\\run_until_stopped.ps1 -Run layered -TrainerArgs "..."
    python scripts/train_layered.py [--steps N] [--workers 16] ...

Every step: one window from each of --workers live histories (World NES with the layer regions, the
RAM-reading bot, scripts/train_dynamics_ui.py's window stride), MaskGIT masking over the layered tokens
(the camera never hidden), the layered losses (background and border pixels, sprite-picture pixels,
backdrop, the next frame's camera). Averaged weights (EMA) as the old trainer. Every --checkpoint-every steps
metrics.csv and model_latest.pt; every --long-dream-every steps the 32 long-dream trials (layered
contexts) dreamed and composed into model frames, scored as every run before (long_dream.csv: the Tetris
scores and the generic coherence scores), and the checkpoint archived. output/stop_training stops it
cleanly. This is the base stage (masked training); rollouts come next (docs/guides/layered_tokens.md).
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from token_world.data.live_batches import LiveBatches
from token_world.data.nes_layers import TRANSPARENT, compose, model_frame
from token_world.data.nes_palette import tetris_palette
from token_world.data.world_nes_tetris import WorldNesTetrisStreams
from token_world.diagnostics import bias
from token_world.diagnostics.coherence import PatchBank
from token_world.diagnostics.coherence import score as coherence
from token_world.diagnostics.long_dream import TrialStream, animation, dream_layered, score
from token_world.diagnostics.long_dream import sheet as dream_sheet
from token_world.diagnostics.run_outputs import (HORIZONS, append_long_dream, horizon_fields, horizon_outputs,
                                               open_metrics, replace, save_gif, write_horizon_curve)
from token_world.models.dynamics import incoming_actions, own_share
from token_world.models.layered import OPTIONS, build, rollout
from train_dynamics_ui import (BANK_EVERY, EMA_DECAY, LONG_DREAM_EVERY, LONG_DREAM_TRIALS, STOP_FILE, WINDOW_STRIDE,
                               LongDreams, background_priority, learning_rate, rollout_depth)

# every layered model's loss names (models/layered_pixels.py, layered_slots.py PARTS): one metrics schema for both
PARTS = ("cells", "sprites", "sprite_flags", "sprite_move", "sprite_place", "sprite_pixels", "backdrop", "camera")
FIELDS = ["step", "seconds", "lr", "train_loss", *(f"train_loss_{p}" for p in PARTS), "masked_fraction",
          "rollout_depth", "rollout_frames", "blend_frames",
          "data_wait_ms_mean", "data_wait_ms_max", "stream_tick_min", "stream_tick_max", "in_game_restarts",
          *horizon_fields(("tetris",))]


class LayeredLongDreams(LongDreams):
    """LongDreams with layered contexts (the same games for the same seed)."""

    def __init__(self, seed: int):
        import queue
        import threading
        self.ready = queue.Queue(maxsize=2 * LONG_DREAM_TRIALS)
        loader = DataLoader(TrialStream(seed, layered=True), batch_size=None, num_workers=1, persistent_workers=True,
                            worker_init_fn=background_priority, prefetch_factor=2)
        threading.Thread(target=self._fill, args=(iter(loader),), daemon=True, name="long-dream-trials").start()


def window_frames(layers: dict, i: int = 0) -> np.ndarray:
    """One window of a batch's layered frames (on the GPU) -> its composed model frames [T, 256, 256]."""
    w = {k: v[i].cpu().numpy() for k, v in layers.items()}
    return np.stack([model_frame(compose({k: v[t] for k, v in w.items()}), w["border"][t])
                     for t in range(len(w["camera"]))])


def composed(layers: dict, i: int, frames: range) -> np.ndarray:
    """Frames `frames` of window i of a batch's layered frames -> composed model frames [len, 256, 256]."""
    w = {k: v[i, frames.start:frames.stop].cpu().numpy() for k, v in layers.items()}
    out = np.stack([model_frame(compose({k: v[t] for k, v in w.items()}), w["border"][t]) for t in range(len(frames))])
    return np.where(out == TRANSPARENT, 0, out)


@torch.no_grad()
def preview(model, layers: dict, incoming: torch.Tensor, out: Path, temperature: float,
            steps: int = 1) -> dict[str, float]:
    """The window's previews, horizon curves and rollout sheet (diagnostics/run_outputs.py): each window's
    last max(HORIZONS) frames dreamed from the real frames before it with its real buttons, as in a long
    dream (committed; "wrong" is 0/1 per pixel, where the pixel model's is its expected wrongness)."""
    b, t = incoming.shape
    start = t - max(HORIZONS)
    model.eval()
    dreamer = model.Dreamer(model, {k: v[:, :start] for k, v in layers.items()}, incoming[:, :start],
                            temperature=temperature, generator=torch.Generator(device=incoming.device).manual_seed(0),
                            steps=steps)
    steps = [{k: v.cpu().numpy() for k, v in dreamer.step(incoming[:, at]).items() if k != "probs"}
             for at in range(start, t)]
    made = [np.stack([model_frame(compose({k: v[i] for k, v in f.items()}), f["border"][i]) for f in steps])
            for i in range(b)]
    made = [np.where(w == TRANSPARENT, 0, w) for w in made]
    model.train()
    real = torch.from_numpy(np.stack([composed(layers, i, range(start - 1, t)) for i in range(b)]))
    made = torch.from_numpy(np.stack(made))
    palette = tetris_palette()
    m = horizon_outputs(real[:, 1:], palette[made.long()], (made != real[:, 1:]).float(), real[:, 0], palette, out,
                        "tetris")
    write_horizon_curve(m, ("tetris",), out)
    return m


def long_dream_test(model, trials, palette, bank, out: Path, step: int, temperature: float, steps: int = 1) -> dict:
    rows, first, likely = [], None, []
    variant = f"layered t={temperature:g}" + (f", {steps} passes" if steps > 1 else "")
    for i, trial in enumerate(trials):
        dreamed = dream_layered(model, trial, palette, temperature, seed=i, steps=steps)
        rows.append({"step": step, "variant": variant, "level": trial.level,
                     **score(trial, dreamed), **coherence(bank, trial.context[-1], dreamed.likely, dreamed.sure)})
        likely.append(dreamed.likely)
        first = first or dreamed
    picks = bias.write(out, step, variant, [np.asarray(t.future) for t in trials], likely,
                       [np.asarray(t.context) for t in trials])
    append_long_dream(out, rows)
    save_gif(animation(trials[0], {"layered": first}, palette), out / "long_dream.gif", 60)
    temporary = out / "long_dream.tmp.png"
    dream_sheet(trials[0], {"layered": first}, palette).save(temporary)
    replace(temporary, out / "long_dream.png")
    return {**{k: float(np.nanmean([r[k] for r in rows])) for k in rows[0] if k not in ("step", "variant", "level")},
            **{f"bias_{k}": v for k, v in picks.items() if k not in ("step", "variant")}}


def train_step(model, forward, layers: dict, incoming: torch.Tensor, depth: int = 0, blend: int = 0,
               drawer=None, temperature: float = 1.0, steps: int = 1) -> dict:
    """One masked step on layered windows (each key [B, T, ...], incoming actions [B, T]) -> the losses
    ("total" has the graph) with "masked", "rollout_frames" and "blend".

    depth 0, stage A (the base): teacher forcing; every frame's tokens hidden at a rate drawn per frame and
    predicted from the real frames before (model.mask: half the windows with a clean past before a cut).
    depth > 0, stage B (scheduled rollout recovery, as train_dynamics_ui.train_step): from a real start s,
    `drawer` (a frozen copy, refreshed every --refresh steps) dreams k frames one after another (k uniform in
    1..depth), as a long dream does (models/layered.py rollout: its categories committed, its pixels as its
    tokens), and they go in as the history; the frames from s + k on are real and scored, every frame
    before visible. blend: the history's last w frames (w uniform in 0..blend) fade from its own to the real
    ones (own_share: the pixel tokens blended; the committed parts its own while its share is at least
    half). Movement is scored from where the history put a thing to where it really is (model.loss seen),
    so the targets lead back to the real game."""
    b, t = incoming.shape
    device = incoming.device
    seen, soft, cut, k, w = layers, None, None, 0, 0
    if depth:
        k = int(torch.randint(1, min(depth, t - 2) + 1, ()))
        s = int(torch.randint(1, t - k, ()))                                # s >= 1, the boundary s + k <= t - 1
        w = int(torch.randint(0, blend + 1, ()))
        boundary = s + k
        own, tokens = rollout(drawer, layers, incoming, s, boundary, temperature, steps=steps)
        part = own_share(s, boundary, w, device)                           # [k] the drawer's share
        mine = part >= 0.5
        seen = dict(layers)                                                  # the model's own fields replaced
        for key in own:
            v = layers[key]
            mixed = torch.where(mine.view(1, k, *[1] * (v.ndim - 2)), own[key].to(v.dtype), v[:, s:boundary])
            seen[key] = torch.cat((v[:, :s], mixed, v[:, boundary:]), 1)
        lo = max(s, boundary - w)                                            # the first frame with a real part
        if lo < boundary:
            real = model.pixel_tokens({key: v[:, lo:boundary] for key, v in layers.items()})
            share = part[lo - s:].view(1, -1, 1, 1)
            tokens = {key: torch.cat((z[:, :lo - s], share * z[:, lo - s:] + (1 - share) * real[key].to(z.dtype)),
                                     1)
                      for key, z in tokens.items()}
        where = torch.zeros(b, t, dtype=torch.bool, device=device)
        where[:, s:boundary] = True
        soft = {"where": where}
        for key, z in tokens.items():
            full = z.new_zeros(b, t, *z.shape[2:])
            full[:, s:boundary] = z
            soft[key] = full
        cut = torch.full((b,), boundary, dtype=torch.long, device=device)
    mask = model.mask(b, t, device, cut=cut)
    losses = model.loss(forward(seen, incoming, mask, soft=soft), layers, incoming, mask, seen=seen)
    losses.update({"masked": mask[:, 1:].float().mean(), "rollout_frames": torch.tensor(float(k)),
                   "blend": torch.tensor(float(w))})
    return losses


def optimizer_state(optimizer, names: list[str]) -> dict:
    """The optimizer's state with each parameter's name (its order in the optimizer's groups): a resume matches
    the state to parameters by name, so the model's code may register them in another order."""
    return {**optimizer.state_dict(), "names": names}


def load_optimizer(optimizer, saved: dict, names: list[str]) -> None:
    """optimizer_state's dict -> the optimizer, each parameter's state by name (the groups' settings, as the
    learning rate, as saved). A state saved without names is matched by position (it must have the same
    order: a mismatch is an error, not a silent swap)."""
    if "names" in saved:
        where = {n: i for i, n in enumerate(saved["names"])}
        missing = [n for n in names if n not in where]
        if missing:
            raise ValueError(f"the saved optimizer has no state for {missing[:5]}")
        groups = optimizer.state_dict()["param_groups"]
        for group, old in zip(groups, saved["param_groups"]):
            group.update({k: v for k, v in old.items() if k != "params"})
        saved = {"state": {i: saved["state"][where[n]] for i, n in enumerate(names) if where[n] in saved["state"]},
                 "param_groups": groups}
    params = [q for g in optimizer.param_groups for q in g["params"]]
    for i, state in saved["state"].items():
        if state["exp_avg"].shape != params[i].shape:
            raise ValueError(f"optimizer state {i} is {tuple(state['exp_avg'].shape)}, its parameter "
                             f"{names[i]} {tuple(params[i].shape)}: saved in another parameter order")
    optimizer.load_state_dict(saved)


def train(args, events: queue.Queue, stop: threading.Event) -> None:
    """The training loop, on the window's worker thread (ui/dynamics_app.py): Stop Training or
    output/stop_training stops it; events update the window's status line and refresh its tabs."""
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        torch.set_num_threads(2)                        # the work is on the GPU; leave the cores to the workers
        # PyTorch keeps freed blocks cached and the masked selections fragment them: without a cap it reserved
        # 19 GB for 8.6 GB in use (2026-10-06); with one it frees its cache before growing past it
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_gb * 2 ** 30 / total))

        config = {"dim": args.dim, "layers": args.layers, "heads": args.heads, "frames": args.frames,
                  "colour_dim": args.colour_dim, "kind": "layered", "model": args.model,
                  **{k: getattr(args, k) for k in OPTIONS}}
        model = build(config, recompute=True).cuda()          # activation checkpointing: memory for time
        forward = torch.compile(model)
        trained = [(n, q) for n, q in model.named_parameters() if q.requires_grad]
        decay = [q for n, q in trained if q.ndim >= 2 and not any(w in n for w in ("pos", "colour", "register",
                                                                                     "rope_freq"))]
        rest = [q for n, q in trained if not any(q is d for d in decay)]
        names = [n for n, q in trained if any(q is d for d in decay)] + [n for n, q in trained if not any(q is d for d in decay)]
        optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                                       {"params": rest, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95), fused=True)
        path = args.out / "model_latest.pt"
        saved = torch.load(path, map_location="cuda", weights_only=False) if path.is_file() else None
        step = 0
        if saved is None and args.grow_from:                  # a new run grown from a trained one (stage B)
            source = torch.load(args.grow_from, map_location="cuda", weights_only=False)
            model.load_state_dict({k: v.to(model.state_dict()[k].dtype) for k, v in source["ema"].items()})
            print(f"layered: grown from {args.grow_from} (step {source['step']})", flush=True)
        if saved is not None:
            model.load_state_dict(saved["model"])
            load_optimizer(optimizer, saved["optimizer"], names)
            step = int(saved["step"])
        live = list(model.state_dict().values())
        ema = ([saved["ema"][k].to("cuda", torch.float32) for k in model.state_dict()] if saved is not None
               else [v.detach().clone() for v in live])
        averaged = build(config).cuda().eval()
        parameters = sum(q.numel() for q in model.parameters())

        stream_seed = args.seed + step
        stream = WorldNesTetrisStreams(args.frames, stream_seed, stride=WINDOW_STRIDE, layered=True)
        batches = LiveBatches(DataLoader(stream, batch_size=None, num_workers=args.workers, pin_memory=True,
                                         persistent_workers=True, worker_init_fn=background_priority, prefetch_factor=2),
                              args.workers, stream.new_frames)
        trial_seed = int(np.random.SeedSequence([args.seed, 0]).generate_state(1)[0])
        trials = LayeredLongDreams(trial_seed % 100_000)
        palette = tetris_palette().numpy().astype(np.uint8)
        bank_path = args.out / "patch_bank.npz"
        bank = PatchBank.load(bank_path if bank_path.is_file() else args.bank) if (bank_path.is_file() or args.bank) \
            else PatchBank()
        (args.out / "run_config.json").write_text(json.dumps({**vars(args), **config, "parameters": parameters,
                                                              "start_step": step, "stream_seed": stream_seed},
                                                             default=str, indent=2))
        metrics_file, writer = open_metrics(args.out / "metrics.csv", FIELDS)
        events.put(("ready", {"layered": parameters}, args.workers))
        print(f"layered: {parameters:,} params (resume {step}); {args.workers} histories x {args.frames}-frame "
              f"windows", flush=True)

        started, start_step, losses, drawer = time.monotonic(), step, None, None
        try:
            while args.steps <= 0 or step < args.steps:
                if stop.is_set():
                    print("Stop Training: stopping", flush=True)
                    break
                if STOP_FILE.exists():
                    STOP_FILE.unlink()
                    print(f"{STOP_FILE} found: stopping", flush=True)
                    break
                layers, actions = batches.next()
                incoming = incoming_actions(actions)
                if step % BANK_EVERY == 0:
                    bank.add(window_frames(layers, (step // BANK_EVERY) % args.workers))
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate(step, args)
                optimizer.zero_grad(set_to_none=True)
                depth = rollout_depth(step, args)
                if depth and (drawer is None or step % args.refresh == 0):
                    if drawer is None:                           # rollouts drawn by weights that hold still for
                        drawer = build(config).cuda().eval()     # --refresh steps (HorizonDrive's cache)
                    drawer.load_state_dict(model.state_dict())
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    losses = train_step(model, forward, layers, incoming, depth, args.blend or 0, drawer,
                                        args.temperature, args.dream_steps)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                with torch.no_grad():
                    torch._foreach_lerp_(ema, live, 1 - EMA_DECAY)
                step += 1
                if step == start_step + 1 or step % args.checkpoint_every == 0:
                    if not torch.isfinite(losses["total"]):
                        raise RuntimeError(f"nonfinite loss at step {step}")
                    seconds = time.monotonic() - started
                    row = {"step": step, "seconds": seconds, "lr": optimizer.param_groups[0]["lr"],
                           "train_loss": losses["total"].item(), "masked_fraction": losses["masked"].item(),
                           "rollout_depth": rollout_depth(step - 1, args),
                           "rollout_frames": losses["rollout_frames"].item(), "blend_frames": losses["blend"].item(),
                           **{f"train_loss_{k}": losses[k].item() for k in model.PARTS}, **batches.take_stats()}
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        row.update(preview(model, layers, incoming, args.out, args.temperature, args.dream_steps))
                    torch.cuda.empty_cache()                     # the preview's one-off buffers
                    writer.writerow(row)
                    metrics_file.flush()
                    temporary = args.out / "model_latest.pt.tmp"
                    torch.save({"step": step, "model": model.state_dict(), "optimizer": optimizer_state(optimizer, names),
                                "ema": dict(zip(model.state_dict(), ema)), "args": {**vars(args), **config},
                                "palettes": {"tetris": tetris_palette()}}, temporary)
                    replace(temporary, path)
                    print(f"layered step={step} loss={row['train_loss']:.4f} " + " ".join(
                        f"{k} {row[f'train_loss_{k}']:.4f}" for k in model.PARTS) +
                          " changed pixels wrong h1/4/16 " + "/".join(f"{row[f'tetris_h{h}_wrong_changed']:.3f}" for h in (1, 4, 16)) +
                          f" steps/s={(step - start_step) / max(seconds, 1):.2f} GPU memory peak "
                          f"{torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB (reserved "
                          f"{torch.cuda.memory_reserved() / 2 ** 30:.1f})", flush=True)
                    events.put(("progress", "layered", step, f"loss {row['train_loss']:.4f}, " + ", ".join(
                        f"{k.replace('_', ' ')} {row[f'train_loss_{k}']:.4f}" for k in model.PARTS[:2])
                        + ", changed pixels wrong +1/+4/+16 " + " / ".join(f"{row[f'tetris_h{h}_wrong_changed']:.0%}" for h in (1, 4, 16))
                        + f", {(step - start_step) / max(seconds, 1):.2f} steps/s"))
                if step % args.long_dream_every == 0:
                    averaged.load_state_dict(dict(zip(model.state_dict(), ema)))
                    bank.save(bank_path)
                    dream_started = time.monotonic()
                    r = long_dream_test(averaged, trials.take(LONG_DREAM_TRIALS), palette, bank, args.out, step,
                                        args.temperature, args.dream_steps)
                    target = args.archive / args.out.name / f"model_step{step}.pt"
                    try:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(torch.load(path, map_location="cpu", weights_only=False), target)
                    except OSError as exc:
                        print(f"could not archive step {step}: {exc}", flush=True)
                    torch.cuda.empty_cache()
                    print(f"layered step={step} long dream: wrong +16/+128 {r['wrong_16']:.2%}/{r['wrong_128']:.2%} "
                          f"mass {r['mass']:.2f} presence {r['presence']:.2f} activation {r['activation']:.2f} "
                          f"hit {r['piece_hit_32']:.2f} ghost {r['piece_ghost_32']:.2f} "
                          f"fall {r['fall']:.2f} spawn {r['spawn']:.2f} timer wrong +16/+128 "
                          f"{r['timer_wrong_16']:.2%}/{r['timer_wrong_128']:.2%} unseen patch/change "
                          f"{r['unseen_patch']:.2%}/{r['unseen_change']:.2%} next pieces chosen "
                          f"{r['bias_choices_dream']:.0f} ({r['bias_types_dream']:.0f} types, top share "
                          f"{r['bias_top_share_dream']:.2f}; real {r['bias_choices_real']:.0f}, "
                          f"{r['bias_types_real']:.0f} types) mix distance {r['bias_type_distance']:.2f} "
                          f"garbled {r['bias_garbled']:.0%} ({time.monotonic() - dream_started:.0f} s)",
                          flush=True)
                    events.put(("longdream", "layered", step, {f"t={args.temperature:g}": r}))
                if args.seconds > 0 and time.monotonic() - started >= args.seconds:
                    break
        finally:
            metrics_file.close()
            if losses is not None and step != start_step:
                torch.save({"step": step, "model": model.state_dict(), "optimizer": optimizer_state(optimizer, names),
                            "ema": dict(zip(model.state_dict(), ema)), "args": {**vars(args), **config},
                            "palettes": {"tetris": tetris_palette()}}, path)
        events.put(("done", {"layered": step}))
    except BaseException as exc:
        import traceback
        traceback.print_exc()
        args.failed = True
        events.put(("error", str(exc)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=("pixels", "slots"), default="pixels",
                   help="the sprites as a picture (models/layered_pixels.py) or as the 64 OAM slots "
                        "(models/layered_slots.py); a resumed run keeps the model it was started with")
    p.add_argument("--registers", type=int, default=0,
                   help="learned tokens per frame with no target (models/layered.py; a resumed run keeps its own)")
    p.add_argument("--space-rope", action="store_true",
                   help="rotary in the spatial attention by screen place, RoPE-Mixed (models/layered.py)")
    p.add_argument("--fixed-camera", action="store_true",
                   help="a game whose camera never moves (Tetris): no camera tokens or camera head (models/layered.py)")
    p.add_argument("--share-pixels", action="store_true",
                   help="pixels: the sprite picture shares the cells' patch and pixel head (models/layered_pixels.py)")
    p.add_argument("--dream-steps", type=int, default=1,
                   help="slots: the passes over which a dream decides the sprites, the surest first (previews, "
                        "long dreams and rollouts)")
    p.add_argument("--out", type=Path, default=None,
                   help="the run folder (default output/world_model_tetris_layered_px, or _slots for --model slots)")
    p.add_argument("--archive", type=Path, default=Path("D:/token_world_checkpoints"))
    p.add_argument("--bank", type=Path, default=None, help="a patch_bank.npz to start the coherence bank from")
    p.add_argument("--dim", type=int, default=96)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--colour-dim", type=int, default=16)
    p.add_argument("--frames", type=int, default=64)
    p.add_argument("--workers", type=int, default=12, help="live histories: the batch")
    p.add_argument("--vram-gb", type=float, default=12.0,
                   help="the most GPU memory training may hold (the allocator frees its cache before more)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--cooldown", type=int, nargs=2, metavar=("START", "LENGTH"), default=None)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--temperature", type=float, default=1.0, help="long dreams' sampling temperature")
    p.add_argument("--checkpoint-every", type=int, default=200)
    p.add_argument("--long-dream-every", type=int, default=LONG_DREAM_EVERY)
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--seconds", type=int, default=0)
    p.add_argument("--rollouts", type=int, nargs=3, metavar=("DEEPEST", "SHALLOWEST", "STEPS"),
                   help="stage B, scheduled rollout recovery: every window a rollout of 1..depth of the model's own "
                        "frames, depth falling from DEEPEST to SHALLOWEST over STEPS steps; without it, stage A "
                        "(teacher forcing)")
    p.add_argument("--refresh", type=int, default=2000,
                   help="with --rollouts: the rollouts' drawer is a frozen copy, refreshed every this many steps")
    p.add_argument("--blend", type=int, metavar="MAX",
                   help="with --rollouts: the history's last w frames fade from its own to the real ones, w in 0..MAX")
    p.add_argument("--grow-from", type=Path, default=None,
                   help="a checkpoint a new run starts from (stage A's, for --rollouts); ignored on resume")
    p.add_argument("--compare", type=Path, nargs="*", default=[ROOT / "output" / "world_model_tetris_base"],
                   help="earlier runs drawn dashed beside this one in the window (by step)")
    p.add_argument("--close-when-done", action="store_true",
                   help="close the window after training finishes or fails (unattended runs)")
    args = p.parse_args()
    if args.blend and not args.rollouts:
        p.error("--blend blends rollouts: it needs --rollouts")
    if args.share_pixels and args.model != "pixels":
        p.error("--share-pixels shares the sprite picture's head: it needs --model pixels")
    if args.cooldown and not args.steps:
        args.steps = sum(args.cooldown)
    if args.out is None:
        args.out = ROOT / "output" / ("world_model_tetris_layered_px" if args.model == "pixels"
                                      else "world_model_tetris_layered_slots")
    latest = args.out / "model_latest.pt"
    if latest.is_file():                                     # a resumed run keeps its model and its options
        saved = torch.load(latest, map_location="cpu", weights_only=False)["args"]
        args.model = saved.get("model") or "pixels"
        for k in OPTIONS:
            setattr(args, k, saved.get(k) or (0 if k == "registers" else False))
    args.out.mkdir(parents=True, exist_ok=True)
    args.models, args.games, args.outputs, args.failed = ["layered"], ["tetris"], {"layered": args.out}, False
    from token_world.ui.dynamics_app import launch
    launch(args, train)
    sys.exit(1 if args.failed else 0)


if __name__ == "__main__":
    main()
