"""Everything a training run writes for its window (ui/run_viewer.py), computed from frames alone.

The window reads a run folder: metrics.csv (and metrics_schema_*.csv), horizon_curve.csv, previews/*.gif,
rollout_<game>.png, long_dream.*. Nothing here knows which model made the frames: a trainer generates the
last max(HORIZONS) frames of some real windows from the frames before them, in the model frames' index
space (data/model_frames.py), and hands real and generated frames over. So every model (the pixel model,
scripts/train_dynamics_ui.py; the layered model, scripts/train_layered.py) gets every chart and preview.
Both trainers also save their optimizer state here (optimizer_state, load_optimizer): by parameter name, so a
resume survives the model's code registering its parameters in another order.
"""
from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from token_world.data.model_frames import BAND, GAME_ROWS, SIZE
from token_world.diagnostics.motion_pictures import motion_overlay

HORIZONS = (1, 2, 4, 8, 16)
KINDS = ("wrong", "changed", "wrong_changed", "wrong_static", "border_wrong", "border_copy")
PREVIEW_GIFS = 4                                  # preview windows saved as animated GIFs


def horizon_fields(games) -> list[str]:
    """metrics.csv's preview columns: <game>_h<h>_<kind>."""
    return [f"{g}_h{h}_{kind}" for g in games for h in HORIZONS for kind in KINDS]


def replace(temporary: Path, path: Path) -> None:
    """temporary -> path, retried for a few seconds: on Windows a viewer reading `path` blocks it briefly.

    The data goes to disk first: a rename can reach the disk before the file's contents, and a crash then leaves
    `path` all zeros (the 128x2 run's model_latest.pt, 2026-10-11)."""
    with open(temporary, "rb+") as f:
        os.fsync(f.fileno())
    for _ in range(50):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            time.sleep(0.1)
    temporary.replace(path)


def gif_palette(colours) -> Image.Image:
    """The one palette every frame of a GIF is drawn in: the game's colours exactly ([N, 3], its frame palette),
    the window's (panel background, labels, the WRONG panel's magenta), then a 5 x 6 x 5 cube for the generated
    frames' blends. Left to itself PIL gives each frame its own 256 colours, and in a frame full of blends a
    small sprite's colours lose their slots: small Mario came out one orange in the REAL panel (2026-10-31)."""
    exact = np.asarray(colours, np.uint8).reshape(-1, 3)
    ui = np.array([[0x18, 0x18, 0x20], [255, 255, 255], [255, 0, 255]], np.uint8)
    cube = np.stack(np.meshgrid(*(np.linspace(0, 255, n).round() for n in (5, 6, 5)), indexing="ij"), -1)
    table = np.concatenate([exact, ui, cube.reshape(-1, 3).astype(np.uint8)])
    if len(table) > 256:
        raise ValueError(f"{len(exact)} exact colours leave no room for the blends in a 256-colour GIF")
    image = Image.new("P", (1, 1))
    image.putpalette(np.concatenate([table, np.zeros((256 - len(table), 3), np.uint8)]).flatten().tolist())
    return image


def save_gif(frames: list, path: Path, ms: int, colours) -> None:
    """RGB frames -> an animated GIF in one fixed palette (gif_palette: the game's colours exact)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    palette = gif_palette(colours)
    frames = [f.convert("RGB").quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    temporary = path.with_name(path.stem + ".tmp.gif")
    frames[0].save(temporary, save_all=True, append_images=frames[1:], duration=ms, loop=0, optimize=False)
    replace(temporary, path)


def open_metrics(path: Path, fields: list[str]):
    """metrics.csv for appending -> (file, DictWriter). A file with other columns is kept as
    metrics_schema_<time>.csv (the window reads those too) and a new one started."""
    if path.exists():
        with path.open(newline="") as f:
            header = next(csv.reader(f), [])
        if header != fields:                            # renamed after closing it: Windows refuses an open file
            path.rename(path.with_name(f"metrics_schema_{int(time.time())}.csv"))
    append = path.exists()
    f = path.open("a" if append else "w", newline="")
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    if not append:
        writer.writeheader()
    return f, writer


def append_long_dream(out: Path, rows: list[dict], name: str = "long_dream.csv") -> None:
    """Append long-dream rows to out/<name> (Tetris's long_dream.csv, SMB's smb_long_dream.csv); a file with other
    columns (an older score) is kept as <stem>_schema_<time>.csv (the window reads those too) and a new one
    started."""
    path = out / name
    if path.exists():
        with path.open(newline="") as file:
            header = next(csv.reader(file), [])
        if header != list(rows[0]):
            replace(path, path.with_name(f"{path.stem}_schema_{int(time.time())}.csv"))
    new = not path.exists()
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        if new:
            writer.writeheader()
        writer.writerows(rows)


def wrong_panel(reference: np.ndarray, wrong: np.ndarray) -> np.ndarray:
    """The real frame dimmed, magenta in proportion to each pixel's expected wrongness."""
    w = wrong.astype(np.float32)[..., None]
    return (reference * 0.35 * (1 - w) + np.array([255, 0, 200]) * w).astype(np.uint8)


def sprite_panel(ref: np.ndarray, truth: np.ndarray, said: np.ndarray) -> np.ndarray:
    """A real frame's colours [256, 256, 3] dimmed, with its sprite pixels as a model said them: green where
    it found a real one, red where it missed one, yellow where it said sprite and there is none."""
    out = (ref * 0.3).astype(np.uint8)
    out[truth & said] = (40, 220, 80)
    out[truth & ~said] = (230, 40, 40)
    out[~truth & said] = (240, 220, 40)
    return out


def save_previews(real, rgb, wrong, last_context, palette, folder: Path, game: str, sprites=None,
                  said=None, motion=None, drawn=None) -> None:
    """PREVIEW_GIFS windows, from the most changing to the quieter ones, as animated GIFs: the last
    context frame, then each generated frame, as REAL | GENERATED (expected colours) | WRONG (magenta), and
    with the real sprite masks and a model's (sprites, said [B, T, 256, 256] bool) SPRITES (sprite_panel), and a
    motion model's REAL MOTION (motion [B, T + 1, 32, 32]: the last context frame's, the motion that went in, then
    each real frame's) and DRAWN MOTION (drawn [B, T, 32, 32]: what it drew for its own frames, which went into
    its history with them), drawn by diagnostics/motion_pictures.py."""
    change = (real[:, -1] != last_context).float().mean((1, 2))
    order = change.argsort(descending=True).tolist()
    picks = list(dict.fromkeys(order[i * len(order) // PREVIEW_GIFS] for i in range(PREVIEW_GIFS)))
    palette = palette.cpu()
    colour = lambda frame: palette[frame.long().cpu()].numpy()
    for slot, w in enumerate(picks):
        frames = []
        for t in range(-1, real.shape[1]):
            ref = colour(last_context[w] if t < 0 else real[w, t])
            gen = ref if t < 0 else rgb[w, t].cpu().numpy()
            miss = wrong_panel(ref, np.zeros(ref.shape[:2]) if t < 0 else wrong[w, t].float().cpu().numpy())
            panels = [("REAL", ref), ("GENERATED", gen), ("WRONG", miss)]
            if sprites is not None and t >= 0:
                panels.append(("SPRITES", sprite_panel(ref, sprites[w, t].cpu().numpy(), said[w, t].cpu().numpy())))
            elif sprites is not None:
                panels.append(("SPRITES", (ref * 0.3).astype(np.uint8)))
            if motion is not None:
                panels.append(("REAL MOTION", motion_overlay(ref, motion[w, t + 1].cpu().numpy(), game=game)))
                panels.append(("DRAWN MOTION", motion_overlay(gen, drawn[w, t].cpu().numpy(), game=game) if t >= 0
                               else (gen * 0.3).astype(np.uint8)))
            image = Image.new("RGB", (len(panels) * (SIZE + 8) - 8, SIZE + 18), "#181820")
            draw = ImageDraw.Draw(image)
            for c, (title, panel) in enumerate(panels):
                image.paste(Image.fromarray(panel, "RGB"), (c * (SIZE + 8), 18))
                draw.text((c * (SIZE + 8) + 2, 3), f"{title} {'context' if t < 0 else f'+{t + 1}'}", fill="white")
            frames.append(image)
        save_gif(frames, folder / f"{game}_{slot}.gif", 160, palette.numpy())


def save_sheet(real, rgb, wrong, last_context, palette, path: Path) -> None:
    """Columns: last context frame, then each horizon. Rows: real, generated, wrong pixels."""
    show = int((real[:, -1] != last_context).float().mean((1, 2)).argmax())   # the most-changing history
    palette = palette.cpu()
    colour = lambda frame: palette[frame.long().cpu()].numpy()
    context = colour(last_context[show])
    columns = [("context t", context, context, np.zeros(context.shape[:2]))]
    columns += [(f"+{h}", colour(real[show, h - 1]), rgb[show, h - 1].cpu().numpy(),
                 wrong[show, h - 1].float().cpu().numpy()) for h in HORIZONS]
    label = 20
    sheet = Image.new("RGB", (len(columns) * SIZE, 3 * (SIZE + label)), "#181820")
    draw = ImageDraw.Draw(sheet)
    for c, (name, ref, gen, miss) in enumerate(columns):
        for r, (title, image) in enumerate((("REAL", ref), ("GENERATED", gen), ("WRONG", wrong_panel(ref, miss)))):
            y = r * (SIZE + label)
            draw.text((c * SIZE + 4, y + 4), f"{title} {name}", fill="white")
            sheet.paste(Image.fromarray(image, "RGB"), (c * SIZE, y + label))
    temporary = path.with_name(path.stem + ".tmp.png")
    sheet.save(temporary)
    replace(temporary, path)


def horizon_outputs(real: torch.Tensor, rgb: torch.Tensor, wrong: torch.Tensor, last_context: torch.Tensor,
                    palette: torch.Tensor, out: Path, game: str, motion: torch.Tensor | None = None,
                    drawn: torch.Tensor | None = None) -> dict[str, float]:
    """One game's generated frames against the real ones -> its <game>_h<h>_<kind> metrics, and its contact
    sheet (rollout_<game>.png) and preview GIFs (previews/<game>_<i>.gif).

    real [B, H, 256, 256] model indices of the last H = max(HORIZONS) frames of B windows; rgb [B, H, 256,
    256, 3] the generated frames' colours; wrong [B, H, 256, 256] each pixel's chance of being wrong (a
    model's expected wrongness, or 0/1 for committed frames); last_context [B, 256, 256] the frame before;
    a motion model's motion and drawn (save_previews)."""
    m = {}
    for h in HORIZONS:
        miss, ref, before = wrong[:, h - 1].float(), real[:, h - 1], last_context
        changed = (ref != before)[:, GAME_ROWS]
        mg = miss[:, GAME_ROWS]
        rate = lambda m_, c: (m_ * c).sum().item() / c.sum().item() if c.any() else float("nan")   # none: no data
        m.update({f"{game}_h{h}_wrong": mg.mean().item(),
                  f"{game}_h{h}_changed": changed.float().mean().item(),
                  f"{game}_h{h}_wrong_changed": rate(mg, changed),
                  f"{game}_h{h}_wrong_static": rate(mg, ~changed),
                  f"{game}_h{h}_border_wrong": torch.cat((miss[:, :BAND], miss[:, -BAND:]), 1).mean().item(),
                  f"{game}_h{h}_border_copy": torch.cat(((ref != before)[:, :BAND], (ref != before)[:, -BAND:]),
                                                        1).float().mean().item()})   # copying the last frame
    save_sheet(real, rgb, wrong, last_context, palette, out / f"rollout_{game}.png")
    save_previews(real, rgb, wrong, last_context, palette, out / "previews", game, motion=motion, drawn=drawn)
    return m


def write_horizon_curve(m: dict[str, float], games, out: Path) -> None:
    """horizon_curve.csv: every game's metrics by horizon (the window's Horizons tab)."""
    temporary = out / "horizon_curve.csv.tmp"
    with temporary.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["game", "horizon", *KINDS])
        for game in games:
            for h in HORIZONS:
                writer.writerow([game, h, *(m[f"{game}_h{h}_{kind}"] for kind in KINDS)])
    replace(temporary, out / "horizon_curve.csv")


class BatchedMuon(torch.optim.Muon):
    """torch.optim.Muon with each step's Newton-Schulz run on all same-shaped matrices at once (batched matmuls):
    torch's runs it one matrix at a time, ~70 matrices x 5 iterations x 3 matmuls and their norms (63 ms of a
    12-row step's 1.46 s, 2026-11-02). The same math, momentum and state (its "momentum_buffer", so a run's saved
    state loads either way): the update in bf16, each matrix wide (transposed when tall), divided by its norm,
    then the quintic iteration with the group's coefficients; the learning rate adjusted as torch's
    (_adjust_lr), the decoupled weight decay."""

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("BatchedMuon takes no closure")
        from torch.optim._muon import _adjust_lr
        for group in self.param_groups:
            params, grads, bufs = [], [], []
            self._init_group(group, params, grads, bufs)
            if not params:
                continue
            lr, momentum = float(group["lr"]), group["momentum"]
            torch._foreach_lerp_(bufs, grads, 1 - momentum)
            updates = torch._foreach_lerp(grads, bufs, momentum) if group["nesterov"] else bufs
            a, b, c = group["ns_coefficients"]
            shapes: dict[tuple, list[int]] = {}
            for i, p in enumerate(params):
                shapes.setdefault(tuple(p.shape), []).append(i)
            for shape, members in shapes.items():
                tall = shape[0] > shape[1]
                x = torch.stack([updates[i] for i in members]).bfloat16()
                x = x.transpose(1, 2) if tall else x
                x = x / x.norm(dim=(1, 2), keepdim=True).clamp(min=group["eps"])
                for _ in range(group["ns_steps"]):
                    gram = torch.bmm(x, x.transpose(1, 2))
                    x = torch.baddbmm(x, torch.baddbmm(gram, gram, gram, beta=b, alpha=c), x, beta=a)
                x = x.transpose(1, 2) if tall else x
                ps = [params[i] for i in members]
                torch._foreach_mul_(ps, 1 - lr * group["weight_decay"])
                torch._foreach_add_(ps, list(x.unbind(0)), alpha=-_adjust_lr(lr, group["adjust_lr_fn"], shape))


class MuonAdamW:
    """Muon on the network's hidden weight matrices and AdamW on the rest (embeddings, the first and last maps,
    biases, norms, positions), as Muon's authors and Moonshot's "Muon is Scalable for LLM Training" (2502.16982)
    train: torch.optim.Muon with its update scaled to AdamW's RMS (adjust_lr_fn "match_rms_adamw"), so both take
    the same learning rate and weight decay. param_groups lists both optimizers' groups (Muon's first), so a
    schedule sets them all; each part keeps its parameters' names for optimizer_state."""

    def __init__(self, muon: list[tuple[str, torch.Tensor]], adamw_groups: list[dict], lr: float,
                 weight_decay: float, betas=(0.9, 0.95)):
        if any(p.ndim != 2 for _, p in muon):
            raise ValueError("Muon takes weight matrices only")
        self.muon_names = [n for n, _ in muon]
        self.adamw_names = [n for g in adamw_groups for n in g["names"]]
        self.muon = BatchedMuon([p for _, p in muon], lr=lr, weight_decay=weight_decay,
                                adjust_lr_fn="match_rms_adamw")
        self.adamw = torch.optim.AdamW([{"params": g["params"], "weight_decay": g["weight_decay"]}
                                        for g in adamw_groups], lr=lr, betas=betas,
                                       fused=all(p.is_cuda for g in adamw_groups for p in g["params"]))

    @property
    def param_groups(self) -> list[dict]:
        return self.muon.param_groups + self.adamw.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self.muon.step()
        self.adamw.step()


def optimizer_state(optimizer, names: list[str] | None = None) -> dict:
    """The optimizer's state with each parameter's name (its order in the optimizer's groups): a resume matches
    the state to parameters by name, so the model's code may register them in another order. A MuonAdamW: each
    part's, under "muon" and "adamw", with the names it keeps."""
    if isinstance(optimizer, MuonAdamW):
        return {"muon": optimizer_state(optimizer.muon, optimizer.muon_names),
                "adamw": optimizer_state(optimizer.adamw, optimizer.adamw_names)}
    return {**optimizer.state_dict(), "names": names}


def load_optimizer(optimizer, saved: dict, names: list[str] | None = None, new: tuple[str, ...] = (),
                   source_of=None, from_adamw: bool = False) -> None:
    """optimizer_state's dict -> the optimizer, each parameter's state by name (the groups' settings, as the
    learning rate, as saved). A state saved without names is matched by position (it must have the same
    order: a mismatch is an error, not a silent swap). new: name prefixes of parameters the saved state may
    lack (a warm start that adds them): they start without state. source_of: a parameter's name -> the saved
    one whose state it takes (a stacked model: each copy takes its source block's, models/dynamics.py
    stacked_source). A MuonAdamW loads each part from its own (names: the part's). from_adamw: a MuonAdamW
    post-training a run AdamW trained alone (2026-11-01: s9 predates Muon): Muon's matrices start without state
    (its momentum fills in ~20 steps, and match_rms_adamw keeps its steps AdamW's size), its AdamW part takes the
    saved state of its own parameters."""
    if isinstance(optimizer, MuonAdamW):
        if "muon" not in saved and from_adamw:
            load_optimizer(optimizer.adamw, saved, optimizer.adamw_names, new, source_of)
            return
        if "muon" not in saved:
            raise ValueError("this optimizer state is AdamW's alone: a Muon run starts from a Muon run's state")
        load_optimizer(optimizer.muon, saved["muon"], optimizer.muon_names, new, source_of)
        load_optimizer(optimizer.adamw, saved["adamw"], optimizer.adamw_names, new, source_of)
        return
    if source_of is not None:
        if "names" not in saved:
            raise ValueError("copying optimizer state by name needs a state saved with names")
        old = {n: i for i, n in enumerate(saved["names"])}
        saved = {**saved, "names": names,
                 "state": {i: {k: v.clone() if torch.is_tensor(v) else v for k, v in saved["state"][old[source_of(n)]].items()}
                           for i, n in enumerate(names) if old.get(source_of(n)) in saved["state"]}}
    if "names" in saved:
        where = {n: i for i, n in enumerate(saved["names"])}
        missing = [n for n in names if n not in where and not (new and n.startswith(new))]
        if missing:
            raise ValueError(f"the saved optimizer has no state for {missing[:5]}")
        groups = optimizer.state_dict()["param_groups"]
        for group, old in zip(groups, saved["param_groups"]):
            group.update({k: v for k, v in old.items() if k != "params"})
        saved = {"state": {i: saved["state"][where[n]] for i, n in enumerate(names)
                           if n in where and where[n] in saved["state"]},
                 "param_groups": groups}
    params = [q for g in optimizer.param_groups for q in g["params"]]
    for i, state in saved["state"].items():
        held = state["exp_avg"] if "exp_avg" in state else state["momentum_buffer"]     # AdamW's or Muon's
        if held.shape != params[i].shape:
            raise ValueError(f"optimizer state {i} is {tuple(held.shape)}, its parameter "
                             f"{names[i]} {tuple(params[i].shape)}: saved in another parameter order")
    optimizer.load_state_dict(saved)
