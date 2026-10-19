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


def save_gif(frames: list, path: Path, ms: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.gif")
    frames[0].save(temporary, save_all=True, append_images=frames[1:], duration=ms, loop=0)
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


def append_long_dream(out: Path, rows: list[dict]) -> None:
    """Append long-dream rows to out/long_dream.csv; a file with other columns (an older score) is kept as
    long_dream_schema_<time>.csv (the window reads those too) and a new one started."""
    path = out / "long_dream.csv"
    if path.exists():
        with path.open(newline="") as file:
            header = next(csv.reader(file), [])
        if header != list(rows[0]):
            replace(path, path.with_name(f"long_dream_schema_{int(time.time())}.csv"))
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
                panels.append(("REAL MOTION", motion_overlay(ref, motion[w, t + 1].cpu().numpy())))
                panels.append(("DRAWN MOTION", motion_overlay(gen, drawn[w, t].cpu().numpy()) if t >= 0
                               else (gen * 0.3).astype(np.uint8)))
            image = Image.new("RGB", (len(panels) * (SIZE + 8) - 8, SIZE + 18), "#181820")
            draw = ImageDraw.Draw(image)
            for c, (title, panel) in enumerate(panels):
                image.paste(Image.fromarray(panel, "RGB"), (c * (SIZE + 8), 18))
                draw.text((c * (SIZE + 8) + 2, 3), f"{title} {'context' if t < 0 else f'+{t + 1}'}", fill="white")
            frames.append(image)
        save_gif(frames, folder / f"{game}_{slot}.gif", 160)


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


def optimizer_state(optimizer, names: list[str]) -> dict:
    """The optimizer's state with each parameter's name (its order in the optimizer's groups): a resume matches
    the state to parameters by name, so the model's code may register them in another order."""
    return {**optimizer.state_dict(), "names": names}


def load_optimizer(optimizer, saved: dict, names: list[str], new: tuple[str, ...] = (), source_of=None) -> None:
    """optimizer_state's dict -> the optimizer, each parameter's state by name (the groups' settings, as the
    learning rate, as saved). A state saved without names is matched by position (it must have the same
    order: a mismatch is an error, not a silent swap). new: name prefixes of parameters the saved state may
    lack (a warm start that adds them): they start without state. source_of: a parameter's name -> the saved
    one whose state it takes (a stacked model: each copy takes its source block's, models/dynamics.py
    stacked_source)."""
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
        if state["exp_avg"].shape != params[i].shape:
            raise ValueError(f"optimizer state {i} is {tuple(state['exp_avg'].shape)}, its parameter "
                             f"{names[i]} {tuple(params[i].shape)}: saved in another parameter order")
    optimizer.load_state_dict(saved)
