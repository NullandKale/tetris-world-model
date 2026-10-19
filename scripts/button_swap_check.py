"""Does each button decide its move? Live play, one frame ahead from 63 real frames (the incoming buttons, as
trained), the same history three times: the real press, the press swapped (A <-> B for a turn, Left <-> Right for
a shift), and no press; what the model draws in the well each time (2026-10-14).

    python scripts/button_swap_check.py D:/token_world_checkpoints/world_model_tetris_onestage/model_step20000.pt
    python scripts/button_swap_check.py <run or checkpoint> [--turns 200] [--shifts 100]

Instances: tap presses (no rotate or shift button the frame before) while a piece falls, where the RAM shows the
move happen and the other direction would also fit (in the well, off the stack). Turns are J, L and T only: the
pieces with four orientations, where a turn can go two ways (I, O, S and Z look the same either way). Each draw
is the real move, the opposite move, no move, or a mix (any other set of cells). A model that uses the button
draws the real move with the real press and the opposite one swapped; every pixel model up to 2026-10-14 drew
the same mix either way for turns (docs/guides/recipe_attempts.md) and followed Left / Right fully. The `gap`
is real-move share with the real press minus with the swap: about 0 when the button is ignored, 1 when it
decides. The p gap is finer, before any drawing flips: the mean probability of the real frame's colour on the
cells only one of the two moves fills, with the real press minus swapped (0 when the button is ignored). Each kind is also split by whether the button is still held on the frame that shows the move (the
frame the model draws gets the button that produced it: a tap is released by then, so the press sits on the
frame before, which looks unchanged). Appends a row per kind to the run folder's button_swap.csv (with the checkpoint's step).
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from token_world.data.nes_layers import TRANSPARENT, compose, model_frame
from token_world.data.tetris_bot import A, B, IDS_OF, LEFT, ORIENTATIONS, PIECE, RIGHT, TYPE_OF, X, Y, cells, settled_board
from token_world.data.tetris_events import FALLING, IN_GAME, MODE, STATE
from token_world.data.tetris_motion import stream_motion
from token_world.data.world_nes_tetris import WorldNesTetrisStreams
from token_world.diagnostics.line_clears import well_cells
from token_world.diagnostics.worlds import load
from token_world.models.dynamics import incoming_actions

T = 64                                   # the model's window: 63 real frames, then the one scored
WELL = (slice(56, 216), slice(96, 176))  # the well's 20 x 10 cells of 8 x 8 pixels in the model frame
ROTATE, SHIFT = A | B, LEFT | RIGHT
DRAWS = ("real move", "opposite move", "no move", "mix")


def mask(o: int, x: int, y: int) -> np.ndarray | None:
    """The piece's cells [20, 10], or None when one is outside the well."""
    m = np.zeros((20, 10), bool)
    for r, c in cells(o, x, y):
        if not (0 <= r < 20 and 0 <= c < 10):
            return None
        m[r, c] = True
    return m


def instance(ram: np.ndarray, act: np.ndarray, want: dict[str, int]):
    """A window whose last RAM frame (T - 2: the last picture shows it) has a tapped move of a kind still wanted
    -> (kind, press, before, real, opposite masks), else None."""
    e = T - 2
    cur, prev = ram[e], ram[e - 1]
    if cur[MODE] != IN_GAME or cur[STATE] != FALLING or prev[STATE] != FALLING:
        return None
    o0, o1 = int(prev[PIECE]), int(cur[PIECE])
    x0, x, y = int(prev[X]), int(cur[X]), int(cur[Y])
    if o0 not in ORIENTATIONS or o1 not in ORIENTATIONS or TYPE_OF[o0] != TYPE_OF[o1] or int(prev[Y]) != y or y < 2:
        return None
    press = int(act[e - 1])                    # lag 0: the press at e - 1 moves the piece in RAM e
    if act[e - 2] & (ROTATE | SHIFT):
        return None
    if o0 != o1 and x0 == x and want["turn"] and TYPE_OF[o1] in "JLT" and press & ROTATE in (A, B) \
            and not press & SHIFT:
        ids = IDS_OF[TYPE_OF[o1]]
        opposite = ids[(2 * ids.index(o0) - ids.index(o1)) % 4]
        kind, before, real, other = "turn", mask(o0, x, y), mask(o1, x, y), mask(opposite, x, y)
    elif o0 == o1 and abs(x - x0) == 1 and want["shift"] and press & SHIFT in (LEFT, RIGHT) and not press & ROTATE:
        kind, before, real, other = "shift", mask(o0, x0, y), mask(o0, x, y), mask(o0, 2 * x0 - x, y)
    else:
        return None
    if before is None or real is None or other is None or (other & settled_board(cur)).any():
        return None
    return kind, press, before, real, other


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path, help="a run folder (its model_latest.pt) or a checkpoint")
    p.add_argument("--turns", type=int, default=200)
    p.add_argument("--shifts", type=int, default=100)
    p.add_argument("--seed", type=int, default=9191)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda", help="cuda, or cpu while a training run holds the GPU (memory: one GPU "
                                                    "process at a time)")
    p.add_argument("--motion-steps", type=int, choices=(1, 2), default=2,
                   help="a motion-first model's motion decoding: 2 steps (confident cells first) or 1")
    p.add_argument("--threads", type=int, default=0, help="CPU threads for the model (0: torch's default)")
    args = p.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    world = load(args.run, args.device)
    world.model.motion_steps = args.motion_steps
    loader = DataLoader(WorldNesTetrisStreams(T, args.seed, stride=1, layered=world.layered), batch_size=None,
                        num_workers=args.workers, prefetch_factor=4)
    want = {"turn": args.turns, "shift": args.shifts}
    tally = Counter()
    sure = Counter()                                   # (kind, press) -> summed p(real) where the moves differ
    for part in loader:
        ram, act = part["ram"].numpy(), part["action"].numpy()
        found = instance(ram, act, want)
        if found is None:
            continue
        kind, press, before, real, other = found
        group = ROTATE if kind == "turn" else SHIFT
        layers = None
        if world.layered:
            layers = {k: np.asarray(v) for k, v in part["layers"].items()}
            x = np.stack([model_frame(compose({k: v[t] for k, v in layers.items()}), layers["border"][t])
                          for t in range(T)])
            frames = np.where(x == TRANSPARENT, 0, x).astype(np.uint8)[None]
            layers = {k: v[None] for k, v in layers.items()}
        else:
            frames = part["x"].numpy()[None]
        empty = int(np.bincount(frames[0, -1][WELL].ravel()).argmax())
        board = settled_board(ram[T - 2])
        motion = stream_motion(frames[0], ram)[0][None] if getattr(world.model, "motion", False) else None
        e = T - 2
        swapped = (press & ~group) | (group & ~press)
        for name, byte in (("real press", press), ("swapped", swapped), ("no press", press & ~group)):
            a = act.copy()
            a[e - 1] = byte
            if act[e] & group:                 # a held press: hold the substituted button
                a[e] = (act[e] & ~group) | (byte & group)
            made, right, _ = world.rollout(frames, incoming_actions(torch.from_numpy(a[None])).numpy(), T - 1,
                                           layers=layers, motion=motion)      # a motion model: the real context's
            p_real = right[0, 0].astype(np.float32)[WELL].reshape(20, 8, 10, 8).mean((1, 3))   # per cell
            differ = real != other                     # the cells only one of the two moves fills
            drawn = well_cells(made[0, 0], empty) & ~board
            draw = ("real move" if (drawn == real).all() else "opposite move" if (drawn == other).all()
                    else "no move" if (drawn == before).all() else "mix")
            tally[(kind, name, draw)] += 1
            held = "held" if act[e] & group else "tapped"        # still down on the frame that shows the move
            tally[(f"{kind}, {held}", name, draw)] += 1
            for k in (kind, f"{kind}, {held}"):
                sure[(k, name)] += float(p_real[differ].mean())
                sure[(k, "n " + name)] += 1
        want[kind] -= 1
        if not any(want.values()):
            break
    step = world.step
    out = (args.run if args.run.is_dir() else ROOT / "output" / args.run.parent.name) / "button_swap.csv"
    rows = []
    print(f"\n{world.name} @ {step:,}: one frame ahead from real history; what the model draws")
    for kind in ("turn", "shift", "turn, held", "turn, tapped", "shift, held", "shift, tapped"):
        share = {}
        for name in ("real press", "swapped", "no press"):
            row = {d: tally[(kind, name, d)] for d in DRAWS}
            n = sum(row.values())
            if not n:
                continue
            share[name] = {d: v / n for d, v in row.items()}
            print(f"{kind:>13} | {name:>10} [{n:3d}]: " + "  ".join(f"{d} {v / n:.0%}" for d, v in row.items()))
        if "real press" in share and "swapped" in share:
            gap = share["real press"]["real move"] - share["swapped"]["real move"]
            p_with, p_swap = (sure[(kind, n)] / max(sure[(kind, "n " + n)], 1) for n in ("real press", "swapped"))
            print(f"{kind:>13} | gap (real move, real press - swapped): {gap:+.2f};  p(real) on the cells the two "
                  f"moves disagree on: {p_with:.3f} real press, {p_swap:.3f} swapped ({p_with - p_swap:+.3f})")
            if "," in kind:
                continue
            rows.append({"step": step, "kind": kind, "gap": gap, "p_gap": p_with - p_swap,
                         **{f"real_{d.replace(' ', '_')}": v for d, v in share["real press"].items()},
                         **{f"swapped_{d.replace(' ', '_')}": v for d, v in share["swapped"].items()},
                         "no_press_no_move": share.get("no press", {}).get("no move", float("nan"))})
    if rows and out.parent.is_dir():
        old = list(csv.DictReader(out.open(newline=""))) if out.exists() else []
        fields = list(dict.fromkeys([*rows[0], *(old[0] if old else [])]))
        with out.open("w", newline="") as f:          # rewritten whole: rows from before a new column keep it blank
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(old + rows)
        print(f"appended to {out}")


if __name__ == "__main__":
    main()
