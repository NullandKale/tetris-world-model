"""Long dreams with no buttons: does each model keep a falling piece intact and let it fall, over 128 frames?

    python scripts/long_dream_check.py [run ...] [--keeps 48 63 ...]

Runs default to the long-trained ones (see play_world_model.py). Each keep is
how many frames a dream keeps when its window slides (models/dynamics.py
Dreamer; default three quarters of the window). Every run x keep dreams the
same trials. Dreams are soft, so scores are expectations.

The bot plays World NES until a fresh piece has spawned, then nobody presses
anything for 128 frames: the real game lets gravity work, and each model
dreams the same 128 frames from the last 48 real ones. Trials are spread over
start-level bands (gravity from one row per 48 frames at level 0 to one per 2
at 19). Per model it prints, averaged over trials: playfield pixels expected
wrong at +16/+32/+64/+128, the falling piece kept at +32, and the dream's
playfield block mass against the real one at +128 (1.0 = as many filled
pixels; pieces that melt or grow move it away).
Writes one strip image per trial (playfield every 8 frames) to --out.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from play_world_model import load_model, runs_with_checkpoints
from token_world.data.nes_palette import tetris_palette
from token_world.diagnostics.long_dream import HORIZONS, dream, score, sheet, trials


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", type=Path, nargs="*")
    p.add_argument("--per-band", type=int, default=2)
    p.add_argument("--keeps", type=int, nargs="+", default=[None], help="frames kept when a dream slides")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=ROOT / "output" / "long_dream" / time.strftime("%Y%m%d_%H%M%S"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runs = args.runs or runs_with_checkpoints()
    models = [(model, f"{name.split(' @')[0]} [keep {keep or 'default'}]", keep)
              for model, name in (load_model(r) for r in runs) for keep in args.keeps]
    palette = tetris_palette().cpu().numpy().astype(np.uint8)
    scores = {name: [] for _, name, _ in models}
    for n, trial in enumerate(trials(args.seed, args.per_band)):
        dreams = {name: dream(model, trial, palette, keep) for model, name, keep in models}
        for name, dreamed in dreams.items():
            scores[name].append(score(trial, dreamed))
        sheet(trial, dreams, palette).save(args.out / f"trial{n}_level{trial.level}.png")
        print(f"trial {n}: level {trial.level}", flush=True)
    print(f"\nplayfield pixels expected wrong at +{'/+'.join(map(str, HORIZONS))}, piece kept at +32, and "
          "block mass at +128 (1.0 = real)")
    for name, rows in scores.items():
        mass = [r["mass"] for r in rows]
        print(f"  {name:48s} " + " / ".join(f"{np.mean([r[f'wrong_{h}'] for r in rows]):6.2%}" for h in HORIZONS)
              + "   " + "  ".join(f"{k} {np.nanmean([r[k] for r in rows]):.2f}"
                                  for k in ("piece_32", "piece_hit_32", "piece_ghost_32", "fall", "spawn"))
              + f"   mass {np.mean(mass):.2f} (range {min(mass):.2f}-{max(mass):.2f})")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
