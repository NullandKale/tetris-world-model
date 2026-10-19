"""Long dreams, with no buttons and with the bot's buttons: does each model keep a falling piece intact, let it
fall and move it as the buttons say, over 128 frames?

    python scripts/long_dream_check.py [run or checkpoint ...] [--keeps 48 63 ...] [--device cpu]

Any model, pixel or layered (diagnostics/worlds.py); every model dreams the same trials (layered trials, so
the layered models have their layers; the pixel model reads the same frames).

Runs default to the long-trained ones (see play_world_model.py). Each keep is
how many frames a dream keeps when its window slides (models/dynamics.py
Dreamer; default three quarters of the window). Every run x keep dreams the
same trials. Dreams are soft, so scores are expectations.

The bot plays World NES until a fresh piece has spawned, then either nobody
presses anything for 1,000 frames (the real game lets gravity work) or the bot
plays on (its presses go to the real game and to every model alike); each model
dreams the same 1,000 frames from the last 48 real ones (diagnostics/long_dream.py:
most scores look at the first 128; lost, morph and full per 100 frames at all 1,000). --buttons picks which
(default both: a model that ignores the buttons can pass the no-button trials). Trials are spread over
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

from play_world_model import runs_with_checkpoints
from token_world.data.nes_palette import tetris_palette
from token_world.diagnostics.long_dream import HORIZONS, dream_world, score, sheet, trials
from token_world.diagnostics.worlds import load


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", type=Path, nargs="*")
    p.add_argument("--per-band", type=int, default=2)
    p.add_argument("--keeps", type=int, nargs="+", default=[None], help="frames kept when a dream slides")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda", help="cuda, or cpu while a training run holds the GPU")
    p.add_argument("--buttons", choices=("both", "none", "pressed"), default="both",
                   help="trials with no buttons, the bot's buttons, or both")
    p.add_argument("--out", type=Path, default=ROOT / "output" / "long_dream" / time.strftime("%Y%m%d_%H%M%S"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runs = args.runs or runs_with_checkpoints()
    worlds = [load(r, args.device) for r in runs]
    models = [(world, f"{world.name}@{world.step} [keep {keep or 'default'}]", keep)
              for world in worlds for keep in args.keeps]
    palette = tetris_palette().cpu().numpy().astype(np.uint8)
    kinds = {"none": [False], "pressed": [True], "both": [False, True]}[args.buttons]
    scores = {(pressed, name): [] for pressed in kinds for _, name, _ in models}
    for pressed in kinds:
        kind = "buttons" if pressed else "none"
        for n, trial in enumerate(trials(args.seed, args.per_band, layered=True, pressed=pressed)):
            dreams = {name: dream_world(world, trial, palette, keep, seed=n) for world, name, keep in models}
            for name, dreamed in dreams.items():
                scores[pressed, name].append(score(trial, dreamed))
            sheet(trial, dreams, palette).save(args.out / f"trial{n}_level{trial.level}_{kind}.png")
            print(f"trial {n} ({trial.what}): level {trial.level}", flush=True)
    for pressed in kinds:
        print(f"\n{'the bot' + chr(39) + 's buttons' if pressed else 'no buttons'}: playfield pixels expected wrong "
              f"at +{'/+'.join(map(str, HORIZONS))}, the falling piece at +32, and block mass at +128 (1.0 = real)")
        for (kind, name), rows in scores.items():
            if kind != pressed:
                continue
            mass = [r["mass"] for r in rows]
            print(f"  {name:48s} " + " / ".join(f"{np.mean([r[f'wrong_{h}'] for r in rows]):6.2%}" for h in HORIZONS)
                  + "   " + "  ".join(f"{k} {np.nanmean([r[k] for r in rows]):.2f}"
                                      for k in ("piece_32", "piece_hit_32", "piece_ghost_32", "fall", "spawn", "shape",
                                                "lost_long", "morph_long", "full_long", "size_long"))
                  + f"   mass {np.mean(mass):.2f} (range {min(mass):.2f}-{max(mass):.2f})")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
