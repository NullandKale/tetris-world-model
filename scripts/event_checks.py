"""Long game events, followed to their end: each model dreams them live, fed the real buttons.

    python scripts/event_checks.py output/world_model_tetris_layered_px output/world_model_tetris_base
    python scripts/event_checks.py D:/token_world_checkpoints/world_model_tetris_layered_px/model_step20000.pt --device cpu
    python scripts/event_checks.py <run> --checks line_clears --trials 16

Checks (--checks, default all):
- restart (diagnostics/restart.py): a top-out, the curtain, the menus and a new game (320 frames);
- line_clears (diagnostics/line_clears.py): from 10 frames before a clear starts, 60 frames: the rows gone, the
  stack dropped (scored per clear size too).

Any model: run folders (their model_latest.pt) or checkpoint files, pixel or layered (diagnostics/worlds.py).
Every model dreams the same --trials events per check (same --seed). Prints each event's scores and the means,
writes one sheet per event (real on top, a row per model) to --out, and appends each run's means with the
checkpoint's step to its run folder's event_checks.csv (one row per check, group and score), which the run
window's Scenarios tab shows.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from token_world.data.nes_palette import tetris_palette
from token_world.diagnostics import line_clears, restart
from token_world.diagnostics.worlds import load, run_folder

CHECKS = {
    # name: (the live events, their scores, sheet, the group an event is also averaged in, or None)
    "restart": (restart.restarts, restart.score, restart.sheet, restart.FIELDS, lambda event: None),
    "line_clears": (line_clears.clears, line_clears.score, line_clears.sheet, line_clears.FIELDS,
                    lambda event: event.name),
}
SUMMARY = ("step", "check", "group", "events", "score", "value")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", type=Path, nargs="+", help="run folders holding model_latest.pt, or checkpoint files")
    p.add_argument("--checks", nargs="+", choices=list(CHECKS), default=list(CHECKS))
    p.add_argument("--trials", type=int, default=8, help="events per check")
    p.add_argument("--seed", type=int, default=9100)
    p.add_argument("--device", default="cuda", help="cuda, or cpu while a training run holds the GPU")
    p.add_argument("--out", type=Path, default=ROOT / "output" / "event_checks" / time.strftime("%Y%m%d_%H%M%S"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    worlds = [load(run, args.device) for run in args.runs]
    names = [f"{world.name}@{world.step}" for world in worlds]
    palette = tetris_palette().cpu().numpy().astype(np.uint8)
    summary = defaultdict(list)                                     # world index -> rows
    for check in args.checks:
        found, score, sheet, fields, group_of = CHECKS[check]
        events = found(args.seed, layered=any(w.layered for w in worlds))
        scores = defaultdict(list)                                  # (world index, group) -> [scores]
        for n in range(args.trials):
            event = next(events)
            group = group_of(event)
            dreams = {}
            for w, world in enumerate(worlds):
                dreams[names[w]] = made = world.follow(event.context, event.actions, event.layers,
                                                       event.future_actions, seed=n)
                s = score(event, made)
                scores[w, "all"].append(s)
                if group is not None:
                    scores[w, group].append(s)
                print(f"{check} {n}{f' ({group})' if group else ''} {names[w]}: "
                      + " ".join(f"{k} {v:.3g}" for k, v in s.items()), flush=True)
            Image.fromarray(sheet(event, dreams, palette)).save(args.out / f"{check}_{n}.png")
        print()
        for (w, group), rows in sorted(scores.items(), key=lambda item: (item[0][0], item[0][1] != "all", item[0][1])):
            means = {k: float(np.mean([r[k] for r in rows])) for k in fields}
            print(f"{check} {group} ({len(rows)}) {names[w]}: " + " ".join(f"{k} {v:.3g}" for k, v in means.items()))
            summary[w] += [{"step": worlds[w].step, "check": check, "group": group, "events": len(rows), "score": k,
                            "value": v} for k, v in means.items()]
        print()
    for w in range(len(worlds)):
        path = run_folder(args.runs[w]) / "event_checks.csv"
        if not path.parent.is_dir():
            continue
        new = not path.exists()
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY)
            if new:
                writer.writeheader()
            writer.writerows(summary[w])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
