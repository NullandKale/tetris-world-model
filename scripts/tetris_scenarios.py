"""Score world-model checkpoints on Tetris scenarios found live (diagnostics/tetris_scenarios.py).

    python scripts/tetris_scenarios.py output/world_model_tetris_self output/world_model_tetris_8m

Every run is scored on the same instances: fresh live streams (same --seed,
same instances), spread over the whole collection time (per stream, one
instance of a scenario every minutes * streams / per), for up to --minutes. Instances are
scored in batches as they arrive and then dropped, so nothing is stored.
Prints each scenario's mean with a 95% interval from resampling whole games
(instances from one game are not independent), and writes scenarios.csv and
one contact sheet per scenario (real, then each model's rollout of its first
instance) to --out.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from token_world.data.nes_palette import tetris_palette
from token_world.diagnostics.tetris_scenarios import (CONTEXT, HORIZONS, LEAD, SCENARIOS, evaluate, live_instances,
                                                      spread)
from token_world.models.dynamics import Dynamics, load_run

SHEET_HORIZONS = (1, LEAD, 4, 8, 16)
BATCH = 16


def load(run: Path) -> tuple[Dynamics, int]:
    model, saved = load_run(run)
    return model, int(saved["step"])


def sheet(path: Path, frames: np.ndarray, rollouts: dict[str, np.ndarray], palette: np.ndarray, title: str) -> None:
    """Columns: last context frame, then SHEET_HORIZONS. Rows: real, then each model's rollout."""
    size, label = 256, 18
    rows = [("real", [frames[CONTEXT - 1]] + [frames[CONTEXT - 1 + h] for h in SHEET_HORIZONS])]
    rows += [(name, [frames[CONTEXT - 1]] + [gen[h - 1] for h in SHEET_HORIZONS]) for name, gen in rollouts.items()]
    image = Image.new("RGB", ((1 + len(SHEET_HORIZONS)) * size, 20 + len(rows) * (size + label)), "#181820")
    draw = ImageDraw.Draw(image)
    draw.text((4, 4), title, fill="white")
    for r, (name, row) in enumerate(rows):
        y = 20 + r * (size + label)
        for c, frame in enumerate(row):
            caption = "context" if c == 0 else f"+{SHEET_HORIZONS[c - 1]}"
            draw.text((c * size + 4, y + 3), f"{name} {caption}", fill="white")
            picture = palette[frame]
            if r and c:                                               # mark this model's wrong pixels
                picture = picture.copy()
                picture[frame != rows[0][1][c]] = (255, 0, 200)
            image.paste(Image.fromarray(picture.astype(np.uint8), "RGB"), (c * size, y + label))
    image.save(path)


def game_interval(values: np.ndarray, games: np.ndarray, rng: np.random.Generator, draws: int = 1000):
    """Mean and 95% interval from resampling whole games (NaNs ignored)."""
    ok = ~np.isnan(values)
    values, games = values[ok], games[ok]
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    ids = np.unique(games)
    groups = [values[games == g] for g in ids]
    sums, counts = np.array([g.sum() for g in groups]), np.array([len(g) for g in groups])
    pick = rng.integers(0, len(ids), (draws, len(ids)))
    means = sums[pick].sum(1) / counts[pick].sum(1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", type=Path, nargs="+", help="run folders holding model_latest.pt")
    p.add_argument("--per", type=int, default=100, help="instances per scenario")
    p.add_argument("--minutes", type=float, default=15, help="collection time limit")
    p.add_argument("--seed", type=int, default=0, help="stream seed (same seed, same instances)")
    p.add_argument("--workers", type=int, default=8, help="live streams")
    p.add_argument("--out", type=Path, default=ROOT / "output" / "scenarios" / time.strftime("%Y%m%d_%H%M%S"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    models = {}
    for run in args.runs:
        model, step = load(run)
        models[f"{run.name}@{step}"] = model
    palette = tetris_palette().cpu().numpy()
    rows, first, pending = [], {}, defaultdict(list)
    counts = defaultdict(int)

    def flush(scenario: str) -> None:
        batch = pending.pop(scenario)
        for name, model in models.items():
            scores, generated = evaluate(model, batch)
            rows.extend({"model": name, "scenario": scenario, "stream": i.stream, "game": f"{i.stream}:{i.game}",
                         "level": i.level, **s} for i, s in zip(batch, scores))
            if scenario not in first or name not in first[scenario][1]:
                first.setdefault(scenario, (batch[0].frames, {}))[1][name] = generated[0]

    started = time.time()
    for instance in spread(live_instances(args.seed, args.workers), args.per, args.minutes * 60, args.workers):
        pending[instance.scenario].append(instance)
        counts[instance.scenario] += 1
        if len(pending[instance.scenario]) == BATCH:
            flush(instance.scenario)
    for scenario in list(pending):
        flush(scenario)
    print(f"collected and scored in {time.time() - started:.0f} s")

    with (args.out / "scenarios.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for scenario, (frames, rollouts) in first.items():
        slug = scenario.replace(" ", "_").replace(":", "").replace("(", "").replace(")", "")
        sheet(args.out / f"{slug}.png", frames, rollouts, palette,
              f"{scenario}: first visible change at +{LEAD}; magenta = wrong pixels")

    rng = np.random.default_rng(0)
    print(f"\nevent pixels wrong at +{LEAD} and +16, and exact at +{LEAD}: mean [95% interval over games]")
    for scenario in SCENARIOS:
        mine = [r for r in rows if r["scenario"] == scenario]
        if not mine:
            print(f"{scenario:22s} none found")
            continue
        base = [r for r in mine if r["model"] == next(iter(models))]
        games = {r["game"] for r in base}
        levels = [r["level"] for r in base]
        print(f"{scenario:22s} n={len(base)} from {len(games)} games, levels {min(levels)}-{max(levels)}")
        for name in models:
            sub = [r for r in mine if r["model"] == name]
            g = np.array([r["game"] for r in sub])
            cells = []
            for key in (f"event_wrong_{LEAD}", "event_wrong_16", "exact"):
                m, lo, hi = game_interval(np.array([r[key] for r in sub], float), g, rng)
                cells.append(f"{m:6.1%} [{lo:5.1%}-{hi:5.1%}]")
            print(f"    {name:32s} " + "   ".join(cells))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
