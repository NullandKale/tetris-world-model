"""Score world-model checkpoints on Tetris scenarios found live (diagnostics/tetris_scenarios.py).

    python scripts/tetris_scenarios.py output/world_model_tetris_layered_px output/world_model_tetris_base
    python scripts/tetris_scenarios.py D:/token_world_checkpoints/world_model_tetris_layered_px/model_step20000.pt

Any model: run folders (their model_latest.pt) or checkpoint files, pixel or layered (diagnostics/worlds.py).
Each run's summary (per scenario: instances, games, levels, the means) is appended to its run folder's
scenarios.csv with the checkpoint's step, which the run window's Scenarios tab shows (ui/run_viewer.py).

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
from token_world.diagnostics.worlds import load, run_folder

SHEET_HORIZONS = (1, LEAD, 4, 8, 16)
BATCH = 16


def sheet(path: Path, frames: np.ndarray, rollouts: dict[str, tuple[np.ndarray, np.ndarray]], palette: np.ndarray,
          title: str) -> None:
    """Columns: last context frame, then SHEET_HORIZONS. Rows: real, then each model's rollout (expected colours,
    magenta in proportion to each pixel's probability of being wrong)."""
    size, label = 256, 18
    context = palette[frames[CONTEXT - 1]]
    rows = [("real", [context] + [palette[frames[CONTEXT - 1 + h]] for h in SHEET_HORIZONS])]
    for name, (rgb, wrong) in rollouts.items():
        pictures = [context]
        for h in SHEET_HORIZONS:
            w = wrong[h - 1].astype(np.float32)[..., None]
            pictures.append(rgb[h - 1] * (1 - w) + np.array([255, 0, 200]) * w)
        rows.append((name, pictures))
    image = Image.new("RGB", ((1 + len(SHEET_HORIZONS)) * size, 20 + len(rows) * (size + label)), "#181820")
    draw = ImageDraw.Draw(image)
    draw.text((4, 4), title, fill="white")
    for r, (name, row) in enumerate(rows):
        y = 20 + r * (size + label)
        for c, picture in enumerate(row):
            caption = "context" if c == 0 else f"+{SHEET_HORIZONS[c - 1]}"
            draw.text((c * size + 4, y + 3), f"{name} {caption}", fill="white")
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


SUMMARY = ("step", "scenario", "instances", "games", "level_min", "level_max", *(f"event_wrong_{h}" for h in HORIZONS),
           *(f"false_change_{h}" for h in HORIZONS), "exact")


def summarize(rows: list[dict], models: dict, folders: dict) -> None:
    """Each model's means per scenario -> appended to its run folder's scenarios.csv (the window's Scenarios
    tab), with the checkpoint's step (run_folder)."""
    for name in models:
        folder, step = folders[name]
        out = []
        for scenario in SCENARIOS:
            mine = [r for r in rows if r["scenario"] == scenario and r["model"] == name]
            if not mine:
                continue
            mean = lambda key: float(np.nanmean([r[key] for r in mine])) if any(
                not np.isnan(r[key]) for r in mine) else float("nan")
            levels = [r["level"] for r in mine]
            out.append({"step": step, "scenario": scenario, "instances": len(mine),
                        "games": len({r["game"] for r in mine}), "level_min": min(levels), "level_max": max(levels),
                        **{key: mean(key) for key in SUMMARY[6:]}})
        path = folder / "scenarios.csv"
        new = not path.exists()
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY)
            if new:
                writer.writeheader()
            writer.writerows(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", type=Path, nargs="+", help="run folders holding model_latest.pt, or checkpoint files")
    p.add_argument("--device", default="cuda", help="cuda, or cpu while a training run holds the GPU")
    p.add_argument("--per", type=int, default=100, help="instances per scenario")
    p.add_argument("--minutes", type=float, default=15, help="collection time limit")
    p.add_argument("--seed", type=int, default=0, help="stream seed (same seed, same instances)")
    p.add_argument("--workers", type=int, default=8, help="live streams")
    p.add_argument("--out", type=Path, default=ROOT / "output" / "scenarios" / time.strftime("%Y%m%d_%H%M%S"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    models, folders = {}, {}
    for run in args.runs:
        world = load(run, args.device)
        models[f"{world.name}@{world.step}"] = world
        folders[f"{world.name}@{world.step}"] = (run_folder(run), world.step)
    palette = tetris_palette().cpu().numpy()
    rows, first, pending = [], {}, defaultdict(list)
    counts = defaultdict(int)

    def flush(scenario: str) -> None:
        batch = pending.pop(scenario)
        for name, world in models.items():
            scores, rgb, wrong = evaluate(world, batch)
            rows.extend({"model": name, "scenario": scenario, "stream": i.stream, "game": f"{i.stream}:{i.game}",
                         "level": i.level, **s} for i, s in zip(batch, scores))
            if scenario not in first or name not in first[scenario][1]:
                first.setdefault(scenario, (batch[0].frames, {}))[1][name] = (rgb[0], wrong[0])

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
              f"{scenario}: first visible change at +{LEAD}; magenta = likely wrong pixels")

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
    summarize(rows, models, folders)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
