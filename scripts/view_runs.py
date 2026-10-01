"""Look at world-model runs while they train or afterwards, without touching the training.

    python scripts/view_runs.py output/world_model_tetris_small [--compare output/world_model_tetris_8m ...]

The same tabs as the training window (ui/run_viewer.py), read from the run folders and refreshed as
the trainer writes them. Compared runs are drawn dashed on the same axes, by step.
"""
from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from token_world.ui.run_viewer import RunViewer
from token_world.ui.theme import setup_dark_theme


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--compare", type=Path, nargs="*", default=[])
    args = p.parse_args()
    root = tk.Tk()
    root.title("World-model runs: " + ", ".join(r.name for r in args.runs))
    root.geometry("1600x980")
    setup_dark_theme(root)
    RunViewer(root, args.runs, args.compare)
    root.mainloop()


if __name__ == "__main__":
    main()
