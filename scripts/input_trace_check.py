"""Check input traces from the browser (the page's Save trace) or OnnxDreamer: every step's buttons and frames
in the order training gives them (diagnostics/input_trace.py). Exits 1 at the first step that differs.

    python scripts/input_trace_check.py trace.json [more.json ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from token_world.diagnostics.input_trace import check


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("traces", type=Path, nargs="+")
    for path in p.parse_args().traces:
        result = check(json.loads(path.read_text()))
        gaps = ", ".join(f"{g} frames: {n}" for g, n in result["gaps"].items())
        print(f"{path}: {result['steps']} steps over {result['frames']} frames as training lays them out "
              f"(steps of {gaps})")


if __name__ == "__main__":
    main()
