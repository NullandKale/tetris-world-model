"""Input traces: does a dream get its buttons and frames in the order training gave them?

The page (web/dreamer.js, web/play.js) and models/onnx_dreamer.py record, for every step, what they fed the step
graph (models/onnx_export.py): the two frames' incoming bytes ("actions"), their steps' earlier bytes, newest first
("skipped"), their frames on the frame clock ("clock") and the new frame's place in the ring ("at"). The page also
records, apart from that, every game frame's byte at 60 a second as its controller gave them ("frames", oldest
first, from the frame after the start's last).

expected() rebuilds the step inputs from the frame bytes and the steps' frames the way training lays out a skip
window (data/world_nes_tetris.py step_bytes, then the shift to incoming bytes: dynamics.incoming_actions,
incoming_skipped), and check() compares them with what was fed, step by step: a reversed byte order, a byte on the
wrong frame, a clock off by one, or a frame dropped from the record is a mismatch.

A trace (JSON): {"skip": the model's, "start": the start's incoming bytes, "frames": [byte, ...],
"steps": [{"actions": [2], "skipped": [[skip - 1], [skip - 1]], "clock": [2], "at": [1]}, ...]}.
"""
from __future__ import annotations

import numpy as np

from token_world.data.world_nes_tetris import step_bytes


def expected(trace: dict) -> list[dict]:
    """A trace's frame bytes and steps' frames -> each step's inputs as training would give them."""
    skip, start, steps = int(trace["skip"]), list(trace["start"]), trace["steps"]
    first = len(start) - 1                               # the start's last frame on the frame clock
    at = np.array([first] + [int(s["clock"][1]) for s in steps])
    frames = list(trace["frames"])
    if len(frames) != at[-1] - first:
        raise ValueError(f"the trace's frames: {len(frames)} bytes for the {at[-1] - first} frames its steps cover")
    acts = np.array([-1] * first + frames)               # acts[i] takes frame i to i + 1 (frame f's byte: acts[f - 1])
    action, skipped = step_bytes(acts, at, skip)         # step k's, to frame at[k + 1]
    none = [-1] * (skip - 1)
    out = []
    for k in range(len(steps)):                          # the previous frame's incoming, then the new one's
        before = (start[-1], none) if k == 0 else (int(action[k - 1]), skipped[k - 1].tolist())
        out.append({"actions": [int(before[0]), int(action[k])], "skipped": [list(before[1]), skipped[k].tolist()],
                    "clock": [int(at[k]), int(at[k + 1])], "at": [first + 1 + k]})
    return out


def check(trace: dict) -> dict:
    """Compare a trace's fed inputs with expected() -> {"steps", "frames", "gaps": {gap: steps}}; raises
    ValueError at the first step that differs, saying how."""
    for k, (fed, want) in enumerate(zip(trace["steps"], expected(trace))):
        for name, value in want.items():
            got = fed[name]
            if name == "skipped" and int(trace["skip"]) == 1:
                continue                                 # a one-frame model has no earlier bytes
            if [list(map(int, v)) if isinstance(v, list) else int(v) for v in got] != value:
                raise ValueError(f"step {k} (frame {want['clock'][1]}): {name} fed {got}, training's {value}")
    gaps = np.diff([len(trace["start"]) - 1] + [int(s["clock"][1]) for s in trace["steps"]])
    return {"steps": len(trace["steps"]), "frames": len(trace["frames"]),
            "gaps": {int(g): int((gaps == g).sum()) for g in np.unique(gaps)}}
