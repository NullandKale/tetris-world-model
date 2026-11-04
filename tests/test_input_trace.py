"""Input traces (diagnostics/input_trace.py): training's own skip windows, as the model reads them, pass the check;
a byte out of order, a clock off by one or a frame missing from the record does not."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from token_world.data.world_nes_tetris import SkipWindows
from token_world.diagnostics.input_trace import check
from token_world.models.dynamics import incoming_actions, incoming_skipped


def chunks(n=8, size=16):
    """Consecutive chunks of a stream: frame i's pixel (0, 0) is i, the byte played on it i % 200."""
    for c in range(n):
        ids = np.arange(c * (size - 1), c * (size - 1) + size)
        x = np.zeros((size, 256, 256), np.uint8)
        x[:, 0, 0] = ids % 256
        yield {"x": torch.from_numpy(x), "ram": torch.zeros(size, 2048, dtype=torch.uint8),
               "action": torch.from_numpy(np.append(ids[:-1] % 200, ids[-2] % 200)),
               "in_game_restarts": 0, "worker_id": 0}


def training_trace(skip=4, steps=9) -> dict:
    """A skip window as the trainer feeds it to the model (incoming bytes, the frame clock), written as a trace
    of a dream whose start's last frame is the window's first."""
    window = next(iter(SkipWindows(steps, skip, np.random.default_rng(3))(chunks())))
    frame = window["x"][:, 0, 0].long()
    incoming = incoming_actions(window["action"]).tolist()
    earlier = incoming_skipped(window["skipped"]).tolist()
    start = [7, 9, incoming[0]]                          # the start: its last frame is the window's first
    clock = (window["times"] + len(start) - 1).tolist()
    fed = [{"actions": incoming[k:k + 2], "skipped": earlier[k:k + 2], "clock": clock[k:k + 2], "at": [len(start) + k]}
           for k in range(steps - 1)]
    played = list(range(int(frame[0]), int(frame[-1])))  # every frame's byte from the first step's to the last's
    return {"skip": skip, "start": start, "frames": [f % 200 for f in played], "steps": fed}


class InputTraceTests(unittest.TestCase):
    def test_training_windows_as_the_model_reads_them_pass(self):
        trace = training_trace()
        result = check(trace)
        self.assertEqual(result["steps"], 8)
        self.assertTrue(set(result["gaps"]) <= {1, 2, 3, 4})
        self.assertEqual(sum(g * n for g, n in result["gaps"].items()), len(trace["frames"]))

    def test_out_of_order_bytes_a_wrong_clock_or_a_lost_frame_fail(self):
        trace = training_trace()
        k = next(i for i, s in enumerate(trace["steps"]) if sum(b >= 0 for b in s["skipped"][1]) >= 2)
        reversed_bytes = copy.deepcopy(trace)
        row = reversed_bytes["steps"][k]["skipped"][1]
        n = sum(b >= 0 for b in row)
        row[:n] = row[:n][::-1]                          # oldest first, not newest first
        late = copy.deepcopy(trace)
        late["steps"][2]["clock"][0] += 1
        lost = copy.deepcopy(trace)
        del lost["frames"][5]
        swapped = copy.deepcopy(trace)                   # the last byte among the earlier ones
        s = swapped["steps"][k]
        s["actions"][1], s["skipped"][1][0] = s["skipped"][1][0], s["actions"][1]
        for bad in (reversed_bytes, late, lost, swapped):
            with self.assertRaises(ValueError):
                check(bad)


if __name__ == "__main__":
    unittest.main()
