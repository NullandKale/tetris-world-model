"""Dreaming from the exported ONNX graphs (models/onnx_export.py): numpy and onnxruntime, no PyTorch.

OnnxDreamer does what the pixel model's Dreamer (models/dynamics.py, one motion step) does, as web/dreamer.js does
in the browser: a start's real frames are embedded as their content, their motion not known, and fill the cache;
each step decodes the next frame for an action and returns its picture (rgb), its content, its motion and the
previous frame's keys and values, which go into the cache in place at slot (t - 1) mod frames. The cache is a ring
(frame t at slot t mod frames, models/dynamics.py TemporalCache), never re-encoded, and holds every frame but the
last, which the next step reads as `previous` (models/onnx_export.py). A frame-skip model's step covers 1..skip
frames: it takes their bytes and the frame clock, its own and the previous frame's (each frame as it was made).
`trace` records what each step fed the graph, as web/dreamer.js does (diagnostics/input_trace.py checks it).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


class OnnxDreamer:
    def __init__(self, folder: Path, frames: np.ndarray, actions: np.ndarray, threads: int = 0,
                 providers: list[str] | None = None):
        """frames: the start's real frames [T0, 256, 256] uint8 (palette indices; a two-layer model's [T0, 2, 256,
        256], its layers); actions [T0] the incoming button bytes (-1 = none); T0 3..frames - 1."""
        folder = Path(folder)
        meta = json.loads((folder / "model.json").read_text())
        self.window, self.prefilled, self.names = meta["frames"], meta["prefill"], meta["cache"]["names"]
        self.skip = meta.get("skip", 1)
        if not 3 <= len(actions) <= self.prefilled + 1:
            raise ValueError(f"the start: 3..{self.prefilled + 1} frames (prefill takes {self.prefilled})")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        providers = providers or ["CPUExecutionProvider"]
        session = lambda name: ort.InferenceSession(str(folder / name), options, providers=providers)
        self.embed, self.prefill, self.decode = session("embed.onnx"), session("prefill.onnx"), session("step.onnx")
        content, self.last = self.embed.run(None, {"frames": np.asarray(frames).astype(np.uint8)})
        content = list(content)
        motion = [np.full(meta["motion"], meta["unknown"], np.float32) for _ in content]
        actions = [int(a) for a in actions]
        t, pad = len(content) - 1, self.prefilled - (len(content) - 1)          # the start but its last, padded
        stacked = lambda xs: np.concatenate([np.stack(xs[:t]), np.zeros((pad, *xs[0].shape), np.float32)])
        cache = self.prefill.run(None, {"content": stacked(content), "motion": stacked(motion),
                                        "actions": np.array(actions[:t] + [-1] * pad, np.int32)})
        half = len(cache) // 2
        self.keys, self.values = cache[:half], cache[half:]               # each layer's [n, h, hd, F], [n, h, F, hd]
        self.content, self.motion, self.action = content[-1], motion[-1], actions[-1]     # the newest frame's
        self.skipped = [-1] * (self.skip - 1)            # its step's earlier bytes (a start's frames: none)
        self.t = len(content)
        self.clock = self.t - 1                          # its frame on the frame clock
        self.trace = {"skip": self.skip, "start": actions, "steps": []}

    def step(self, action: int, earlier=()) -> np.ndarray:
        """The frame that `action` (the controller byte, -1 for none) produces -> its colours [256, 256, 3]
        float32 (whole palette colours: each pixel's most likely colour). earlier (a frame-skip model's): the
        bytes of the frames the step skips, newest first: the frame is 1 + len(earlier) frames on."""
        at, earlier = self.t, [int(b) for b in earlier]
        if len(earlier) >= self.skip and earlier:
            raise ValueError(f"a step covers at most {self.skip} frames: {1 + len(earlier)} asked")
        skipped = earlier + [-1] * (self.skip - 1 - len(earlier))
        clock = self.clock + 1 + len(earlier)
        inputs = {"actions": [self.action, int(action)], "at": [at], "clock": [self.clock, clock],
                  "skipped": [self.skipped, skipped]}
        self.trace["steps"].append(inputs)
        rgb, content, motion, self.last, keys, values = self.decode.run(None, {
            "previous": self.content, "previous_motion": self.motion,
            "actions": np.array(inputs["actions"], np.int32), "at": np.array([at], np.int32), "last": self.last,
            **({"skipped": np.array(inputs["skipped"], np.int32).reshape(2, -1),
                "clock": np.array(inputs["clock"], np.int32)} if self.skip > 1 else {}),
            **dict(zip(self.names, self.keys + self.values))})
        slot = (at - 1) % self.window                                   # the previous frame's
        for cached, new in zip(self.keys, keys):
            cached[..., slot] = new
        for cached, new in zip(self.values, values):
            cached[..., slot, :] = new
        self.content, self.motion, self.action, self.skipped, self.clock = content, motion, int(action), skipped, clock
        self.t += 1
        return rgb
