"""Dreaming from the exported ONNX graphs (models/onnx_export.py): numpy and onnxruntime, no PyTorch.

OnnxDreamer does what models/dynamics.py's Dreamer does, as web/dreamer.js does in the browser: real
frames fill the cache, each step decodes the next frame for an action and returns the cache with it, and
when the window is full the last `keep` frames are re-encoded at positions 0..keep-1. Generated frames
carry `level`, real ones 0. The cache always holds every frame but the last, which the next step writes
in as it decodes (one pass a frame, models/onnx_export.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

GENERATED_LEVEL = 2             # models/dynamics.py's default for generated frames


class OnnxDreamer:
    def __init__(self, folder: Path, frames: np.ndarray, actions: np.ndarray, level: int = GENERATED_LEVEL,
                 keep: int = 48, threads: int = 0, providers: list[str] | None = None):
        """frames [T0, 256, 256] uint8 real frames, actions [T0] the incoming button bytes (-1 = none)."""
        folder = Path(folder)
        meta = json.loads((folder / "model.json").read_text())
        self.window = meta["frames"]
        if not 3 <= keep < self.window or len(frames) < 3:
            raise ValueError("keep must be 3..frames - 1, from 3 or more frames (prefill takes 2 or more)")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        providers = providers or ["CPUExecutionProvider"]
        self.prefill = ort.InferenceSession(str(folder / "prefill.onnx"), options, providers=providers)
        self.decode = ort.InferenceSession(str(folder / "step.onnx"), options, providers=providers)
        self.level, self.keep = level, keep
        self.frames = [np.asarray(f, np.uint8) for f in frames]
        self.actions = [int(a) for a in actions]
        self.levels = [0] * len(self.frames)
        self._encode()

    def _encode(self) -> None:
        self.frames, self.actions, self.levels = (self.frames[-self.keep:], self.actions[-self.keep:],
                                                  self.levels[-self.keep:])
        self.keys, self.values = self.prefill.run(None, {"frames": np.stack(self.frames[:-1]),
                                                         "actions": np.array(self.actions[:-1], np.int32),
                                                         "levels": np.array(self.levels[:-1], np.int32)})

    def step(self, action: int) -> np.ndarray:
        """The frame that `action` (the controller byte, -1 for none) produces -> [256, 256] uint8."""
        if len(self.frames) == self.window:
            self._encode()
        at = len(self.frames)
        frame, self.keys, self.values = self.decode.run(None, {
            "previous": self.frames[-1].astype(np.int32), "actions": np.array([self.actions[-1], action], np.int32),
            "levels": np.array([self.levels[-1], self.level], np.int32), "at": np.array([at], np.int32),
            "keys": self.keys, "values": self.values})
        frame = frame.astype(np.uint8)
        self.frames.append(frame)
        self.actions.append(action)
        self.levels.append(self.level)
        return frame
