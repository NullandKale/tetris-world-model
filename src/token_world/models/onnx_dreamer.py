"""Dreaming from the exported ONNX graphs (models/onnx_export.py): numpy and onnxruntime, no PyTorch.

OnnxDreamer does what the pixel model's Dreamer (models/dynamics.py, one motion step) does, as web/dreamer.js does
in the browser: a start's real frames are embedded as their content, their motion not known, and fill the cache;
each step decodes the next frame for an action and returns its picture (rgb), its content, its motion and the
previous frame's keys and values, which go into the cache in place; when the window is full the last `keep`
frames (content, motion, actions) are encoded again at positions 0..keep-1. The cache holds every frame but the
last, which the next step reads as `previous` (models/onnx_export.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


class OnnxDreamer:
    def __init__(self, folder: Path, frames: np.ndarray, actions: np.ndarray, keep: int | None = None,
                 threads: int = 0, providers: list[str] | None = None):
        """frames: the start's real frames [T0, 256, 256] uint8 (palette indices); actions [T0] the incoming
        button bytes (-1 = none)."""
        folder = Path(folder)
        meta = json.loads((folder / "model.json").read_text())
        self.window, self.prefilled, self.names = meta["frames"], meta["prefill"], meta["cache"]["names"]
        keep = meta["keep"] if keep is None else keep
        if not 3 <= keep <= self.prefilled + 1 or not 3 <= len(actions) <= self.prefilled + 1:
            raise ValueError(f"keep and the start: 3..{self.prefilled + 1} frames (prefill takes {self.prefilled})")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        providers = providers or ["CPUExecutionProvider"]
        session = lambda name: ort.InferenceSession(str(folder / name), options, providers=providers)
        self.embed, self.prefill, self.decode = session("embed.onnx"), session("prefill.onnx"), session("step.onnx")
        self.keep = keep
        self.content = list(self.embed.run(None, {"frames": np.asarray(frames).astype(np.uint8)})[0])
        self.motion = [np.full(meta["motion"], meta["unknown"], np.float32) for _ in self.content]
        self.actions = [int(a) for a in actions]
        self._encode()

    def _encode(self) -> None:
        self.content, self.motion, self.actions = (x[-self.keep:] for x in (self.content, self.motion, self.actions))
        t, pad = len(self.content) - 1, self.prefilled - (len(self.content) - 1)    # padded after the frames
        stacked = lambda xs: np.concatenate([np.stack(xs[:t]), np.zeros((pad, *xs[0].shape), np.float32)])
        cache = self.prefill.run(None, {"content": stacked(self.content), "motion": stacked(self.motion),
                                        "actions": np.array(self.actions[:t] + [-1] * pad, np.int32)})
        half = len(cache) // 2
        self.keys, self.values = cache[:half], cache[half:]               # each layer's [n, h, hd, F], [n, h, F, hd]

    def step(self, action: int) -> np.ndarray:
        """The frame that `action` (the controller byte, -1 for none) produces -> its colours [256, 256, 3]
        float32 (whole palette colours: each pixel's most likely colour)."""
        if len(self.content) == self.window:
            self._encode()
        at = len(self.content)
        rgb, content, motion, keys, values = self.decode.run(None, {
            "previous": self.content[-1], "previous_motion": self.motion[-1],
            "actions": np.array([self.actions[-1], action], np.int32), "at": np.array([at], np.int32),
            **dict(zip(self.names, self.keys + self.values))})
        for cached, new in zip(self.keys, keys):                         # the previous frame, at at - 1
            cached[..., at - 1] = new
        for cached, new in zip(self.values, values):
            cached[..., at - 1, :] = new
        self.content.append(content)
        self.motion.append(motion)
        self.actions.append(action)
        return rgb
