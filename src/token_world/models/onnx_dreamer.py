"""Dreaming from the exported ONNX graphs (models/onnx_export.py): numpy and onnxruntime, no PyTorch.

OnnxDreamer does what models/dynamics.py's Dreamer does, as web/dreamer.js does in the browser: real
frames are embedded as patch tokens and fill the cache, each step decodes the next frame for an action,
returns its expected colours and its tokens, and returns the cache with it; when the window is full the
last `keep` frames' tokens are re-encoded at positions 0..keep-1. The cache always holds every frame but
the last, which the next step writes in as it decodes (one pass a frame, models/onnx_export.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


class OnnxDreamer:
    def __init__(self, folder: Path, frames: np.ndarray, actions: np.ndarray, keep: int | None = None,
                 threads: int = 0, providers: list[str] | None = None):
        """frames [T0, 256, 256] uint8 real frames, actions [T0] the incoming button bytes (-1 = none)."""
        folder = Path(folder)
        meta = json.loads((folder / "model.json").read_text())
        self.window = meta["frames"]
        keep = meta["keep"] if keep is None else keep
        if not 3 <= keep < self.window or len(frames) < 3:
            raise ValueError("keep must be 3..frames - 1, from 3 or more frames (prefill takes 2 or more)")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        providers = providers or ["CPUExecutionProvider"]
        session = lambda name: ort.InferenceSession(str(folder / name), options, providers=providers)
        self.embed, self.prefill, self.decode = session("embed.onnx"), session("prefill.onnx"), session("step.onnx")
        self.keep = keep
        self.tokens = list(self.embed.run(None, {"frames": np.asarray(frames, np.uint8)})[0])
        self.actions = [int(a) for a in actions]
        self._encode()

    def _encode(self) -> None:
        self.tokens, self.actions = self.tokens[-self.keep:], self.actions[-self.keep:]
        self.keys, self.values = self.prefill.run(None, {"tokens": np.stack(self.tokens[:-1]),
                                                         "actions": np.array(self.actions[:-1], np.int32)})

    def step(self, action: int) -> np.ndarray:
        """The frame that `action` (the controller byte, -1 for none) produces -> each pixel's expected
        colour [256, 256, 3] float32."""
        if len(self.tokens) == self.window:
            self._encode()
        at = len(self.tokens)
        rgb, tokens, self.keys, self.values = self.decode.run(None, {
            "previous": self.tokens[-1], "actions": np.array([self.actions[-1], action], np.int32),
            "at": np.array([at], np.int32), "keys": self.keys, "values": self.values})
        self.tokens.append(tokens)
        self.actions.append(action)
        return rgb
