"""Dreaming from the exported ONNX graphs (models/onnx_export.py): numpy and onnxruntime, no PyTorch.

OnnxDreamer does what the layered model's Dreamer (models/layered.py) does, as web/dreamer.js does in the
browser: a start's real layered frames are embedded as their content and fill the cache, each step decodes the
next frame for an action, returns its picture (rgb) and its content, and returns the cache with it; when the
window is full the last `keep` frames' content is encoded again at positions 0..keep-1. The cache always holds
every frame but the last, which the next step writes in as it decodes (one pass a frame, models/onnx_export.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


class OnnxDreamer:
    def __init__(self, folder: Path, layers: dict, actions: np.ndarray, keep: int | None = None,
                 threads: int = 0, providers: list[str] | None = None):
        """layers: the start's layered frames, each of model.json's inputs [T0, ...] uint8; actions [T0] the
        incoming button bytes (-1 = none)."""
        folder = Path(folder)
        meta = json.loads((folder / "model.json").read_text())
        self.window, self.prefilled = meta["frames"], meta["prefill"]
        keep = meta["keep"] if keep is None else keep
        if not 3 <= keep <= self.prefilled + 1 or not 3 <= len(actions) <= self.prefilled + 1:
            raise ValueError(f"keep and the start: 3..{self.prefilled + 1} frames (prefill takes {self.prefilled})")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        providers = providers or ["CPUExecutionProvider"]
        session = lambda name: ort.InferenceSession(str(folder / name), options, providers=providers)
        self.embed, self.prefill, self.decode = session("embed.onnx"), session("prefill.onnx"), session("step.onnx")
        self.keep = keep
        feeds = {name: np.asarray(layers[name]).astype(np.uint8) for name in meta["inputs"]}
        self.content = list(self.embed.run(None, feeds)[0])
        self.actions = [int(a) for a in actions]
        self._encode()

    def _encode(self) -> None:
        self.content, self.actions = self.content[-self.keep:], self.actions[-self.keep:]
        t, pad = len(self.content) - 1, self.prefilled - (len(self.content) - 1)    # padded after the frames
        content = np.concatenate([np.stack(self.content[:t]), np.zeros((pad, *self.content[0].shape), np.float32)])
        actions = np.array(self.actions[:t] + [-1] * pad, np.int32)
        self.keys, self.values = self.prefill.run(None, {"content": content, "actions": actions})

    def step(self, action: int) -> np.ndarray:
        """The frame that `action` (the controller byte, -1 for none) produces -> its colours [256, 256, 3]
        float32 (whole palette colours: each pixel's most likely colour)."""
        if len(self.content) == self.window:
            self._encode()
        at = len(self.content)
        rgb, content, self.keys, self.values = self.decode.run(None, {
            "previous": self.content[-1], "actions": np.array([self.actions[-1], action], np.int32),
            "at": np.array([at], np.int32), "keys": self.keys, "values": self.values})
        self.content.append(content)
        self.actions.append(action)
        return rgb
