"""The dynamics model as two ONNX graphs, for dreaming in the browser on WebGPU (web/) and in Python
(models/onnx_dreamer.py).

    prefill.onnx  frames [T, 256, 256] uint8, actions [T] int32 (incoming, -1 = none), levels [T] int32
                  -> keys, values [layers, tokens, heads, frames, head_dim]: positions 0..T-1, the rest
                  zero (2 <= T < frames)
    step.onnx     previous [256, 256] int32 (the last frame), actions [2] int32 and levels [2] int32 (the
                  previous frame's, the new one's), at [1] int32 (the new frame's position), keys, values
                  (positions before at - 1) -> frame [256, 256] int32 palette indices, new_keys,
                  new_values: the cache with the previous frame at at - 1

One step is one pass over two frames: the previous frame, written into the cache at at - 1, and the masked
new frame at `at`, which sees it. Dreamer (models/dynamics.py) does the same in two passes, decoding a
frame and then a pass to write it into the cache; merged, a step dispatches half the kernels, and the
frames are the same. So prefill takes every real frame but the last, which is the first step's
`previous`.

The cache goes in and comes out whole, so a WebGPU host keeps it in GPU buffers from call to call (as
onnxruntime-web runs LLMs' key/value caches) and only the frame crosses to the CPU; rewriting the whole
cache each step is cheap on a GPU (on the CPU it is most of a 200 ms step). Every node of step.onnx runs
on WebGPU (int32 and float32 only: WebGPU has no int64 or uint8 kernels for most ops). Both graphs run the model's
own forward against GraphCache, which keeps TemporalCache's interface over tensors passed in. step.onnx
decodes with decode_frame at the exported Decoding's steps (argmax; temperature needs random numbers, so
it stays 0).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from token_world.models.dynamics import Decoding, Dynamics, decode_frame


class GraphCache:
    """TemporalCache's interface over a layer's cache [tokens, heads, frames, head_dim] passed into the
    graph, never written: each call's result (prefill: padded to `frames` positions; step: the cache as
    passed in with this pass's frame at `at`) is kept in k, v for the graph's outputs."""

    def __init__(self, frames: int, k: torch.Tensor | None = None, v: torch.Tensor | None = None):
        self.frames, self.base, self.k, self.v = frames, (k, v), k, v

    def prefill(self, k: torch.Tensor, v: torch.Tensor):
        pad = (0, 0, 0, self.frames - k.shape[2])
        self.k, self.v = F.pad(k, pad), F.pad(v, pad)
        return k, v

    def step(self, k: torch.Tensor, v: torch.Tensor, at):
        positions = torch.arange(self.frames, device=k.device, dtype=at.dtype)
        keys, values = self.base
        for j in range(k.shape[2]):                     # frames at, at + 1, ...
            here = (positions == at + j).view(1, 1, -1, 1)
            keys, values = torch.where(here, k[:, :, j:j + 1], keys), torch.where(here, v[:, :, j:j + 1], values)
        self.k, self.v = keys, values
        # additive, not boolean: the exporter guards a boolean mask's softmax with IsNaN, which
        # onnxruntime-web runs on the CPU, a GPU round trip per attention
        own = at + torch.arange(k.shape[2], device=k.device, dtype=at.dtype)
        seen = torch.zeros(k.shape[2], self.frames, dtype=k.dtype, device=k.device)
        return keys, values, seen.masked_fill(positions[None] > own[:, None], float("-inf"))


class PrefillGraph(nn.Module):
    def __init__(self, model: Dynamics):
        super().__init__()
        self.model = model

    def forward(self, frames, actions, levels):
        caches = [GraphCache(self.model.frames) for _ in self.model.blocks]
        visible = torch.zeros(1, frames.shape[0], self.model.grid ** 2, dtype=torch.bool, device=frames.device)
        self.model(frames[None], actions[None], visible, levels[None], caches)
        return torch.stack([c.k for c in caches]), torch.stack([c.v for c in caches])


class StepGraph(nn.Module):
    def __init__(self, model: Dynamics, decoding: Decoding):
        super().__init__()
        self.model, self.decoding = model, decoding

    def forward(self, previous, actions, levels, at, keys, values):
        model, n = self.model, self.model.grid ** 2
        caches = [GraphCache(model.frames, keys[i], values[i]) for i in range(len(model.blocks))]
        actions, levels, first = actions.view(1, 2), levels.view(1, 2), at.view(()) - 1
        visible = torch.zeros(1, 1, n, dtype=torch.bool, device=keys.device)

        def logits_of(frame, hidden):                   # the previous frame, written in, and the new one
            pair = torch.stack((previous[None], frame), 1)
            mask = torch.cat((visible, hidden[:, None]), 1)
            return model.logits(model(pair, actions, mask, levels, caches, first)[:, 1])

        frame = decode_frame(logits_of, 1, n, model.patch, self.decoding, keys.device)
        return frame[0], torch.stack([c.k for c in caches]), torch.stack([c.v for c in caches])


def cache_shape(model: Dynamics) -> list[int]:
    """[layers, tokens, heads, frames, head_dim]: the cache the graphs pass around."""
    attention = model.blocks[0].temporal
    dim = model.blocks[0].norm_t.normalized_shape[0]
    return [len(model.blocks), model.grid ** 2, attention.heads, model.frames, dim // attention.heads]


@torch.no_grad()
def export(model: Dynamics, folder: Path, decoding: Decoding = Decoding(steps=1)) -> None:
    """Write prefill.onnx, step.onnx and model.json (cache shape, frames, steps) for a CPU float32 copy of
    `model` in eval mode (the model passed in is left as it is)."""
    if decoding.temperature > 0 or decoding.refine > 0:
        raise ValueError("step.onnx decodes with argmax and no refinement")
    model = copy.deepcopy(model).float().cpu().eval()
    model.embed_frames = model.frames                    # one embedding chunk: a traced loop would fix T
    folder.mkdir(parents=True, exist_ok=True)
    shape = cache_shape(model)
    t = torch.export.Dim("frames", min=2, max=model.frames - 1)
    example = model.frames // 2                          # inside the dynamic range, for any window
    frames = torch.zeros(example, 256, 256, dtype=torch.uint8)
    actions = torch.zeros(example, dtype=torch.int32)
    torch.onnx.export(PrefillGraph(model).eval(), (frames, actions, actions.clone()), folder / "prefill.onnx",
                      input_names=["frames", "actions", "levels"], output_names=["keys", "values"],
                      dynamic_shapes={"frames": {0: t}, "actions": {0: t}, "levels": {0: t}},
                      dynamo=True, external_data=False, verbose=False)
    cache = torch.zeros(shape)
    torch.onnx.export(StepGraph(model, decoding).eval(),
                      (torch.zeros(256, 256, dtype=torch.int32), torch.zeros(2, dtype=torch.int32),
                       torch.zeros(2, dtype=torch.int32), torch.tensor([example], dtype=torch.int32), cache,
                       cache.clone()), folder / "step.onnx",
                      input_names=["previous", "actions", "levels", "at", "keys", "values"],
                      output_names=["frame", "new_keys", "new_values"], dynamo=True, external_data=False, verbose=False)
    (folder / "model.json").write_text(json.dumps({"cache": shape, "frames": model.frames,
                                                   "steps": decoding.steps}, indent=1))
