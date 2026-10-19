"""The pixel world model (models/dynamics.py Dynamics, motion first) as three ONNX graphs, for dreaming in the
browser on WebGPU (web/) and in Python (models/onnx_dreamer.py). A frame is kept in the context as its content,
its patch tokens before the model adds anything to them (the patch conv of each pixel's colour embedding; a dreamed
frame's soft: its colour probabilities through the colour table, Dynamics.soft_tokens) [N, dim] float32, and its
motion, each token's 8 x 8 cells' codes (data/tetris_motion.py) [N, cells] float32, 255 where not known (a start's
real frames: the page ships no RAM, the motion's source).

    embed.onnx    real frames [T, 256, 256] uint8 (palette indices) -> content [T, N, dim]
    prefill.onnx  content [P, N, dim], motion [P, N, cells], actions [P] int32 (incoming, -1 = none) -> the cache,
                  each layer's keys_i [tokens, heads, head_dim, frames] (transposed, as attention reads them) and
                  values_i [tokens, heads, frames, head_dim], at positions 0..P-1, the rest zero. P is fixed, keep - 1
                  (the frames a window slide encodes): a start with fewer frames is padded after them, which no real
                  frame sees (causal) and the steps write over (a dynamic length put its shape arithmetic on the CPU
                  in onnxruntime-web)
    step.onnx     previous [N, dim] and previous_motion [N, cells] (the last frame's), actions [2] int32 (the
                  previous frame's, the new one's), at [1] int32 (the new frame's position), the cache (keys_i,
                  values_i: positions before at - 1 are read) -> rgb [256, 256, 3] float32 (the new frame, each
                  pixel's most likely colour through the palette in the graph), content [N, dim] and motion [N,
                  cells] (the new frame's), new_keys, new_values [layers, tokens, heads, head_dim]: the previous
                  frame's, which the host writes into the cache at at - 1

A step is FrameDecoder's frame with one motion step (Dynamics.motion_steps = 1: at s9_cool140 152k one step dreamed
as two, 2026-10-18, docs/guides/recipe_attempts.md): pass 1 is over two frames, the previous one with its motion
(the decoder's pass that writes a frame into the cache) and the new one hidden at `at`, and gives each of the new
frame's cells its most likely motion (the border's token rows STILL, Dynamics.drawn_motion); pass 2 is over the new
frame alone, hidden with that motion added, attending to the cache and to the previous frame's keys and values from
pass 1 (a frame never sees a later one, so its keys are the same in both), and gives each pixel's colour
probabilities: the picture (most likely colours) and the soft content. Three frames' work a step, as the decoder's
three passes. A model trained with a change weight other than 1 is refused (its decoding reweighs changes from the
last frame, which the graphs leave out).

A step attends over the cache's earlier frames and its own together (one softmax over all) and never copies the
cache: the host writes the previous frame's keys and values in place (web/dreamer.js: a compute shader on the GPU
buffers the cache stays in). Passing the cache through whole was a rewrite of all of it a step, 40% of a step's
GPU time on a 3090, more on an integrated GPU's shared memory. Masks are additive (a boolean mask's softmax guard
runs on the CPU in onnxruntime-web); argmax is float one-hots (WebGPU runs no int64: ArgMax's indices sent the work
to the CPU, a round trip a frame).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from token_world.models.dynamics import (COLOURS, MOTION_CODES, MOTION_STILL, MOTION_UNKNOWN, ROPE_BASE, SIZE,
                                         Dynamics, button_bits)


def first_max(scores: torch.Tensor) -> torch.Tensor:
    """[..., C] -> [..., C] float one-hot of the largest (the first at the maximum, as argmax), in float ops."""
    hit = (scores >= scores.amax(-1, keepdim=True)).float()
    ranked = hit * torch.arange(scores.shape[-1], 0, -1, dtype=hit.dtype)               # earlier ones higher
    return (ranked >= ranked.amax(-1, keepdim=True)).float()


def rotation(times: torch.Tensor, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Frames `times` [T] -> the cos and sin [T, 1, 1, head_dim / 2] that rotate their queries and keys
    (dynamics.rotary's), made once for every layer of a pass."""
    half = head_dim // 2
    angle = times.float()[:, None] * ROPE_BASE ** -(torch.arange(half, dtype=torch.float32) / half)
    return angle.cos()[:, None, None], angle.sin()[:, None, None]


def heads(layer, h: torch.Tensor, turn: tuple[torch.Tensor, torch.Tensor]):
    """A temporal layer's queries, keys and values for h [T, n, d], rotated by `turn` (rotation) -> each [n,
    heads, T, head_dim]."""
    t, n, d = h.shape
    q, k, v = layer.qkv(h).view(t, n, 3, layer.heads, d // layer.heads).unbind(2)
    if layer.q_norm is not None:
        q, k = layer.q_norm(q), layer.k_norm(k)
    cos, sin = turn
    half = q.shape[-1] // 2
    rotate = lambda z: torch.cat((z[..., :half] * cos - z[..., half:] * sin,
                                  z[..., :half] * sin + z[..., half:] * cos), -1)
    each = lambda z: z.permute(1, 2, 0, 3)
    return each(rotate(q)), each(rotate(k)), each(v)


def merge(layer, y: torch.Tensor) -> torch.Tensor:
    """[n, heads, T, head_dim] -> the layer's output [T, n, d]."""
    n, h, t, hd = y.shape
    return layer.out(y.permute(2, 0, 1, 3).reshape(t, n, h * hd))


class Graph(nn.Module):
    """What the three graphs share: a frame's tokens from its content, motion and buttons, and the blocks."""

    def __init__(self, model: Dynamics):
        super().__init__()
        self.model = model
        border = torch.zeros(model.grid, model.grid)
        border[0] = border[-1] = 1
        self.register_buffer("border", border.flatten()[:, None, None])                 # [N, 1, 1] the border rows
        self.register_buffer("codes", torch.arange(MOTION_CODES).float())
        self.register_buffer("still", F.one_hot(torch.tensor(MOTION_STILL), MOTION_CODES).float())

    def motion(self, picks: torch.Tensor) -> torch.Tensor:
        """Motion one-hots [..., N, cells, CODES] (all zero: not known) -> what they add to each token [..., N, dim]
        (Dynamics.motion_embedding: each cell's code by its place in the token)."""
        return picks.flatten(-2) @ self.model.motion_in.weight

    def known(self, codes: torch.Tensor) -> torch.Tensor:
        """Motion codes [..., N, cells] float (255: not known) -> one-hots [..., N, cells, CODES]."""
        return (codes[..., None] == self.codes).float()

    def tokens(self, content: torch.Tensor, added: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """content [T, N, d], what the motion adds [T, N, d], each frame's button embedding [T, d] (Dynamics.action
        of its button_bits) -> the tokens [T, N, d] (Dynamics.forward)."""
        model = self.model
        return content + added + model.space_pos + (action[:, None] if model.action_blocks is None else 0)

    def run(self, x: torch.Tensor, action: torch.Tensor, temporal) -> torch.Tensor:
        """x [T, N, d] -> the model's features: each block's spatial attention within each frame, `temporal(i,
        layer, normed)` (per token over frames), its MLP, the norms set by the buttons (action_norm); the final
        norm (Block.forward)."""
        model = self.model
        for i, block in enumerate(model.blocks):
            norms = (block.norm_s, block.norm_t, block.norm_m)
            if block.modulation is not None:
                mods = block.modulation(action)[:, None].chunk(6, -1)                   # each [T, 1, d]
                norms = tuple((lambda z, norm=norm, scale=mods[2 * j], shift=mods[2 * j + 1]:
                               norm(z) * (1 + scale) + shift) for j, norm in enumerate(norms))
            x = x + block.spatial(norms[0](x), model.place)
            x = x + temporal(i, block.temporal, norms[1](x))
            x = x + block.mlp(norms[2](x))
        return model.norm(x)


class EmbedGraph(nn.Module):
    def __init__(self, model: Dynamics):
        super().__init__()
        self.model = model

    def forward(self, frames):
        model = self.model
        pixels = F.embedding(frames.int(), model.colour.weight)                         # [T, 256, 256, colour_dim]
        return model.patchify(pixels.permute(0, 3, 1, 2)).flatten(2).transpose(1, 2)


class PrefillGraph(Graph):
    def forward(self, content, motion, actions):
        model, t = self.model, content.shape[0]
        action = model.action(button_bits(actions))
        x = self.tokens(content, self.motion(self.known(motion)), action)
        times = torch.arange(t)
        turn = rotation(times, model.blocks[0].temporal.qkv.in_features // model.blocks[0].temporal.heads)
        later = times[None, :] > times[:, None]
        causal = torch.zeros(t, t).masked_fill(later, float("-inf"))
        keys, values = [], []

        def temporal(i, layer, h):
            q, k, v = heads(layer, h, turn)
            pad = (0, 0, 0, model.frames - t)
            keys.append(F.pad(k, pad).transpose(-1, -2))
            values.append(F.pad(v, pad))
            return merge(layer, F.scaled_dot_product_attention(q, k, v, attn_mask=causal))

        self.run(x, action, temporal)
        return (*keys, *values)


class StepGraph(Graph):
    def __init__(self, model: Dynamics, palette: torch.Tensor):
        super().__init__(model)
        self.register_buffer("colours", palette.float()[:COLOURS])                     # [COLOURS, 3]

    def forward(self, previous, previous_motion, actions, at, *cache):
        model = self.model
        keys, values = cache[:len(model.blocks)], cache[len(model.blocks):]
        hidden = model.mask_token.expand_as(previous)
        positions = torch.arange(model.frames, dtype=at.dtype)
        earlier = torch.zeros(model.frames).masked_fill(positions >= at.view(()) - 1, float("-inf"))   # the cache's
        head_dim = model.blocks[0].temporal.qkv.in_features // model.blocks[0].temporal.heads
        scale = head_dim ** -0.5
        new_keys, new_values = [], []

        # pass 1: the previous frame, with its motion, and the new one hidden -> the new frame's motion
        action = model.action(button_bits(actions))                                      # [2, d]
        x = self.tokens(torch.stack((previous, hidden)),
                        torch.stack((self.motion(self.known(previous_motion)), torch.zeros_like(previous))), action)
        cos, sin = rotation(at.view(()) - 1 + torch.arange(2, dtype=at.dtype), head_dim)  # the pair's frames
        own = torch.tensor([[0.0, float("-inf")], [0.0, 0.0]])                          # the pair: causal

        def first(i, layer, h):
            q, k, v = heads(layer, h, (cos, sin))                                         # [n, heads, 2, hd]
            new_keys.append(k[:, :, 0])                         # the previous frame's, for the cache at at - 1
            new_values.append(v[:, :, 0])
            weights = torch.cat((q @ keys[i] * scale + earlier, q @ k.transpose(-1, -2) * scale + own), -1).softmax(-1)
            return merge(layer, weights[..., :model.frames] @ values[i] + weights[..., model.frames:] @ v)

        f = self.run(x, action, first)[1]
        picks = first_max(model.motion_logits(f))                                         # [N, cells, CODES]
        picks = picks * (1 - self.border) + self.still * self.border                     # the border rows STILL

        # pass 2: the new frame alone, hidden with that motion -> its pixels
        x = self.tokens(hidden[None], self.motion(picks)[None], action[1:])
        now = (cos[1:], sin[1:])                                                          # the new frame's, at

        def second(i, layer, h):
            q, k, v = heads(layer, h, now)                                                # [n, heads, 1, hd]
            before_k, before_v = new_keys[i][:, :, None], new_values[i][:, :, None]       # [n, heads, 1, hd]
            weights = torch.cat((q @ keys[i] * scale + earlier, q @ before_k.transpose(-1, -2) * scale,
                                 q @ k.transpose(-1, -2) * scale), -1).softmax(-1)
            f = model.frames
            return merge(layer, weights[..., :f] @ values[i] + weights[..., f:f + 1] @ before_v
                         + weights[..., f + 1:] @ v)

        f = self.run(x, action[1:], second)[0]
        probs = model.logits(f).float().softmax(-1)                                       # [N, P*P, COLOURS]
        g, p = model.grid, model.patch
        rgb = (first_max(probs) @ self.colours).view(g, g, p, p, 3).permute(0, 2, 1, 3, 4).reshape(SIZE, SIZE, 3)
        # its content from the same probabilities (Dynamics.soft_tokens): each pixel's expected colour embedding,
        # through the patch conv
        pixels = (probs @ model.colour.weight).transpose(1, 2).flatten(1)                 # [N, colour_dim * P * P]
        content = F.linear(pixels, model.patchify.weight.flatten(1), model.patchify.bias)
        motion = (picks * self.codes).sum(-1)                                            # [N, cells] the codes
        return rgb, content, motion, torch.stack(new_keys), torch.stack(new_values)


def cache_layout(model: Dynamics) -> dict:
    """The cache the graphs read, one keys and one values tensor a layer (keys transposed, as attention reads
    them), and their names in the graphs."""
    layer = model.blocks[0].temporal
    n, h, f = model.grid ** 2, layer.heads, model.frames
    hd = layer.qkv.in_features // h
    count = len(model.blocks)
    return {"layers": count, "keys": [n, h, hd, f], "values": [n, h, f, hd],
            "names": [f"keys_{i}" for i in range(count)] + [f"values_{i}" for i in range(count)]}


@torch.no_grad()
def export(model: Dynamics, folder: Path, palette: torch.Tensor) -> None:
    """Write embed.onnx, prefill.onnx, step.onnx and model.json for a CPU float32 copy of `model` in eval mode;
    palette [COLOURS, 3] is the game's."""
    if not isinstance(model, Dynamics) or not model.motion_first:
        raise NotImplementedError("the export is for the pixel model with motion first (models/dynamics.py)")
    if model.latent or model.change_weight != 1.0:
        raise NotImplementedError("the export has no latent choice and no change weight")
    model = copy.deepcopy(model).float().cpu().eval()
    folder.mkdir(parents=True, exist_ok=True)
    cache = cache_layout(model)
    n, dim = model.grid ** 2, model.blocks[0].temporal.qkv.in_features
    t = torch.export.Dim("frames", min=2, max=model.frames - 1)
    torch.onnx.export(EmbedGraph(model).eval(), (torch.zeros(model.frames // 2, SIZE, SIZE, dtype=torch.uint8),),
                      folder / "embed.onnx", input_names=["frames"], output_names=["content"],
                      dynamic_shapes={"frames": {0: t}}, dynamo=True, external_data=False, verbose=False)
    prefill = model.frames * 3 // 4 - 1                   # keep - 1
    torch.onnx.export(PrefillGraph(model).eval(),
                      (torch.zeros(prefill, n, dim), torch.full((prefill, n, model.motion_cells), float(MOTION_UNKNOWN)),
                       torch.zeros(prefill, dtype=torch.int32)),
                      folder / "prefill.onnx", input_names=["content", "motion", "actions"],
                      output_names=cache["names"], dynamo=True, external_data=False, verbose=False)
    empty = [torch.zeros(cache[name.split("_")[0]]) for name in cache["names"]]
    torch.onnx.export(StepGraph(model, palette).eval(),
                      (torch.zeros(n, dim), torch.zeros(n, model.motion_cells), torch.zeros(2, dtype=torch.int32),
                       torch.tensor([model.frames // 2], dtype=torch.int32), *empty), folder / "step.onnx",
                      input_names=["previous", "previous_motion", "actions", "at", *cache["names"]],
                      output_names=["rgb", "content", "motion", "new_keys", "new_values"], dynamo=True,
                      external_data=False, verbose=False)
    (folder / "model.json").write_text(json.dumps({
        "cache": cache, "frames": model.frames, "content": [n, dim], "motion": [n, model.motion_cells],
        "unknown": MOTION_UNKNOWN, "keep": model.frames * 3 // 4, "prefill": prefill,
        "parameters": sum(p.numel() for p in model.parameters())}, indent=1))
