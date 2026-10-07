"""The layered world model (models/layered_pixels.py) as three ONNX graphs, for dreaming in the browser on WebGPU
(web/) and in Python (models/onnx_dreamer.py). A frame is kept in the context as its content: what each token
holds before the model adds its place (models/layered.py LayeredBase.content: the soft pixel tokens and the
backdrop colour), [n, dim] float32.

    embed.onnx    a real frame's layers, uint8: cells and cell_known [T, BANDS, VIEW, 16, 16], sprite_layer
                  [T, 240, 256], border [T, 32, 256], backdrop [T] -> content [T, n, dim]
    prefill.onnx  content [P, n, dim], actions [P] int32 (incoming, -1 = none) -> the cache, each layer's
                  keys_i [tokens, heads, head_dim, frames] (transposed, as attention reads them) and values_i
                  [tokens, heads, frames, head_dim], at positions 0..P-1, the rest zero. P is fixed, keep - 1
                  (the frames a window slide encodes): a start with fewer frames is padded after them, which no
                  real frame sees (causal) and the steps write over (a dynamic length put its shape arithmetic,
                  about 190 nodes, on the CPU in onnxruntime-web)
    step.onnx     previous [n, dim] (the last frame's content), actions [2] int32 (the previous frame's, the
                  new one's), at [1] int32 (the new frame's position), the cache (keys_i, values_i: positions
                  before at - 1 are read) -> rgb [256, 256, 3] float32 (the new model frame: its layers
                  composed, each pixel's most likely colour, through the palette in the graph), content [n, dim]
                  (the new frame's, soft, as the Dreamer's commit pass keeps it), new_keys, new_values [layers,
                  tokens, heads, head_dim]: the previous frame's, which the host writes into the cache at at - 1

A dream holds its camera, the start frame's (the page's start is one frame's layers; docs/guides/
layered_tokens.md): the export is for a game whose camera does not move, as Tetris (a model trained with
fixed_camera, or one whose camera head only learned "no move": the first layered-pixels run's never moved it
in 1,024 dream frames). Every token then keeps its identity, so the routed temporal attention (RoutedTemporal)
is attention per token over the earlier frames, and each token's place is a constant of the graphs.

One step is one pass over two frames, as the pixel model's export was: the previous frame, written into the
cache at at - 1 from its soft content (the Dreamer's commit pass), and the hidden new frame at `at` (its decide
pass), which sees it. So prefill takes every frame but the last, which is the first step's `previous`. A step
attends over the cache's earlier frames and its own two together (one softmax over both) and never copies the
cache: the host writes the previous frame's keys and values in place (web/dreamer.js: a compute shader on the
GPU buffers the cache stays in). Passing the cache through whole was a rewrite of all of it a step (214 MB,
copied about five times: Gather, Where, Concat, Transpose), 40% of a step's GPU time on a 3090, more on an
integrated GPU's shared memory. Masks are additive (a boolean mask's softmax guard runs on the CPU in
onnxruntime-web).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from token_world.data.nes_layers import BANDS, CELL, LAYER_COLOURS, TRANSPARENT, VIEW
from token_world.models.layered import BORDER_TOKENS, CELLS, rotary
from token_world.models.layered_pixels import SPRITE_TOKENS, PixelLayers

INPUTS = ("cells", "cell_known", "sprite_layer", "border", "backdrop")      # embed.onnx's, in this order


def held_camera(model: PixelLayers, layers: dict) -> dict:
    """A start's layered frames (each key [T, ...]) -> its camera fields for the graphs' constants; the camera
    must be the same in every frame, with no fine scroll (the picture is composed from whole cells)."""
    camera = torch.as_tensor(layers["camera"]).long()
    if not (camera == camera[-1]).all():
        raise ValueError("the start's camera moves: the export is for a fixed camera")
    if (camera[-1, :, 0] % CELL).any():
        raise ValueError("the start's camera has a fine scroll")
    return {"camera": camera[-1], "cell_col": torch.as_tensor(layers["cell_col"])[-1].long()}


class Constants(nn.Module):
    """The held camera's place for every token (and its screen place, for space_rope)."""

    def __init__(self, model: PixelLayers, camera: dict):
        super().__init__()
        self.model = model
        frame = {k: v[None, None] for k, v in camera.items()}
        self.register_buffer("place", model.place(frame)[0, 0])                       # [border_end, dim]
        if model.space_rope:
            screen, placed = model.screen(frame)
            self.register_buffer("screen", screen[0, 0])                              # [tokens, 2]
            self.register_buffer("placed", placed)
        else:
            self.screen = self.placed = None

    def run(self, x: torch.Tensor, temporal) -> torch.Tensor:
        """x [1, T, tokens, dim] -> the model's features: each block's spatial attention, `temporal(i, layer,
        normed)` (per token over frames), its MLP; then the final norm."""
        model = self.model
        b, t, n, d = x.shape
        place = None if self.screen is None else self.screen.expand(b * t, -1, -1)
        for i, block in enumerate(model.blocks):
            x = x + block.spatial(block.norm_s(x).reshape(b * t, n, d), place, self.placed).view(b, t, n, d)
            x = x + temporal(i, block.temporal, block.norm_t(x))
            x = x + block.mlp(block.norm_m(x))
        return model.norm(x)


def heads(layer, h: torch.Tensor, times: torch.Tensor):
    """A temporal layer's queries, keys and values for x [1, T, n, d] at frames `times` -> each [n, heads, T,
    head_dim], queries and keys rotated by their frame."""
    b, t, n, d = h.shape
    q, k, v = layer.qkv(h).view(b, t, n, 3, layer.heads, d // layer.heads).unbind(3)
    if layer.q_norm is not None:
        q, k = layer.q_norm(q), layer.k_norm(k)
    q, k = rotary(q, times), rotary(k, times)
    each = lambda z: z[0].permute(1, 2, 0, 3)
    return each(q), each(k), each(v)


def merge(layer, y: torch.Tensor) -> torch.Tensor:
    """[n, heads, T, head_dim] -> the layer's output [1, T, n, d]."""
    n, h, t, hd = y.shape
    return layer.out(y.permute(2, 0, 1, 3).reshape(1, t, n, h * hd))


class EmbedGraph(nn.Module):
    def __init__(self, model: PixelLayers):
        super().__init__()
        self.model = model

    def forward(self, cells, cell_known, sprite_layer, border, backdrop):
        t = cells.shape[0]
        layers = {"cells": cells[None], "cell_known": cell_known[None], "sprite_layer": sprite_layer[None],
                  "border": border[None], "backdrop": backdrop[None],
                  "camera": torch.zeros(1, t, BANDS, 2, dtype=torch.int32)}               # (its shape only)
        return self.model.content(layers)[0]


class PrefillGraph(Constants):
    def forward(self, content, actions):
        model, t = self.model, content.shape[0]
        hidden = torch.zeros(1, t, model.tokens, dtype=torch.bool)
        x = model.join(content[None], self.place.expand(1, t, -1, -1), actions[None], hidden)
        times = torch.arange(t)
        later = times[None, :] > times[:, None]
        causal = torch.zeros(t, t).masked_fill(later, float("-inf"))
        keys, values = [], []

        def temporal(i, layer, h):
            q, k, v = heads(layer, h, times)
            pad = (0, 0, 0, model.frames - t)
            keys.append(F.pad(k, pad).transpose(-1, -2))
            values.append(F.pad(v, pad))
            return merge(layer, F.scaled_dot_product_attention(q, k, v, attn_mask=causal))

        self.run(x, temporal)
        return (*keys, *values)


class StepGraph(Constants):
    def __init__(self, model: PixelLayers, camera: dict, palette: torch.Tensor):
        super().__init__(model, camera)
        colours = torch.zeros(LAYER_COLOURS, 3)               # each layer colour's RGB; TRANSPARENT's is unused
        colours[:TRANSPARENT] = palette.float()[:TRANSPARENT]
        self.register_buffer("colours", colours)

    def forward(self, previous, actions, at, *cache):
        model = self.model
        keys, values = cache[:len(model.blocks)], cache[len(model.blocks):]
        hidden = torch.zeros(1, 2, model.tokens, dtype=torch.bool)
        hidden[0, 1] = True
        hidden[0, 1, model.camera_at:model.frame_at] = False                            # the camera is no content
        pair = torch.stack((previous, torch.zeros_like(previous)))[None]
        x = model.join(pair, self.place.expand(1, 2, -1, -1), actions.view(1, 2), hidden)
        times = at.view(()) - 1 + torch.arange(2, dtype=at.dtype)
        positions = torch.arange(model.frames, dtype=at.dtype)
        earlier = torch.zeros(model.frames).masked_fill(positions >= times[0], float("-inf"))   # the cache's
        own = torch.tensor([[0.0, float("-inf")], [0.0, 0.0]])                     # the pair: causal
        new_keys, new_values = [], []

        def temporal(i, layer, h):
            q, k, v = heads(layer, h, times)                                          # [n, heads, 2, hd]
            new_keys.append(k[:, :, 0])                         # the previous frame's, for the cache at at - 1
            new_values.append(v[:, :, 0])
            scale = q.shape[-1] ** -0.5
            weights = torch.cat((q @ keys[i] * scale + earlier, q @ k.transpose(-1, -2) * scale + own), -1)
            weights = weights.softmax(-1)
            return merge(layer, weights[..., :model.frames] @ values[i] + weights[..., model.frames:] @ v)

        f = self.run(x, temporal)[0, 1]                                                 # the new frame's
        cells = model.pixel_logits(f[:CELLS]).float().softmax(-1)                       # [CELLS, 256, C]
        sprites = model.pixel_logits(f[model.sprite_at:model.camera_at], sprite=True).float().softmax(-1)
        border = model.pixel_logits(f[model.border_at:model.border_end]).float().softmax(-1)
        backdrop = first_max(model.backdrop_head(f[model.frame_at]))                    # [C] one-hot
        # its content, as the Dreamer's commit pass keeps it: the pixels soft, the backdrop decided
        soft = lambda probs, n: model.embed_probs(probs).view(n, CELL, CELL, -1)
        conv, _ = model.sprite_kind()
        content = torch.cat((model._pixels(soft(cells, CELLS), model.cell_patch),
                             model._pixels(soft(sprites, SPRITE_TOKENS), conv),
                             f.new_zeros(model.frame_at - model.camera_at, f.shape[-1]),
                             (backdrop @ model.backdrop.weight)[None],
                             model._pixels(soft(border, BORDER_TOKENS), model.cell_patch)))
        # its picture: the background's cells (backdrop where transparent), the sprites over them, the border
        # bands above and below NES lines 8-231 (data/nes_layers.py compose, model_frame; a transparent border
        # pixel is colour 0), each pixel its most likely colour; all in float, so the step stays on the GPU
        # (WebGPU runs no int64: ArgMax's indices sent the composition to the CPU, a round trip a frame)
        def paint(probs):                                                                # -> RGB, transparent
            pick = first_max(probs)
            return pick @ self.colours, pick[..., TRANSPARENT:TRANSPARENT + 1]

        rgb, clear = paint(cells)                                                       # [CELLS, 256, 3], [.., 1]
        lay = lambda z: z.view(BANDS, VIEW, CELL, CELL, -1).permute(0, 2, 1, 3, 4).reshape(BANDS, CELL, VIEW * CELL, -1)
        background = lay(rgb + clear * (backdrop @ self.colours))[:, :, :CELL * (VIEW - 1)].reshape(
            BANDS * CELL, CELL * (VIEW - 1), 3)
        rgb, clear = paint(sprites)
        picture = pictures(rgb) + pictures(clear) * background
        rgb, clear = paint(border)
        bands = pictures(rgb + clear * self.colours[0])
        frame = torch.cat((bands[:CELL], picture[8:232], bands[CELL:]))
        return frame, content, torch.stack(new_keys), torch.stack(new_values)


def first_max(scores: torch.Tensor) -> torch.Tensor:
    """[..., C] -> [..., C] float one-hot of the largest (the first at the maximum, as argmax), in float ops."""
    hit = (scores >= scores.amax(-1, keepdim=True)).float()
    ranked = hit * torch.arange(scores.shape[-1], 0, -1, dtype=hit.dtype)               # earlier colours higher
    return (ranked >= ranked.amax(-1, keepdim=True)).float()


def pictures(tokens: torch.Tensor) -> torch.Tensor:
    """16 x 16 patches with channels [n, 256, k] (row-major, 16 across) -> the picture [n / 16 * 16, 256, k]."""
    rows, k = tokens.shape[0] // 16, tokens.shape[-1]
    return tokens.view(rows, 16, CELL, CELL, k).permute(0, 2, 1, 3, 4).reshape(rows * CELL, 16 * CELL, k)


def cache_layout(model: PixelLayers) -> dict:
    """The cache the graphs read, one keys and one values tensor a layer (keys transposed, as attention reads
    them), and their names in the graphs."""
    layer = model.blocks[0].temporal
    n, h, f, hd = model.tokens, layer.heads, model.frames, model.dim // layer.heads
    count = len(model.blocks)
    return {"layers": count, "keys": [n, h, hd, f], "values": [n, h, f, hd],
            "names": [f"keys_{i}" for i in range(count)] + [f"values_{i}" for i in range(count)]}


@torch.no_grad()
def export(model: PixelLayers, folder: Path, palette: torch.Tensor, start: dict) -> None:
    """Write embed.onnx, prefill.onnx, step.onnx and model.json for a CPU float32 copy of `model` in eval mode;
    palette [colours, 3] is the game's; start is the page's start (layered frames, each key [T, ...]), whose
    camera the dreams hold."""
    if not isinstance(model, PixelLayers):
        raise NotImplementedError("the export is for the sprite-picture model (models/layered_pixels.py)")
    model = copy.deepcopy(model).float().cpu().eval()
    camera = held_camera(model, start)
    folder.mkdir(parents=True, exist_ok=True)
    cache = cache_layout(model)
    n, dim = model.border_end, model.dim
    t = torch.export.Dim("frames", min=2, max=model.frames - 1)
    example = model.frames // 2
    pixels = lambda *s: torch.zeros(example, *s, dtype=torch.uint8)
    torch.onnx.export(EmbedGraph(model).eval(),
                      (pixels(BANDS, VIEW, CELL, CELL), pixels(BANDS, VIEW, CELL, CELL), pixels(240, 256),
                       pixels(2 * CELL, 256), pixels()), folder / "embed.onnx", input_names=list(INPUTS),
                      output_names=["content"], dynamic_shapes={k: {0: t} for k in INPUTS},
                      dynamo=True, external_data=False, verbose=False)
    prefill = model.frames * 3 // 4 - 1                   # keep - 1
    torch.onnx.export(PrefillGraph(model, camera).eval(),
                      (torch.zeros(prefill, n, dim), torch.zeros(prefill, dtype=torch.int32)),
                      folder / "prefill.onnx", input_names=["content", "actions"], output_names=cache["names"],
                      dynamo=True, external_data=False, verbose=False)
    empty = [torch.zeros(cache[name.split("_")[0]]) for name in cache["names"]]
    torch.onnx.export(StepGraph(model, camera, palette).eval(),
                      (torch.zeros(n, dim), torch.zeros(2, dtype=torch.int32), torch.tensor([example], dtype=torch.int32),
                       *empty), folder / "step.onnx",
                      input_names=["previous", "actions", "at", *cache["names"]],
                      output_names=["rgb", "content", "new_keys", "new_values"], dynamo=True, external_data=False,
                      verbose=False)
    (folder / "model.json").write_text(json.dumps({"cache": cache, "frames": model.frames, "content": [n, dim],
                                                   "keep": model.frames * 3 // 4, "prefill": prefill,
                                                   "inputs": list(INPUTS),
                                                   "parameters": sum(p.numel() for p in model.parameters())},
                                                  indent=1))
