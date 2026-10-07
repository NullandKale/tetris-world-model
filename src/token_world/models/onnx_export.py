"""The layered world model (models/layered_pixels.py) as three ONNX graphs, for dreaming in the browser on WebGPU
(web/) and in Python (models/onnx_dreamer.py). A frame is kept in the context as its content: what each token
holds before the model adds its place (models/layered.py LayeredBase.content: the soft pixel tokens and the
backdrop colour), [n, dim] float32.

    embed.onnx    a real frame's layers, uint8: cells and cell_known [T, BANDS, VIEW, 16, 16], sprite_layer
                  [T, 240, 256], border [T, 32, 256], backdrop [T] -> content [T, n, dim]
    prefill.onnx  content [T, n, dim], actions [T] int32 (incoming, -1 = none) -> keys, values [layers,
                  tokens, heads, frames, head_dim]: positions 0..T-1, the rest zero (2 <= T < frames)
    step.onnx     previous [n, dim] (the last frame's content), actions [2] int32 (the previous frame's, the
                  new one's), at [1] int32 (the new frame's position), keys, values (positions before at - 1)
                  -> rgb [256, 256, 3] float32 (the new model frame: its layers composed, each pixel's most
                  likely colour, through the palette in the graph), content [n, dim] (the new frame's, soft, as
                  the Dreamer's commit pass keeps it), new_keys, new_values (the previous frame at at - 1)

A dream holds its camera, the start frame's (the page's start is one frame's layers; docs/guides/
layered_tokens.md): the export is for a game whose camera does not move, as Tetris (a model trained with
fixed_camera, or one whose camera head only learned "no move": the first layered-pixels run's never moved it
in 1,024 dream frames). Every token then keeps its identity, so the routed temporal attention (RoutedTemporal)
is attention per token over the earlier frames, and each token's place is a constant of the graphs.

One step is one pass over two frames, as the pixel model's export was: the previous frame, written into the
cache at at - 1 from its soft content (the Dreamer's commit pass), and the hidden new frame at `at` (its decide
pass), which sees it. So prefill takes every frame but the last, which is the first step's `previous`. The
cache goes in and comes out whole, so a WebGPU host keeps it in GPU buffers; masks are additive (a boolean
mask's softmax guard runs on the CPU in onnxruntime-web).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from token_world.data.nes_layers import BANDS, CELL, TRANSPARENT, VIEW
from token_world.models.layered import BORDER_TOKENS, CELLS, rotary, unpatch
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
            keys.append(F.pad(k, pad))
            values.append(F.pad(v, pad))
            return merge(layer, F.scaled_dot_product_attention(q, k, v, attn_mask=causal))

        self.run(x, temporal)
        return torch.stack(keys), torch.stack(values)


class StepGraph(Constants):
    def __init__(self, model: PixelLayers, camera: dict, palette: torch.Tensor):
        super().__init__(model, camera)
        self.register_buffer("palette", palette.float())

    def forward(self, previous, actions, at, keys, values):
        model = self.model
        hidden = torch.zeros(1, 2, model.tokens, dtype=torch.bool)
        hidden[0, 1] = True
        hidden[0, 1, model.camera_at:model.frame_at] = False                            # the camera is no content
        pair = torch.stack((previous, torch.zeros_like(previous)))[None]
        x = model.join(pair, self.place.expand(1, 2, -1, -1), actions.view(1, 2), hidden)
        times = at.view(()) - 1 + torch.arange(2, dtype=at.dtype)
        positions = torch.arange(model.frames, dtype=at.dtype)
        seen = torch.zeros(2, model.frames).masked_fill(positions[None] > times[:, None], float("-inf"))
        new_keys, new_values = [], []

        def temporal(i, layer, h):
            q, k, v = heads(layer, h, times)
            kk, vv = keys[i], values[i]
            for j in range(2):                                   # the previous frame at at - 1, the new at at
                here = (positions == times[j]).view(1, 1, -1, 1)
                kk, vv = torch.where(here, k[:, :, j:j + 1], kk), torch.where(here, v[:, :, j:j + 1], vv)
            new_keys.append(kk)
            new_values.append(vv)
            return merge(layer, F.scaled_dot_product_attention(q, kk, vv, attn_mask=seen))

        f = self.run(x, temporal)[0, 1]                                                 # the new frame's
        cells = model.pixel_logits(f[:CELLS]).float().softmax(-1)                       # [CELLS, 256, C]
        sprites = model.pixel_logits(f[model.sprite_at:model.camera_at], sprite=True).float().softmax(-1)
        border = model.pixel_logits(f[model.border_at:model.border_end]).float().softmax(-1)
        backdrop = model.backdrop_head(f[model.frame_at]).argmax(-1)
        # its content, as the Dreamer's commit pass keeps it: the pixels soft, the backdrop decided
        soft = lambda probs, n: model.embed_probs(probs).view(n, CELL, CELL, -1)
        conv, _ = model.sprite_kind()
        content = torch.cat((model._pixels(soft(cells, CELLS), model.cell_patch),
                             model._pixels(soft(sprites, SPRITE_TOKENS), conv),
                             f.new_zeros(model.frame_at - model.camera_at, f.shape[-1]),
                             model.backdrop(backdrop)[None],
                             model._pixels(soft(border, BORDER_TOKENS), model.cell_patch)))
        # its picture: the background's cells (backdrop where transparent), the sprites over them, the border
        # bands above and below NES lines 8-231 (data/nes_layers.py compose, model_frame)
        strip = cells.argmax(-1).view(BANDS, VIEW, CELL, CELL).permute(0, 2, 1, 3).reshape(BANDS, CELL, VIEW * CELL)
        picture = strip[:, :, :CELL * (VIEW - 1)].reshape(BANDS * CELL, CELL * (VIEW - 1))
        picture = torch.where(picture != TRANSPARENT, picture, backdrop)
        drawn = unpatch(sprites.argmax(-1).view(SPRITE_TOKENS, CELL, CELL))
        picture = torch.where(drawn != TRANSPARENT, drawn, picture)
        bands = unpatch(border.argmax(-1).view(BORDER_TOKENS, CELL, CELL))
        frame = torch.cat((bands[:CELL], picture[8:232], bands[CELL:]))
        frame = torch.where(frame != TRANSPARENT, frame, torch.zeros_like(frame))
        rgb = F.embedding(frame, self.palette)
        return rgb, content, torch.stack(new_keys), torch.stack(new_values)


def cache_shape(model: PixelLayers) -> list[int]:
    """[layers, tokens, heads, frames, head_dim]: the cache the graphs pass around."""
    layer = model.blocks[0].temporal
    return [len(model.blocks), model.tokens, layer.heads, model.frames, model.dim // layer.heads]


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
    shape = cache_shape(model)
    n, dim = model.border_end, model.dim
    t = torch.export.Dim("frames", min=2, max=model.frames - 1)
    example = model.frames // 2
    pixels = lambda *s: torch.zeros(example, *s, dtype=torch.uint8)
    torch.onnx.export(EmbedGraph(model).eval(),
                      (pixels(BANDS, VIEW, CELL, CELL), pixels(BANDS, VIEW, CELL, CELL), pixels(240, 256),
                       pixels(2 * CELL, 256), pixels()), folder / "embed.onnx", input_names=list(INPUTS),
                      output_names=["content"], dynamic_shapes={k: {0: t} for k in INPUTS},
                      dynamo=True, external_data=False, verbose=False)
    torch.onnx.export(PrefillGraph(model, camera).eval(),
                      (torch.zeros(example, n, dim), torch.zeros(example, dtype=torch.int32)),
                      folder / "prefill.onnx", input_names=["content", "actions"], output_names=["keys", "values"],
                      dynamic_shapes={"content": {0: t}, "actions": {0: t}}, dynamo=True, external_data=False,
                      verbose=False)
    cache = torch.zeros(shape)
    torch.onnx.export(StepGraph(model, camera, palette).eval(),
                      (torch.zeros(n, dim), torch.zeros(2, dtype=torch.int32), torch.tensor([example], dtype=torch.int32),
                       cache, cache.clone()), folder / "step.onnx",
                      input_names=["previous", "actions", "at", "keys", "values"],
                      output_names=["rgb", "content", "new_keys", "new_values"], dynamo=True, external_data=False,
                      verbose=False)
    (folder / "model.json").write_text(json.dumps({"cache": shape, "frames": model.frames, "content": [n, dim],
                                                   "keep": model.frames * 3 // 4, "inputs": list(INPUTS),
                                                   "parameters": sum(p.numel() for p in model.parameters())},
                                                  indent=1))
