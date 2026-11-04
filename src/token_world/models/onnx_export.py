"""The pixel world model (models/dynamics.py Dynamics, motion first) as three ONNX graphs, for dreaming in the
browser on WebGPU (web/) and in Python (models/onnx_dreamer.py). A frame is kept in the context as its content,
its patch tokens before the model adds anything to them (the patch conv of each pixel's colour embedding; a dreamed
frame's soft: its colour probabilities through the colour table, Dynamics.soft_tokens) [N, dim] float32, and its
motion, each token's 8 x 8 cells' codes (data/tetris_motion.py) [N, cells] float32, 255 where not known (a start's
real frames: the page ships no RAM, the motion's source).

    embed.onnx    real frames [T, 256, 256] uint8 (palette indices; a two-layer model's [T, 2, 256, 256], each
                  frame's background and sprite layers) -> content [T, N, dim], and last [layers, N, P*P] float32,
                  the last frame's colours (each layer's), which the first step reads
    prefill.onnx  content [P, N, dim], motion [P, N, cells], actions [P] int32 (incoming, -1 = none) -> the cache,
                  each layer's keys_i [tokens, heads, head_dim, frames] (transposed, as attention reads them) and
                  values_i [tokens, heads, frames, head_dim], frames 0..P-1 at slots 0..P-1, the rest zero. P is
                  fixed, frames - 2 (the longest start, frames - 1, but its last frame, which the first step reads
                  as `previous`): a start with fewer frames is padded after them, which no real frame sees (causal)
                  and the steps write over (a dynamic length put its shape arithmetic on the CPU in onnxruntime-web)
    step.onnx     previous [N, dim] and previous_motion [N, cells] (the last frame's), actions [2] int32 (the
                  previous frame's, the new one's), at [1] int32 (the new frame: frames since the start's first),
                  last [layers, N, P*P] (the previous frame's colours, each layer's), a frame-skip model's
                  skipped [2, skip - 1] int32 (each of the two frames' step's earlier bytes, newest first, -1 past
                  its frames) and clock [2] int32 (their frames on the frame clock: what rotary turns by, while
                  `at` places them in the ring), the cache (keys_i, values_i: a ring, frame t at slot t mod frames, holding the frames up to at - 2) -> rgb
                  [256, 256, 3] float32 (the new frame, each pixel's most likely colour through the palette in the
                  graph), content [N, dim], motion [N, cells] and last_out (the new frame's), new_keys, new_values
                  [layers, tokens, heads, head_dim]: the previous frame's, which the host writes into the cache at
                  slot (at - 1) mod frames

A step is FrameDecoder's frame with one motion step (Dynamics.motion_steps = 1: at s9_cool140 152k one step dreamed
as two, 2026-10-18, docs/guides/recipe_attempts.md): pass 1 is over two frames, the previous one with its motion
(the decoder's pass that writes a frame into the cache) and the new one hidden at `at`, and gives each of the new
frame's cells its most likely motion (the border's token rows STILL, Dynamics.drawn_motion); pass 2 is over the new
frame alone, hidden with that motion added, attending to the cache and to the previous frame's keys and values from
pass 1 (a frame never sees a later one, so its keys are the same in both), and gives each pixel's colour
probabilities: the picture (most likely colours) and the soft content. Three frames' work a step, as the decoder's
three passes. A model trained with a change weight w has the odds of each game pixel's change from `last` times
1 / w (FrameDecoder's weigh_changes, undoing the weighted loss; a two-layer model's sprite layer weighed as its
background). A two-layer model (two_layers) has two pixel heads, the background's and the sprite layer's: the
picture is their composite (compose_probs) and the content the sum of both layers' soft pixels, as FrameDecoder.

With conv ends (conv_ends) both pixel ends are pyramids of 2 x 2 steps: a token's pixels are kept in the
pyramids' nested order (nested_order: each 2 x 2 group's four together, then each group of groups'), so the
head's output goes into the content's pyramid with no reordering, and only the picture is put back in rows.

The cache is the Dreamer's ring (models/dynamics.py TemporalCache): nothing is ever re-encoded, and each frame
attends to its own and the frames - 1 before it, exactly as there (the previous frame one older slot than the new
one; each slot's frame worked out from `at`, in float: exact to 2^24 frames, 77 hours at 60 a second).

A step attends over the cache's earlier frames and its own together (one softmax over all) and never copies the
cache: the host writes the previous frame's keys and values in place (web/dreamer.js: a compute shader on the GPU
buffers the cache stays in). Passing the cache through whole was a rewrite of all of it a step, 40% of a step's
GPU time on a 3090, more on an integrated GPU's shared memory. Masks are additive (a boolean mask's softmax guard
runs on the CPU in onnxruntime-web); argmax is float one-hots (WebGPU runs no int64: ArgMax's indices sent the work
to the CPU, a round trip a frame).

The graphs are plain ONNX ops, the RMSNorms fused (fuse_norms) and nothing else. A rewrite with folded constants,
fused residual norms (SkipLayerNormalization) and com.microsoft MultiHeadAttention ran 57 frames/s against 41 on a
desktop 3090, but slower on a laptop 3070 (30 against 40), and a phone's WebGPU could not build its shaders
("Failed to create a WebGPU compute pipeline", 2026-11-02 and again 11-03): test a graph change on the user's
devices before it ships.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from token_world.models.dynamics import (COLOURS, MOTION_CODES, MOTION_STILL, MOTION_UNKNOWN, ROPE_BASE, SIZE,
                                         TRANSPARENT, Dynamics, button_bits, patch_pixels, pixel_logits)


def first_max(scores: torch.Tensor) -> torch.Tensor:
    """[..., C] -> [..., C] float one-hot of the largest (the first at the maximum, as argmax), in float ops."""
    hit = (scores >= scores.amax(-1, keepdim=True)).float()
    ranked = hit * torch.arange(scores.shape[-1], 0, -1, dtype=hit.dtype)               # earlier ones higher
    return (ranked >= ranked.amax(-1, keepdim=True)).float()


def pairing(head_dim: int, pairs: int) -> torch.Tensor:
    """[head_dim, head_dim]: z @ it holds, for each rotated pair of channels (i, pairs + i), -z[pairs + i] at i and
    z[i] at pairs + i, and 0 on the channels not rotated, so a rotation by an angle is z * cos + (z @ it) * sin
    (cos 1 and sin 0 where not rotated): one product and three element-wise kernels for queries and keys together,
    where slicing the halves apart took about twenty a layer."""
    out = torch.zeros(head_dim, head_dim)
    i = torch.arange(pairs)
    out[pairs + i, i] = -1.0
    out[i, pairs + i] = 1.0
    return out


def rotation(times: torch.Tensor, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Frames `times` [T] -> the cos and sin [T, 1, 1, 1, head_dim] that rotate their queries and keys
    (dynamics.rotary's: channel pairs (i, head_dim / 2 + i)) with `pairing`, made once for every layer of a pass."""
    half = head_dim // 2
    angle = times.float()[:, None] * ROPE_BASE ** -(torch.arange(half, dtype=torch.float32) / half)
    angle = torch.cat((angle, angle), -1)[:, None, None, None]
    return angle.cos(), angle.sin()


def merge(layer, y: torch.Tensor) -> torch.Tensor:
    """[n, heads, T, head_dim] -> the layer's output [T, n, d]."""
    n, h, t, hd = y.shape
    return layer.out(y.permute(2, 0, 1, 3).reshape(t, n, h * hd))


def norm_eps(norm) -> float:
    return norm.eps if norm.eps is not None else torch.finfo(torch.float32).eps


def rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm over the last axis as Pow, ReduceMean, Add, Sqrt, Div and Mul by the weight, which fuse_norms makes
    one kernel. ONNX before opset 23 has no RMSNorm, and torch's (or an rsqrt) exports as seven kernels: the
    QK-norms and the conv ends' norms were 315 of a step's 1,680 (s9_postC, 2026-11-02)."""
    return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def fuse_norms(path: Path) -> int:
    """Each `rms` in the ONNX graph at `path` -> one SimplifiedLayerNormalization (onnxruntime's RMSNorm, which its
    WebGPU provider runs as one LayerNorm kernel, as the training's fused norm, models/norm_kernel.py). Its
    optimiser makes the same fusion on the CPU but not on WebGPU (the browser ran all six ops, 2026-11-02).
    Rewrites the file; -> the norms fused."""
    import onnx
    from onnx import helper, numpy_helper

    graph_path = str(path)
    proto = onnx.load(graph_path)
    graph = proto.graph
    values = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    users: dict[str, list] = {}
    for node in graph.node:
        for name in node.input:
            users.setdefault(name, []).append(node)

    def only(node, op: str):
        after = users.get(node.output[0], [])
        return after[0] if len(after) == 1 and after[0].op_type == op else None

    gone, fused = set(), []
    for pow_ in graph.node:
        if pow_.op_type != "Pow" or values.get(pow_.input[1]) is None or float(values[pow_.input[1]]) != 2.0:
            continue
        mean = only(pow_, "ReduceMean")
        add = mean and only(mean, "Add")
        sqrt = add and only(add, "Sqrt")
        div = sqrt and only(sqrt, "Div")
        mul = div and only(div, "Mul")
        if not mul or div.input[0] != pow_.input[0] or list(values.get(mean.input[1], [])) != [-1]:
            continue
        eps = values.get(add.input[1])
        weight = next((name for name in mul.input if name != div.output[0]), None)
        if eps is None or eps.size != 1 or weight not in values:
            continue
        gone.update(id(n) for n in (pow_, mean, add, sqrt, div, mul))
        fused.append((mul, helper.make_node("SimplifiedLayerNormalization", [pow_.input[0], weight], [mul.output[0]],
                                            name=f"{mul.name}_rms", axis=-1, epsilon=float(eps), stash_type=1)))
    at_mul = {id(mul): node for mul, node in fused}
    nodes = [at_mul.get(id(n), n) for n in graph.node if id(n) not in gone or id(n) in at_mul]
    del graph.node[:]
    graph.node.extend(nodes)
    used = {name for node in nodes for name in node.input}           # the fused ops' constants go too
    kept = [i for i in graph.initializer if i.name in used or i.name in {o.name for o in graph.output}]
    del graph.initializer[:]
    graph.initializer.extend(kept)
    onnx.save(proto, graph_path)
    return len(fused)


def rms_gelu(x: torch.Tensor, norm) -> torch.Tensor:
    """F.gelu(norm(x)) for an RMSNorm module, the conv ends' step between maps (dynamics.norm_gelu): two kernels."""
    return F.gelu(rms(x, norm.weight, norm_eps(norm)))


class Graph(nn.Module):
    """What the three graphs share: a frame's tokens from its content, motion and buttons, and the blocks.

    The attentions are the model's, written out for the browser's kernel count (each kernel costs a dispatch in
    onnxruntime-web, which bounded a step at about 2,800 of them): queries and keys normalised together (QK-norm,
    one fused norm with unit weights, then each one's weights) and rotated together (pairing: by frame in the
    temporal attention, by place in the spatial one, RoPE-Mixed's angles fixed for the graph), and each layer's
    local bias gathered once for both of a step's passes. The RMSNorms are `rms`, which onnxruntime fuses."""

    def __init__(self, model: Dynamics):
        super().__init__()
        self.model = model
        border = torch.zeros(model.grid, model.grid)
        border[0] = border[-1] = 1
        self.register_buffer("border", border.flatten()[:, None, None])                 # [N, 1, 1] the border rows
        self.register_buffer("codes", torch.arange(MOTION_CODES).float())
        self.register_buffer("still", F.one_hot(torch.tensor(MOTION_STILL), MOTION_CODES).float())
        first = model.blocks[0]
        self.heads_, self.head_dim = first.temporal.heads, first.temporal.qkv.in_features // first.temporal.heads
        self.register_buffer("unit", torch.ones(self.head_dim))                         # the QK-norms' fused part
        self.register_buffer("temporal_pairs", pairing(self.head_dim, self.head_dim // 2))
        self.locals = first.spatial.local
        p = model.patch
        nest = model.nest if model.conv_ends else torch.arange(p * p)          # the graph's order of a token's pixels
        self.register_buffer("unnest", nest.argsort())                          # each row-major place's
        undo = model.change_undo("cpu")                                          # [N, P*P, 1] or None
        self.weighs = undo is not None
        if self.weighs:
            self.register_buffer("undo", undo[:, nest])
        if model.conv_ends:                       # the first step, 2 x 2 pixels (ky, kx, colour) -> its width
            self.register_buffer("patch_weight", model.patchify.weight.permute(0, 2, 3, 1).flatten(1).clone())
        if self.locals:                                   # one offsets table for every layer
            self.register_buffer("offsets", first.spatial.offsets.clone())
        for i, block in enumerate(model.blocks):
            for kind, attn in (("s", block.spatial), ("t", block.temporal)):
                if attn.q_norm is not None:               # [2, 1, head_dim]: the queries' and the keys' weights
                    self.register_buffer(f"qk_{kind}{i}", torch.stack((attn.q_norm.weight, attn.k_norm.weight))[:, None])
            spatial = block.spatial
            if spatial.rope:                              # RoPE-Mixed at each token's place, [N, 1, heads, hd]
                angle = torch.einsum("nc,hpc->nhp", model.place[0].float(), spatial.rope_freq.float())
                pairs = angle.shape[-1]
                cos, sin = torch.ones(angle.shape[0], self.heads_, self.head_dim), torch.zeros(angle.shape[0],
                                                                                              self.heads_, self.head_dim)
                cos[..., :pairs] = cos[..., pairs:2 * pairs] = angle.cos()
                sin[..., :pairs] = sin[..., pairs:2 * pairs] = angle.sin()
                self.register_buffer(f"space_cos{i}", cos[:, None])
                self.register_buffer(f"space_sin{i}", sin[:, None])
                self.register_buffer("space_pairs", pairing(self.head_dim, pairs))

    def buttons(self, actions: torch.Tensor, skipped: torch.Tensor | None = None) -> torch.Tensor:
        """Each frame's button embedding [T, d] from its incoming byte [T] and, a frame-skip model's, its step's
        earlier bytes [T, skip - 1] (None: none, every frame a one-frame step; Dynamics.forward)."""
        model = self.model
        action = model.action(button_bits(actions))
        if model.skip > 1:
            if skipped is None:
                skipped = torch.full((actions.shape[0], model.skip - 1), -1, dtype=actions.dtype)
            action = action + model.skipped_action(button_bits(skipped).flatten(-2))
        return action

    def biases(self) -> list:
        """Each layer's local bias [heads, N, N] (SpatialAttention), gathered once a graph; None without."""
        return [block.spatial.local_bias[:, self.offsets] if self.locals else None for block in self.model.blocks]

    def qk(self, i: int, kind: str, attn, x: torch.Tensor):
        """An attention's queries and keys together [T, N, 2, heads, hd], normalised (QK-norm), and its values
        [T, N, heads, hd], for x [T, N, d]."""
        t, n, d = x.shape
        qk, v = attn.qkv(x).view(t, n, 3, self.heads_, self.head_dim).split([2, 1], 2)
        if attn.q_norm is not None:
            qk = rms(qk, self.unit, norm_eps(attn.q_norm)) * getattr(self, f"qk_{kind}{i}")
        return qk, v.squeeze(2)

    def heads(self, i: int, layer, h: torch.Tensor, turn: tuple[torch.Tensor, torch.Tensor]):
        """A temporal layer's queries, keys and values for h [T, n, d], rotated by `turn` (rotation) -> each [n,
        heads, T, head_dim]."""
        qk, v = self.qk(i, "t", layer, h)
        cos, sin = turn
        qk = (qk * cos + (qk @ self.temporal_pairs) * sin).permute(2, 1, 3, 0, 4)       # [2, n, heads, T, hd]
        return qk[0], qk[1], v.permute(1, 2, 0, 3)

    def spatial(self, i: int, attn, x: torch.Tensor, bias) -> torch.Tensor:
        """A block's spatial attention within each frame (SpatialAttention) for x [T, N, d]."""
        t, n, d = x.shape
        qk, v = self.qk(i, "s", attn, x)
        if attn.rope:
            qk = qk * getattr(self, f"space_cos{i}") + (qk @ self.space_pairs) * getattr(self, f"space_sin{i}")
        qk = qk.permute(2, 0, 3, 1, 4)                                                    # [2, T, heads, N, hd]
        scores = qk[0] @ qk[1].transpose(-1, -2) * self.head_dim ** -0.5
        if bias is not None:
            scores = scores + bias
        y = scores.softmax(-1) @ v.transpose(1, 2)                                        # [T, heads, N, hd]
        return attn.out(y.transpose(1, 2).reshape(t, n, d))

    def content(self, pixels: torch.Tensor) -> torch.Tensor:
        """Each pixel's colour embedding [N, P*P, colour_dim] in the graph's order -> the tokens [N, dim] (the
        patch conv; with conv ends its pyramid, Dynamics._steps)."""
        model, n = self.model, pixels.shape[0]
        if model.conv_ends is None:
            return F.linear(pixels.transpose(1, 2).flatten(1), model.patchify.weight.flatten(1), model.patchify.bias)
        x = F.linear(pixels.reshape(n, -1, 4 * pixels.shape[-1]), self.patch_weight, model.patchify.bias)
        for norm, step in zip(model.patch_norms, model.patch_steps):
            x = step(rms_gelu(x, norm).reshape(n, -1, 4 * x.shape[-1]))
        return x[:, 0]

    def pixel_probs(self, f: torch.Tensor, sprite: bool = False) -> torch.Tensor:
        """Features [N, dim] -> each pixel's colour probabilities [N, P*P, colours] in the graph's order (sprite:
        the sprite layer's head and colours)."""
        model = self.model
        head = model.sprite_pixel if sprite else model.pixel
        if model.conv_ends is None:
            vectors = head(f).unflatten(-1, (model.patch ** 2, -1))
        else:                                     # down the head's steps (Dynamics.pixel_groups)
            steps, norms = ((model.sprite_pixel_steps, model.sprite_pixel_norms) if sprite and not model.shared_heads
                            else (model.pixel_steps, model.pixel_norms))
            h = f[:, None]
            for step, norm in zip(steps, norms):
                h = rms_gelu(step(h).unflatten(-1, (4, -1)).flatten(-3, -2), norm)
            vectors = head(h).unflatten(-1, (4, -1)).flatten(-3, -2)
        table, bias = (model.sprite_table(), model.sprite_bias) if sprite else (model.colour_table(), model.colour_bias)
        return pixel_logits(vectors, table, bias).softmax(-1)

    def weigh(self, probs: torch.Tensor, last: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """A layer's probabilities [N, P*P, C] and its last colours [N, P*P] -> the probabilities with the odds of
        each change from `last` times the model's undo (weigh_changes; as they are without a change weight), and
        their most likely colours [N, P*P] (the next step's last)."""
        if self.weighs:
            seen = (last[..., None] == torch.arange(probs.shape[-1], dtype=torch.float32)).float()
            probs = probs * (self.undo + (1 - self.undo) * seen)
            probs = probs / probs.sum(-1, keepdim=True)
        return probs, (first_max(probs) * torch.arange(probs.shape[-1], dtype=torch.float32)).sum(-1)

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

    def run(self, x: torch.Tensor, action: torch.Tensor, temporal, biases: list) -> torch.Tensor:
        """x [T, N, d] -> the model's features: each block's spatial attention within each frame (its local bias
        from `biases`), `temporal(i, layer, normed)` (per token over frames), its MLP, the norms set by the
        buttons (action_norm); the final norm (Block.forward)."""
        model = self.model
        for i, block in enumerate(model.blocks):
            norms = (block.norm_s, block.norm_t, block.norm_m)
            if block.modulation is not None:
                mods = block.modulation(action)[:, None].chunk(6, -1)                   # each [T, 1, d]
                norms = tuple((lambda z, norm=norm, scale=mods[2 * j], shift=mods[2 * j + 1]:
                               norm(z) * (1 + scale) + shift) for j, norm in enumerate(norms))
            x = x + self.spatial(i, block.spatial, norms[0](x), biases[i])
            x = x + temporal(i, block.temporal, norms[1](x))
            x = x + block.mlp(norms[2](x))
        return model.norm(x)


class EmbedGraph(nn.Module):
    def __init__(self, model: Dynamics):
        super().__init__()
        self.model = model

    def forward(self, frames):
        model, p = self.model, self.model.patch
        if model.two_layers:                      # the conv is linear: both layers' embeddings, summed
            pixels = (F.embedding(frames[:, 0].int(), model.colour.weight)
                      + F.embedding(frames[:, 1].int(), model.sprite_colour.weight))
            last = torch.stack([patch_pixels(frames[-1, i].float(), p) for i in range(2)])
        else:
            pixels = F.embedding(frames.int(), model.colour.weight)                     # [T, 256, 256, colour_dim]
            last = patch_pixels(frames[-1].float(), p)[None]
        first = model.patchify(pixels.permute(0, 3, 1, 2)).flatten(2).transpose(1, 2)
        nest = model.nest if model.conv_ends else torch.arange(p * p)
        return model._steps(first), last[:, :, nest]


class PrefillGraph(Graph):
    def forward(self, content, motion, actions):
        model, t = self.model, content.shape[0]
        action = self.buttons(actions)                    # a start's real frames: one-frame steps, frames 0..t-1
        x = self.tokens(content, self.motion(self.known(motion)), action)
        times = torch.arange(t)
        turn = rotation(times, self.head_dim)
        later = times[None, :] > times[:, None]
        causal = torch.zeros(t, t).masked_fill(later, float("-inf"))
        keys, values = [], []

        def temporal(i, layer, h):
            q, k, v = self.heads(i, layer, h, turn)
            pad = (0, 0, 0, model.frames - t)
            keys.append(F.pad(k, pad).transpose(-1, -2))
            values.append(F.pad(v, pad))
            return merge(layer, F.scaled_dot_product_attention(q, k, v, attn_mask=causal))

        self.run(x, action, temporal, self.biases())
        return (*keys, *values)


class StepGraph(Graph):
    def __init__(self, model: Dynamics, palette: torch.Tensor):
        super().__init__(model)
        self.register_buffer("colours", palette.float()[:COLOURS])                     # [COLOURS, 3]

    def forward(self, previous, previous_motion, actions, at, last, *rest):
        model = self.model
        skipped, clock, cache = (rest[0], rest[1], rest[2:]) if model.skip > 1 else (None, None, rest)
        keys, values = cache[:len(model.blocks)], cache[len(model.blocks):]
        hidden = model.mask_token.expand_as(previous)
        window = model.frames
        newest = at.view(()).float() - 2                                                # the cache's newest frame
        behind = newest - torch.arange(window, dtype=torch.float32)
        stamp = newest - (behind - torch.floor(behind / window) * window)                # each slot's frame
        rows = at.view(()).float() - 1 + torch.arange(2, dtype=torch.float32)            # the previous frame, the new
        seen = (stamp[None] > rows[:, None] - window) & (stamp[None] >= 0)               # its last frames - 1 before
        earlier = torch.zeros(2, window).masked_fill(~seen, float("-inf"))               # [2, frames]
        head_dim, biases = self.head_dim, self.biases()
        scale = head_dim ** -0.5
        new_keys, new_values = [], []

        # pass 1: the previous frame, with its motion, and the new one hidden -> the new frame's motion
        action = self.buttons(actions, skipped)                                          # [2, d]
        x = self.tokens(torch.stack((previous, hidden)),
                        torch.stack((self.motion(self.known(previous_motion)), torch.zeros_like(previous))), action)
        times = clock if clock is not None else at.view(()) - 1 + torch.arange(2, dtype=at.dtype)
        cos, sin = rotation(times, head_dim)                                              # the pair's frames
        own = torch.tensor([[0.0, float("-inf")], [0.0, 0.0]])                          # the pair: causal

        def first(i, layer, h):
            q, k, v = self.heads(i, layer, h, (cos, sin))                                       # [n, heads, 2, hd]
            new_keys.append(k[:, :, 0])                         # the previous frame's, for the cache at at - 1
            new_values.append(v[:, :, 0])
            weights = torch.cat((q @ keys[i] * scale + earlier, q @ k.transpose(-1, -2) * scale + own), -1).softmax(-1)
            return merge(layer, weights[..., :model.frames] @ values[i] + weights[..., model.frames:] @ v)

        f = self.run(x, action, first, biases)[1]
        picks = first_max(model.motion_logits(f))                                         # [N, cells, CODES]
        picks = picks * (1 - self.border) + self.still * self.border                     # the border rows STILL

        # pass 2: the new frame alone, hidden with that motion -> its pixels
        x = self.tokens(hidden[None], self.motion(picks)[None], action[1:])
        now = (cos[1:], sin[1:])                                                          # the new frame's, at

        def second(i, layer, h):
            q, k, v = self.heads(i, layer, h, now)                                              # [n, heads, 1, hd]
            before_k, before_v = new_keys[i][:, :, None], new_values[i][:, :, None]       # [n, heads, 1, hd]
            weights = torch.cat((q @ keys[i] * scale + earlier[1:], q @ before_k.transpose(-1, -2) * scale,
                                 q @ k.transpose(-1, -2) * scale), -1).softmax(-1)
            return merge(layer, weights[..., :window] @ values[i] + weights[..., window:window + 1] @ before_v
                         + weights[..., window + 1:] @ v)

        f = self.run(x, action[1:], second, biases)[0]
        probs, drawn = self.weigh(self.pixel_probs(f), last[0])                          # [N, P*P, COLOURS]
        # its content from the same probabilities (FrameDecoder): each pixel's expected colour embedding (both
        # layers'), through the patch conv
        pixels = probs @ model.colour.weight
        if model.two_layers:
            sprite, sprite_drawn = self.weigh(self.pixel_probs(f, sprite=True), last[1])
            pixels = pixels + sprite @ model.sprite_colour.weight
            probs = sprite[..., :TRANSPARENT] + sprite[..., TRANSPARENT:] * probs         # compose_probs
            drawn = torch.stack((drawn, sprite_drawn))
        else:
            drawn = drawn[None]
        g, p = model.grid, model.patch
        rgb = (first_max(probs) @ self.colours)[:, self.unnest]                          # [N, P*P, 3] in rows
        rgb = rgb.view(g, g, p, p, 3).permute(0, 2, 1, 3, 4).reshape(SIZE, SIZE, 3)
        motion = (picks * self.codes).sum(-1)                                            # [N, cells] the codes
        return rgb, self.content(pixels), motion, drawn, torch.stack(new_keys), torch.stack(new_values)


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
    if model.latent:
        raise NotImplementedError("the export has no latent choice")
    model = copy.deepcopy(model).float().cpu().eval()
    folder.mkdir(parents=True, exist_ok=True)
    cache = cache_layout(model)
    n, dim = model.grid ** 2, model.blocks[0].temporal.qkv.in_features
    t = torch.export.Dim("frames", min=2, max=model.frames - 1)
    layers, p = (2 if model.two_layers else 1), model.patch
    frame = (2, SIZE, SIZE) if model.two_layers else (SIZE, SIZE)
    torch.onnx.export(EmbedGraph(model).eval(), (torch.zeros(model.frames // 2, *frame, dtype=torch.uint8),),
                      folder / "embed.onnx", input_names=["frames"], output_names=["content", "last"],
                      dynamic_shapes={"frames": {0: t}}, dynamo=True, external_data=False, verbose=False)
    prefill = model.frames - 2                            # the longest start but its last frame
    torch.onnx.export(PrefillGraph(model).eval(),
                      (torch.zeros(prefill, n, dim), torch.full((prefill, n, model.motion_cells), float(MOTION_UNKNOWN)),
                       torch.zeros(prefill, dtype=torch.int32)),
                      folder / "prefill.onnx", input_names=["content", "motion", "actions"],
                      output_names=cache["names"], dynamo=True, external_data=False, verbose=False)
    empty = [torch.zeros(cache[name.split("_")[0]]) for name in cache["names"]]
    skip = ((torch.full((2, model.skip - 1), -1, dtype=torch.int32), torch.tensor([3, 4], dtype=torch.int32))
            if model.skip > 1 else ())                    # a frame-skip model's: each frame's earlier bytes, clock
    torch.onnx.export(StepGraph(model, palette).eval(),
                      (torch.zeros(n, dim), torch.zeros(n, model.motion_cells), torch.zeros(2, dtype=torch.int32),
                       torch.tensor([model.frames // 2], dtype=torch.int32), torch.zeros(layers, n, p * p), *skip,
                       *empty),
                      folder / "step.onnx",
                      input_names=["previous", "previous_motion", "actions", "at", "last",
                                   *(["skipped", "clock"] if skip else []), *cache["names"]],
                      output_names=["rgb", "content", "motion", "last_out", "new_keys", "new_values"], dynamo=True,
                      external_data=False, verbose=False)
    for name in ("prefill.onnx", "step.onnx"):
        fuse_norms(folder / name)
    (folder / "model.json").write_text(json.dumps({
        "cache": cache, "frames": model.frames, "content": [n, dim], "motion": [n, model.motion_cells],
        "last": [layers, n, p * p], "frame": list(frame), "transparent": TRANSPARENT if model.two_layers else None,
        "unknown": MOTION_UNKNOWN, "prefill": prefill, "skip": model.skip,
        "parameters": sum(p.numel() for p in model.parameters())}, indent=1))
