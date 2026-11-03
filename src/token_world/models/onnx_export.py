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
                  last [layers, N, P*P] (the previous frame's colours, each layer's), the cache (keys_i,
                  values_i: a ring, frame t at slot t mod frames, holding the frames up to at - 2) -> rgb
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
runs on the CPU in onnxruntime-web); argmax is in float ops (first_index; WebGPU runs no int64: ArgMax's indices
sent the work to the CPU, a round trip a frame), and the picture's colours are gathered by int32 index.
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


def first_index(scores: torch.Tensor) -> torch.Tensor:
    """[..., C] -> [...] float: the index of the largest (the first at the maximum, as argmax), in float ops (an
    ArgMax's int64 indices run on the CPU in onnxruntime-web): five kernels, where first_max's one-hot and a sum
    took nine passes over the scores."""
    c = scores.shape[-1]
    ranked = torch.where(scores >= scores.amax(-1, keepdim=True), torch.arange(c, 0, -1, dtype=torch.float32), 0.0)
    return c - ranked.amax(-1)


def pairing(head_dim: int, pairs: int) -> torch.Tensor:
    """[head_dim, head_dim]: z @ it holds, for each rotated pair of channels (i, pairs + i), -z[pairs + i] at i and
    z[i] at pairs + i, and 0 on the channels not rotated, so a rotation by an angle is z * cos + (z @ it) * sin
    (cos 1 and sin 0 where not rotated)."""
    out = torch.zeros(head_dim, head_dim)
    i = torch.arange(pairs)
    out[pairs + i, i] = -1.0
    out[i, pairs + i] = 1.0
    return out


def rotation(times: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    """Frames `times` [T] and the temporal pairing [hd, hd] -> each frame's rotation [T, hd, hd] of its queries and
    keys (dynamics.rotary's: channel pairs (i, hd / 2 + i)), z @ it = z * cos + (z @ pairs) * sin: one product a
    layer where the element-wise form took four, the matrices made once a pass for every layer."""
    head_dim = pairs.shape[0]
    half = head_dim // 2
    angle = times.float()[:, None] * ROPE_BASE ** -(torch.arange(half, dtype=torch.float32) / half)
    angle = torch.cat((angle, angle), -1)[:, None]                                       # [T, 1, hd]
    return torch.eye(head_dim) * angle.cos() + pairs * angle.sin()


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


def fuse_skip_norms(path: Path) -> int:
    """Each LayerNormalization of a residual sum (an Add of two tensors of one shape) in the ONNX graph at `path`
    -> one SkipLayerNormalization (onnxruntime's, com.microsoft: the sum and its norm in one kernel, the sum an
    output for the next residual); its optimiser fused only the sums used once (1 of a step's 56, 2026-11-02).
    Rewrites the file; -> the norms fused."""
    import onnx
    from onnx import helper

    graph_path = str(path)
    proto = onnx.load(graph_path)
    inferred = onnx.shape_inference.infer_shapes(proto).graph
    shapes = {v.name: [d.dim_value if d.HasField("dim_value") else None for d in v.type.tensor_type.shape.dim]
              for v in (*inferred.value_info, *inferred.input, *inferred.output)}
    graph = proto.graph
    made = {o: n for n in graph.node for o in n.output}
    fused, gone = {}, set()
    for norm in graph.node:
        add = made.get(norm.input[0])
        attributes = {a.name: helper.get_attribute_value(a) for a in norm.attribute}
        if (norm.op_type != "LayerNormalization" or add is None or add.op_type != "Add" or id(add) in gone
                or attributes.get("axis", -1) != -1 or len(norm.input) < 2):
            continue
        a, b = (shapes.get(name) for name in add.input)
        if not a or a != b or None in a or len(a) != 3:              # SkipLayerNormalization: [B, S, H] both
            continue
        gone.update((id(add), id(norm)))
        fused[id(norm)] = helper.make_node(
            "SkipLayerNormalization", [add.input[0], add.input[1], *norm.input[1:3]],       # (a bias if it has one)
            [norm.output[0], "", "", add.output[0]], name=f"{norm.name}_skip", domain="com.microsoft",
            epsilon=float(attributes.get("epsilon", 1e-5)))
    nodes = [fused.get(id(n), n) for n in graph.node if id(n) not in gone or id(n) in fused]
    del graph.node[:]
    graph.node.extend(nodes)
    if fused and not any(o.domain == "com.microsoft" for o in proto.opset_import):
        proto.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    onnx.save(proto, graph_path)
    return len(fused)


@torch.library.custom_op("token_world_onnx::attention", mutates_args=())
def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor | None, heads: int) -> torch.Tensor:
    """Attention within each of B sequences: q, k, v [B, S, heads * hd] (the scale already in q), bias [1, heads,
    S, S] additive or None -> [B, S, heads * hd]. In the graphs onnxruntime's MultiHeadAttention (attention_op):
    one op where the transposes, products, bias and softmax were nine kernels."""
    b, s, d = q.shape
    split = lambda x: x.view(b, s, heads, d // heads).transpose(1, 2)
    scores = split(q) @ split(k).transpose(-1, -2)
    if bias is not None:
        scores = scores + bias
    return (scores.softmax(-1) @ split(v)).transpose(1, 2).reshape(b, s, d)


@attention.register_fake
def _(q, k, v, bias, heads):
    return torch.empty_like(q)


def attention_op():
    """{attention: its ONNX translation} for torch.onnx.export: com.microsoft MultiHeadAttention (onnxruntime's
    contrib op, which its WebGPU provider runs; onnx has no schema for it, so one is given here) with scale 1."""
    import onnxscript
    from onnx.defs import OpSchema

    parameter, optional = OpSchema.FormalParameter, OpSchema.FormalParameterOption.Optional
    schema = OpSchema(
        "MultiHeadAttention", "com.microsoft", 1,
        inputs=[parameter("query", "T"), parameter("key", "T", param_option=optional),
                parameter("value", "T", param_option=optional), parameter("bias", "T", param_option=optional),
                parameter("key_padding_mask", "M", param_option=optional),
                parameter("attention_bias", "T", param_option=optional)],
        outputs=[parameter("output", "T")],
        type_constraints=[("T", ["tensor(float)"], ""), ("M", ["tensor(int32)"], "")],
        attributes=[OpSchema.Attribute("num_heads", OpSchema.AttrType.INT, ""),
                    OpSchema.Attribute("scale", OpSchema.AttrType.FLOAT, "")])
    mha = onnxscript.values.Op(onnxscript.values.Opset("com.microsoft", 1), "MultiHeadAttention", schema)

    def translate(q, k, v, bias, heads: int):
        return mha(q, k, v, None, None, bias, num_heads=heads, scale=1.0)

    return {torch.ops.token_world_onnx.attention.default: translate}


def rms_gelu(x: torch.Tensor, norm) -> torch.Tensor:
    """F.gelu(norm(x)) for an RMSNorm module, the conv ends' step between maps (dynamics.norm_gelu): two kernels."""
    return F.gelu(rms(x, norm.weight, norm_eps(norm)))


class Graph(nn.Module):
    """What the three graphs share: a frame's tokens from its content, motion and buttons, and the blocks.

    The blocks are the model's, written out for the browser's kernel count (each kernel costs a dispatch in
    onnxruntime-web): what is fixed is folded into constants once, at export.
    - Attention: queries and keys normalised together (QK-norm: one fused norm with unit weights) and then
      weighted, the attention's scale folded into the queries' weights. The temporal attention rotates them by
      frame with one product (rotation); the spatial one by place (RoPE-Mixed, its angles fixed for the graph) with
      the weights folded into its cos and pairing. One additive mask a pass, one Split for the softmax's parts.
    - The buttons' norms (action_norm): every block's (scale, shift) in one product a pass, with each LayerNorm's
      own weight and bias folded in, so a norm is the plain norm, one Mul and one Add.
    - Linear maps take 2-D inputs (the bias in the product, Gemm). The RMSNorms are `rms`, which fuse_norms makes
      one kernel each."""

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
        hd = self.head_dim
        self.register_buffer("unit", torch.ones(hd))                                    # the QK-norms' fused part
        self.register_buffer("temporal_pairs", pairing(hd, hd // 2))
        self.locals = first.spatial.local
        p = model.patch
        nest = model.nest if model.conv_ends else torch.arange(p * p)          # the graph's order of a token's pixels
        self.register_buffer("unnest", nest.argsort())                          # each row-major place's
        undo = model.change_undo("cpu")                                          # [N, P*P, 1] or None
        self.weighs = undo is not None
        if self.weighs:                               # weigh_changes on the logits: log(undo) where not the last
            self.register_buffer("log_undo", undo[:, nest].log())
        if model.conv_ends:                       # the first step, 2 x 2 pixels (ky, kx, colour) -> its width
            self.register_buffer("patch_weight", model.patchify.weight.permute(0, 2, 3, 1).flatten(1).clone())
        if self.locals:                           # one offsets table for every layer (each layer's own was 4.7 MB
            self.register_buffer("offsets", first.spatial.offsets.clone())             # more a graph, 2026-11-02)
        scale = hd ** -0.5
        for i, block in enumerate(model.blocks):
            for kind, attn in (("s", block.spatial), ("t", block.temporal)):
                weights = (torch.stack((attn.q_norm.weight, attn.k_norm.weight)) if attn.q_norm is not None
                           else torch.ones(2, hd))
                weights = (weights * torch.tensor([[scale], [1.0]]))[:, None]           # [2, 1, hd]: q, k
                spatial = kind == "s" and attn.rope
                if not spatial:
                    self.register_buffer(f"qk_{kind}{i}", weights)
                    continue
                # RoPE-Mixed at each token's place: (z * w) * cos + ((z * w) @ pairs) * sin
                # = z * (w * cos) + (z @ (diag(w) pairs)) * sin, for queries and keys at once
                angle = torch.einsum("nc,hpc->nhp", model.place[0].float(), attn.rope_freq.float())
                pairs = angle.shape[-1]
                cos, sin = torch.ones(angle.shape[0], self.heads_, hd), torch.zeros(angle.shape[0], self.heads_, hd)
                cos[..., :pairs] = cos[..., pairs:2 * pairs] = angle.cos()
                sin[..., :pairs] = sin[..., pairs:2 * pairs] = angle.sin()
                self.register_buffer(f"space_cos{i}", cos[:, None] * weights)           # [N, 2, heads, hd]
                self.register_buffer(f"space_sin{i}", sin[:, None])                     # [N, 1, heads, hd]
                self.register_buffer(f"space_pairs{i}", weights.view(2, hd, 1) * pairing(hd, pairs))   # [2, hd, hd]
        # the buttons' norms: Block.modulation (SiLU, then a linear map to (scale, shift) of each of the three
        # norms) and each LayerNorm's (weight, bias), (z * g + b) * (1 + s) + t = z * (g + g s) + (b + b s + t), every
        # block's in one map
        self.modulated = [block.modulation is not None for block in model.blocks]
        if any(self.modulated):
            rows, biases = [], []
            for block in model.blocks:
                if block.modulation is None:
                    continue
                linear = block.modulation[1]
                w, c = linear.weight.view(6, -1, linear.in_features), linear.bias.view(6, -1)
                for j, norm in enumerate((block.norm_s, block.norm_t, block.norm_m)):
                    g = norm.weight if norm.weight is not None else torch.ones_like(c[0])
                    b = norm.bias if getattr(norm, "bias", None) is not None else torch.zeros_like(c[0])
                    s_w, s_c, t_w, t_c = w[2 * j], c[2 * j], w[2 * j + 1], c[2 * j + 1]
                    rows += [g[:, None] * s_w, b[:, None] * s_w + t_w]
                    biases += [g + g * s_c, b + b * s_c + t_c]
            self.register_buffer("modulation_weight", torch.cat(rows))
            self.register_buffer("modulation_bias", torch.cat(biases))

    def biases(self) -> list:
        """Each layer's local bias [heads, N, N] (SpatialAttention), gathered once a graph; None without."""
        return [block.spatial.local_bias[:, self.offsets] if self.locals else None for block in self.model.blocks]

    def qk(self, attn, x: torch.Tensor):
        """An attention's queries and keys together [T, N, 2, heads, hd], normalised (QK-norm, unit weights), and
        its values [T, N, 1, heads, hd], for x [T, N, d]."""
        t, n, d = x.shape
        qk, v = attn.qkv(x.reshape(t * n, d)).view(t, n, 3, self.heads_, self.head_dim).split([2, 1], 2)
        if attn.q_norm is not None:
            qk = rms(qk, self.unit, norm_eps(attn.q_norm))
        return qk, v

    def heads(self, i: int, attn, x: torch.Tensor, turn: torch.Tensor):
        """A temporal layer's queries [n, heads, T, hd], keys transposed [n, heads, hd, T] and values [n, heads, T,
        hd] for x [T, n, d], rotated by `turn` (rotation, [T, hd, hd])."""
        qk, v = self.qk(attn, x)
        qk = qk * getattr(self, f"qk_t{i}")
        qk = qk @ turn[:, None, None]             # [T, 1, 1, hd, hd] (reshaping to [T, rows, hd] around the product
        #                                           exported wrongly: the rewriter broadcast it over heads, 2026-11-02)
        q, k = (part.squeeze(2) for part in qk.split(1, 2))
        if q.shape[0] == 1:                       # one frame (pass 2): the permutes only move its size-1 axis
            n, h, hd = q.shape[1:]
            return q.reshape(n, h, 1, hd), k.reshape(n, h, hd, 1), v.reshape(n, h, 1, hd)
        return q.permute(1, 2, 0, 3), k.permute(1, 2, 3, 0), v.squeeze(2).permute(1, 2, 0, 3)

    def attend(self, layer, q, keys_t, values, cache_keys, cache_values, mask) -> torch.Tensor:
        """Temporal attention for q [n, heads, T, hd] over the cache's frames and the recent ones (keys_t [n, heads,
        hd, R], values [n, heads, R, hd]), mask [T, frames + R] additive -> the layer's output [T, n, d]."""
        n, h, t, hd = q.shape
        frames = cache_keys.shape[-1]
        weights = (torch.cat((q @ cache_keys, q @ keys_t), -1) + mask).softmax(-1)
        old, recent = weights.split([frames, weights.shape[-1] - frames], -1)
        y = old @ cache_values + recent @ values
        y = y.reshape(n, h * hd) if t == 1 else y.permute(2, 0, 1, 3).reshape(t * n, h * hd)
        return layer.out(y).view(t, n, h * hd)

    def spatial(self, i: int, attn, x: torch.Tensor, bias) -> torch.Tensor:
        """A block's spatial attention within each frame (SpatialAttention) for x [T, N, d]."""
        t, n, d = x.shape
        qk, v = self.qk(attn, x)
        if attn.rope:
            qk = (qk * getattr(self, f"space_cos{i}") + (qk @ getattr(self, f"space_pairs{i}"))
                  * getattr(self, f"space_sin{i}"))
        else:
            qk = qk * getattr(self, f"qk_s{i}")
        q, k = (part.reshape(t, n, d) for part in qk.split(1, 2))
        y = attention(q, k, v.reshape(t, n, d), bias[None] if bias is not None else None, self.heads_)
        return attn.out(y.reshape(t * n, d)).view(t, n, d)

    def content(self, pixels: torch.Tensor) -> torch.Tensor:
        """Each pixel's colour embedding [N, P*P, colour_dim] in the graph's order -> the tokens [N, dim] (the
        patch conv; with conv ends its pyramid, Dynamics._steps)."""
        model, n = self.model, pixels.shape[0]
        if model.conv_ends is None:
            return F.linear(pixels.transpose(1, 2).flatten(1), model.patchify.weight.flatten(1), model.patchify.bias)
        x = F.linear(pixels.reshape(-1, 4 * pixels.shape[-1]), self.patch_weight, model.patchify.bias)
        for norm, step in zip(model.patch_norms, model.patch_steps):
            x = step(rms_gelu(x, norm).reshape(-1, 4 * x.shape[-1]))
        return x.view(n, -1)

    def pixel_logits(self, f: torch.Tensor, sprite: bool = False) -> torch.Tensor:
        """Features [N, dim] -> each pixel's colour logits [N, P*P, colours] in the graph's order (sprite: the
        sprite layer's head and colours)."""
        model, n = self.model, f.shape[0]
        head = model.sprite_pixel if sprite else model.pixel
        if model.conv_ends is None:
            vectors = head(f).view(n, model.patch ** 2, -1)
        else:                                     # down the head's steps (Dynamics.pixel_groups)
            steps, norms = ((model.sprite_pixel_steps, model.sprite_pixel_norms) if sprite and not model.shared_heads
                            else (model.pixel_steps, model.pixel_norms))
            h = f
            for step, norm in zip(steps, norms):
                h = rms_gelu(step(h).view(-1, norm.weight.shape[0]), norm)               # [N * groups, width]
            vectors = head(h).view(n, model.patch ** 2, -1)
        table, bias = (model.sprite_table(), model.sprite_bias) if sprite else (model.colour_table(), model.colour_bias)
        return pixel_logits(vectors, table, bias)

    def weigh(self, logits: torch.Tensor, last: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """A layer's logits [N, P*P, C] and its last colours [N, P*P] -> its probabilities with the odds of each
        change from `last` times the model's undo (weigh_changes, as log(undo) added to every logit but the last
        colour's; as they are without a change weight), and their most likely colours [N, P*P] (the next step's
        last)."""
        if self.weighs:
            stays = last[..., None] == torch.arange(logits.shape[-1], dtype=torch.float32)
            logits = logits + torch.where(stays, 0.0, self.log_undo)
        return logits.softmax(-1), first_index(logits)

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
        model, (t, n, d) = self.model, x.shape
        if any(self.modulated):                   # every modulated norm's (scale, shift), each [T, 1, d], split
            # by block, then by norm (one Split of all 54 needed 55 buffers: WebGPU allows a shader 16, 2026-11-02)
            every = F.linear(F.silu(action), self.modulation_weight, self.modulation_bias)[:, None]
            mods = iter([m for block in every.split(6 * d, -1) for m in block.split(d, -1)])
        for i, block in enumerate(model.blocks):
            norms = (block.norm_s, block.norm_t, block.norm_m)
            if self.modulated[i]:                 # the plain norm (its weights are in the scale and shift); one
                # frame's (pass 2) scale and shift are the LayerNorm's own weight and bias (fuse_skip_norms then
                # makes the residual's sum, the norm and both one kernel)
                def modulated(z, norm, scale, shift):
                    if isinstance(norm, nn.LayerNorm) and t == 1:
                        return F.layer_norm(z, (d,), scale.reshape(d), shift.reshape(d), norm.eps)
                    plain = (F.layer_norm(z, (d,), eps=norm.eps) if isinstance(norm, nn.LayerNorm)
                             else rms(z, torch.ones(d), norm_eps(norm)))
                    return plain * scale + shift
                norms = tuple((lambda z, norm=norm, scale=next(mods), shift=next(mods):
                               modulated(z, norm, scale, shift)) for norm in norms)
            x = x + self.spatial(i, block.spatial, norms[0](x), biases[i])
            x = x + temporal(i, block.temporal, norms[1](x))
            x = x + block.mlp(norms[2](x).reshape(t * n, d)).view(t, n, d)
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
        action = model.action(button_bits(actions))
        x = self.tokens(content, self.motion(self.known(motion)), action)
        times = torch.arange(t)
        turn = rotation(times, self.temporal_pairs)
        causal = torch.zeros(t, t).masked_fill(times[None, :] > times[:, None], float("-inf"))
        keys, values = [], []

        def temporal(i, layer, h):
            q, k, v = self.heads(i, layer, h, turn)
            keys.append(F.pad(k, (0, model.frames - t)))
            values.append(F.pad(v, (0, 0, 0, model.frames - t)))
            y = ((q @ k + causal).softmax(-1) @ v).permute(2, 0, 1, 3).reshape(-1, h.shape[-1])
            return layer.out(y).view(h.shape)

        self.run(x, action, temporal, self.biases())
        return (*keys, *values)


class StepGraph(Graph):
    def __init__(self, model: Dynamics, palette: torch.Tensor):
        super().__init__(model)
        self.register_buffer("colours", palette.float()[:COLOURS])                     # [COLOURS, 3]

    def forward(self, previous, previous_motion, actions, at, last, *cache):
        model = self.model
        keys, values = cache[:len(model.blocks)], cache[len(model.blocks):]
        hidden = model.mask_token.expand_as(previous)
        window = model.frames
        newest = at.view(()).float() - 2                                                # the cache's newest frame
        behind = newest - torch.arange(window, dtype=torch.float32)
        stamp = newest - (behind - torch.floor(behind / window) * window)                # each slot's frame
        rows = at.view(()).float() - 1 + torch.arange(2, dtype=torch.float32)            # the previous frame, the new
        seen = (stamp[None] > rows[:, None] - window) & (stamp[None] >= 0)               # its last frames - 1 before
        earlier = torch.zeros(2, window).masked_fill(~seen, float("-inf"))               # [2, frames]
        # pass 1's mask: the cache, then the pair (causal); pass 2's: the cache, then both
        first_mask = torch.cat((earlier, torch.tensor([[0.0, float("-inf")], [0.0, 0.0]])), -1)
        second_mask = torch.cat((earlier[1:], torch.zeros(1, 2)), -1)
        biases = self.biases()
        new_keys, new_values = [], []

        # pass 1: the previous frame, with its motion, and the new one hidden -> the new frame's motion
        action = model.action(button_bits(actions))                                      # [2, d]
        x = self.tokens(torch.stack((previous, hidden)),
                        torch.stack((self.motion(self.known(previous_motion)), torch.zeros_like(previous))), action)
        turns = rotation(at.view(()) - 1 + torch.arange(2, dtype=at.dtype), self.temporal_pairs)   # the pair's

        def first(i, layer, h):
            q, k, v = self.heads(i, layer, h, turns)                     # [n, heads, 2, hd], [n, heads, hd, 2], ..
            new_keys.append(k[..., :1])                         # the previous frame's, for the cache at at - 1
            new_values.append(v[:, :, :1])
            return self.attend(layer, q, k, v, keys[i], values[i], first_mask)

        f = self.run(x, action, first, biases)[1]
        picks = first_max(model.motion_logits(f))                                         # [N, cells, CODES]
        picks = picks * (1 - self.border) + self.still * self.border                     # the border rows STILL

        # pass 2: the new frame alone, hidden with that motion -> its pixels
        x = self.tokens(hidden[None], self.motion(picks)[None], action[1:])

        def second(i, layer, h):
            q, k, v = self.heads(i, layer, h, turns[1:])                 # [n, heads, 1, hd], [n, heads, hd, 1], ..
            return self.attend(layer, q, torch.cat((new_keys[i], k), -1), torch.cat((new_values[i], v), 2),
                               keys[i], values[i], second_mask)

        f = self.run(x, action[1:], second, biases)[0]
        probs, drawn = self.weigh(self.pixel_logits(f), last[0])                         # [N, P*P, COLOURS]
        # its content from the same probabilities (FrameDecoder): each pixel's expected colour embedding (both
        # layers'), through the patch conv
        pixels = probs @ model.colour.weight
        if model.two_layers:
            sprite, sprite_drawn = self.weigh(self.pixel_logits(f, sprite=True), last[1])
            pixels = pixels + sprite @ model.sprite_colour.weight
            probs = sprite[..., :TRANSPARENT] + sprite[..., TRANSPARENT:] * probs         # compose_probs
            drawn = torch.stack((drawn, sprite_drawn))
        else:
            drawn = drawn[None]
        g, p = model.grid, model.patch                  # the picture: each pixel's most likely colour, in rows
        shown = first_index(probs)[:, self.unnest].view(g, g, p, p).transpose(1, 2).reshape(SIZE, SIZE)
        rgb = F.embedding(shown.int(), self.colours)                                       # Gather, int32
        motion = (picks * self.codes).sum(-1)                                            # [N, cells] the codes
        return (rgb, self.content(pixels), motion, drawn, torch.cat(new_keys, -1).permute(3, 0, 1, 2),
                torch.cat(new_values, 2).permute(2, 0, 1, 3))


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
                      output_names=cache["names"], dynamo=True, external_data=False, verbose=False,
                      custom_translation_table=attention_op())
    empty = [torch.zeros(cache[name.split("_")[0]]) for name in cache["names"]]
    torch.onnx.export(StepGraph(model, palette).eval(),
                      (torch.zeros(n, dim), torch.zeros(n, model.motion_cells), torch.zeros(2, dtype=torch.int32),
                       torch.tensor([model.frames // 2], dtype=torch.int32), torch.zeros(layers, n, p * p), *empty),
                      folder / "step.onnx",
                      input_names=["previous", "previous_motion", "actions", "at", "last", *cache["names"]],
                      output_names=["rgb", "content", "motion", "last_out", "new_keys", "new_values"], dynamo=True,
                      external_data=False, verbose=False, custom_translation_table=attention_op())
    for name in ("prefill.onnx", "step.onnx"):
        fuse_norms(folder / name)
        fuse_skip_norms(folder / name)
    (folder / "model.json").write_text(json.dumps({
        "cache": cache, "frames": model.frames, "content": [n, dim], "motion": [n, model.motion_cells],
        "last": [layers, n, p * p], "frame": list(frame), "transparent": TRANSPARENT if model.two_layers else None,
        "unknown": MOTION_UNKNOWN, "prefill": prefill,
        "parameters": sum(p.numel() for p in model.parameters())}, indent=1))
