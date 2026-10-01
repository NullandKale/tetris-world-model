"""Action-conditioned world model: a Genie-style spatiotemporal MaskGIT transformer on exact pixels.

A frame is a [256, 256] image of palette indices (data/nes_palette.py: NES
colours 0-54, the game's three border shades 55-57), so input and target are
exact. Each 16 x 16 patch is one transformer token (256 per frame): a
per-pixel colour embedding and a 16 x 16 stride-16 conv, as in a ViT. Each
block attends spatially within a frame (bidirectional), then temporally at
the same position over the frames so far (causal), then applies an MLP. The
action that produced frame t is added to frame t's tokens. The head predicts
all 256 pixels of every token as palette indices. It is tied to the input
colour embedding, as BERT and MaskGIT tie their output to their token
embeddings: each pixel gets a colour_dim vector, and its logits are that
vector against every colour's embedding (plus a bias per colour), so the head
costs dim x 256 x colour_dim weights instead of dim x 256 x 58.

Training is MaskGIT with a masking rate drawn per frame: tokens are replaced
by a learned mask embedding at their frame's rate, and in half the windows
every frame before a random cut is left fully visible (a clean context, as in
generation). Cross-entropy is taken over the pixels of masked tokens (the
letterbox tick border included, at full weight). The context window is the
only memory.

Rollouts build on the model's own frames, which contain confident mistakes,
while masking only ever taught it about missing information. So, as GameNGen
and Diffusion Forcing corrupt their context and tell the model by how much,
every frame carries a corruption level (0 = real, an embedding per level).
In half of each batch each frame draws a level and that share of its tokens
is replaced by wrong but real content (corrupt_context); in the other half
the model generates 1-32 frames one after another from the real past, as in
a rollout, and those frames carry GENERATED_LEVEL (scheduled sampling: its
own mistakes, compounding). Either way it trains on the real frames after
them, so it learns to repair its context, not only to tolerate it. Generated
frames carry Decoding.level in generation too. Generation decodes a frame from a fully masked frame in a few
steps, unmasking the most confident tokens first. Temporal attention is causal
and spatial attention stays within a frame, so a rollout keeps each layer's
temporal keys and values of the frames so far (TemporalCache) and runs only
the frame being decoded, not the whole window, through the model.
See docs/guides/dynamics.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from token_world.data.nes_palette import BORDER_SLOTS, NES_COLOURS

COLOURS = len(NES_COLOURS) + BORDER_SLOTS
SIZE = 256


def button_bits(action: torch.Tensor) -> torch.Tensor:
    """[...] action bytes (-1 = none) -> [..., 9]: the 8 button bits and a 'no action' flag.

    In float32 (exact for bytes): the exported graph then runs it on WebGPU,
    which has no int64 shift, division or remainder."""
    byte = action.float()
    none = byte < 0
    powers = 2.0 ** torch.arange(8, device=action.device)
    bits = torch.floor(byte.clamp_min(0)[..., None] / powers)
    bits = bits - 2 * torch.floor(bits / 2)                            # bit k of the byte
    return torch.cat((bits * ~none[..., None], none[..., None].float()), -1)


def incoming_actions(actions: torch.Tensor) -> torch.Tensor:
    """Stream actions (action t takes frame t to t + 1) -> the action that produced each frame."""
    return F.pad(actions[..., :-1], (1, 0), value=-1)


class ColourEmbedding(torch.autograd.Function):
    """F.embedding of palette indices, with the table's gradient as one-hot matmuls.

    Every pixel's gradient goes to one of 58 rows, most of them to the
    background's, so the usual scatter-add contended on a few rows (81 ms a
    step, the most expensive kernel); a one-hot [pixels, 58] matmul against the
    gradients gives the same sums 4x faster (0.4 ms per million pixels).
    """

    @staticmethod
    def forward(ctx, index: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(index)
        ctx.colours = table.shape[0]
        return F.embedding(index.int(), table)           # int32 indices: WebGPU has few int64 kernels

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        if not ctx.needs_input_grad[1]:                  # a frozen colour table (deep runs' frozen head)
            return None, None
        (index,) = ctx.saved_tensors
        g = grad.reshape(-1, grad.shape[-1])
        onehot = (index.reshape(-1, 1) == torch.arange(ctx.colours, device=index.device)).to(g.dtype)
        return None, onehot.T @ g


class TemporalCache:
    """One layer's temporal keys and values for the window's frames: [B*N, heads, frames, head_dim].

    The buffers always hold all `frames` positions (allocated on the first
    write), so a one-frame call has the same shapes at every position and can
    be captured as a CUDA graph; attention masks the positions after it.
    """

    def __init__(self, frames: int):
        self.frames, self.k, self.v, self.positions = frames, None, None, None

    def prefill(self, k: torch.Tensor, v: torch.Tensor):
        """Store frames 0..t-1 and return their keys and values (causal attention among them)."""
        self._allocate(k)
        t = k.shape[2]
        self.k[:, :, :t], self.v[:, :, :t] = k, v
        return self.k[:, :, :t], self.v[:, :, :t]

    def step(self, k: torch.Tensor, v: torch.Tensor, at):
        """Store t frames at positions at..at+t-1 (`at` an int or 0-dim tensor) -> all keys and values,
        and the attention mask [t, frames]: each frame sees the positions up to its own."""
        self._allocate(k)
        index = torch.as_tensor(at, device=k.device).view(1) + torch.arange(k.shape[2], device=k.device)
        self.k.index_copy_(2, index, k)
        self.v.index_copy_(2, index, v)
        return self.k, self.v, self.positions[None] <= index[:, None]

    def _allocate(self, k: torch.Tensor) -> None:
        if self.k is None:
            self.k = k.new_zeros(*k.shape[:2], self.frames, k.shape[3])
            self.v = torch.zeros_like(self.k)
            self.positions = torch.arange(self.frames, device=k.device)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, causal: bool, cache: TemporalCache | None = None, at=0):
        """With a cache (temporal attention only): x holds frames 0..t-1 (at = 0, causal among them),
        or frames at..at+t-1 (`at` an int or 0-dim tensor), each seeing the positions up to its own."""
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        if cache is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        elif n > 1 and isinstance(at, int) and at == 0:
            k, v = cache.prefill(k, v)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            k, v, seen = cache.step(k, v, at)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=seen)
        return self.out(y.transpose(1, 2).reshape(b, n, d))


class Block(nn.Module):
    """Spatial attention within each frame, causal temporal attention per position, MLP."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm_s, self.norm_t, self.norm_m = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.spatial, self.temporal = Attention(dim, heads), Attention(dim, heads)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x: torch.Tensor, cache: TemporalCache | None = None, at=0) -> torch.Tensor:
        """x [B, T, N, d]: window frames at..at+T-1. With a cache, frames before `at` come from it."""
        b, t, n, d = x.shape
        x = x + self.spatial(self.norm_s(x).reshape(b * t, n, d), causal=False).view(b, t, n, d)
        y = self.norm_t(x).transpose(1, 2).reshape(b * n, t, d)
        x = x + self.temporal(y, causal=True, cache=cache, at=at).view(b, n, t, d).transpose(1, 2)
        return x + self.mlp(self.norm_m(x))


class Dynamics(nn.Module):
    def __init__(self, dim: int = 512, layers: int = 12, heads: int = 8, patch: int = 16,
                 frames: int = 64, colour_dim: int = 32, recompute: bool = False):
        super().__init__()
        self.patch, self.frames = patch, frames
        self.recompute = recompute     # keep only stage inputs for backward and recompute the rest: memory for time
        self.grid = SIZE // patch
        tokens = self.grid ** 2
        self.colour = nn.Embedding(COLOURS, colour_dim)
        self.patchify = nn.Conv2d(colour_dim, dim, patch, stride=patch)
        self.mask_token = nn.Parameter(torch.randn(dim) * 0.02)
        self.level = nn.Embedding(LEVELS, dim)               # a frame's corruption level, added to its tokens
        nn.init.zeros_(self.level.weight)                    # zero: every level starts out read as real
        self.space_pos = nn.Parameter(torch.randn(tokens, dim) * 0.02)
        self.time_pos = nn.Parameter(torch.randn(frames, dim) * 0.02)
        self.action = nn.Linear(9, dim)
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)
        self.pixel = nn.Linear(dim, patch * patch * colour_dim)       # each pixel's vector in colour space
        self.colour_bias = nn.Parameter(torch.zeros(COLOURS))

    def forward(self, index: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor,
                level: torch.Tensor | None = None, cache: list[TemporalCache] | None = None,
                at=0) -> torch.Tensor:
        """index [B, T, 256, 256] palette indices; actions [B, T] bytes that produced each frame
        (-1 = none); mask [B, T, N] tokens hidden from the model; level [B, T] each frame's corruption
        level (None: all real) -> features [B, T, N, dim].

        With a cache (new_cache()), index holds window frames at..at+T-1, the
        frames before `at` are read from the cache, and these frames' keys and
        values are written into it. `at` may be a 0-dim tensor (FrameDecoder's
        CUDA graph, the exported step graph).
        """
        b, t = index.shape[:2]
        remember = self.recompute and self.training and torch.is_grad_enabled()
        x = checkpoint(self._patches, index, use_reentrant=False) if remember else self._patches(index)
        if level is None:                                    # real frames: level 0
            level = torch.zeros(b, t, dtype=torch.long, device=index.device)
        x = x + self.level(level).to(x.dtype)[:, :, None]
        x = torch.where(mask[..., None], self.mask_token.to(x.dtype), x)
        time = (self.time_pos[at:at + t] if isinstance(at, int) else
                self.time_pos.index_select(0, at.view(1) + torch.arange(t, device=at.device, dtype=at.dtype)))
        x = x + self.space_pos + time[:, None] + self.action(button_bits(actions))[:, :, None]
        if cache is not None:
            for block, layer in zip(self.blocks, cache):
                x = block(x, layer, at)
            return self.norm(x)
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False) if remember else block(x)
        return self.norm(x)

    embed_frames = 16                  # frames embedded at a time; ONNX export uses the window (one chunk)

    def _patches(self, index: torch.Tensor) -> torch.Tensor:
        """[B, T, 256, 256] -> [B, T, N, dim]: per-pixel colour embedding, then the patch conv.

        The embedded pixels are 16 values per pixel, so they are built
        embed_frames frames at a time, already in the conv's autocast dtype (the
        conv casts its input to it anyway, so the result is the same).
        """
        b, t = index.shape[:2]
        table = self.colour.weight
        if torch.is_autocast_enabled(index.device.type):
            table = table.to(torch.get_autocast_dtype(index.device.type))
        flat = index.flatten(0, 1)
        chunks = flat.split(self.embed_frames) if flat.shape[0] > self.embed_frames else (flat,)
        out = [self.patchify(ColourEmbedding.apply(chunk, table).permute(0, 3, 1, 2)).flatten(2).transpose(1, 2)
               for chunk in chunks]
        return torch.cat(out).unflatten(0, (b, t))

    def new_cache(self) -> list[TemporalCache]:
        return [TemporalCache(self.frames) for _ in self.blocks]

    def decoder(self, batch: int, decoding: "Decoding", device, compiled: bool = False) -> "FrameDecoder":
        """This model's FrameDecoder for a batch size, Decoding, the caller's autocast setting and
        `compiled` (see FrameDecoder), made on first use (its cache and CUDA graph are kept)."""
        device = torch.device(device)
        autocast = (torch.is_autocast_enabled(device.type), torch.get_autocast_dtype(device.type))
        key = (batch, decoding, device, autocast, compiled)
        decoders = self.__dict__.setdefault("_decoders", {})
        if key not in decoders:
            decoders[key] = FrameDecoder(self, batch, decoding, device, autocast, compiled)
        return decoders[key]

    def colour_table(self) -> torch.Tensor:
        """[COLOURS, colour_dim]: the head's colour vectors, the input embedding over sqrt(colour_dim)
        (as T5 scales its tied output, so the first logits are as small as an untied head's)."""
        return self.colour.weight * self.colour.embedding_dim ** -0.5

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        """[..., dim] -> [..., patch * patch pixels, COLOURS]: each pixel's colour_dim vector against every
        colour's vector, plus a bias per colour (never the full [P*P*COLOURS, dim] map)."""
        pixels = self.pixel(features).unflatten(-1, (self.patch * self.patch, -1))
        return pixel_logits(pixels, self.colour_table(), self.colour_bias)


def pixel_logits(pixels: torch.Tensor, table: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """pixels [..., colour_dim], table [COLOURS, colour_dim], bias [COLOURS] -> [..., COLOURS]."""
    return pixels @ table.T.to(pixels.dtype) + bias.to(pixels.dtype)


def build(config: dict, recompute: bool = False) -> Dynamics:
    """A Dynamics from a run's saved args (checkpoint["args"] or run_config.json)."""
    return Dynamics(config["dim"], config["layers"], config["heads"], config["patch"], config["frames"],
                    config["colour_dim"], recompute)


HEAD = ("pixel.weight", "pixel.bias", "colour.weight", "colour_bias")   # the pixel head (colour table tied in)


def deepen(model: Dynamics, source: dict[str, torch.Tensor]) -> None:
    """Grow a trained model deeper, keeping what it computes (depth up-scaling): `source` is a shallower
    model's state dict (same width); its block i becomes block i * r of `model` (r = the depth ratio)
    and everything else is copied. The blocks between keep their fresh weights but start with zero
    output projections (both attentions' and the MLP's), so each adds nothing to the residual stream:
    the deeper model starts as the shallower one and the new blocks learn from there."""
    old = 1 + max(int(k.split(".")[1]) for k in source if k.startswith("blocks."))
    ratio, rest = divmod(len(model.blocks), old)
    if rest:
        raise ValueError(f"{len(model.blocks)} blocks cannot take {old} evenly")
    state = model.state_dict()
    for key, value in source.items():
        if key.startswith("blocks."):
            _, i, name = key.split(".", 2)
            key = f"blocks.{int(i) * ratio}.{name}"
        state[key] = value
    model.load_state_dict(state)
    with torch.no_grad():
        for i, block in enumerate(model.blocks):
            if i % ratio:
                for layer in (block.spatial.out, block.temporal.out, block.mlp[2]):
                    layer.weight.zero_()
                    if layer.bias is not None:
                        layer.bias.zero_()


def load_run(run, device: str = "cuda") -> tuple[Dynamics, dict]:
    """A run's model_latest.pt as an eval model with its averaged weights -> (model, checkpoint).
    Refuses runs trained on another frame layout (model_frames.BORDER_VERSION): they would read the
    frames' border wrongly."""
    from token_world.data.model_frames import BORDER_VERSION
    saved = torch.load(Path(run) / "model_latest.pt", map_location=device, weights_only=False)
    if saved["args"].get("border") != BORDER_VERSION:
        raise ValueError(f"{run} was trained on another frame layout than {BORDER_VERSION}")
    model = build(saved["args"]).to(device).eval()
    model.load_state_dict(saved["ema"])
    return model, saved


def patch_pixels(index: torch.Tensor, patch: int) -> torch.Tensor:
    """[..., H, W] -> [..., N, patch * patch]: each token's pixels, row-major, tokens row-major."""
    *lead, h, w = index.shape
    g = h // patch
    return (index.reshape(*lead, g, patch, w // patch, patch).transpose(-3, -2)
            .reshape(*lead, g * (w // patch), patch * patch))


def unpatch_pixels(pixels: torch.Tensor, patch: int) -> torch.Tensor:
    """Inverse of patch_pixels for a square frame."""
    *lead, n, _ = pixels.shape
    g = math.isqrt(n)
    return (pixels.reshape(*lead, g, g, patch, patch).transpose(-3, -2)
            .reshape(*lead, g * patch, g * patch))


def training_mask(batch: int, frames: int, tokens: int, device, generator=None,
                  clean_context: float = 0.5, cut: torch.Tensor | None = None) -> torch.Tensor:
    """[B, T, N] training mask; frame 0 is always visible.

    Every masked frame draws its own rate from MaskGIT's cosine schedule,
    cos(pi/2 * u) for u ~ U(0, 1), which favours high rates (generation starts
    from a fully masked frame). In a `clean_context` fraction of windows a cut
    is drawn and every frame before it is fully visible: the rollout condition,
    a clean past followed by frames to predict. Without such windows no
    training window ever had a fully visible past, and the model, 1-2% wrong
    one frame ahead when the past was 25-50% masked, was 23% wrong with a clean
    past, which is what rollouts give it (docs/guides/dynamics.md). cut [B]:
    windows with a positive entry use it as their clean cut.
    """
    u = torch.rand(batch, frames, 1, device=device, generator=generator)
    mask = torch.rand(batch, frames, tokens, device=device, generator=generator) < torch.cos(math.pi / 2 * u)
    drawn = torch.randint(1, frames, (batch, 1), device=device, generator=generator)
    clean = torch.rand(batch, 1, device=device, generator=generator) < clean_context
    if cut is not None:
        forced = (cut > 0)[:, None]
        drawn, clean = torch.where(forced, cut[:, None], drawn), clean | forced
    before_cut = torch.arange(frames, device=device)[None] < torch.where(clean, drawn, 1)
    mask &= ~before_cut[..., None]
    return mask


def masked_loss(model: Dynamics, features: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                chunk: int = 4096):
    """Cross-entropy over the pixels of masked tokens.

    features [B, T, N, dim], target [B, T, N, P*P], mask [B, T, N] -> (mean over
    all scored pixels, per-pixel loss [M, P*P] without gradient, the masked
    tokens' (b, t, n)). The head's [M, P*P, COLOURS] logits are 58 values per
    pixel, so they are never stored: see HeadCrossEntropy.
    """
    where = mask.nonzero(as_tuple=True)
    total, per_pixel = HeadCrossEntropy.apply(features[where], model.pixel.weight, model.pixel.bias,
                                              model.colour_table(), model.colour_bias, target[where].long(), chunk)
    return total, per_pixel, where


class HeadCrossEntropy(torch.autograd.Function):
    """Mean cross-entropy of the tied head's pixel logits, with the gradient taken during the forward pass.

    Each chunk of tokens gets its pixel vectors [c, P*P, colour_dim], their
    logits against the colour table, the loss and the loss's gradients for the
    tokens and every head part, then the logits are dropped (the fused linear
    cross-entropy used for large vocabularies). Computing the two factors
    separately takes 2.3-2.8x fewer operations than the equivalent
    [P*P*COLOURS, dim] map. The matmuls run under the caller's autocast and the
    softmax in float32.
    """

    @staticmethod
    def forward(ctx, h, weight, bias, table, colour_bias, y, chunk: int):
        """h [M, dim], weight [P*P*e, dim], bias [P*P*e], table [COLOURS, e], colour_bias [COLOURS],
        y [M, P*P] long."""
        scale = 1.0 / max(y.numel(), 1)
        per_pixel = torch.empty(y.shape, dtype=torch.float32, device=h.device)
        grads = [torch.empty_like(h)] + [torch.zeros_like(t, dtype=torch.float32)
                                         for t in (weight, bias, table, colour_bias)]
        step = _head_chunk_compiled if h.is_cuda else _head_chunk
        for i in range(0, h.shape[0], chunk):
            loss, g_h, g_w, g_b, g_t, g_cb = step(h[i:i + chunk], weight, bias, table, colour_bias,
                                                  y[i:i + chunk], scale)
            per_pixel[i:i + chunk] = loss
            grads[0][i:i + chunk] = g_h
            for total, part in zip(grads[1:], (g_w, g_b, g_t, g_cb)):
                total += part
        ctx.save_for_backward(*grads)
        ctx.mark_non_differentiable(per_pixel)
        return per_pixel.mean(), per_pixel

    @staticmethod
    def backward(ctx, grad_total, grad_per_pixel):
        return (*(g * grad_total for g in ctx.saved_tensors), None, None)


def _head_chunk(h, weight, bias, table, colour_bias, y, scale: float):
    """One chunk of HeadCrossEntropy -> per-pixel loss [c, P*P] and the gradients (float32 for the head)."""
    c, e = h.shape[0], table.shape[1]
    pixels = F.linear(h, weight, bias).view(c, -1, e)                            # [c, P*P, e]
    loss, g = _ce_and_grad(pixel_logits(pixels, table, colour_bias), y, table.shape[0], scale)
    g = g.view(c, -1, table.shape[0])                                          # [c, P*P, COLOURS]
    g_pixels = (g @ table.to(g.dtype)).view(c, -1)                              # [c, P*P*e]
    return (loss, g_pixels @ weight.to(g_pixels.dtype), (g_pixels.T @ h.to(g_pixels.dtype)).float(),
            g_pixels.float().sum(0), (g.flatten(0, 1).T @ pixels.flatten(0, 1).to(g.dtype)).float(),
            g.float().sum((0, 1)))


def _ce_and_grad(logits: torch.Tensor, y: torch.Tensor, colours: int, scale: float):
    """logits [c, P*P, colours], y [c, P*P] -> per-pixel loss [c, P*P] float32 and the mean loss's
    gradient for the logits [c, P*P, colours] in the logits' dtype (softmax minus one-hot, over count)."""
    logp = logits.float().log_softmax(-1)
    loss = -logp.gather(-1, y[..., None])[..., 0]
    grad = logp.exp() - F.one_hot(y, colours)
    return loss, (grad * scale).to(logits.dtype)


_head_chunk_compiled = torch.compile(_head_chunk, dynamic=True)


@torch.no_grad()
def corrupt_context(index: torch.Tensor, patch: int, generator: torch.Generator | None = None):
    """GameNGen / Diffusion Forcing context corruption, for exact tokens: index [B, T, 256, 256] ->
    (corrupted index, level [B, T] long, changed [B, T, N] tokens whose content changed).

    Each frame draws a level 0..LEVELS-1 (frame 0 stays real), and that share of
    its tokens, level / (LEVELS - 1) * MAX_CORRUPTION, is replaced by real but
    wrong content: half by the same place 1-8 frames earlier (timing errors, a
    piece that lags) and half by a random place in the same frame (a block
    where there is none, a hole where there is one). Sources are only this or
    earlier frames, so no later content leaks in. Static tokens often swap
    with an identical one: the corruption lands where the frames change, as
    the model's own errors do.
    """
    tokens = patch_pixels(index, patch)                                  # [B, T, N, P*P]
    b, t, n = tokens.shape[:3]
    device = index.device
    rand = lambda *shape: torch.rand(*shape, device=device, generator=generator)
    level = torch.randint(0, LEVELS, (b, t), device=device, generator=generator)
    level[:, 0] = 0
    swap = rand(b, t, n) < (level.float() / (LEVELS - 1) * MAX_CORRUPTION)[..., None]
    earlier = rand(b, t, n) < 0.5
    back = torch.randint(1, 9, (b, t, n), device=device, generator=generator)
    frame = torch.arange(t, device=device)[None, :, None]
    place = torch.arange(n, device=device)[None, None, :]
    source_t = torch.where(earlier, (frame - back).clamp_min(0), frame)
    source_n = torch.where(earlier, place, torch.randint(0, n, (b, t, n), device=device, generator=generator))
    moved = tokens[torch.arange(b, device=device)[:, None, None], source_t, source_n]
    changed = swap & (moved != tokens).any(-1)
    return unpatch_pixels(torch.where(swap[..., None], moved, tokens), patch), level, changed


LEVELS = 10                     # corruption levels of a frame (GameNGen buckets its noise levels in 10)
MAX_CORRUPTION = 0.3            # share of a frame's tokens replaced at the top level
GENERATED_LEVEL = 2             # the level generated frames carry, in training rollouts and by default in play


@dataclass(frozen=True)
class Decoding:
    """How a frame is generated; training rollouts use the defaults.

    steps: MaskGIT steps. refine: once the frame is complete, this fraction of
    its least confident tokens is masked again and predicted against the rest
    of the frame (0: no refinement pass). temperature: 0 picks each pixel's
    most likely colour; above 0 samples it at that temperature. level: the
    corruption level the generated frames carry (0 reads them as real;
    training rollouts use GENERATED_LEVEL).
    """
    steps: int = 4
    refine: float = 0.0
    temperature: float = 0.0
    level: int = GENERATED_LEVEL


def decode_frame(logits_of, batch: int, tokens: int, patch: int, decoding: Decoding, device) -> torch.Tensor:
    """MaskGIT-decode one frame: logits_of(frame [B, 256, 256], hidden [B, N]) -> [B, N, P*P, COLOURS].

    Starts fully masked; each step unmasks the most confident tokens (summed
    pixel log-probability of their picks) on a cosine schedule, then the
    optional refinement pass (Decoding). Returns [B, 256, 256] int32 palette
    indices (callers keep frames as uint8; int32 until then keeps the exported
    graph's frame on the GPU, where WebGPU has no uint8 or int64 Where).
    """
    frame = torch.zeros(batch, tokens, patch * patch, dtype=torch.int32, device=device)
    hidden = torch.ones(batch, tokens, dtype=torch.bool, device=device)
    held = torch.zeros(batch, tokens, device=device)                   # each token's confidence when committed
    for s in range(decoding.steps):
        best, pick = _pick(logits_of(unpatch_pixels(frame, patch), hidden), decoding.temperature)
        token = best.sum(-1)
        still = int(tokens * math.cos(math.pi / 2 * (s + 1) / decoding.steps))  # left masked after this step
        if still == 0:                                   # the last step reveals everything left: no ordering
            reveal = hidden.clone()
        else:
            order = token.masked_fill(~hidden, float("inf")).argsort(1, descending=True)
            reveal = torch.zeros_like(hidden)
            reveal.scatter_(1, order[:, :tokens - still], True)
            reveal &= hidden
        frame = torch.where(reveal[..., None], pick.to(frame.dtype), frame)
        held = torch.where(reveal, token, held)
        hidden &= ~reveal
    if decoding.refine > 0:
        redo = torch.zeros_like(hidden)
        redo.scatter_(1, held.argsort(1)[:, :max(1, round(tokens * decoding.refine))], True)
        _, pick = _pick(logits_of(unpatch_pixels(frame, patch), redo), decoding.temperature)
        frame = torch.where(redo[..., None], pick.to(frame.dtype), frame)
    return unpatch_pixels(frame, patch)


def _pick(logits: torch.Tensor, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    """[B, N, P*P, COLOURS] -> (log-probability of each pixel's pick, the pick): argmax at temperature 0,
    else a sample at that temperature (Gumbel-max)."""
    logits = logits.float()
    if temperature <= 0:
        # max - logsumexp: log_softmax's max without its [..., COLOURS] tensor (onnxruntime-web runs
        # LogSoftmax on the CPU); the pick as the first colour at the max, in float32 then int32, as
        # argmax picks (ONNX's ArgMax gives int64, which WebGPU cannot convert)
        best = logits.max(-1, keepdim=True).values
        count = logits.shape[-1]
        first = (logits == best) * torch.arange(count, 0, -1, device=logits.device, dtype=logits.dtype)
        return best[..., 0] - logits.logsumexp(-1), (count - first.amax(-1)).int()
    logp = logits.log_softmax(-1)
    gumbel = -torch.log(-torch.log(torch.rand_like(logp).clamp_min(1e-20)))
    pick = (logp / temperature + gumbel).argmax(-1)
    return logp.gather(-1, pick[..., None])[..., 0], pick


class FrameDecoder:
    """Generates frames one at a time against a TemporalCache, one CUDA graph per frame.

    prefill() runs real (or earlier) frames through the model at positions
    0..t-1; next() decodes the frame at position `at` in `steps` MaskGIT steps,
    writes it into the cache at Decoding.level and returns it. Every call of
    next() has the same shapes (the cache always holds model.frames positions),
    so on CUDA its whole work (steps + 1 one-frame passes and the confidence
    ordering, ~2,000 kernels) is captured once as a CUDA graph and replayed:
    the launches, not the GPU, bounded training rollouts. The graph reads the
    parameters in place, so it follows training. Autocast runs without its
    weight cache inside (a cached cast would be captured stale). One decoder
    per model, batch size, Decoding, autocast setting and `compiled`
    (Dynamics.decoder). compiled: torch.compile the frame's work before
    capturing it (23% less GPU time per frame, about a minute to compile).
    Only training rollouts use it, one decoder for the whole run; previews,
    play and checks capture the eager kernels and start at once, whatever
    Decoding they ask for.
    """

    def __init__(self, model: Dynamics, batch: int, decoding: Decoding, device, autocast, compiled: bool = False):
        self.model, self.batch, self.decoding, self.autocast = model, batch, decoding, autocast
        self.cache = model.new_cache()
        self.act = torch.zeros(batch, 1, dtype=torch.long, device=device)
        self.at = torch.zeros((), dtype=torch.long, device=device)
        self.level = torch.full((batch, 1), decoding.level, dtype=torch.long, device=device)
        self.graph = self.out = None
        self._step = torch.compile(self._decode) if compiled and self.act.is_cuda else self._decode

    def prefill(self, frames: torch.Tensor, actions: torch.Tensor, level: torch.Tensor | None = None) -> None:
        """frames [B, t, 256, 256], incoming actions [B, t], level [B, t] or None (real) -> positions 0..t-1."""
        b, t = frames.shape[:2]
        visible = torch.zeros(b, t, self.model.grid ** 2, dtype=torch.bool, device=frames.device)
        with torch.no_grad():
            self.model(frames, actions, visible, level, cache=self.cache)

    @torch.no_grad()
    def next(self, action: torch.Tensor, at: int) -> torch.Tensor:
        """action [B, 1] incoming, at: the frame's position -> [B, 256, 256] uint8."""
        self.act.copy_(action)
        self.at.fill_(at)
        if not self.act.is_cuda:
            return self._step()
        if self.graph is None:                          # compile and run once on a side stream, then capture
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                frame = self._step()
            torch.cuda.current_stream().wait_stream(side)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.out = self._step()
            return frame
        self.graph.replay()
        return self.out.clone()

    def _decode(self) -> torch.Tensor:
        model, b, n = self.model, self.batch, self.model.grid ** 2
        enabled, dtype = self.autocast
        with torch.autocast(self.act.device.type, dtype=dtype, enabled=enabled, cache_enabled=False):
            frame = decode_frame(lambda f, hidden: model.logits(model(f[:, None], self.act, hidden[:, None],
                                                                      self.level, self.cache, self.at)[:, 0]),
                                 b, n, model.patch, self.decoding, self.act.device)
            committed = torch.zeros(b, 1, n, dtype=torch.bool, device=self.act.device)
            model(frame[:, None], self.act, committed, self.level, self.cache, self.at)
        return frame.to(torch.uint8)


@torch.no_grad()
def rollout(model: Dynamics, window: torch.Tensor, actions: torch.Tensor, start: int,
            decoding: Decoding = Decoding(), compiled: bool = False) -> torch.Tensor:
    """Generate frames start..T-1 one after another from the real frames before `start`.

    window [B, T, 256, 256] (frames from `start` on are ignored), actions [B, T]
    incoming. The real frames fill the decoder's cache once; each generated
    frame is decoded against it and then written into it. Returns [B, T - start,
    256, 256] uint8. compiled: see FrameDecoder (training rollouts).
    """
    b, t = window.shape[:2]
    decoder = model.decoder(b, decoding, window.device, compiled)
    decoder.prefill(window[:, :start], actions[:, :start])
    return torch.stack([decoder.next(actions[:, at:at + 1], at) for at in range(start, t)], 1)


class Dreamer:
    """The model as a game engine: one generated frame per action, for as long as it is played.

    Starts from real frames [T0, 256, 256] and the incoming actions [T0] that
    produced them. Each step decodes the next frame against a TemporalCache;
    when the window is full (model.frames), the last `keep` frames are
    re-encoded at positions 0..keep-1 and generation goes on (keep up to
    model.frames - 1: more history, re-encoded more often). Generated frames
    carry Decoding.level, real ones 0.
    """

    def __init__(self, model: Dynamics, frames: torch.Tensor, actions: torch.Tensor,
                 decoding: Decoding = Decoding(), keep: int = 48):
        if not 1 <= keep < model.frames:
            raise ValueError("keep must be 1..model.frames - 1")
        self.model, self.decoding, self.keep = model, decoding, keep
        self.frames = list(frames.unbind(0))
        self.actions = [int(a) for a in actions]
        self.levels = [0] * len(self.frames)
        self._encode()

    def _encode(self) -> None:
        """Re-encode the window at positions 0..keep-1 (after the start, and whenever it slides)."""
        self.frames, self.actions, self.levels = (self.frames[-self.keep:], self.actions[-self.keep:],
                                                  self.levels[-self.keep:])
        device = self.frames[0].device
        if not hasattr(self, "decoder"):                # its own: a dream keeps state in its cache between
            autocast = (torch.is_autocast_enabled(device.type), torch.get_autocast_dtype(device.type))
            self.decoder = FrameDecoder(self.model, 1, self.decoding, device, autocast)   # steps
        self.decoder.prefill(torch.stack(self.frames)[None], torch.tensor([self.actions], device=device),
                             torch.tensor([self.levels], device=device))

    @torch.no_grad()
    def step(self, action: int) -> torch.Tensor:
        """The frame that `action` (the controller byte, -1 for none) produces -> [256, 256] uint8."""
        if len(self.frames) == self.model.frames:
            self._encode()
        device = self.frames[0].device
        frame = self.decoder.next(torch.tensor([[action]], device=device), len(self.frames))
        self.frames.append(frame[0])
        self.actions.append(action)
        self.levels.append(self.decoding.level)
        return frame[0]
