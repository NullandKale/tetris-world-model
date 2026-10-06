"""Action-conditioned world model: a Genie-style spatiotemporal MaskGIT transformer on exact pixels, soft.

A frame is a [256, 256] image of palette indices (data/nes_palette.py: NES
colours 0-54, the game's three border shades 55-57), so input and target are
exact. Each 16 x 16 patch is one transformer token (256 per frame): a
per-pixel colour embedding and a 16 x 16 stride-16 conv, as in a ViT. Each
block attends spatially within a frame (bidirectional), then temporally at
the same position over the frames so far (causal), then applies an MLP. The
action that produced frame t is added to frame t's tokens. The head predicts
every pixel's colour probabilities. It is tied to the input colour embedding,
as BERT and MaskGIT tie their output to their token embeddings: each pixel
gets a colour_dim vector, and its logits are that vector against every
colour's embedding (plus a bias per colour).

Soft, never argmax: a generated frame is its colour probabilities. It goes back
into the context as each pixel's probability-weighted colour embedding (a soft
frame, read like a real one), and it is shown as each pixel's expected colour.
Nothing ever picks one colour, so an unsure piece stays in the context as a
ghost for the next frame to resolve, instead of being rounded away.

Training is MaskGIT with a masking rate drawn per frame: tokens are replaced
by a learned mask embedding at their frame's rate, and in half the windows
every frame before a random cut is left fully visible (a clean context, as in
generation). Cross-entropy is taken over the pixels of masked tokens (the
letterbox border included, at full weight). The context window is the only
memory.

Training is in two stages, as HorizonDrive (examples/papers) trains its
rollout-capable model. The base model learns on real context only (teacher
forcing: the frames before each predicted one are real and clean). Then
scheduled rollout recovery: every window's history is the model's own soft
frames, generated from a real start as in a dream, and the frames after them
are scored against the real ones; the last frames before that boundary fade
from the model's own to the real ones (own_share), so a rollout that drifted
out of step is not scored against a real frame it contradicts. A dream decodes a frame from a fully masked frame in one
pass.

With a latent (`latent=(groups, classes)`), each frame also gets a sampled
choice: groups x classes one-hot categoricals (DreamerV3's latents), added to
its tokens like the action. In training a posterior reads the real frame and
the one before it and says which choice happened, so the frame is predicted
given its outcome and a random event (a next piece) need not be averaged; a
prior learns to predict the choice from the previous frame's features
(KL-balanced, with free bits), and a dream samples it from the prior, so the
dream commits to one outcome. The layer that adds the choice starts at zero, so
a model grown from one without a latent starts as it. Temporal attention is causal and spatial attention stays within a frame,
so a dream keeps each layer's temporal keys and values of the frames so far
(TemporalCache) and runs only the frame being decoded through the model.
See docs/guides/dynamics.md.
"""
from __future__ import annotations

import math
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
    def __init__(self, dim: int, heads: int, qk_norm: bool = False):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        # QK-norm (as in recent large transformers): queries and keys at a fixed size, so attention
        # logits cannot grow without bound
        self.q_norm = RMSNorm(dim // heads) if qk_norm else None
        self.k_norm = RMSNorm(dim // heads) if qk_norm else None

    def forward(self, x: torch.Tensor, causal: bool, cache: TemporalCache | None = None, at=0):
        """With a cache (temporal attention only): x holds frames 0..t-1 (at = 0, causal among them),
        or frames at..at+t-1 (`at` an int or 0-dim tensor), each seeing the positions up to its own."""
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        if cache is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        elif n > 1 and isinstance(at, int) and at == 0:
            k, v = cache.prefill(k, v)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            k, v, seen = cache.step(k, v, at)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=seen)
        return self.out(y.transpose(1, 2).reshape(b, n, d))


class RMSNorm(nn.RMSNorm):
    """RMSNorm in the input's dtype: under bf16 autocast the float32 weight would keep it off the fused
    kernel (1.5x slower dreams, measured)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, self.normalized_shape, self.weight.to(x.dtype), self.eps)


class SwiGLU(nn.Module):
    """The gated MLP of Llama / PaLM: (silu(x W_gate) * x W_up) W_out, hidden 8/3 x dim (the parameters of
    a 4 x dim GELU MLP)."""

    def __init__(self, dim: int):
        super().__init__()
        hidden = 8 * dim // 3
        self.gate_up = nn.Linear(dim, 2 * hidden)
        self.out = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, -1)
        return self.out(F.silu(gate) * up)


class Block(nn.Module):
    """Spatial attention within each frame, causal temporal attention per position, MLP. modern: RMSNorm,
    QK-norm and a SwiGLU MLP (Llama-style) instead of LayerNorm and a GELU MLP."""

    def __init__(self, dim: int, heads: int, modern: bool = False):
        super().__init__()
        norm = RMSNorm if modern else nn.LayerNorm
        self.norm_s, self.norm_t, self.norm_m = norm(dim), norm(dim), norm(dim)
        self.spatial, self.temporal = Attention(dim, heads, modern), Attention(dim, heads, modern)
        self.mlp = SwiGLU(dim) if modern else nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                                            nn.Linear(4 * dim, dim))

    def forward(self, x: torch.Tensor, cache: TemporalCache | None = None, at=0) -> torch.Tensor:
        """x [B, T, N, d]: window frames at..at+T-1. With a cache, frames before `at` come from it."""
        b, t, n, d = x.shape
        x = x + self.spatial(self.norm_s(x).reshape(b * t, n, d), causal=False).view(b, t, n, d)
        y = self.norm_t(x).transpose(1, 2).reshape(b * n, t, d)
        x = x + self.temporal(y, causal=True, cache=cache, at=at).view(b, n, t, d).transpose(1, 2)
        return x + self.mlp(self.norm_m(x))


class Dynamics(nn.Module):
    def __init__(self, dim: int = 512, layers: int = 12, heads: int = 8, patch: int = 16,
                 frames: int = 64, colour_dim: int = 32, recompute: bool = False, modern: bool = False,
                 latent: tuple[int, int] | None = None):
        super().__init__()
        self.patch, self.frames = patch, frames
        self.recompute = recompute     # keep only stage inputs for backward and recompute the rest: memory for time
        self.grid = SIZE // patch
        tokens = self.grid ** 2
        self.colour = nn.Embedding(COLOURS, colour_dim)
        self.patchify = nn.Conv2d(colour_dim, dim, patch, stride=patch)
        self.mask_token = nn.Parameter(torch.randn(dim) * 0.02)
        self.space_pos = nn.Parameter(torch.randn(tokens, dim) * 0.02)
        self.time_pos = nn.Parameter(torch.randn(frames, dim) * 0.02)
        self.action = nn.Linear(9, dim)
        self.blocks = nn.ModuleList(Block(dim, heads, modern) for _ in range(layers))
        self.norm = RMSNorm(dim) if modern else nn.LayerNorm(dim)
        self.pixel = nn.Linear(dim, patch * patch * colour_dim)       # each pixel's vector in colour space
        self.colour_bias = nn.Parameter(torch.zeros(COLOURS))
        self.latent = tuple(latent) if latent else None                # a sampled choice per frame (groups, classes)
        if self.latent:
            size = self.latent[0] * self.latent[1]
            self.choice = nn.Linear(size, dim, bias=False)             # the choice, added to its frame's tokens:
            nn.init.zeros_(self.choice.weight)                         # zero at first, so it starts as without
            self.posterior_in = nn.Linear(2 * dim, dim)                # (the frame's change, the frame) per token
            self.posterior_out = nn.Linear(2 * dim, size)              # pooled (mean, max) -> logits
            self.prior_out = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, size))

    def forward(self, frames: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor,
                cache: list[TemporalCache] | None = None, at=0,
                soft: tuple[torch.Tensor, torch.Tensor] | None = None, z: torch.Tensor | None = None) -> torch.Tensor:
        """frames [B, T, 256, 256] palette indices, soft frames float [B, T, 256, 256, colour_dim]
        (embed_probs), or frames already embedded as patch tokens float [B, T, N, dim]; actions [B, T] bytes
        that produced each frame (-1 = none); mask [B, T, N] tokens hidden from the model; soft (training)
        (where [B, T] bool, tokens [B, T, N, dim]): those frames' patch embeddings, made outside (a rollout's
        soft frames; frames holds the real frame there), in a fixed shape so a compiled forward sees the same
        shapes every step; z [B, T, groups * classes] each frame's choice (with a latent; None: none) ->
        features [B, T, N, dim].

        With a cache (new_cache()), frames holds window frames at..at+T-1, the
        frames before `at` are read from the cache, and these frames' keys and
        values are written into it. `at` may be a 0-dim tensor (FrameDecoder's
        CUDA graph, the exported step graph).
        """
        b, t = frames.shape[:2]
        remember = self.recompute and self.training and torch.is_grad_enabled()
        x = checkpoint(self._patches, frames, use_reentrant=False) if remember else self._patches(frames)
        if soft is not None:
            x = torch.where(soft[0][:, :, None, None], soft[1].to(x.dtype), x)
        x = torch.where(mask[..., None], self.mask_token.to(x.dtype), x)
        time = (self.time_pos[at:at + t] if isinstance(at, int) else
                self.time_pos.index_select(0, at.view(1) + torch.arange(t, device=at.device, dtype=at.dtype)))
        x = x + self.space_pos + time[:, None] + self.action(button_bits(actions))[:, :, None]
        if z is not None:
            x = x + self.choice(z.to(x.dtype))[:, :, None]
        if cache is not None:
            for block, layer in zip(self.blocks, cache):
                x = block(x, layer, at)
            return self.norm(x)
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False) if remember else block(x)
        return self.norm(x)

    embed_frames = 16                  # frames embedded at a time; ONNX export uses the window (one chunk)

    def _patches(self, frames: torch.Tensor) -> torch.Tensor:
        """[B, T, 256, 256] indices or [B, T, 256, 256, colour_dim] soft frames -> [B, T, N, dim]: per-pixel
        colour embedding, then the patch conv. Patch tokens [B, T, N, dim] pass through.

        The embedded pixels are 16 values per pixel, so they are built
        embed_frames frames at a time, already in the conv's autocast dtype (the
        conv casts its input to it anyway, so the result is the same).
        """
        if frames.is_floating_point() and frames.dim() == 4:             # patch tokens already
            return frames
        b, t = frames.shape[:2]
        flat = frames.flatten(0, 1)
        chunks = flat.split(self.embed_frames) if flat.shape[0] > self.embed_frames else (flat,)
        if frames.is_floating_point():                  # embedded pixels already (soft frames, embed_probs)
            out = [self.patchify(chunk.permute(0, 3, 1, 2)).flatten(2).transpose(1, 2) for chunk in chunks]
            return torch.cat(out).unflatten(0, (b, t))
        table = self.colour.weight
        if torch.is_autocast_enabled(frames.device.type):
            table = table.to(torch.get_autocast_dtype(frames.device.type))
        out = [self.patchify(ColourEmbedding.apply(chunk, table).permute(0, 3, 1, 2)).flatten(2).transpose(1, 2)
               for chunk in chunks]
        return torch.cat(out).unflatten(0, (b, t))

    def embed_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """Each pixel's colour probabilities [B, N, P*P, COLOURS] -> its probability-weighted colour embedding
        in frame layout [B, 256, 256, colour_dim]: a soft frame, which forward() reads like a real one."""
        x = probs.to(self.colour.weight.dtype) @ self.colour.weight
        return unpatch_pixels(x.permute(0, 3, 1, 2), self.patch).permute(0, 2, 3, 1)

    def soft_frame(self, features: torch.Tensor) -> torch.Tensor:
        """A frame's features [B, N, dim] -> its soft frame [B, 256, 256, colour_dim] (its probabilities'
        colour embeddings)."""
        return self.embed_probs(self.logits(features).float().softmax(-1))

    def soft_tokens(self, features: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
        """Tokens' features [M, dim] -> [M, dim]: the token their predicted colours make, the patch conv of
        each pixel's probability-weighted colour embedding (a soft token, as _patches embeds one), computed a
        chunk at a time (the probabilities are 58 values per pixel). The exported step graph's output."""
        weight = self.patchify.weight.flatten(1)                     # [dim, colour_dim * P * P]
        out = []
        for h in features.split(chunk):
            probs = self.logits(h).float().softmax(-1)               # [c, P*P, COLOURS]
            pixels = probs.to(self.colour.weight.dtype) @ self.colour.weight            # [c, P*P, colour_dim]
            pixels = pixels.transpose(1, 2).flatten(1)               # [c, colour_dim * P * P], the conv's order
            out.append(F.linear(pixels.to(weight.dtype), weight, self.patchify.bias))
        return torch.cat(out) if out else features.new_zeros(0, features.shape[-1])

    def posterior(self, tokens: torch.Tensor) -> torch.Tensor:
        """Real frames' patch tokens [B, T, N, dim] -> logits [B, T, groups, classes] of each frame's choice,
        read from the frame and the one before it (frame 0's from itself: its logits go unused)."""
        before = torch.cat((tokens[:, :1], tokens[:, :-1]), 1)
        h = F.gelu(self.posterior_in(torch.cat((tokens - before, tokens), -1)) + self.space_pos)
        return self.posterior_out(torch.cat((h.mean(-2), h.amax(-2)), -1)).unflatten(-1, self.latent)

    def prior(self, features: torch.Tensor) -> torch.Tensor:
        """A frame's features [..., N, dim] -> logits [..., groups, classes] of the next frame's choice."""
        return self.prior_out(torch.cat((features.mean(-2), features.amax(-2)), -1)).unflatten(-1, self.latent)

    def new_cache(self) -> list[TemporalCache]:
        return [TemporalCache(self.frames) for _ in self.blocks]

    def decoder(self, batch: int, device, compiled: bool = False, scored: bool = False) -> "FrameDecoder":
        """This model's FrameDecoder for a batch size, the caller's autocast setting, `compiled` and `scored`
        (see FrameDecoder), made on first use (its cache and CUDA graph are kept)."""
        device = torch.device(device)
        autocast = (torch.is_autocast_enabled(device.type), torch.get_autocast_dtype(device.type))
        key = (batch, device, autocast, compiled, scored)
        decoders = self.__dict__.setdefault("_decoders", {})
        if key not in decoders:
            decoders[key] = FrameDecoder(self, batch, device, autocast, compiled, scored)
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
                    config["colour_dim"], recompute, config.get("modern", False), config.get("latent"))


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
                mlp_out = block.mlp.out if isinstance(block.mlp, SwiGLU) else block.mlp[2]
                for layer in (block.spatial.out, block.temporal.out, mlp_out):
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


GHOST = 0.9         # a generated pixel whose most likely colour is below this probability is unsure (a ghost)
UNIMIX = 0.01       # each choice keeps 1% uniform probability (DreamerV3): no option is ever impossible
FREE_NATS = 1.0     # KL below this many nats per frame is free (DreamerV3's free bits)


def weigh_changes(probs: torch.Tensor, last: torch.Tensor, weight: torch.Tensor | float) -> torch.Tensor:
    """Colour probabilities [..., COLOURS] and the last frame's colours [...] -> the same with the odds of
    every change times `weight` (1: as they were): a pixel that changes with probability p is drawn
    changing with w p / (w p + 1 - p), as a loss weighting changed pixels w times would train it. An
    unsure piece is then drawn with too many cells, not with its four spread thin."""
    stay = probs.gather(-1, last[..., None])
    weighted = (probs * weight).scatter(-1, last[..., None], stay)
    return weighted / weighted.sum(-1, keepdim=True)


def choice_probs(logits: torch.Tensor) -> torch.Tensor:
    """Choice logits [..., groups, classes] -> probabilities, with UNIMIX uniform mixed in."""
    return (1 - UNIMIX) * logits.float().softmax(-1) + UNIMIX / logits.shape[-1]


def sample_choice(logits: torch.Tensor, straight_through: bool = False) -> torch.Tensor:
    """Choice logits [..., groups, classes] -> one sampled option per group, one-hot, flattened
    [..., groups * classes] (Gumbel-max, so it can run inside a CUDA graph); straight_through: the
    probabilities' gradient passes through the sample (training)."""
    probs = choice_probs(logits)
    u = torch.rand_like(probs).clamp(1e-9, 1 - 1e-9)
    pick = (probs.log() - (-u.log()).log()).argmax(-1)
    one_hot = F.one_hot(pick, probs.shape[-1]).to(probs.dtype)
    if straight_through:
        one_hot = one_hot + probs - probs.detach()
    return one_hot.flatten(-2)


def choice_kl(posterior: torch.Tensor, prior: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Logits [..., groups, classes] -> (KL(sg(posterior) || prior), KL(posterior || sg(prior))) [...], nats
    summed over groups: DreamerV3's dynamics and representation losses."""
    def kl(p, q):
        return (p * (p.log() - q.log())).sum((-2, -1))
    post, pri = choice_probs(posterior), choice_probs(prior)
    return kl(post.detach(), pri), kl(post, pri.detach())


class FrameDecoder:
    """Generates soft frames one at a time against a TemporalCache, one CUDA graph per frame.

    prefill() runs real (or soft) frames through the model at positions
    0..t-1; next() decodes the frame at position `at` from a fully masked
    frame in one pass, writes its soft frame into the cache and returns it.
    Every call of next() has the same shapes (the cache always holds
    model.frames positions), so on CUDA its whole work is captured once as a
    CUDA graph and replayed: the launches, not the GPU, bounded training
    rollouts. The graph reads the parameters in place, so it follows training.
    Autocast runs without its weight cache inside (a cached cast would be
    captured stale). One decoder per model, batch size, autocast setting,
    `compiled` and `scored` (Dynamics.decoder). compiled: torch.compile the
    frame's work before capturing it (23% less GPU time per frame, about a
    minute to compile); only training rollouts use it. scored: next() also
    takes the real frame and reports the pixels expected wrong (training
    rollouts); otherwise it returns the frame's colour probabilities (dreams,
    previews).
    """

    def __init__(self, model: Dynamics, batch: int, device, autocast, compiled: bool = False,
                 scored: bool = False):
        self.model, self.batch, self.autocast = model, batch, autocast
        self.cache = model.new_cache()
        self.act = torch.zeros(batch, 1, dtype=torch.long, device=device)
        self.at = torch.zeros((), dtype=torch.long, device=device)
        self.target = torch.zeros(batch, SIZE, SIZE, dtype=torch.long, device=device) if scored else None
        # with a latent: the next frame's choice logits, from the last frame's features (prefill, then each frame)
        self.choices = torch.zeros(batch, *model.latent, device=device) if model.latent else None
        # dreams: the last frame's most likely colours and the weight on changes from them (weigh_changes; the
        # graph reads both in place, so the weight can change between frames)
        self.last = None if scored else torch.zeros(batch, SIZE, SIZE, dtype=torch.long, device=device)
        self.change_weight = None if scored else torch.ones((), device=device)
        self.graph = self.out = None
        self._step = torch.compile(self._decode) if compiled and self.act.is_cuda else self._decode

    def prefill(self, frames: torch.Tensor, actions: torch.Tensor) -> None:
        """frames [B, t, 256, 256] (or soft frames or patch tokens, Dynamics.forward), incoming actions
        [B, t] -> positions 0..t-1."""
        b, t = frames.shape[:2]
        visible = torch.zeros(b, t, self.model.grid ** 2, dtype=torch.bool, device=frames.device)
        with torch.no_grad():
            features = self.model(frames, actions, visible, cache=self.cache)
            if self.choices is not None:
                self.choices.copy_(self.model.prior(features[:, -1]).float())

    @torch.no_grad()
    def next(self, action: torch.Tensor, at: int, target: torch.Tensor | None = None) -> dict:
        """action [B, 1] incoming, at: the frame's position (, target [B, 256, 256] the real frame when
        scored) -> {"soft": [B, 256, 256, colour_dim] the soft frame that went into the context, "unsure":
        [B] its ghost pixels, "tokens": [B, N, dim] its patch tokens, and "wrong": [B] its pixels expected
        wrong (scored) or "probs": [B, 256, 256, COLOURS] its colour probabilities}."""
        self.act.copy_(action)
        self.at.fill_(at)
        if self.target is not None:
            self.target.copy_(target)
        if not self.act.is_cuda:
            out = self._step()
        elif self.graph is None:                        # compile and run once on a side stream, then capture
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                out = self._step()
            torch.cuda.current_stream().wait_stream(side)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.out = self._step()
        else:
            self.graph.replay()
            out = {k: v.clone() for k, v in self.out.items()}
        return out

    def _decode(self) -> dict:
        model, b, n = self.model, self.batch, self.model.grid ** 2
        enabled, dtype = self.autocast
        device = self.act.device
        with torch.autocast(device.type, dtype=dtype, enabled=enabled, cache_enabled=False):
            blank = torch.zeros(b, 1, SIZE, SIZE, dtype=torch.uint8, device=device)
            hidden = torch.ones(b, 1, n, dtype=torch.bool, device=device)
            z = sample_choice(self.choices)[:, None] if self.choices is not None else None   # this frame's choice
            features = model(blank, self.act, hidden, self.cache, self.at, z=z)[:, 0]
            probs = model.logits(features).float().softmax(-1)                   # [B, N, P*P, COLOURS]
            if self.last is not None:
                probs = weigh_changes(probs, patch_pixels(self.last, model.patch), self.change_weight)
                self.last.copy_(unpatch_pixels(probs.argmax(-1), model.patch))
            soft = model.embed_probs(probs)
            tokens = model._patches(soft[:, None])
            written = model(tokens, self.act, torch.zeros_like(hidden), self.cache, self.at, z=z)
            if self.choices is not None:                                         # the next frame's choice logits
                self.choices.copy_(model.prior(written[:, 0]).float())
            out = {"soft": soft, "tokens": tokens[:, 0], "unsure": (probs.amax(-1) < GHOST).sum((1, 2)).float()}
            if self.target is not None:
                right = probs.gather(-1, patch_pixels(self.target, model.patch)[..., None])
                out["wrong"] = 1 - right.mean((1, 2, 3))
            else:
                out["probs"] = unpatch_pixels(probs.permute(0, 3, 1, 2), model.patch).permute(0, 2, 3, 1)
        return out


def own_share(start: int, boundary: int, w: int, device=None) -> torch.Tensor:
    """The blend on a rollout's history, frames start..boundary-1 -> each frame's share of the model's own
    [boundary - start]: 1, then over the last w frames falling linearly from w/(w+1) to 1/(w+1), so the frame
    before the first scored one is nearly real (w = 0: all its own, as in a dream). The scored frames, from
    `boundary` on, are real. HorizonDrive's pred-to-real blend (eq. 8) on its history side only."""
    i = torch.arange(start, boundary, dtype=torch.float32, device=device)
    return (1 - (i - (boundary - w) + 1) / (w + 1)).clamp(0, 1)


@torch.no_grad()
def rollout(model: Dynamics, window: torch.Tensor, actions: torch.Tensor, start: int,
            compiled: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate frames start..T-1 one after another from the real frames before `start`, scored against the
    real ones in `window`.

    window [B, T, 256, 256], actions [B, T] incoming. The real frames fill the
    decoder's cache once; each generated frame is decoded against it and its
    soft frame written into it. -> (the soft frames as patch tokens [B, T -
    start, N, dim], each one's ghost pixels [B, T - start], each one's pixels
    expected wrong [B, T - start]). compiled: see FrameDecoder (training rollouts).
    """
    b, t = window.shape[:2]
    decoder = model.decoder(b, window.device, compiled, scored=True)
    decoder.prefill(window[:, :start], actions[:, :start])
    out = [decoder.next(actions[:, at:at + 1], at, window[:, at]) for at in range(start, t)]
    return tuple(torch.stack([o[k] for o in out], 1) for k in ("tokens", "unsure", "wrong"))


class Dreamer:
    """The model as a game engine: one generated frame per action, for as long as it is played.

    Starts from real frames [T0, 256, 256] and the incoming actions [T0] that
    produced them. Each step decodes the next frame against a TemporalCache
    and keeps its soft frame's patch tokens as context (the frames as the
    model reads them, 98 KB a frame); when the window is full
    (model.frames), the last `keep` frames are re-encoded at positions
    0..keep-1 and generation goes on (keep up to model.frames - 1: more
    history, re-encoded more often).
    """

    def __init__(self, model: Dynamics, frames: torch.Tensor, actions: torch.Tensor, keep: int | None = None,
                 change_weight: float = 1.0):
        keep = model.frames * 3 // 4 if keep is None else keep
        if not 1 <= keep < model.frames:
            raise ValueError("keep must be 1..model.frames - 1")
        self.model, self.keep = model, keep
        self.frames = list(model._patches(frames[None]).detach()[0].float().unbind(0))   # patch tokens
        self.actions = [int(a) for a in actions]
        self._encode()
        self.decoder.last.copy_(frames[-1][None])
        self.set_change_weight(change_weight)

    def set_change_weight(self, weight: float) -> None:
        """The odds of every pixel change times `weight` from the next frame on (weigh_changes; 1: none)."""
        self.decoder.change_weight.fill_(weight)

    def _encode(self) -> None:
        """Re-encode the window at positions 0..keep-1 (after the start, and whenever it slides)."""
        self.frames, self.actions = self.frames[-self.keep:], self.actions[-self.keep:]
        device = self.frames[0].device
        if not hasattr(self, "decoder"):                # its own: a dream keeps state in its cache between
            autocast = (torch.is_autocast_enabled(device.type), torch.get_autocast_dtype(device.type))
            self.decoder = FrameDecoder(self.model, 1, device, autocast)   # steps
        self.decoder.prefill(torch.stack(self.frames)[None], torch.tensor([self.actions], device=device))

    @torch.no_grad()
    def step(self, action: int) -> torch.Tensor:
        """The frame that `action` (the controller byte, -1 for none) produces -> its colour probabilities
        [256, 256, COLOURS] float32."""
        if len(self.frames) == self.model.frames:
            self._encode()
        device = self.frames[0].device
        out = self.decoder.next(torch.tensor([[action]], device=device), len(self.frames))
        self.frames.append(out["tokens"][0].float())
        self.actions.append(action)
        return out["probs"][0]
