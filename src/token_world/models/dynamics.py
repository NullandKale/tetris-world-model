"""Action-conditioned world model: a Genie-style spatiotemporal MaskGIT transformer on exact pixels, soft.

A frame is a [256, 256] image of palette indices (data/nes_palette.py: NES
colours 0-54, the game's three border shades 55-57), so input and target are
exact. Each 16 x 16 patch is one transformer token (256 per frame): a
per-pixel colour embedding and a 16 x 16 stride-16 conv, as in a ViT. Each
block attends spatially within a frame (bidirectional), then temporally at
the same position over the frames so far (causal), then applies an MLP. Time
is rotary (RoPE) inside the temporal attention, as in Open-Oasis: queries and
keys are rotated by their frame, so attention sees how many frames apart two
frames are, and nothing else of time (the layered model's, which beat this
model's one learned embedding per window slot: docs/guides/recipe_attempts.md).
The action that produced frame t is added to frame t's tokens. The head predicts
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
prior learns to predict the choice from the previous frame's features and the
buttons going into the frame (KL-balanced, every nat paid for), and a dream
samples it from the prior, so the dream commits to one outcome. The prior
must see whatever decides the frame (the buttons, the timer in the border):
what it cannot see looks random to it, and the choice carries it instead. The layer that adds the choice starts at zero, so
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

from token_world.data.model_frames import BAND
from token_world.data.tetris_motion import CELL as MOTION_CELL, CODES as MOTION_CODES, OFFSETS as MOTION_OFFSETS
from token_world.data.tetris_motion import STILL as MOTION_STILL, UNKNOWN as MOTION_UNKNOWN
MOTION_SURE = 0.9      # motion_first's first step commits the cells at least this sure (and the most confident other)
from token_world.data.nes_palette import BORDER_SLOTS, NES_COLOURS

COLOURS = len(NES_COLOURS) + BORDER_SLOTS
SIZE = 256
ROPE_BASE = 10_000.0                                    # rotary's longest wavelength scale (frames)
SPACE_ROPE_BASE = 10.0                                  # RoPE-Mixed's frequency scale (its code's), places in 16 px


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


def rotary(x: torch.Tensor, times: torch.Tensor, axis: int = 1) -> torch.Tensor:
    """RoPE over time (Su et al., 2104.09864; Open-Oasis's temporal attention): x [..., hd] whose axis `axis`
    holds frames `times` [T] -> x with each pair of channels rotated by an angle proportional to its frame, so
    a query and a key score by how many frames apart they are, whatever window slot they sit in."""
    half = x.shape[-1] // 2
    freq = ROPE_BASE ** -(torch.arange(half, device=x.device, dtype=torch.float32) / half)
    angle = times.to(torch.float32)[:, None] * freq                                   # [T, hd / 2]
    shape = [1] * x.dim()
    shape[axis], shape[-1] = -1, half
    cos, sin = angle.cos().view(shape).to(x.dtype), angle.sin().view(shape).to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), -1)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, qk_norm: bool = False, rotate: bool = False):
        """rotate: queries and keys rotated by their frame (rotary time; the temporal attention)."""
        super().__init__()
        self.heads, self.rotate = heads, rotate
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
        if self.rotate:                                                    # [b, heads, n frames, hd]
            times = at + torch.arange(n, device=x.device)
            q, k = rotary(q, times, axis=2), rotary(k, times, axis=2)
        if cache is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        elif n > 1 and isinstance(at, int) and at == 0:
            k, v = cache.prefill(k, v)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            k, v, seen = cache.step(k, v, at)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=seen)
        return self.out(y.transpose(1, 2).reshape(b, n, d))


def mixed_rotary(x: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """x [..., h, hd] whose first 2P channels are rotated as P pairs (i, P + i) by angle [..., h, P]; the rest
    pass unchanged."""
    p = angle.shape[-1]
    cos, sin = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
    x1, x2, rest = x[..., :p], x[..., p:2 * p], x[..., 2 * p:]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos, rest), -1)


class SpatialAttention(Attention):
    """Attention within a frame; with rope (the space_rope option), half its channels rotated by each token's
    place on screen along learned 2D directions (RoPE-Mixed: per layer and head, initialised as its code does,
    two perpendicular directions at a random angle per head, magnitudes SPACE_ROPE_BASE ** -(i / P)); with
    local_grid (the token grid's side), a learned bias per head on each (dy, dx) offset between tokens added to
    the attention logits (Swin's relative position bias), from zero: a head can make itself as local as it needs.
    Every model's spatial attention was near-global (2026-10-15: the most local heads put 0.10-0.22 of their
    weight on the 3 x 3 tokens around, uniform 0.035), and a turning piece's cells barely conditioned on each other
    (2026-10-16)."""

    def __init__(self, dim: int, heads: int, rope: bool = False, qk_norm: bool = False, local_grid: int | None = None):
        super().__init__(dim, heads, qk_norm)
        self.rope = rope
        self.local = local_grid is not None
        if self.local:
            g = local_grid
            self.local_bias = nn.Parameter(torch.zeros(heads, (2 * g - 1) ** 2))
            yx = torch.stack(torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij"), -1).view(-1, 2)
            dy, dx = (yx[None] - yx[:, None] + g - 1).unbind(-1)                          # [N, N] offsets
            self.register_buffer("offsets", dy * (2 * g - 1) + dx, persistent=False)
        if rope:
            pairs = dim // heads // 4                                                  # half the channels
            mag = SPACE_ROPE_BASE ** -(torch.arange(pairs // 2 + pairs % 2).float() / (pairs // 2 + pairs % 2))
            turn = torch.rand(heads, 1) * 2 * torch.pi
            angles = torch.cat((turn.expand(-1, len(mag)), turn.expand(-1, len(mag)) + torch.pi / 2), 1)[:, :pairs]
            mags = torch.cat((mag, mag))[:pairs]
            self.rope_freq = nn.Parameter(torch.stack((mags * angles.sin(), mags * angles.cos()), -1))  # [h, P, (y, x)]

    def forward(self, x: torch.Tensor, place: torch.Tensor | None = None, placed: torch.Tensor | None = None):
        """x [M, N, d]; place [M or 1, N, 2] (y, x in 16-pixel units) and placed [N] (rope only; None: every
        token has its place)."""
        m, n, d = x.shape
        h = self.heads
        q, k, v = self.qkv(x).view(m, n, 3, h, d // h).unbind(2)                        # [M, N, h, hd]
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        if self.rope:
            angle = torch.einsum("mnc,hpc->mnhp", place.float(), self.rope_freq.float())
            rotated = 2 * self.rope_freq.shape[1]
            q, k = mixed_rotary(q, angle), mixed_rotary(k, angle)
            if placed is not None:
                keep = torch.ones(n, 1, d // h, device=x.device, dtype=x.dtype)
                keep[~placed, :, :rotated] = 0                                          # no place: content only
                q, k = q * keep, k * keep
        bias = self.local_bias[:, self.offsets].to(q.dtype) if self.local else None    # [h, N, N]
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=bias)
        return self.out(y.transpose(1, 2).reshape(m, n, d))

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
    QK-norm and a SwiGLU MLP (Llama-style) instead of LayerNorm and a GELU MLP; qk_norm: QK-norm alone;
    space_rope: the spatial attention rotated by each patch's place (RoPE-Mixed, SpatialAttention);
    action_norm: the frame's buttons set a scale and shift on each of the block's three norms (adaptive
    LayerNorm, as DiT and Open-Oasis condition every block), from zero, so a new model starts as without;
    local_grid: the spatial attention's learned bias per token offset (SpatialAttention)."""

    def __init__(self, dim: int, heads: int, modern: bool = False, space_rope: bool = False, qk_norm: bool = False,
                 action_norm: bool = False, local_grid: int | None = None):
        super().__init__()
        if action_norm:
            self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))     # (scale, shift) per norm
            nn.init.zeros_(self.modulation[1].weight)
            nn.init.zeros_(self.modulation[1].bias)
        else:
            self.modulation = None
        norm = RMSNorm if modern else nn.LayerNorm
        qk = modern or qk_norm
        self.norm_s, self.norm_t, self.norm_m = norm(dim), norm(dim), norm(dim)
        self.spatial = SpatialAttention(dim, heads, space_rope, qk, local_grid)
        self.temporal = Attention(dim, heads, qk, rotate=True)
        self.mlp = SwiGLU(dim) if modern else nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                                            nn.Linear(4 * dim, dim))

    def forward(self, x: torch.Tensor, cache: TemporalCache | None = None, at=0,
                place: torch.Tensor | None = None, action: torch.Tensor | None = None) -> torch.Tensor:
        """x [B, T, N, d]: window frames at..at+T-1. With a cache, frames before `at` come from it. place [1, N,
        2]: each patch's (row, column), for space_rope. action [B, T, d]: each frame's button embedding, for
        action_norm."""
        b, t, n, d = x.shape
        if self.modulation is None:
            ns, nt, nm = self.norm_s, self.norm_t, self.norm_m
        else:
            mods = self.modulation(action)[:, :, None].to(x.dtype).chunk(6, -1)        # each [B, T, 1, d]

            def modulated(norm, scale, shift):
                return lambda z: norm(z) * (1 + scale) + shift
            ns, nt, nm = (modulated(norm, mods[2 * i], mods[2 * i + 1])
                          for i, norm in enumerate((self.norm_s, self.norm_t, self.norm_m)))
        x = x + self.spatial(ns(x).reshape(b * t, n, d), place).view(b, t, n, d)
        y = nt(x).transpose(1, 2).reshape(b * n, t, d)
        x = x + self.temporal(y, causal=True, cache=cache, at=at).view(b, n, t, d).transpose(1, 2)
        return x + self.mlp(nm(x))


class Dynamics(nn.Module):
    def __init__(self, dim: int = 512, layers: int = 12, heads: int = 8, patch: int = 16,
                 frames: int = 64, colour_dim: int = 32, recompute: bool = False, modern: bool = False,
                 latent: tuple[int, int] | None = None, space_rope: bool = False, qk_norm: bool = False,
                 action_norm: bool = False, action_blocks: tuple[int, ...] | None = None,
                 change_weight: float = 1.0, motion: bool = False, motion_first: bool = False,
                 local_attention: bool = False):
        """space_rope: space is also rotary in the spatial attention (RoPE-Mixed: learned 2D directions per
        layer and head, by each patch's row and column), on top of the learned place per patch (space_pos):
        RoPE-Mixed + APE, the paper's choice at a fixed resolution (2403.13298: APE about doubles RoPE's gain
        there; a Tetris token's absolute place carries meaning, the clock digits, the next-piece box). The first
        space_rope runs (2026-10-07, without space_pos) learned slower than rotary time alone;
        qk_norm: queries and keys normalised per head in both attentions (as modern, without its other
        changes); action_norm: the buttons also condition every block's norms (Block); action_blocks (with
        action_norm): only these blocks' norms see the buttons, and they are not added to the input tokens, so
        the buttons enter nowhere else (2026-10-15: every actnorm model read a shift's direction in its first
        block only, before any piece's shape is built, and never learned a turn's); change_weight: the loss
        weight the model was trained with on game pixels that change from the frame before (masked_loss): its
        probabilities lean that much to change, and FrameDecoder takes the odds of every change back down by
        it; motion: each frame's motion (data/tetris_motion.py: per 8 x 8 cell, still, moved from where, or new)
        goes in with its visible tokens and comes out of every token, VideoJAM's joint appearance and motion
        (examples/papers, 2502.02492; motion_loss); motion_first (with motion): a generated frame's motion is drawn
        first, in two steps (committed_motion), and its pixels then given it (FrameDecoder: VideoJAM's motion
        guiding the video; training gives a share of masked frames all or part of their cells' motion);
        local_attention: the spatial attention gets a learned bias per token offset, from zero (SpatialAttention).
        All off when absent from a run's config (build)."""
        super().__init__()
        self.patch, self.frames, self.action_norm = patch, frames, action_norm
        self.change_weight = float(change_weight)
        if motion_first and not motion:
            raise ValueError("motion_first draws the motion first: it needs motion")
        self.motion, self.motion_first = motion, motion_first
        # motion_first's decoding: the motion in 2 steps (committed_motion, then the rest given it) or 1 (every
        # cell from the first pass), then the pixels; a decoding setting, not saved (2026-10-18: does s9 still
        # need the second step, 4 forwards a frame against 3?)
        self.motion_steps = 2
        self.motion_cells = (patch // MOTION_CELL) ** 2                  # motion cells per token
        if action_blocks is not None and not action_norm:
            raise ValueError("action_blocks chooses which blocks' norms see the buttons: it needs action_norm")
        self.action_blocks = tuple(action_blocks) if action_blocks is not None else None
        self.recompute = recompute     # keep only stage inputs for backward and recompute the rest: memory for time
        self.grid = SIZE // patch
        tokens = self.grid ** 2
        self.colour = nn.Embedding(COLOURS, colour_dim)
        self.patchify = nn.Conv2d(colour_dim, dim, patch, stride=patch)
        self.mask_token = nn.Parameter(torch.randn(dim) * 0.02)
        self.space_pos = nn.Parameter(torch.randn(tokens, dim) * 0.02)
        if space_rope:
            rows = torch.arange(self.grid).float()
            self.register_buffer("place", torch.stack(torch.meshgrid(rows, rows, indexing="ij"), -1).view(1, tokens, 2),
                                 persistent=False)
        else:
            self.place = None
        self.action = nn.Linear(9, dim)
        self.blocks = nn.ModuleList(Block(dim, heads, modern, space_rope, qk_norm,
                                          action_norm and (action_blocks is None or i in action_blocks),
                                          SIZE // patch if local_attention else None)
                                    for i in range(layers))
        self.norm = RMSNorm(dim) if modern else nn.LayerNorm(dim)
        self.pixel = nn.Linear(dim, patch * patch * colour_dim)       # each pixel's vector in colour space
        self.colour_bias = nn.Parameter(torch.zeros(COLOURS))
        if motion:                     # each cell's code, by its place in the token, into the token (from zero,
            self.motion_in = nn.Embedding(self.motion_cells * MOTION_CODES, dim)    # as VideoJAM's added rows)
            nn.init.zeros_(self.motion_in.weight)
            self.motion_out = nn.Linear(dim, self.motion_cells * MOTION_CODES)
        self.latent = tuple(latent) if latent else None                # a sampled choice per frame (groups, classes)
        if self.latent:
            size = self.latent[0] * self.latent[1]
            self.choice = nn.Linear(size, dim, bias=False)             # the choice, added to its frame's tokens:
            nn.init.zeros_(self.choice.weight)                         # zero at first, so it starts as without
            self.posterior_in = nn.Linear(2 * dim, dim)                # (the frame's change, the frame) per token
            self.posterior_out = nn.Linear(2 * dim, size)              # pooled (mean, max) -> logits
            # the prior: per group, attention pooling over the last frame's tokens (a learned query each, so it
            # can read a few tokens, such as the timer's), plus the incoming buttons' embedding -> logits
            self.prior_key = nn.Linear(dim, dim)
            self.prior_query = nn.Parameter(torch.randn(self.latent[0], dim) * dim ** -0.5)
            self.prior_out = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, self.latent[1]))

    def forward(self, frames: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor,
                cache: list[TemporalCache] | None = None, at=0,
                soft: tuple[torch.Tensor, torch.Tensor] | None = None, z: torch.Tensor | None = None,
                motion: torch.Tensor | None = None) -> torch.Tensor:
        """frames [B, T, 256, 256] palette indices, soft frames float [B, T, 256, 256, colour_dim]
        (embed_probs), or frames already embedded as patch tokens float [B, T, N, dim]; actions [B, T] bytes
        that produced each frame (-1 = none); mask [B, T, N] tokens hidden from the model; soft (training)
        (where [B, T] bool, tokens [B, T, N, dim]): those frames' patch embeddings, made outside (a rollout's
        soft frames; frames holds the real frame there), in a fixed shape so a compiled forward sees the same
        shapes every step; z [B, T, groups * classes] each frame's choice (with a latent; None: none); motion
        [B, T, N, motion_cells] each token's motion codes (motion_tokens; with motion; None or UNKNOWN: not
        given), added to the tokens, masked ones too (the caller withholds what the model may not see) ->
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
        if motion is not None and self.motion:
            x = x + self.motion_embedding(motion).to(x.dtype)
        action = self.action(button_bits(actions))                                   # [B, T, d]
        x = x + self.space_pos + (action[:, :, None] if self.action_blocks is None else 0)
        action = action if self.action_norm else None
        if z is not None:
            x = x + self.choice(z.to(x.dtype))[:, :, None]
        if cache is not None:
            for block, layer in zip(self.blocks, cache):
                x = block(x, layer, at, self.place, action)
            return self.norm(x)
        for block in self.blocks:
            x = (checkpoint(block, x, None, 0, self.place, action, use_reentrant=False) if remember
                 else block(x, place=self.place, action=action))
        return self.norm(x)

    embed_frames = 16                  # frames embedded at a time; ONNX export uses the window (one chunk)

    def _patches(self, frames: torch.Tensor) -> torch.Tensor:
        """[B, T, 256, 256] indices or [B, T, 256, 256, colour_dim] soft frames -> [B, T, N, dim]: per-pixel
        colour embedding, then the patch conv (index frames on the GPU: the same sums as an embedding bag,
        models/patch_kernel.py). Patch tokens [B, T, N, dim] pass through.

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
        if flat.is_cuda:                               # the same sums without building the embedded pixels
            from token_world.models.patch_kernel import bag_table, patch_bag
            with torch.autocast("cuda", enabled=False):                          # M in float32 (5.7 MB at 96 wide)
                table = bag_table(self.patchify.weight.float(), self.colour.weight.float())
            dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else table.dtype
            return (patch_bag(flat, table, dtype) + self.patchify.bias.to(dtype)).unflatten(0, (b, t))
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

    def read_choice(self, frames: torch.Tensor) -> torch.Tensor:
        """Real frames [B, T, 256, 256] -> posterior logits [B, T, groups, classes] (training compiles it)."""
        return self.posterior(self._patches(frames))

    def prior(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """A frame's features [..., N, dim] and the next frame's incoming buttons [...] -> logits
        [..., groups, classes] of the next frame's choice."""
        scores = torch.einsum("...nd,gd->...gn", self.prior_key(features), self.prior_query)
        scores = scores * self.prior_query.shape[-1] ** -0.5
        pooled = torch.einsum("...gn,...nd->...gd", scores.softmax(-1), features)    # [..., groups, dim]
        return self.prior_out(pooled + self.action(button_bits(actions))[..., None, :])

    def new_cache(self) -> list[TemporalCache]:
        return [TemporalCache(self.frames) for _ in self.blocks]

    def decoder(self, batch: int, device, compiled: bool = False, scored: bool = False,
                resync: int | None = None) -> "FrameDecoder":
        """This model's FrameDecoder for a batch size, the caller's autocast setting, `compiled`, `scored` and
        `resync` (see FrameDecoder), made on first use (its cache and CUDA graph are kept)."""
        device = torch.device(device)
        autocast = (torch.is_autocast_enabled(device.type), torch.get_autocast_dtype(device.type))
        key = (batch, device, autocast, compiled, scored, resync)
        decoders = self.__dict__.setdefault("_decoders", {})
        if key not in decoders:
            decoders[key] = FrameDecoder(self, batch, device, autocast, compiled, scored, resync)
        return decoders[key]

    def colour_table(self) -> torch.Tensor:
        """[COLOURS, colour_dim]: the head's colour vectors, the input embedding over sqrt(colour_dim)
        (as T5 scales its tied output, so the first logits are as small as an untied head's)."""
        return self.colour.weight * self.colour.embedding_dim ** -0.5

    def motion_embedding(self, motion: torch.Tensor) -> torch.Tensor:
        """Motion codes [B, T, N, motion_cells] -> what they add to each token [B, T, N, dim]: each known cell's code
        embedded by its place in the token, summed; nothing for UNKNOWN cells."""
        codes = motion.long()
        known = codes != MOTION_UNKNOWN
        place = torch.arange(self.motion_cells, device=codes.device) * MOTION_CODES
        added = self.motion_in(codes.clamp(max=MOTION_CODES - 1) + place) * known[..., None]
        return added.sum(-2)

    def drawn_motion(self, features: torch.Tensor) -> torch.Tensor:
        """A generated frame's features [..., N, dim] -> the motion it draws [..., N, motion_cells]: each cell's most
        likely code, the border's token rows STILL (as the data has them: drawn state, not picture; motion_loss
        never scores them, so their logits mean nothing). Most likely, not sampled with one number a frame: on the
        same 16 dreams at s3_motionfirst 100k, sampling broke pressed pieces more (lost / morph 0.91 / 1.20
        against 0.50 / 0.83 per 100 frames, wrong +128 2.2% against 1.5%; 2026-10-16)."""
        codes = self.motion_logits(features).argmax(-1)
        border = torch.zeros(self.grid, self.grid, dtype=torch.bool, device=codes.device)
        border[0] = border[-1] = True
        return codes.masked_fill(border.flatten()[:, None], MOTION_STILL)

    def committed_motion(self, features: torch.Tensor) -> torch.Tensor:
        """A generated frame's features [B, N, dim] -> the motion its first motion step commits [B, N, motion_cells]:
        each cell whose most likely code has at least MOTION_SURE probability (the still background, the stack),
        the border's STILL, and of the rest the one most confident cell, the others UNKNOWN for the second step to
        draw given these. A piece's cells drawn each on its own come out half one turn, half the other (s3_motionfirst
        104k: 46% of turns); one committed cell decides the turn for the rest (MaskGIT's confident-first, by cell)."""
        probs = self.motion_logits(features).float().softmax(-1)
        top, codes = probs.max(-1)                                          # [B, N, cells]
        border = torch.zeros(self.grid, self.grid, dtype=torch.bool, device=codes.device)
        border[0] = border[-1] = True
        border = border.flatten()[:, None].expand_as(top)
        sure = (top >= MOTION_SURE) | border
        unsure = torch.where(sure, -1.0, top).flatten(1)                     # [B, N * cells]
        first = F.one_hot(unsure.argmax(1), unsure.shape[1]).bool().view_as(sure) & ~sure
        codes = codes.masked_fill(border, MOTION_STILL)
        return torch.where(sure | first, codes, torch.full_like(codes, MOTION_UNKNOWN))

    def motion_logits(self, features: torch.Tensor) -> torch.Tensor:
        """Features [..., N, dim] -> each token's cells' motion logits [..., N, motion_cells, CODES]."""
        return self.motion_out(features).unflatten(-1, (self.motion_cells, MOTION_CODES))

    def loss_weights(self, target: torch.Tensor) -> torch.Tensor | None:
        """Real frames' pixels [B, T, N, P*P] (patch_pixels) -> each pixel's loss weight (masked_loss):
        change_weight on game pixels whose colour differs from the frame before's, 1 on the rest (the border,
        frame 0); None when change_weight is 1. Pieces move on a few game pixels a frame (0.16% of the masked
        pixels, 2026-10-02), and the cells that tell one turn from the other are a few of those: weighted, the
        moves get the loss's attention (recipe_attempts.md, 2026-10-15)."""
        if self.change_weight == 1.0:
            return None
        changed = torch.zeros_like(target, dtype=torch.bool)
        changed[:, 1:] = target[:, 1:] != target[:, :-1]
        changed &= ~border_pixels(self.patch, target.device)
        return 1.0 + (self.change_weight - 1.0) * changed.float()

    def change_undo(self, device) -> torch.Tensor | None:
        """[N, P*P, 1]: what weigh_changes multiplies the odds of each pixel's change by to give back the
        probabilities an unweighted loss trains to (1 / change_weight on game pixels, 1 on the border, as
        loss_weights weighs them); None when change_weight is 1."""
        if self.change_weight == 1.0:
            return None
        undo = torch.full((self.grid ** 2, self.patch ** 2), 1.0 / self.change_weight, device=device)
        return undo.masked_fill(border_pixels(self.patch, device), 1.0)[..., None]

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
                    config["colour_dim"], recompute, config.get("modern", False), config.get("latent"),
                    config.get("space_rope", False), config.get("qk_norm", False), config.get("action_norm", False),
                    config.get("action_blocks"), config.get("change_weight", 1.0), config.get("motion", False),
                    config.get("motion_first", False), config.get("local_attention", False))


def stacked_source(name: str, old: int) -> str:
    """A parameter's name in a model stacked from one of `old` blocks (deepen stack) -> its source's name:
    block j copies block j mod old."""
    if not name.startswith("blocks."):
        return name
    _, i, rest = name.split(".", 2)
    return f"blocks.{int(i) % old}.{rest}"


def deepen(model: Dynamics, source: dict[str, torch.Tensor], new_at: int | None = None, copy: bool = False,
           stack: int | None = None) -> None:
    """Grow a trained model deeper, keeping what it computes (depth up-scaling): `source` is a shallower
    model's state dict (same width) with L blocks; for `model`'s L' blocks, its block i becomes block
    round(i L' / L) (spread evenly: i * r when L' = r L; 3 -> 4 puts the new block third), or, with `new_at`
    (one block more), the new block goes at that position (0: first, L: last) and the old ones keep their
    order around it; everything else is copied. copy (with new_at): the new block starts as a copy of its
    neighbour (the block before it; the one after, for the first) and both have their output projections
    halved, so the pair adds about what the one block did (x + f(x)/2 + f(x + f(x)/2)/2) and the new block
    trains from working weights (so new_at 0 and 1 make the same model: a copy of block 0 twice; on s3_plain the
    features move by 0.17-0.25 and the loss by 0.64-1.07x at the growth); zero-start blocks stayed near no-ops, adding 0.003-0.05 of the stream and
    nothing to the loss when skipped, 10k steps after growing (2026-10-15). The blocks between keep their fresh weights but start with zero output projections (both
    attentions' and the MLP's), so each adds nothing to the residual stream: the deeper model starts as the
    shallower one and the new blocks learn from there."""
    old = 1 + max(int(k.split(".")[1]) for k in source if k.startswith("blocks."))
    new = len(model.blocks)
    if new < old:
        raise ValueError(f"{new} blocks cannot take {old}")
    if stack is not None:
        # the whole model stacked `stack` times, in order, first and last blocks included (G_stack: Du et al.,
        # NeurIPS 2024, 2405.15319; whole >> middle ~ back >> front, and in order beats each block twice in
        # place), plain copies, nothing zeroed or halved (2026-10-17, docs/guides/turn_direction.md)
        if new != old * stack or new_at is not None or copy:
            raise ValueError(f"stack {stack}: {old} blocks make {old * stack}, not {new}; no new_at or copy")
        model.load_state_dict({name: source[stacked_source(name, old)] for name in model.state_dict()})
        return
    if new_at is not None:
        if new != old + 1 or not 0 <= new_at <= old:
            raise ValueError(f"new_at {new_at}: one block more ({old} -> {new}) at 0..{old}")
        place = [i + (i >= new_at) for i in range(old)]
    else:
        place = [round(i * new / old) for i in range(old)]
    state = model.state_dict()
    for key, value in source.items():
        if key.startswith("blocks."):
            _, i, name = key.split(".", 2)
            key = f"blocks.{place[int(i)]}.{name}"
        state[key] = value
    if copy:
        if new_at is None:
            raise ValueError("copy starts one new block: give new_at")
        neighbour = new_at - 1 if new_at > 0 else 1
        for key in [k for k in state if k.startswith(f"blocks.{neighbour}.")]:
            state[f"blocks.{new_at}." + key.split(".", 2)[2]] = state[key].clone()
    model.load_state_dict(state)
    with torch.no_grad():
        if copy:
            for block in (model.blocks[new_at], model.blocks[neighbour]):
                mlp_out = block.mlp.out if isinstance(block.mlp, SwiGLU) else block.mlp[2]
                for layer in (block.spatial.out, block.temporal.out, mlp_out):
                    layer.weight.mul_(0.5)
                    if layer.bias is not None:
                        layer.bias.mul_(0.5)
            return
        for i, block in enumerate(model.blocks):
            if i not in place:
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


def motion_tokens(codes: torch.Tensor, patch: int) -> torch.Tensor:
    """Motion codes per 8 x 8 cell [..., 32, 32] -> per token [..., N, motion_cells] (each token's cells
    row-major, tokens row-major, as patch_pixels)."""
    return patch_pixels(codes, patch // MOTION_CELL)


def motion_cells(tokens: torch.Tensor, patch: int) -> torch.Tensor:
    """motion_tokens' inverse: [..., N, motion_cells] -> [..., 32, 32]."""
    return unpatch_pixels(tokens, patch // MOTION_CELL)


# per code: moved at all; moved along both axes, which only a turn's cells do
MOVED = torch.tensor([False, *[True] * len(MOTION_OFFSETS), False])
TURNED = torch.tensor([False, *[dy != 0 and dx != 0 for dy, dx in MOTION_OFFSETS], False])


def motion_loss(model: "Dynamics", features: torch.Tensor, motion: torch.Tensor, scored: torch.Tensor) -> dict:
    """Cross-entropy of the scored cells' motion: features [B, T, N, dim], motion [B, T, N, motion_cells] the real
    codes, scored [B, T, N, motion_cells] the cells to score (masked ones whose motion was not given) -> {"motion":
    the mean over the scored cells (known ones in the picture: not UNKNOWN, not in the border's token rows),
    "motion_moved" / "motion_turned": the share of those that moved / of a turn's whose most likely code is the
    real one (no gradient)}."""
    picture = torch.ones(model.grid, model.grid, dtype=torch.bool, device=scored.device)
    picture[0] = picture[-1] = False                                    # the border's token rows
    where = (scored.any(-1) & picture.flatten()).nonzero(as_tuple=True)
    real = motion[where].long()                                         # [M, cells]
    known = (real != MOTION_UNKNOWN) & scored[where]
    logits = model.motion_logits(features[where]).float()               # [M, cells, CODES]
    ce = F.cross_entropy(logits[known], real[known]) if known.any() else logits.sum() * 0
    with torch.no_grad():
        right = logits.argmax(-1) == real
        code = real.clamp(max=MOTION_CODES - 1)
        moved, turned = MOVED.to(real.device)[code] & known, TURNED.to(real.device)[code] & known
        nan = torch.tensor(float("nan"), device=real.device)
        return {"motion": ce, "motion_moved": right[moved].float().mean() if moved.any() else nan,
                "motion_turned": right[turned].float().mean() if turned.any() else nan}


def border_pixels(patch: int, device=None) -> torch.Tensor:
    """[N, P*P] bool: which pixels of each token lie in the letterbox border."""
    rows = torch.arange(SIZE, device=device)[:, None].expand(SIZE, SIZE)
    return patch_pixels((rows < BAND) | (rows >= SIZE - BAND), patch)


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
                  clean_context: float = 0.5, cut: torch.Tensor | None = None, hidden: float = 0.0,
                  hide: torch.Tensor | None = None) -> torch.Tensor:
    """[B, T, N] training mask; frame 0 is always visible.

    Every masked frame draws its own rate from MaskGIT's cosine schedule,
    cos(pi/2 * u) for u ~ U(0, 1), which favours high rates (generation starts
    from a fully masked frame). In a `clean_context` fraction of windows a cut
    is drawn and every frame before it is fully visible: the rollout condition,
    a clean past followed by frames to predict. Without such windows no
    training window ever had a fully visible past, and the model, 1-2% wrong
    one frame ahead when the past was 25-50% masked, was 23% wrong with a clean
    past, which is what rollouts give it (docs/guides/dynamics.md). cut [B]:
    windows with a positive entry use it as their clean cut. hidden: the share of masked frames hidden
    entirely, as a dream draws every frame: with part of a frame visible, the model can read what changed
    (a turn's direction) off the frame itself instead of the buttons (recipe_attempts.md, 2026-10-14). hide
    [B, T]: frames hidden entirely whenever they are predicted (not when before a clean cut): the frames a
    press reaches, so a turn is learned from the button as a dream must draw it (2026-10-16).
    """
    u = torch.rand(batch, frames, 1, device=device, generator=generator)
    mask = torch.rand(batch, frames, tokens, device=device, generator=generator) < torch.cos(math.pi / 2 * u)
    drawn = torch.randint(1, frames, (batch, 1), device=device, generator=generator)
    clean = torch.rand(batch, 1, device=device, generator=generator) < clean_context
    if cut is not None:
        forced = (cut > 0)[:, None]
        drawn, clean = torch.where(forced, cut[:, None], drawn), clean | forced
    if hidden:                                    # drawn last: the cuts and rates are the same without it
        mask |= torch.rand(batch, frames, 1, device=device, generator=generator) < hidden
    if hide is not None:
        mask |= hide[..., None]
    before_cut = torch.arange(frames, device=device)[None] < torch.where(clean, drawn, 1)
    mask &= ~before_cut[..., None]
    return mask


def masked_loss(model: Dynamics, features: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                chunk: int = 4096, weight: torch.Tensor | None = None):
    """Cross-entropy over the pixels of masked tokens.

    features [B, T, N, dim], target [B, T, N, P*P], mask [B, T, N] (, weight [B, T, N, P*P] each pixel's
    weight, default all equal) -> (the weighted mean over all scored pixels, per-pixel loss [M, P*P] without
    gradient and unweighted, the masked tokens' (b, t, n)). The head's [M, P*P, COLOURS] logits are 58 values
    per pixel, so they are never stored: see HeadCrossEntropy.
    """
    where = mask.nonzero(as_tuple=True)
    y = target[where].long()
    if weight is None:
        share = even_share(y)
    else:
        share = weight[where].float()
        share = share / share.sum().clamp_min(1e-12)
    total, per_pixel = HeadCrossEntropy.apply(features[where], model.pixel.weight, model.pixel.bias,
                                              model.colour_table(), model.colour_bias, y, share, chunk)
    return total, per_pixel, where


def even_share(y: torch.Tensor) -> torch.Tensor:
    """Targets [...] -> every pixel's equal share of HeadCrossEntropy's total (the plain mean)."""
    return torch.full(y.shape, 1.0 / max(y.numel(), 1), device=y.device)


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
    def forward(ctx, h, weight, bias, table, colour_bias, y, share, chunk: int):
        """h [M, dim], weight [P*P*e, dim], bias [P*P*e], table [COLOURS, e], colour_bias [COLOURS],
        y [M, P*P] long, share [M, P*P] float32 each pixel's share of the total (summing to 1)."""
        per_pixel = torch.empty(y.shape, dtype=torch.float32, device=h.device)
        grads = [torch.empty_like(h)] + [torch.zeros_like(t, dtype=torch.float32)
                                         for t in (weight, bias, table, colour_bias)]
        step = _head_chunk_fused if h.is_cuda else _head_chunk
        for i in range(0, h.shape[0], chunk):
            loss, g_h, g_w, g_b, g_t, g_cb = step(h[i:i + chunk], weight, bias, table, colour_bias,
                                                  y[i:i + chunk], share[i:i + chunk])
            per_pixel[i:i + chunk] = loss
            grads[0][i:i + chunk] = g_h
            for total, part in zip(grads[1:], (g_w, g_b, g_t, g_cb)):
                total += part
        ctx.save_for_backward(*grads)
        ctx.mark_non_differentiable(per_pixel)
        return (per_pixel * share).sum(), per_pixel

    @staticmethod
    def backward(ctx, grad_total, grad_per_pixel):
        return (*(g * grad_total for g in ctx.saved_tensors), None, None, None)


def _head_chunk(h, weight, bias, table, colour_bias, y, share):
    """One chunk of HeadCrossEntropy -> per-pixel loss [c, P*P] and the gradients (float32 for the head)."""
    c, e = h.shape[0], table.shape[1]
    pixels = F.linear(h, weight, bias).view(c, -1, e)                            # [c, P*P, e]
    loss, g = _ce_and_grad(pixel_logits(pixels, table, colour_bias), y, table.shape[0], share)
    g = g.view(c, -1, table.shape[0])                                          # [c, P*P, COLOURS]
    g_pixels = (g @ table.to(g.dtype)).view(c, -1)                              # [c, P*P*e]
    return (loss, g_pixels @ weight.to(g_pixels.dtype), (g_pixels.T @ h.to(g_pixels.dtype)).float(),
            g_pixels.float().sum(0), (g.flatten(0, 1).T @ pixels.flatten(0, 1).to(g.dtype)).float(),
            g.float().sum((0, 1)))


def _ce_and_grad(logits: torch.Tensor, y: torch.Tensor, colours: int, share: torch.Tensor):
    """logits [c, P*P, colours], y [c, P*P], share [c, P*P] -> per-pixel loss [c, P*P] float32 and the
    gradient of the share-weighted loss for the logits [c, P*P, colours] in the logits' dtype (softmax minus
    one-hot, times each pixel's share)."""
    logp = logits.float().log_softmax(-1)
    loss = -logp.gather(-1, y[..., None])[..., 0]
    grad = logp.exp() - F.one_hot(y, colours)
    return loss, (grad * share[..., None]).to(logits.dtype)


def _head_chunk_fused(h, weight, bias, table, colour_bias, y, share, precision: str | None = None):
    """_head_chunk on the GPU: the same results, with everything per pixel (the logits against the colour table,
    the softmax, the loss and its gradients) in one fused kernel (models/head_kernel.py; 41% of a 96 x 3 model's
    step as separate ops, 2026-10-14)."""
    from token_world.models.head_kernel import PRECISION, head_ce
    c, e = h.shape[0], table.shape[1]
    pixels = F.linear(h, weight, bias)                                           # [c, P*P*e]
    loss, g_pixels, g_table, g_colour_bias = head_ce(pixels.view(-1, e), table, colour_bias, y.reshape(-1),
                                                     share.reshape(-1), precision or PRECISION)
    g_pixels = g_pixels.view(c, -1)
    return (loss.view(c, -1), g_pixels @ weight.to(g_pixels.dtype), (g_pixels.T @ h.to(g_pixels.dtype)).float(),
            g_pixels.float().sum(0), g_table, g_colour_bias)


GHOST = 0.9         # a generated pixel whose most likely colour is below this probability is unsure (a ghost)
UNIMIX = 0.01       # each choice keeps 1% uniform probability (DreamerV3): no option is ever impossible


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
    previews). resync (scored only): before a frame goes into the cache, each
    patch with at least this many pixels whose most likely colour is not the
    real one is replaced by the real patch, so a rollout's history stays in
    sync with the real game (and the real buttons that follow it) and keeps
    only the model's small errors (its haze, a few pixels off); the rest of
    the patch keeps the model's own soft pixels (and a motion model's
    replaced patches take the real motion, so pixels and motion agree). A model trained with a
    change_weight w (masked_loss) is decoded with the odds of every change
    from the frame before (the last prefilled frame, then each frame's most
    likely colours) times 1/w, which gives back the probabilities an
    unweighted loss trains to; dreams can weigh changes further on top
    (Dreamer.set_change_weight).
    """

    def __init__(self, model: Dynamics, batch: int, device, autocast, compiled: bool = False,
                 scored: bool = False, resync: int | None = None):
        self.model, self.batch, self.autocast, self.resync = model, batch, autocast, resync
        self.cache = model.new_cache()
        self.act = torch.zeros(batch, 1, dtype=torch.long, device=device)
        self.at = torch.zeros((), dtype=torch.long, device=device)
        self.target = torch.zeros(batch, SIZE, SIZE, dtype=torch.long, device=device) if scored else None
        # a motion model's real motion codes for the frame (scored with resync: the replaced patches' motion)
        self.target_motion = (torch.full((batch, SIZE // MOTION_CELL, SIZE // MOTION_CELL), MOTION_UNKNOWN,
                                         dtype=torch.uint8, device=device)
                              if scored and resync is not None and model.motion else None)
        # with a latent: the last frame's features, which the prior reads with the next frame's buttons
        self.features = (torch.zeros(batch, model.grid ** 2, model.prior_key.in_features, device=device)
                         if model.latent else None)
        # the last frame's most likely colours and the weight on changes from them (weigh_changes; the graph
        # reads both in place, so the weight can change between frames): in dreams always, in scored rollouts
        # only to undo the model's own change_weight
        self.undo = model.change_undo(device)
        weighs = not scored or self.undo is not None
        self.last = torch.zeros(batch, SIZE, SIZE, dtype=torch.long, device=device) if weighs else None
        self.change_weight = torch.ones((), device=device) if weighs else None
        self.graph = self.out = None
        self._step = torch.compile(self._decode) if compiled and self.act.is_cuda else self._decode

    def prefill(self, frames: torch.Tensor, actions: torch.Tensor, motion: torch.Tensor | None = None) -> None:
        """frames [B, t, 256, 256] (or soft frames or patch tokens, Dynamics.forward), incoming actions
        [B, t] (, motion codes [B, t, 32, 32], with a motion model; None: not known, as from a screenshot) ->
        positions 0..t-1."""
        b, t = frames.shape[:2]
        if self.last is not None and not frames.is_floating_point():     # palette frames: the last one's colours
            self.last.copy_(frames[:, -1])
        visible = torch.zeros(b, t, self.model.grid ** 2, dtype=torch.bool, device=frames.device)
        with torch.no_grad():
            tokens = motion_tokens(motion, self.model.patch) if motion is not None else None
            features = self.model(frames, actions, visible, cache=self.cache, motion=tokens)
            if self.features is not None:
                self.features.copy_(features[:, -1])

    @torch.no_grad()
    def next(self, action: torch.Tensor, at: int, target: torch.Tensor | None = None,
             target_motion: torch.Tensor | None = None) -> dict:
        """action [B, 1] incoming, at: the frame's position (, target [B, 256, 256] the real frame when
        scored; target_motion [B, 32, 32] its motion codes, for a motion model's resync; None: not known) -> {"soft": [B, 256, 256, colour_dim] the soft frame that went into the context, "unsure":
        [B] its ghost pixels, "tokens": [B, N, dim] its patch tokens, and "wrong": [B] its pixels expected
        wrong (scored) or "probs": [B, 256, 256, COLOURS] its colour probabilities}."""
        self.act.copy_(action)
        self.at.fill_(at)
        if self.target is not None:
            self.target.copy_(target)
        if self.target_motion is not None:
            if target_motion is None:
                self.target_motion.fill_(MOTION_UNKNOWN)
            else:
                self.target_motion.copy_(target_motion)
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
            z = (sample_choice(model.prior(self.features, self.act[:, 0]))[:, None]           # this frame's choice
                 if self.features is not None else None)
            features = model(blank, self.act, hidden, self.cache, self.at, z=z)[:, 0]
            moved = model.drawn_motion(features) if model.motion else None              # [B, N, cells]: its own
            if model.motion_first:                       # the motion (in motion_steps), then the pixels given it
                if model.motion_steps == 2:
                    committed = model.committed_motion(features)
                    second = model(blank, self.act, hidden, self.cache, self.at, z=z, motion=committed[:, None])[:, 0]
                    moved = torch.where(committed != MOTION_UNKNOWN, committed, model.drawn_motion(second))
                features = model(blank, self.act, hidden, self.cache, self.at, z=z, motion=moved[:, None])[:, 0]
            probs = model.logits(features).float().softmax(-1)                   # [B, N, P*P, COLOURS]
            if self.last is not None:
                weight = self.change_weight if self.undo is None else self.change_weight * self.undo
                probs = weigh_changes(probs, patch_pixels(self.last, model.patch), weight)
                self.last.copy_(unpatch_pixels(probs.argmax(-1), model.patch))
            soft = model.embed_probs(probs)
            tokens = model._patches(soft[:, None])
            if self.resync is not None:                 # the patches that went off: the real ones instead
                real = patch_pixels(self.target, model.patch)                    # [B, N, P*P]
                diverged = (probs.argmax(-1) != real).sum(-1) >= self.resync      # [B, N]
                tokens = torch.where(diverged[:, None, :, None],
                                     model._patches(self.target[:, None]).to(tokens.dtype), tokens)
                resynced = diverged.float().mean(1)
                if self.target_motion is not None:      # and their real motion
                    real_moved = motion_tokens(self.target_motion, model.patch).to(moved.dtype)   # [B, N, cells]
                    moved = torch.where(diverged[..., None], real_moved, moved)
            written = model(tokens, self.act, torch.zeros_like(hidden), self.cache, self.at, z=z,
                            motion=moved[:, None] if moved is not None else None)
            if self.features is not None:                                        # for the next frame's prior
                self.features.copy_(written[:, 0])
            out = {"soft": soft, "tokens": tokens[:, 0], "unsure": (probs.amax(-1) < GHOST).sum((1, 2)).float()}
            if z is not None:
                out["z"] = z[:, 0]                                                # the choice it drew
            if moved is not None:
                out["motion"] = motion_cells(moved, model.patch)                 # [B, 32, 32] the motion it drew
            if self.target is not None:
                right = probs.gather(-1, patch_pixels(self.target, model.patch)[..., None])
                out["wrong"] = 1 - right.mean((1, 2, 3))
                if self.resync is not None:
                    out["resynced"] = resynced                                    # [B] its patches replaced
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
            compiled: bool = False, resync: int | None = None, motion: torch.Tensor | None = None) -> tuple:
    """Generate frames start..T-1 one after another from the real frames before `start`, scored against the
    real ones in `window`; resync: each frame's diverged patches replaced by the real ones before the next
    frame reads it (FrameDecoder).

    window [B, T, 256, 256], actions [B, T] incoming. The real frames fill the
    decoder's cache once; each generated frame is decoded against it and its
    soft frame written into it. -> (the soft frames as patch tokens [B, T -
    start, N, dim], each one's ghost pixels [B, T - start], each one's pixels
    expected wrong [B, T - start], and with a latent the choice each was drawn with [B, T - start,
    groups * classes], else None, and with resync each frame's share of patches replaced [B, T - start], else
    None, and with a motion model the motion each frame went into the cache with [B, T - start, 32, 32], else
    None). compiled: see FrameDecoder (training rollouts). motion: the window's motion codes [B, T, 32, 32] (a
    motion model's context before `start`, and with resync the replaced patches' motion; None: not known).
    """
    b, t = window.shape[:2]
    decoder = model.decoder(b, window.device, compiled, scored=True, resync=resync)
    decoder.prefill(window[:, :start], actions[:, :start], motion[:, :start] if motion is not None else None)
    out = [decoder.next(actions[:, at:at + 1], at, window[:, at], motion[:, at] if motion is not None else None)
           for at in range(start, t)]
    tokens, unsure, wrong = (torch.stack([o[k] for o in out], 1) for k in ("tokens", "unsure", "wrong"))
    return (tokens, unsure, wrong, torch.stack([o["z"] for o in out], 1) if model.latent else None,
            torch.stack([o["resynced"] for o in out], 1) if resync is not None else None,
            torch.stack([o["motion"] for o in out], 1) if model.motion else None)


class Dreamer:
    """The model as a game engine: one generated frame per action, for as long as it is played.

    Starts from real frames [T0, 256, 256] and the incoming actions [T0] that
    produced them (or a batch of dreams, side by side: [B, T0, 256, 256] and
    [B, T0]). Each step decodes the next frame against a TemporalCache
    and keeps its soft frame's patch tokens as context (the frames as the
    model reads them, 98 KB a frame); when the window is full
    (model.frames), the last `keep` frames are re-encoded at positions
    0..keep-1 and generation goes on (keep up to model.frames - 1: more
    history, re-encoded more often).
    """

    def __init__(self, model: Dynamics, frames: torch.Tensor, actions: torch.Tensor, keep: int | None = None,
                 change_weight: float = 1.0, motion: torch.Tensor | None = None):
        keep = model.frames * 3 // 4 if keep is None else keep
        if not 1 <= keep < model.frames:
            raise ValueError("keep must be 1..model.frames - 1")
        self.model, self.keep = model, keep
        self.batched = frames.dim() == 4
        if not self.batched:
            frames, actions = frames[None], torch.as_tensor(actions)[None]
        self.frames = list(model._patches(frames).detach().float().unbind(1))    # patch tokens, each [B, N, dim]
        self.actions = list(torch.as_tensor(actions, device=frames.device).long().unbind(1))     # each [B]
        # a motion model's history: each frame's motion codes [B, 32, 32] (the start's as given, else UNKNOWN, as
        # a screenshot's; then each frame's own), re-encoded with the frames; `motion` is the last frame's
        self.motions = None
        if model.motion:
            known = motion if motion is None or self.batched else motion[None]
            self.motions = (list(torch.as_tensor(known, device=frames.device).unbind(1)) if known is not None else
                            [torch.full((len(frames), SIZE // MOTION_CELL, SIZE // MOTION_CELL), MOTION_UNKNOWN,
                                        dtype=torch.uint8, device=frames.device)]
                            * frames.shape[1])
        self.motion = None
        self._encode()
        self.decoder.last.copy_(frames[:, -1])
        self.set_change_weight(change_weight)

    def set_change_weight(self, weight: float) -> None:
        """The odds of every pixel change times `weight` from the next frame on (weigh_changes; 1: none, on top of
        undoing the model's own change_weight)."""
        self.decoder.change_weight.fill_(weight)

    def _encode(self) -> None:
        """Re-encode the window at positions 0..keep-1 (after the start, and whenever it slides)."""
        self.frames, self.actions = self.frames[-self.keep:], self.actions[-self.keep:]
        if self.motions is not None:
            self.motions = self.motions[-self.keep:]
        device = self.frames[0].device
        if not hasattr(self, "decoder"):                # its own: a dream keeps state in its cache between
            autocast = (torch.is_autocast_enabled(device.type), torch.get_autocast_dtype(device.type))
            self.decoder = FrameDecoder(self.model, self.frames[0].shape[0], device, autocast)   # steps
        self.decoder.prefill(torch.stack(self.frames, 1), torch.stack(self.actions, 1),
                             torch.stack(self.motions, 1) if self.motions is not None else None)

    @torch.no_grad()
    def step(self, action) -> torch.Tensor:
        """The frame that `action` (the controller byte, -1 for none; batched: [B], one per dream, or one for
        all) produces -> its colour probabilities [256, 256, COLOURS] ([B, ...] batched) float32."""
        if len(self.frames) == self.model.frames:
            self._encode()
        device = self.frames[0].device
        act = torch.as_tensor(action, device=device).long().reshape(-1, 1).expand(self.decoder.batch, 1)
        out = self.decoder.next(act, len(self.frames))
        self.frames.append(out["tokens"].float())
        self.actions.append(act[:, 0].clone())
        if self.motions is not None:
            self.motion = out["motion"] if self.batched else out["motion"][0]
            self.motions.append(out["motion"].to(torch.uint8))
        return out["probs"] if self.batched else out["probs"][0]
