"""A world model over the video VAE's latents: EDM diffusion on clean context, as DIAMOND (2026-10-10).

The frozen VAE (models/video_vae.py, run video_vae_n1) turns each frame into 256 latents of 16 numbers (one
per 16 x 16 patch, unit RMS), causal in time. This model predicts them, frame after frame, from the frames
before and the buttons: the transformer of the pixel model it is compared with (world_model_tetris_
rope_actnorm: models/dynamics.py Block: spatial attention, causal temporal attention with rotary time,
RoPE-Mixed space, QK-norm, the frame's conditioning in every block's norms), narrowed from dim 96 to 64 so
that with the VAE it has the pixel model's parameter budget (2.42M vs 2.45M).

DIAMOND's recipe (2405.12399; examples/repos/diamond: models/diffusion/denoiser.py, config/trainer.yaml):
- EDM (Karras et al. 2022): the network F sees the noisy frame scaled by c_in and the noise level as
  c_noise = log(sigma) / 4, and the denoised frame is D = c_skip x + c_out F, so under heavy noise it predicts
  the clean frame and under light noise the noise; DIAMOND found this stable down to one sampling step, where
  noise prediction (DDPM) drifted. Its settings, for data of standard deviation 0.5, are scaled to our
  unit-RMS latents (SIGMA_DATA 1: every sigma doubled): training sigmas log-normal (LOC, SCALE, clipped to
  TRAIN_SIGMAS), offset noise (one random shift per frame and channel, OFFSET: a whole channel at once, as a
  palette change moves every cell), and `steps` = 3 Euler steps from SAMPLE_SIGMAS[1] to SAMPLE_SIGMAS[0]
  (rho 7) when sampling.
- Clean context (teacher forcing): only the frame being predicted is noised; the frames before it are the
  real ones, unnoised, as when dreaming (the dream's frames are clean in the cache). The flow-matching model
  before this (run world_model_tetris_vae_flow, to 17k) gave every frame its own noise level (Diffusion
  Forcing), so the frame before the predicted one was more than half noise half the time: it held the board
  but never moved a piece. Trained on whole windows in parallel as Self Forcing (2506.08009) describes
  teacher forcing: every frame twice, a clean context copy (causal among the clean copies) and a noisy target
  copy that sees the clean copies before it and itself (pair_block). Both copies are conditioned by the
  incoming buttons, the target by its noise level, the context by a learned clean embedding.
- Context noise (GameNGen, 2408.14837 "noise augmentation"; `context_noise`, 0 = off): each training window's
  context copies get Gaussian noise of one level drawn uniformly from 0 to context_noise, and the context
  conditioning carries that level; dreams run at level 0. A model taught only on real history never sees
  its own errors in the cache: the fast run (2026-10-11) spawned a new piece given real frames nearly as well
  as the pixel model, but in its own long dreams only 0.3-0.5 of the time (the pixel model 1.0).

Tokens (group): a VAE of 8 x 8 latents (one per NES tile and well cell, models/video_vae.py per frame) has
1,024 a frame; the generator takes each 2 x 2 of them as one token of 4 D numbers (DiT's patch 2 over a
latent grid), so it keeps 256 tokens and its speed while every latent it predicts is one cell.

Dreaming (LatentDreamer): the clean frames are in the temporal cache; the next frame starts as noise and takes
`steps` Euler steps (each a pass at `at` that sees the cache and itself, exactly the target copy's view), then
a clean pass writes it into the cache; the VAE decodes it (a per-frame VAE from its own latents, n1 from the
last WINDOW) into colour probabilities, so every check made for the pixel model applies unchanged.

On a per-frame VAE with sprite numbers (video_vae_sprite, 2026-10-12) a token is 2 x 2 cells of 16 colour and 2
sprite numbers (72), and the trainer (scripts/train_latent_ui.py) encodes its streams continuously
(video_vae.StreamEncoder), so each training frame's sprite numbers look back 128 frames as the VAE's did.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from token_world.models.dynamics import Block, TemporalCache, button_bits, rotary
from token_world.models.video_vae import build as build_vae
from token_world.models.video_vae import frames as untile

WINDOW, STRIDE = 16, 8                          # the VAE's windows (scripts/train_vae_ui.py): latents see 8-15 frames
SIGMA_DATA = 1.0                                # the latents' standard deviation (unit RMS); DIAMOND's 0.5 x 2
LOC, SCALE = -0.4 + math.log(2), 1.2            # training sigmas: log-normal (DIAMOND's, scaled)
TRAIN_SIGMAS = (0.004, 40.0)                    # clipped to (DIAMOND's 2e-3, 20, scaled)
OFFSET = 0.6                                    # offset noise (DIAMOND's 0.3, scaled)
SAMPLE_SIGMAS, RHO = (0.004, 10.0), 7           # the sampler's sigmas, smallest and largest (DIAMOND's 2e-3, 5)
MOVED = 0.05                                    # a patch's latent change (RMS) that counts as moved: still patches
                                                # change by 0.000, moved ones by ~1.5 (scripts/latent_check.py)


def sample_sigmas(steps: int) -> list[float]:
    """The sampler's noise levels, largest first, ending at 0 (EDM's rho schedule, as DIAMOND build_sigmas)."""
    lo, hi = SAMPLE_SIGMAS[0] ** (1 / RHO), SAMPLE_SIGMAS[1] ** (1 / RHO)
    return [(hi + i / max(steps - 1, 1) * (lo - hi)) ** RHO for i in range(steps)] + [0.0]


def fourier(level: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
    """Levels [...] -> their random Fourier features [..., 2 len(freq)] (DIAMOND's NoiseEmbedding)."""
    angle = 2 * math.pi * level.float()[..., None] * freq
    return torch.cat((angle.cos(), angle.sin()), -1)


def conditioners(sigma: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Noise levels [...] -> EDM's (c_in, c_skip, c_out, c_noise), with the offset noise counted in (DIAMOND)."""
    s = (sigma ** 2 + OFFSET ** 2).sqrt()
    c_in = 1 / (s ** 2 + SIGMA_DATA ** 2).sqrt()
    c_skip = SIGMA_DATA ** 2 / (s ** 2 + SIGMA_DATA ** 2)
    return c_in, c_skip, s * c_skip.sqrt(), s.log() / 4


def pair_temporal(att, yc: torch.Tensor, yt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """models/dynamics.py Attention (rotary time) over both copies: yc, yt [M, T, d] the clean and the noisy
    copies of T frames -> each one's output. A clean copy sees the clean copies up to its own frame; a noisy one
    the clean copies before its frame and itself (the dream's view: the cache, then the frame being made)."""
    m, t, d = yc.shape
    heads = att.heads
    times = torch.arange(t, device=yc.device)

    def qkv(y):
        q, k, v = att.qkv(y).view(m, t, 3, heads, d // heads).permute(2, 0, 3, 1, 4)
        if att.q_norm is not None:
            q, k = att.q_norm(q), att.k_norm(k)
        return rotary(q, times, axis=2), rotary(k, times, axis=2), v

    (qc, kc, vc), (qt, kt, vt) = qkv(yc), qkv(yt)
    oc = F.scaled_dot_product_attention(qc, kc, vc, is_causal=True)
    seen = torch.cat((times[None] < times[:, None], torch.eye(t, dtype=torch.bool, device=yc.device)), 1)
    ot = F.scaled_dot_product_attention(qt, torch.cat((kc, kt), 2), torch.cat((vc, vt), 2), attn_mask=seen)
    return tuple(att.out(o.transpose(1, 2).reshape(m, t, d)) for o in (oc, ot))


def pair_block(block: Block, xc: torch.Tensor, xt: torch.Tensor, cond_c: torch.Tensor, cond_t: torch.Tensor,
               place: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    """models/dynamics.py Block (with action_norm) over both copies [B, T, N, d], each with its conditioning
    [B, T, d]: the same weights and the same work as Block.forward, but the temporal attention is pair_temporal."""
    b, t, n, d = xc.shape

    def norms(cond):
        mods = block.modulation(cond)[:, :, None].to(xc.dtype).chunk(6, -1)
        return [lambda z, norm=norm, scale=mods[2 * i], shift=mods[2 * i + 1]: norm(z) * (1 + scale) + shift
                for i, norm in enumerate((block.norm_s, block.norm_t, block.norm_m))]

    (sc, tc, mc), (st, tt, mt) = norms(cond_c), norms(cond_t)
    both = torch.cat((sc(xc), st(xt))).reshape(2 * b * t, n, d)
    spatial = block.spatial(both, place).view(2 * b, t, n, d)
    xc, xt = xc + spatial[:b], xt + spatial[b:]
    oc, ot = pair_temporal(block.temporal, tc(xc).transpose(1, 2).reshape(b * n, t, d),
                           tt(xt).transpose(1, 2).reshape(b * n, t, d))
    xc = xc + oc.view(b, n, t, d).transpose(1, 2)
    xt = xt + ot.view(b, n, t, d).transpose(1, 2)
    return xc + block.mlp(mc(xc)), xt + block.mlp(mt(xt))


class LatentDynamics(nn.Module):
    def __init__(self, vae_config: dict, dim: int = 64, layers: int = 8, heads: int = 4, frames: int = 64,
                 space_rope: bool = True, qk_norm: bool = True, steps: int = 3, recompute: bool = False,
                 context_noise: float = 0.0, group: int = 1):
        """recompute: keep only each block's inputs in training and recompute the rest in the backward pass
        (width 96 needed 23 GB of the 3090's 24 without it, and spilled into system memory at a tenth of the
        speed, 2026-10-11). context_noise: the largest context noise level in training (module docstring).
        group: the VAE latents per token side (module docstring: Tokens)."""
        super().__init__()
        self.frames, self.steps, self.recompute, self.context_noise = frames, steps, recompute, context_noise
        self.vae = build_vae(vae_config).eval()
        self.vae.requires_grad_(False)
        self.group = group
        self.latent, self.grid = self.vae.numbers * group ** 2, self.vae.grid // group    # a token's numbers, side
        self.span = WINDOW if self.vae.enc_time else 1                # the latents a frame is decoded from
        tokens = self.grid ** 2
        self.inp = nn.Linear(self.latent, dim)
        self.space_pos = nn.Parameter(torch.randn(tokens, dim) * 0.02)
        if space_rope:
            rows = torch.arange(self.grid).float()
            self.register_buffer("place", torch.stack(torch.meshgrid(rows, rows, indexing="ij"), -1).view(1, tokens, 2),
                                 persistent=False)
        else:
            self.place = None
        self.action = nn.Linear(9, dim)
        # the noise level, as DIAMOND's NoiseEmbedding: random Fourier features of c_noise, then an MLP; a clean
        # context frame has a learned embedding instead
        self.register_buffer("noise_freq", torch.randn(dim // 2))
        self.noise = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.clean = nn.Parameter(torch.zeros(dim))
        if context_noise:                           # the context's noise level: Fourier features, zero at the start
            self.context_level = nn.Linear(dim, dim)
            nn.init.zeros_(self.context_level.weight)
            nn.init.zeros_(self.context_level.bias)
        self.blocks = nn.ModuleList(Block(dim, heads, False, space_rope, qk_norm, action_norm=True)
                                    for _ in range(layers))
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, self.latent)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.vae.eval()                             # frozen: always as it was trained
        return self

    def trained_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("vae.")]

    @torch.no_grad()
    def latents(self, frames: torch.Tensor) -> torch.Tensor:
        """Model frames [B, T, 256, 256] -> their tokens [B, T, N, D]. A per-frame VAE's: each frame's from
        itself and, its sprite numbers, the frames before it here (a trainer streams instead, models/video_vae.py
        StreamEncoder). A time-block VAE's (n1): each from the WINDOW frames up to it (STRIDE new a window, as
        it was trained: the first WINDOW from fewer), the convolutions once per frame (VideoVAE.features); with
        sprite numbers (n1sn) those also see up to `history` frames before the window, as in training."""
        if not self.vae.enc_time:
            return self.tokens(self.vae.encode(frames))
        b, t = frames.shape[:2]
        x = self.vae.features(frames)
        parts = [self.vae.encode_features(x[:, :min(t, WINDOW)])]
        for start in range(WINDOW, t, STRIDE):
            end = min(start + STRIDE, t)
            before = x[:, max(0, end - WINDOW - self.vae.history):end - WINDOW] if self.vae.sprite else None
            parts.append(self.vae.encode_features(x[:, end - WINDOW:end], before)[:, -(end - start):])
        return self.tokens(torch.cat(parts, 1))

    def tokens(self, z: torch.Tensor) -> torch.Tensor:
        """The VAE's latents [B, T, G, G, numbers] -> tokens [B, T, N, D]: each group x group of them as one."""
        b, t = z.shape[:2]
        g, side = self.group, self.grid
        return (z.float().view(b, t, side, g, side, g, -1).transpose(3, 4)
                .reshape(b, t, side * side, self.latent))

    @torch.no_grad()
    def decode_probs(self, latents: torch.Tensor) -> torch.Tensor:
        """The last `span` frames' latents [B, T, N, D] -> the last frame's colour probabilities [B, 256, 256,
        COLOURS] (a time-block VAE decodes a frame from the latents up to it, a per-frame VAE from its own)."""
        z = self.cells(latents[:, -self.span:])
        logits = self.vae.logits(z, last=1)                                                 # [B*G*G, C, P, P]
        return untile(logits.float().softmax(1).permute(0, 2, 3, 1), z.shape[0])

    def cells(self, latents: torch.Tensor) -> torch.Tensor:
        """tokens' inverse: tokens [B, T, N, D] -> the VAE's latents [B, T, G, G, numbers]."""
        b, t = latents.shape[:2]
        g, side = self.group, self.grid
        return latents.view(b, t, side, side, g, g, -1).transpose(3, 4).reshape(b, t, side * g, side * g, -1)

    def condition(self, actions: torch.Tensor, c_noise: torch.Tensor | None = None,
                  context: torch.Tensor | None = None) -> torch.Tensor:
        """Incoming action bytes [B, T] and (a noisy frame) its c_noise [B, T], or None (a context frame, at
        its context noise level [B, T], None for 0) -> each frame's conditioning [B, T, dim]."""
        cond = self.action(button_bits(actions))
        if c_noise is None:
            cond = cond + self.clean
            if self.context_noise:
                level = torch.zeros(actions.shape, device=cond.device) if context is None else context
                cond = cond + self.context_level(fourier(level, self.noise_freq))
            return cond
        return cond + self.noise(fourier(c_noise, self.noise_freq))

    def _embed(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.inp(x.to(self.inp.weight.dtype)) + self.space_pos + cond[:, :, None]

    def run(self, x: torch.Tensor, cond: torch.Tensor, cache: list[TemporalCache] | None = None, at=0) -> torch.Tensor:
        """One copy alone (dreaming: frames at..at+T-1 against the cache) -> F [B, T, N, D]."""
        h = self._embed(x, cond)
        for i, block in enumerate(self.blocks):
            h = block(h, None if cache is None else cache[i], at, self.place, cond)
        return self.out(self.norm(h)).float()

    def denoise(self, x: torch.Tensor, sigma: torch.Tensor, actions: torch.Tensor, cache: list[TemporalCache],
                at) -> torch.Tensor:
        """A noisy frame [B, 1, N, D] at sigma [B, 1] -> its denoised latents D (EDM), against the clean cache."""
        c_in, c_skip, c_out, c_noise = conditioners(sigma)
        f = self.run(c_in[..., None, None] * x, self.condition(actions, c_noise), cache, at)
        return c_skip[..., None, None] * x + c_out[..., None, None] * f

    def remember(self, z: torch.Tensor, actions: torch.Tensor, cache: list[TemporalCache], at=0) -> None:
        """Clean frames [B, T, N, D] into the cache at at..at+T-1 (as the clean context copies)."""
        self.run(z / SIGMA_DATA, self.condition(actions), cache, at)

    def forward(self, z: torch.Tensor, actions: torch.Tensor, sigma: torch.Tensor, noise: torch.Tensor,
                offset: torch.Tensor, moved_weight: float = 1.0, context: torch.Tensor | None = None,
                context_draw: torch.Tensor | None = None) -> torch.Tensor:
        """Clean latents [B, T, N, D], incoming actions [B, T], each frame's noise level [B, T], its noise [B, T,
        N, D] and offset noise [B, T, 1, D] (standard normal) -> the EDM loss over every frame, each predicted
        from the clean frames before it (both copies in one pass: pair_block). moved_weight: the weight of the
        patches whose latents changed from the frame before (MOVED), against 1 for the rest, the weights
        normalised to a mean of 1: about 0.6% of patches carried 23% of the loss at 52k steps (EDM-64), the
        pixel model's changed pixels 30-90% of its own. context [B]: each window's context noise level, with
        its noise context_draw [B, T, N, D] (standard normal); None: clean context. The targets stay clean."""
        c_in, c_skip, c_out, c_noise = conditioners(sigma)
        x = z + OFFSET * offset + sigma[..., None, None] * noise
        seen, level = z, None
        if context is not None:
            seen = z + context[:, None, None, None] * context_draw
            level = context[:, None].expand(sigma.shape)
        f = self.pair(seen, c_in[..., None, None] * x, actions, c_noise, level)
        per = (f - (z - c_skip[..., None, None] * x) / c_out[..., None, None]).pow(2).mean(-1)      # [B, T, N]
        if moved_weight == 1.0:
            return per.mean()
        moved = torch.zeros_like(per, dtype=torch.bool)
        moved[:, 1:] = (z[:, 1:] - z[:, :-1]).pow(2).mean(-1).sqrt() > MOVED
        weight = 1 + (moved_weight - 1) * moved.float()
        return (weight * per).sum() / weight.sum()

    def pair(self, z: torch.Tensor, x_in: torch.Tensor, actions: torch.Tensor, c_noise: torch.Tensor,
             context: torch.Tensor | None = None) -> torch.Tensor:
        """Context latents and the scaled noisy ones (c_in x) [B, T, N, D], incoming actions and c_noise [B, T]
        (and the context's noise level [B, T], None for 0) -> F of every noisy frame, each seeing the context
        frames before it and itself [B, T, N, D]."""
        cond_c, cond_t = self.condition(actions, context=context), self.condition(actions, c_noise)
        hc, ht = self._embed(z / SIGMA_DATA, cond_c), self._embed(x_in, cond_t)
        remember = self.recompute and self.training and torch.is_grad_enabled()
        for block in self.blocks:
            if remember:
                hc, ht = checkpoint(pair_block, block, hc, ht, cond_c, cond_t, self.place, use_reentrant=False)
            else:
                hc, ht = pair_block(block, hc, ht, cond_c, cond_t, self.place)
        return self.out(self.norm(ht)).float()

    def loss(self, frames: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Frames [B, T, 256, 256] and incoming actions [B, T] -> the EDM loss (the noise drawn here)."""
        z = self.latents(frames)
        b, t, _, d = z.shape
        sigma = (torch.randn(b, t, device=z.device) * SCALE + LOC).exp().clamp(*TRAIN_SIGMAS)
        context = torch.rand(b, device=z.device) * self.context_noise if self.context_noise else None
        return self(z, actions, sigma, torch.randn_like(z), torch.randn(b, t, 1, d, device=z.device),
                    context=context, context_draw=None if context is None else torch.randn_like(z))

    def new_cache(self) -> list[TemporalCache]:
        return [TemporalCache(self.frames) for _ in self.blocks]


def build(config: dict) -> LatentDynamics:
    keys = ("dim", "layers", "heads", "frames", "space_rope", "qk_norm", "steps", "recompute", "context_noise",
            "group")
    return LatentDynamics(config["vae"], **{k: config[k] for k in keys if k in config})


class LatentDreamer:
    """The latent model as a game engine, as models/dynamics.py Dreamer: real frames [B, T0, 256, 256] (or
    [T0, ...]) and the incoming actions that produced them [B, T0] start it; step(action) returns the next
    frame's colour probabilities. The context is clean latents in the temporal cache; when the window is full
    the last `keep` frames are written again at positions 0..keep-1. change_weight is accepted for the pixel
    Dreamer's interface and must be 1."""

    CAPTURE = True                                  # CUDA graphs on CUDA (off: each frame run as it goes, the test's)

    def __init__(self, model: LatentDynamics, frames: torch.Tensor, actions: torch.Tensor, keep: int | None = None,
                 change_weight: float = 1.0, steps: int | None = None, generator: torch.Generator | None = None):
        if change_weight != 1.0:
            raise ValueError("the latent model has no change weight")
        keep = model.frames * 3 // 4 if keep is None else keep
        if not model.span <= keep < model.frames:
            raise ValueError(f"keep must be {model.span}..model.frames - 1 (the VAE decodes from {model.span} "
                             "latents)")
        self.model, self.keep, self.generator = model, keep, generator
        self.sigmas = sample_sigmas(steps or model.steps)
        self.batched = frames.dim() == 4
        if not self.batched:
            frames, actions = frames[None], torch.as_tensor(actions)[None]
        if frames.shape[1] < model.span:
            raise ValueError(f"a dream starts from at least {model.span} real frames (the checks give 48)")
        self.latents = list(model.latents(frames).unbind(1))                       # each [B, N, D]
        self.actions = list(torch.as_tensor(actions, device=frames.device).long().unbind(1))
        b, n, d = self.latents[0].shape
        device = frames.device
        # one frame's work reads and writes these in place, so on CUDA it is captured once as a CUDA graph and
        # replayed (as models/dynamics.py FrameDecoder): launching its ~40 kernels a pass from Python, not the
        # GPU, bounded dreams (the GPU 15-40% busy, 2026-10-10). The noise is drawn outside the graph.
        self.act = torch.zeros(b, 1, dtype=torch.long, device=device)
        self.at = torch.zeros((), dtype=torch.long, device=device)
        self.noise = torch.zeros(b, 1, n, d, device=device)
        self.recent = torch.zeros(b, model.span, n, d, device=device)              # the latents decoded from
        self.autocast = (torch.is_autocast_enabled(device.type), torch.get_autocast_dtype(device.type))
        self.graph = self.out = None
        self.cache = model.new_cache()
        self._encode()

    def _encode(self) -> None:
        """The last `keep` frames into the cache at positions 0..keep-1 (at the start, and whenever the window
        is full), into the same buffers, so a captured graph stays valid."""
        self.latents, self.actions = self.latents[-self.keep:], self.actions[-self.keep:]
        z, a = torch.stack(self.latents, 1), torch.stack(self.actions, 1)
        self.model.remember(z, a, self.cache, 0)
        if len(self.latents) >= self.model.span:
            self.recent.copy_(z[:, -self.model.span:])

    def _frame(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The next frame from self.noise, self.act at self.at -> (its latents [B, N, D], its colour
        probabilities [B, 256, 256, COLOURS]); writes it into the cache and self.recent."""
        model, b = self.model, self.act.shape[0]
        enabled, dtype = self.autocast
        device = self.act.device
        with torch.autocast(device.type, dtype=dtype, enabled=enabled, cache_enabled=False):
            x = self.noise * self.sigmas[0]
            for sigma, after in zip(self.sigmas, self.sigmas[1:]):     # Euler, as DIAMOND's sampler
                denoised = model.denoise(x, torch.full((b, 1), sigma, device=device), self.act, self.cache, self.at)
                x = x + (x - denoised) / sigma * (after - sigma)
            model.remember(x, self.act, self.cache, self.at)            # the clean frame, as context
            self.recent.copy_(torch.cat((self.recent[:, 1:], x.float()), 1))
            probs = model.decode_probs(self.recent)
        return x[:, 0].float(), probs

    @torch.no_grad()
    def step(self, action) -> torch.Tensor:
        if len(self.latents) == self.model.frames:
            self._encode()
        b = self.act.shape[0]
        self.act.copy_(torch.as_tensor(action, device=self.act.device).long().reshape(-1, 1).expand(b, 1))
        self.at.fill_(len(self.latents))
        self.noise.normal_(generator=self.generator)
        if not self.act.is_cuda or not self.CAPTURE:
            latent, probs = self._frame()
        elif self.graph is None:                    # run once on a side stream (this frame), then capture: the
            side = torch.cuda.Stream()              # capture records the work without running it
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                latent, probs = self._frame()
            torch.cuda.current_stream().wait_stream(side)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.out = self._frame()
        else:
            self.graph.replay()
            latent, probs = (t.clone() for t in self.out)
        self.latents.append(latent)
        self.actions.append(self.act[:, 0].clone())
        return probs if self.batched else probs[0]


LatentDynamics.Dreamer = LatentDreamer       # what diagnostics/long_dream.py dreams() makes for it
