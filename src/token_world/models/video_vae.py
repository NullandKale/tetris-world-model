"""A causal video VAE on exact palette pixels: every frame adds the same number of latents (2026-10-10).

The world model would then predict latents, not pixels (latent diffusion / flow, as GameNGen and Oasis do): a
latent is soft by nature, so an unsure piece is a point between two pictures, not an argmax that erases it
(pixel-head-erases-uncertain-pieces). As Genie's video tokenizer (ST-ViViT), it is causal in time: frame t's
latents are encoded from frames 0..t and frame t is decoded from latents 0..t, so a frame's latents need only
carry what the frames before it do not, and every frame adds G x G latents (G = 256 / P), whatever happens.
No chunks of frames share latents: the world model steps one frame and one controller byte at a time.

    index [B, T, 256, 256] uint8 (model frames: NES colours 0-54, border shades 55-57; data/model_frames.py)
      -> per patch and frame: colour embedding, stride-2 convolutions to 4 x 4, linear -> [B, T, G, G, E]
      -> per patch place: causal attention over the frames (rotary time) -> linear, unit RMS -> [B, T, G, G, D]
    latents -> linear -> per place: causal attention over the frames -> per patch and frame: transposed
      convolutions -> colour logits for its own patch's pixels only

Local in space both ways: a patch's latents depend on its own place's pixels alone (now and before), and they
decode to that place's pixels alone, so a second, spatial world model cannot hide in it; everything that moves
across places is the predictor's. Colours are exact: cross-entropy over the 58 colours (patch_vae.py measured).
Every convolution is followed by GroupNorm (as Stable Diffusion's VAE): without norms both sizes diverged at
steps 800-1,400 (output/video_vae_p*_nonorm, 2026-10-10). The colour logits are soft-capped at +-CAP (Gemma 2):
with GroupNorm and a z-loss the overnight run was exact for 30k steps, then the logits of the wrong colours
drifted ever lower (the z-loss holds only their log-partition), a wrong pixel came to cost hundreds of nats and
both sizes diverged slowly (output/video_vae_p*_zloss). The cap leaves the most likely colour as it is.

Each latent is normalized to unit RMS over its D numbers, and training decodes it with Gaussian noise of a
fixed `noise` per number added, instead of a learned posterior and a KL to N(0, 1): the predictor gets latents
of a fixed scale by construction, and the decoder must forgive noise, as it must forgive a predictor's small
errors. The noise is also what makes history matter: it caps what one frame's latents can carry (about
D/2 log2(1 + 1/noise^2) bits a patch). At noise 0.1 (53 bits a patch, the overnight run) a frame's latents held
the whole picture and decoding was as exact with no frames before as with 31: an image VAE in time. At noise 1
or 2 (8 or 2.6 bits) a patch's picture (some 13 bits) no longer fits one frame, so what stays the same has to
come from the frames before and a frame's latents carry what is new; a cut to a new screen fills in over a few
frames. (The first, per-frame version with the standard KL at weight 1e-3 let the scale grow to 290 before it
diverged, output/frame_vae_p16_kl; it had no norms either.)

Per frame (layers 0, 2026-10-12): no time blocks, so a frame's latents come from that frame alone and decode
to it alone. The video VAE n1 writes a falling piece's recent motion into its latents (scripts/latent_tokens.py:
the same drawn piece cells spread 0.26 per number, wider than the gap to other contents, against 0.000 for the
stack), which a generator has to predict too. A per-frame VAE needs a patch's whole picture in one frame.
Run video_vae_frame: 8 x 8 patches, one per NES tile (the well's cells are tiles at x = 96 + 8 col, y = 56 + 8
row), 8 numbers each at noise 0.5 (about 9 bits a cell; a cell is one of some 50-100 tiles and palettes), so a
falling piece is four latents that move as it does (at 16 x 16 the same drop changed a patch in one of four
ways, by where in it the piece sat).

Sprite numbers (`sprite`, 2026-10-12): each cell's latent also holds `sprite` numbers saying whether the PPU
drew the cell as a sprite (data/nes_layers.sprite_mask is the training signal, never an input), so a token says
"this is the piece" and a generator has to carry that forward. The first version (run video_vae_frame) read it
off the colour latents of one cell: it found the next-box piece (85%: half-cell offsets only drawn there) and
0.1% of the falling piece, whose blocks are the same pixels as the stack's. A falling piece is what moves, so
the sprite numbers see the whole frame and its past: each cell's features through `sprite_layers` Blocks of
the generator's kind (models/dynamics.Block: attention across the frame's cells, then causal attention back
over time per cell, rotary time) of width `sprite_width`, with a learned place per cell, looking back up to
`history` frames (128: a piece hangs 48 frames between drops at level 0). Only these numbers see other cells or
frames: the colour numbers still come from the cell's own pixels in that frame, and the colours are decoded
from them alone, so every frame still decodes on its own; a small head (sprite_out) turns a cell's colour and
sprite numbers into its pixels' sprite logits, the loss that teaches the encoder. Training's history frames
pass without gradient and only fill each Block's temporal cache (Transformer-XL's memory), so 128 frames back
cost activations for the trained frames only; encoding one frame at a time against that cache gives the same
numbers (tests/test_video_vae.py). Both kinds of number get training's noise.

The sprite numbers are bounded by tanh, not normalised: one number normalised to unit RMS keeps only its sign
and passes no gradient (2026-10-13), and the colour numbers keep their own normalisation, so they still come
from the cell's own pixels alone.

The sprite head (`sprite_head`, 2026-10-13: run video_vae_n1s): a VAE with time blocks (n1's) learns the
sprite layer too, with no numbers added: a head on the decoder's features (after its time blocks, so from the
latents up to that frame, as the colours are) gives each pixel's sprite logit, at equal weight. The tokens stay
n1's 16 numbers a 16 x 16 patch (the one latent generator that worked, fast 64 x 8, fits them), and the loss
asks the encoder to keep the falling piece told apart from the stack it looks like. The per-frame VAEs sized
from the generator's side (below) could not carry Tetris's cells in a frame at a noise a generator's errors
fit in; n1 can because most of a frame is its past (docs/guides/latent_world_model.md, the c6 probe).

Sprite numbers on n1 (2026-10-13: run video_vae_n1sn): the sprite head on n1 (n1s) missed a falling piece that
had not moved within its 16-frame window (65% of its pixels when it last moved 16-47 frames ago: each patch sees
only its own past), so n1 gets the per-frame VAEs' sprite numbers instead: 4 a 16 x 16 patch (about one a cell),
from the whole frame and its last 128 frames, while its colour numbers stay time-coded over the 16-frame window
(encode's `held`: frames only the sprite numbers see). Tokens of 20 numbers. The sprite numbers are bounded by
tanh, so they get noise of their own (sprite_noise 0.3): n1's 1.0 would drown them.

Sizing (2026-10-13): run video_vae_sprite (16 colour + 2 sprite numbers a cell, sized to Tetris's information,
about 18 bits a cell) reconstructed nearly exactly, but its 72-number tokens (2 x 2 cells) were wider than the
64-wide generator that used them, which could not even copy a still cell (latent error 0.11 a frame ahead,
the old 16-number tokens' generator 0.005). The latent is sized from the generator's side now, as latent
diffusion does (Stable Diffusion's f8c4: an 8 x 8 cell to 4 numbers): candidates of 4 and 6 numbers a cell,
judged by reconstruction and by a short generator probe each (docs/guides/video_vae.md).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from token_world.models.dynamics import COLOURS, SIZE, Attention, Block, RMSNorm, TemporalCache, rotary

LOW = 4                                         # the convolutions' smallest picture: 4 x 4
GROUPS = 8                                      # GroupNorm's groups after every convolution
CAP = 30.0                                      # the colour logits' soft cap, CAP * tanh(logits / CAP)
FEATURE_CHUNK = 128                             # frames whose convolutions run at once (features)


def patches(index: torch.Tensor, patch: int) -> torch.Tensor:
    """[N, H, W] -> [N * (H / P) * (W / P), P, P], row by row."""
    n, h, w = index.shape
    return index.view(n, h // patch, patch, w // patch, patch).transpose(2, 3).reshape(-1, patch, patch)


def frames(tiles: torch.Tensor, n: int) -> torch.Tensor:
    """patches(...)'s inverse: [N * G * G, P, P, ...] -> [N, G * P, G * P, ...]."""
    p = tiles.shape[1]
    g = math.isqrt(tiles.shape[0] // n)
    rest = tiles.shape[3:]
    return tiles.view(n, g, g, p, p, *rest).transpose(2, 3).reshape(n, g * p, g * p, *rest)


class TimeBlock(nn.Module):
    """Pre-norm causal attention over frames (rotary time, QK-norm) and an MLP, per patch place."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.attention = Attention(dim, heads, qk_norm=True, rotate=True)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), causal=True)
        return x + self.mlp(self.norm2(x))


class VideoVAE(nn.Module):
    def __init__(self, patch: int = 16, latent: int = 16, width: int = 32, colour_dim: int = 16,
                 dim: int = 128, heads: int = 4, layers: int = 2, noise: float = 1.0, sprite: int = 0,
                 sprite_width: int = 96, sprite_layers: int = 2, history: int = 128, sprite_head: bool = False,
                 sprite_noise: float | None = None):
        """latent: each cell's colour numbers; sprite: its sprite numbers (0: none); sprite_noise: training's
        noise on them (None: `noise`); sprite_head: a VAE with time blocks learns the sprite layer from its
        decoder's features (no numbers added)."""
        super().__init__()
        if patch < LOW or SIZE % patch or patch & (patch - 1):
            raise ValueError(f"patch must be a power of two from {LOW} dividing {SIZE}")
        if sprite and sprite_head:
            raise ValueError("sprite numbers or a sprite head, not both")
        if sprite_head and not layers:
            raise ValueError("the sprite head reads the decoder's time blocks (layers > 0); per frame: sprite numbers")
        self.patch, self.latent, self.noise = patch, latent, noise
        self.sprite_noise = noise if sprite_noise is None else sprite_noise
        self.sprite, self.history, self.sprite_head = int(sprite), history, bool(sprite_head)
        self.learns_sprites = bool(sprite or sprite_head)               # trained on the PPU's sprite layer
        self.numbers = latent + self.sprite                         # a cell's latent: colour, then sprite numbers
        self.grid = SIZE // patch
        stages = int(math.log2(patch // LOW))
        chans = [min(width * 2 ** i, 4 * width) for i in range(stages + 1)]
        self.top = chans[-1]
        flat = self.top * LOW * LOW
        self.colour = nn.Embedding(COLOURS, colour_dim)
        norm = lambda c: nn.GroupNorm(GROUPS, c)
        down = [nn.Conv2d(colour_dim, chans[0], 3, padding=1), norm(chans[0]), nn.SiLU()]
        for a, b in zip(chans, chans[1:]):
            down += [nn.Conv2d(a, b, 4, stride=2, padding=1), norm(b), nn.SiLU()]
        self.down = nn.Sequential(*down)
        self.enc_in = nn.Linear(flat, dim)
        self.enc_time = nn.ModuleList(TimeBlock(dim, heads) for _ in range(layers))
        self.to_latent = nn.Sequential(RMSNorm(dim), nn.Linear(dim, latent))
        self.dec_in = nn.Linear(latent, dim)
        self.dec_time = nn.ModuleList(TimeBlock(dim, heads) for _ in range(layers))
        self.dec_out = nn.Sequential(RMSNorm(dim), nn.Linear(dim, flat))
        up = []
        for a, b in zip(chans[::-1], chans[::-1][1:]):
            up += [nn.SiLU(), nn.ConvTranspose2d(a, b, 4, stride=2, padding=1), norm(b)]
        self.up = nn.Sequential(*up, nn.SiLU(), nn.Conv2d(chans[0], COLOURS, 1))
        if self.sprite:
            self.sprite_in = nn.Linear(dim, sprite_width)
            self.sprite_place = nn.Parameter(0.02 * torch.randn(self.grid ** 2, sprite_width))
            self.sprite_blocks = nn.ModuleList(Block(sprite_width, heads, qk_norm=True) for _ in range(sprite_layers))
            self.to_sprite = nn.Sequential(RMSNorm(sprite_width), nn.Linear(sprite_width, self.sprite))
            self.sprite_out = nn.Sequential(nn.Linear(self.numbers, 128), nn.SiLU(), nn.Linear(128, patch * patch))
        if self.sprite_head:
            self.sprite_out = nn.Sequential(RMSNorm(dim), nn.Linear(dim, 128), nn.SiLU(), nn.Linear(128, patch * patch))

    @staticmethod
    def _over_time(blocks: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
        """[B, T, S, E] -> each place's frames through the causal blocks -> [B, T, S, E]."""
        b, t, s, e = x.shape
        x = x.transpose(1, 2).reshape(b * s, t, e)
        for block in blocks:
            x = block(x)
        return x.view(b, s, t, e).transpose(1, 2)

    def encode(self, index: torch.Tensor, held: int = 0) -> torch.Tensor:
        """Frames [B, T, 256, 256] -> latents [B, T - held, G, G, numbers]: colour numbers (unit RMS) then sprite
        numbers (tanh); frame t's from frames 0..t (per frame: its colour numbers from frame t alone). The first
        `held` frames are only the sprite numbers' history (with time blocks the colour numbers see the frames
        after them only, their training window)."""
        x = self.features(index)
        return self.encode_features(x[:, held:], x[:, :held] if held else None)

    def features(self, index: torch.Tensor) -> torch.Tensor:
        """encode's per-frame half: frames [B, T, 256, 256] -> each patch's features [B, T, G * G, dim], from its
        frame alone (the convolutions: nearly all of encode's work), FEATURE_CHUNK frames at a time: all of a
        preview's 768 frames at once wanted 6 GB for one GroupNorm and ran the latent run out of memory
        (2026-10-11)."""
        b, t = index.shape[:2]
        flat = index.reshape(b * t, SIZE, SIZE)
        out = []
        for i in range(0, b * t, FEATURE_CHUNK):
            x = self.colour(patches(flat[i:i + FEATURE_CHUNK], self.patch).long()).permute(0, 3, 1, 2)
            x = self.down(x.contiguous(memory_format=torch.channels_last))
            out.append(self.enc_in(x.flatten(1)))
        return torch.cat(out).view(b, t, -1, self.enc_in.out_features)

    def encode_features(self, x: torch.Tensor, before: torch.Tensor | None = None) -> torch.Tensor:
        """encode's half over time: features [B, T, G * G, dim] -> latents [B, T, G, G, numbers], frame t's from
        frames 0..t (so a model encoding overlapping windows runs features() once per frame). before [B, H, G * G,
        dim]: the features of the H frames before x, seen only by the sprite numbers, without gradient."""
        b, t = x.shape[:2]
        z = F.rms_norm(self.to_latent(self._over_time(self.enc_time, x)).float(), (self.latent,), eps=1e-6)
        if self.sprite:
            z = torch.cat((z, self.sprite_numbers(x, before)), -1)
        return z.view(b, t, self.grid, self.grid, self.numbers)

    def sprite_numbers(self, x: torch.Tensor, before: torch.Tensor | None = None) -> torch.Tensor:
        """Features [B, T, G * G, dim] -> each cell's sprite numbers [B, T, G * G, sprite], in (-1, 1): frame t's from
        frames 0..t of x and all of `before` (its frames pass without gradient, through each Block's temporal
        cache only)."""
        h = self.sprite_in(x) + self.sprite_place
        if before is None or before.shape[1] == 0:
            for block in self.sprite_blocks:
                h = block(h)
        else:
            held = before.shape[1]
            with torch.no_grad():
                old = self.sprite_in(before) + self.sprite_place
            for block in self.sprite_blocks:
                cache = TemporalCache(held + x.shape[1])
                with torch.no_grad():
                    old = block(old, cache)
                h = block(h, cache, at=held)
        return torch.tanh(self.to_sprite(h).float())

    def decode_features(self, z: torch.Tensor, last: int | None = None) -> torch.Tensor:
        """Latents [B, T, G, G, numbers] -> the decoder's features of the last `last` frames (all: None), each from
        the colour numbers up to its own: [B, last, G * G, dim]."""
        b, t = z.shape[:2]
        x = self._over_time(self.dec_time, self.dec_in(z[..., :self.latent].reshape(b, t, -1, self.latent)))
        return x[:, t - (last or t):]

    def logits(self, z: torch.Tensor, last: int | None = None, features: torch.Tensor | None = None) -> torch.Tensor:
        """Latents [B, T, G, G, numbers] (or their decode_features) -> the colour logits of the last `last` frames
        (all: None): [B * last * G * G, COLOURS, P, P] (frames, then patches, in order)."""
        x = self.decode_features(z, last) if features is None else features
        x = self.dec_out(x).reshape(-1, self.top, LOW, LOW)
        # float32: under bf16 autocast the compiled training graph (torch.compile, with gradient) computed these
        # logits so coarsely that pixels' cross-entropy came out as low as -0.047 and the logged loss at 6x the
        # true one (0.0075 against 0.0013 on the same batch; the n1-family runs logged negative losses), the
        # rounding larger than what was left to learn; eager bf16 and float32 agreed (2026-10-13)
        with torch.autocast(x.device.type, enabled=False):
            x = self.up(x.float().contiguous(memory_format=torch.channels_last))
            return CAP * torch.tanh(x / CAP)

    def sprite_logits(self, z: torch.Tensor, last: int | None = None, features: torch.Tensor | None = None
                      ) -> torch.Tensor:
        """Latents -> each pixel's sprite logit [B * last * G * G, P, P] (a model that learns_sprites): per frame
        from its cell's colour and sprite numbers; with the sprite head from the decoder's features (or the given
        decode_features), so from the latents up to its own frame."""
        if self.sprite_head:
            x = self.decode_features(z, last) if features is None else features
            return self.sprite_out(x).float().view(-1, self.patch, self.patch)
        t = z.shape[1]
        cells = z[:, t - (last or t):].reshape(-1, self.numbers)
        return self.sprite_out(cells).float().view(-1, self.patch, self.patch)

    def decode(self, z: torch.Tensor, last: int | None = None) -> torch.Tensor:
        """Latents [B, T, G, G, D] -> the most likely last `last` frames [B, last, 256, 256] uint8."""
        b, t = z.shape[:2]
        shown = last or t
        return frames(self.logits(z, last).argmax(1).byte(), b * shown).view(b, shown, SIZE, SIZE)

    def noisy(self, z: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """z plus Gaussian noise of scale * noise per colour number and scale * sprite_noise per sprite number
        (training's noise at scale 1)."""
        sigma = torch.full((self.numbers,), self.noise, device=z.device)
        sigma[self.latent:] = self.sprite_noise
        return z + scale * sigma * torch.randn_like(z)

    def forward(self, index: torch.Tensor, last: int | None = None, sprites: torch.Tensor | None = None,
                held: int = 0) -> dict[str, torch.Tensor]:
        """Frames [B, T, 256, 256] (and with `sprite` their sprite masks [B, T, 256, 256] bool) -> the loss (mean
        pixel cross-entropy of the last `last` frames, decoded from noisy latents; with `sprite`, plus the mean
        sprite cross-entropy, at equal weight), their per-pixel colour cross-entropy [B, last, 256, 256] and the
        latents (with time blocks all frames' after the first `held`, encoded with gradient, the `held` before
        them only the sprite numbers' history, without gradient; per frame the last `last` frames', the frames
        before them only the sprite numbers' history)."""
        b, t = index.shape[:2]
        shown = last or t
        if self.enc_time:
            before = None
            if self.sprite and held:
                with torch.no_grad():
                    before = self.features(index[:, :held])
            z = self.encode_features(self.features(index[:, held:]), before)
        else:
            before = None
            if self.sprite and t > shown:
                with torch.no_grad():
                    before = self.features(index[:, :t - shown])
            z = self.encode_features(self.features(index[:, t - shown:]), before)
        noisy = self.noisy(z)
        target = patches(index[:, t - shown:].reshape(b * shown, SIZE, SIZE), self.patch).long()
        features = self.decode_features(noisy, shown)
        ce = F.cross_entropy(self.logits(noisy, shown, features), target, reduction="none")
        result = {"loss": ce.mean(), "ce": frames(ce, b * shown).view(b, shown, SIZE, SIZE), "latent": z}
        if self.learns_sprites:
            mask = patches(sprites[:, t - shown:].reshape(b * shown, SIZE, SIZE), self.patch).float()
            said = self.sprite_logits(noisy, shown, features)
            result["sprite_loss"] = F.binary_cross_entropy_with_logits(said, mask)
            result["loss"] = result["loss"] + result["sprite_loss"]
        return result


KEYS = ("patch", "latent", "width", "colour_dim", "dim", "heads", "layers", "noise", "sprite", "sprite_width",
        "sprite_layers", "history", "sprite_head", "sprite_noise")


def config_of(args: dict) -> dict:
    """A run's saved args (or any dict) -> the VAE's own settings in it."""
    return {k: args[k] for k in KEYS if k in args}


def build(config: dict) -> VideoVAE:
    return VideoVAE(**config_of(config))


class StreamEncoder:
    """A frozen VideoVAE with sprite numbers encoding B continuous streams a chunk of frames at a time, as a
    generator's trainer gets them (2026-10-12). Colour numbers are per frame; for the sprite numbers each Block
    keeps, per stream, its temporal attention's inputs for the last `history` frames, and a new chunk's frames
    attend back over them (each frame up to `history` frames back), so a stream's frames get the numbers that
    encoding its whole past would give, for a forward pass over the new frames only. reset(rows): those
    streams start again (nothing before)."""

    def __init__(self, vae: VideoVAE, streams: int):
        if not vae.sprite or vae.enc_time:
            raise ValueError("streaming is for per-frame VAEs with sprite numbers")
        self.vae, self.streams = vae, streams
        self.past: list[torch.Tensor | None] = [None] * len(vae.sprite_blocks)     # each [B, history, N, d]
        self.held = torch.zeros(streams, dtype=torch.long)                          # valid frames of history

    def reset(self, rows) -> None:
        self.held[rows] = 0

    @torch.no_grad()
    def __call__(self, index: torch.Tensor) -> torch.Tensor:
        """The streams' next frames [B, S, 256, 256] -> their latents [B, S, G, G, numbers]."""
        vae, (b, s) = self.vae, index.shape[:2]
        x = vae.features(index)
        colour = F.rms_norm(vae.to_latent(x).float(), (vae.latent,), eps=1e-6)
        h = vae.sprite_in(x) + vae.sprite_place
        held = self.held.to(index.device)
        for i, block in enumerate(vae.sprite_blocks):
            h, self.past[i] = self._block(block, h, self.past[i], held)
        self.held = (self.held + s).clamp(max=vae.history)
        sprite = torch.tanh(vae.to_sprite(h).float())
        return torch.cat((colour, sprite), -1).view(b, s, vae.grid, vae.grid, vae.numbers)

    def _block(self, block: Block, x: torch.Tensor, past: torch.Tensor | None, held: torch.Tensor):
        """dynamics.Block's forward for new frames x [B, S, N, d] against this layer's kept temporal inputs ->
        (its output, the temporal inputs to keep)."""
        b, s, n, d = x.shape
        history = self.vae.history
        if past is None:
            past = x.new_zeros(b, history, n, d)
        x = x + block.spatial(block.norm_s(x).reshape(b * s, n, d)).view(b, s, n, d)
        y = torch.cat((past.to(x.dtype), block.norm_t(x)), 1)                       # [B, H + S, N, d]
        length = history + s
        attention = block.temporal
        heads = attention.heads
        q, k, v = (attention.qkv(y.transpose(1, 2).reshape(b * n, length, d))
                   .view(b * n, length, 3, heads, d // heads).permute(2, 0, 3, 1, 4))
        if attention.q_norm is not None:
            q, k = attention.q_norm(q), attention.k_norm(k)
        times = torch.arange(length, device=x.device)
        q, k = rotary(q, times, axis=2)[:, :, history:], rotary(k, times, axis=2)
        at = history + torch.arange(s, device=x.device)                              # the new frames' places
        keys = times[None]
        seen = (keys <= at[:, None]) & (keys >= at[:, None] - history)                # up to `history` back
        valid = keys[None, 0] >= history - held[:, None]                             # [B, H + S]: not before start
        mask = (seen[None] & valid[:, None]).repeat_interleave(n, 0)[:, None]        # [B * N, 1, S, H + S]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + attention.out(out.transpose(1, 2).reshape(b * n, s, d)).view(b, n, s, d).transpose(1, 2)
        return x + block.mlp(block.norm_m(x)), y[:, -history:]


class TimeStreamEncoder:
    """A frozen time-block VAE (n1, n1sn) encoding B continuous streams a chunk of frames at a time, as a
    generator's trainer gets them (2026-10-13), giving the latents LatentDynamics.latents gives the whole stream:
    the first `window` frames encoded together (causal, so each from those before it), then each `stride` new
    ones from the `window` frames up to them, the sprite numbers also seeing up to `history` frames before that
    window. It keeps each stream's last history + window frames' features (the convolutions run once a frame).
    Every call has to end on a stride boundary after the first window, as the windows of whole-stream encoding do."""

    def __init__(self, vae: VideoVAE, window: int, stride: int):
        if not vae.enc_time:
            raise ValueError("the time stream encoder is for VAEs with time blocks (per frame: StreamEncoder)")
        self.vae, self.window, self.stride = vae, window, stride
        self.past: torch.Tensor | None = None        # features [B, L, N, dim] of frames base .. seen - 1
        self.base = self.seen = 0

    @torch.no_grad()
    def __call__(self, index: torch.Tensor) -> torch.Tensor:
        """The streams' next frames [B, S, 256, 256] -> their latents [B, S, G, G, numbers]."""
        vae, window, stride = self.vae, self.window, self.stride
        new = vae.features(index)
        x = new if self.past is None else torch.cat((self.past, new), 1)
        seen, end_all = self.seen, self.seen + new.shape[1]
        if end_all > window and (end_all - window) % stride:
            raise ValueError(f"a chunk must end on a stride boundary: frame {end_all} (window {window}, stride {stride})")
        at = lambda a, b: x[:, a - self.base:b - self.base]
        out = []
        if seen < window:
            end = min(window, end_all)
            out.append(vae.encode_features(at(0, end))[:, seen:end])
        start = max(window, seen)
        while start < end_all:
            end = min(window + ((start - window) // stride + 1) * stride, end_all)
            first = max(self.base, end - window - vae.history) if vae.sprite else end - window
            before = at(first, end - window) if vae.sprite and end - window > first else None
            out.append(vae.encode_features(at(end - window, end), before)[:, -(end - start):])
            start = end
        keep = window + (vae.history if vae.sprite else 0)
        self.past = x[:, -keep:]
        self.base, self.seen = end_all - self.past.shape[1], end_all
        return torch.cat(out, 1)
