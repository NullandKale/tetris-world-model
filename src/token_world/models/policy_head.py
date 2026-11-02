"""A policy head on a world model's features: frame t's features (the frames up to t and the presses that made
them; press t is not among them) -> the 8 button logits of press t, the byte that takes frame t to t + 1.

Trained on the perfect brain's presses with the world model frozen (scripts/policy_probe.py, 2026-10-30); it plays
a dream in the play app (scripts/play_world_model.py --policy). Bits: A = bit 0 ... Right = bit 7, as the
controller byte.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class PolicyHead(nn.Module):
    """A frame's features [M, N, dim] -> its press's 8 button logits [M, 8]: an attention pool (`queries` learned
    queries over the frame's tokens), then two layers."""

    def __init__(self, dim: int, queries: int = 4, hidden: int = 512):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.queries = nn.Parameter(torch.randn(queries, dim) * 0.02)
        self.pool = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.out = nn.Sequential(nn.Linear(queries * dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 8))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.norm(features.float())
        q = self.queries.expand(len(x), -1, -1)
        pooled, _ = self.pool(q, x, x, need_weights=False)
        return self.out(pooled.flatten(1))


class ChunkHead(nn.Module):
    """A frame's features [M, N, dim] -> its next `chunk` presses (the bytes that take frame t to t + 1 ... t +
    chunk), one after another: the attention pool, then a GRU over the chunk fed each press before it (the press
    that made frame t first), a 256-way byte at each step. Within a chunk each press sees the ones before it, so
    "turned already" is known; a press per frame on its own spreads a turn's odds over the frames it might come on
    and almost never turns (scripts/policy_probe.py, 2026-10-30: A 0.2%, B 2.3% of the bot's turns). ACT (Zhao et
    al. 2304.13705) predicts chunks for the same reason; here autoregressively, as the presses are discrete."""

    NONE = 256                                         # the press before is not known

    def __init__(self, dim: int, chunk: int = 16, queries: int = 4, hidden: int = 512):
        super().__init__()
        self.chunk = chunk
        self.norm = nn.LayerNorm(dim)
        self.queries = nn.Parameter(torch.randn(queries, dim) * 0.02)
        self.pool = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.start = nn.Linear(queries * dim, hidden)
        self.embed = nn.Embedding(257, 64)
        self.rnn = nn.GRU(64, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 256)
        self.register_buffer("size", torch.tensor([chunk]))     # saved with the weights

    def _state(self, features: torch.Tensor) -> torch.Tensor:
        x = self.norm(features.float())
        pooled, _ = self.pool(self.queries.expand(len(x), -1, -1), x, x, need_weights=False)
        return torch.tanh(self.start(pooled.flatten(1)))[None]          # [1, M, hidden]

    def _index(self, presses: torch.Tensor) -> torch.Tensor:
        return torch.where(presses < 0, torch.full_like(presses, self.NONE), presses).long()

    def forward(self, features: torch.Tensor, before: torch.Tensor, presses: torch.Tensor) -> torch.Tensor:
        """Teacher-forced: features [M, N, dim], the press that made the frame [M], the chunk's real presses
        [M, K] -> each press's logits [M, K, 256]."""
        inputs = torch.cat([before[:, None], presses[:, :-1]], 1)
        y, _ = self.rnn(self.embed(self._index(inputs)), self._state(features))
        return self.out(y)

    @torch.no_grad()
    def sample(self, features: torch.Tensor, before: torch.Tensor, steps: int,
               generator: torch.Generator | None = None, greedy: bool = False) -> torch.Tensor:
        """The next `steps` presses [M, steps], each drawn from its distribution given the ones before (greedy:
        the most likely)."""
        h = self._state(features)
        prev, out = self._index(before), []
        for _ in range(steps):
            y, h = self.rnn(self.embed(prev)[:, None], h)
            p = self.out(y[:, 0]).float().softmax(-1)
            prev = p.argmax(-1) if greedy else torch.multinomial(p, 1, generator=generator)[:, 0]
            out.append(prev)
        return torch.stack(out, 1)


class Player:
    """Presses for B dreams from a head, frame by frame: a ChunkHead's chunk, its first `execute` presses played
    before it plans again from the newest frame; a PolicyHead's press each frame (choose)."""

    def __init__(self, head: nn.Module, execute: int = 8, generator: torch.Generator | None = None):
        self.head, self.execute, self.generator = head, execute, generator
        self.plan, self.at = None, 0

    def press(self, features: torch.Tensor, before: torch.Tensor) -> torch.Tensor:
        """features [B, N, dim] of each dream's newest frame, before [B] the press that made it -> presses [B]."""
        if not isinstance(self.head, ChunkHead):
            return choose(self.head, features, self.generator)
        if self.plan is None or self.at == self.execute:
            self.plan, self.at = self.head.sample(features, before, self.execute, self.generator), 0
        self.at += 1
        return self.plan[:, self.at - 1]


def load_head(path: Path, device) -> nn.Module:
    """A saved head (its state dict), a PolicyHead or a ChunkHead, its sizes from the saved weights."""
    state = torch.load(path, map_location=device)
    if "rnn.weight_ih_l0" in state:
        head = ChunkHead(state["norm.weight"].shape[0], chunk=int(state["size"][0]), queries=state["queries"].shape[0],
                         hidden=state["start.weight"].shape[0])
    else:
        head = PolicyHead(state["norm.weight"].shape[0], queries=state["queries"].shape[0],
                          hidden=state["out.0.weight"].shape[0])
    head = head.to(device)
    head.load_state_dict(state)
    return head.eval().requires_grad_(False)


def bits(byte: torch.Tensor) -> torch.Tensor:
    """[...] bytes -> [..., 8] float bits."""
    return ((byte.long()[..., None] >> torch.arange(8, device=byte.device)) & 1).float()


def byte_of(b: torch.Tensor) -> torch.Tensor:
    """[..., 8] bits (bool or 0/1) -> [...] bytes."""
    return (b.long() << torch.arange(8, device=b.device)).sum(-1)


@torch.no_grad()
def last_features(model, dreamer) -> torch.Tensor:
    """The features of a dream's newest frame [B, N, dim] (models/dynamics.py Dreamer: its tokens through the model
    again at its place in the cache, the same keys rewritten)."""
    at = dreamer.t - 1
    tokens = dreamer.tokens[:, None]
    act = dreamer.action[:, None]
    hidden = torch.zeros(tokens.shape[0], 1, model.grid ** 2, dtype=torch.bool, device=tokens.device)
    from token_world.models.dynamics import motion_tokens
    motion = motion_tokens(dreamer.codes, model.patch)[:, None] if dreamer.codes is not None else None
    with torch.autocast(tokens.device.type, dtype=torch.bfloat16, enabled=tokens.is_cuda):
        return model(tokens, act, hidden, dreamer.decoder.cache, at, motion=motion)[:, 0]


@torch.no_grad()
def choose(head: PolicyHead, features: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """The head's press for each dream's next frame [B] from its newest frame's features [B, N, dim]
    (models/decoding.py OptionDreamer.newest, or last_features for a plain Dreamer), each button drawn from its
    probability."""
    p = torch.sigmoid(head(features))
    return byte_of(torch.rand(p.shape, generator=generator, device=p.device) < p)
