"""The patch embedding of index frames as an embedding bag: two fused GPU kernels (2026-10-14).

Dynamics._patches embeds each pixel's colour (a colour_dim vector) and runs the
patch conv over the result. For index frames that is a sum: with
M[pos, colour] = conv weight at pos x the colour's embedding ([P*P, COLOURS,
dim], a small matmul), a token is the conv bias plus M[pos, its pixel at pos's
colour] summed over its P*P pixels. The kernels compute that sum and, for the
backward pass, M's gradient (each position's tokens' gradients added up by the
colour they hold there); autograd takes M's gradient back to the conv weight and
the colour table. Neither the embedded pixels (colour_dim values per pixel, 2.1 GB
for a 16 x 64 batch) nor their gradient is ever built: the conv, its two
backward passes and the colour embedding's one-hot matmul were 25% of a 96 x 3
model's training step (profiled 2026-10-14). tests/test_patch_kernel.py checks it
against the conv.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

SIZE = 256           # model frame side, pixels
BLOCK_D = 32         # feature columns per program
TOKENS = 16384       # tokens per backward program: each adds its share of M's gradient once
TILE = 64            # tokens per backward inner step (the one-hot matmul's K)


@triton.jit
def _bag_forward(index_ptr, table_ptr, out_ptr, D, colours,
                 P: tl.constexpr, GRID: tl.constexpr, BLOCK_D: tl.constexpr):
    """One program: one row of GRID tokens of one frame, BLOCK_D features."""
    row, dblock = tl.program_id(0), tl.program_id(1)
    frame, ty = row // GRID, row % GRID
    tx = tl.arange(0, GRID)
    d = dblock * BLOCK_D + tl.arange(0, BLOCK_D)
    inside = d < D
    acc = tl.zeros((GRID, BLOCK_D), dtype=tl.float32)
    base = index_ptr + frame.to(tl.int64) * (GRID * P) * (GRID * P) + ty * P * (GRID * P)
    for pos in range(0, P * P):
        i, j = pos // P, pos % P
        colour = tl.load(base + i * (GRID * P) + tx * P + j).to(tl.int32)
        rows = (pos * colours + colour).to(tl.int64)
        acc += tl.load(table_ptr + rows[:, None] * D + d[None, :], mask=inside[None, :], other=0.0).to(tl.float32)
    tokens = frame.to(tl.int64) * GRID * GRID + ty * GRID + tx
    tl.store(out_ptr + tokens[:, None] * D + d[None, :], acc.to(out_ptr.dtype.element_ty), mask=inside[None, :])


@triton.jit
def _bag_backward(index_ptr, grad_ptr, out_ptr, n_tokens, D, colours, per_program,
                  P: tl.constexpr, GRID: tl.constexpr, C: tl.constexpr, BLOCK_D: tl.constexpr, TILE: tl.constexpr,
                  EXACT: tl.constexpr):
    """One program: one position, a run of tokens, BLOCK_D features -> its share of M's gradient at that
    position, [colours, BLOCK_D], from one-hot matmuls (the tokens' colours there against their gradients)."""
    pos, chunk, dblock = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i, j = pos // P, pos % P
    c = tl.arange(0, C)
    d = dblock * BLOCK_D + tl.arange(0, BLOCK_D)
    inside_d = d < D
    acc = tl.zeros((C, BLOCK_D), dtype=tl.float32)
    start = chunk * per_program
    for offset in range(0, per_program, TILE):
        t = start + offset + tl.arange(0, TILE)
        inside = (t < n_tokens) & (t < start + per_program)
        t64 = t.to(tl.int64)
        frame, n = t64 // (GRID * GRID), t64 % (GRID * GRID)
        ty, tx = n // GRID, n % GRID
        pixel = frame * (GRID * P) * (GRID * P) + (ty * P + i) * (GRID * P) + tx * P + j
        colour = tl.load(index_ptr + pixel, mask=inside, other=0).to(tl.int32)
        g = tl.load(grad_ptr + t64[:, None] * D + d[None, :], mask=inside[:, None] & inside_d[None, :],
                    other=0.0)                                                                     # [TILE, BLOCK_D]
        onehot = ((colour[:, None] == c[None, :]) & inside[:, None]).to(g.dtype)                   # [TILE, C]
        if EXACT:                                   # float32 gradients: strict float32 (bf16 ones are exact)
            acc += tl.dot(tl.trans(onehot), g, input_precision="ieee")
        else:
            acc += tl.dot(tl.trans(onehot), g)
    rows = (pos * colours + c).to(tl.int64)
    tl.atomic_add(out_ptr + rows[:, None] * D + d[None, :], acc, mask=(c < colours)[:, None] & inside_d[None, :])


@torch.library.custom_op("token_world::patch_bag", mutates_args=())
def patch_bag(index: torch.Tensor, table: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """index [F, 256, 256] uint8 colours, table [P*P, colours, dim] (M, float32) -> tokens [F, (256 / P)^2, dim]
    in `dtype`: each token the sum of its pixels' rows, in float32, rounded once."""
    pp, colours, dim = table.shape
    p = int(round(pp ** 0.5))
    grid = SIZE // p
    index, table = index.contiguous(), table.contiguous()
    out = torch.empty(index.shape[0], grid * grid, dim, dtype=dtype, device=index.device)
    _bag_forward[(index.shape[0] * grid, triton.cdiv(dim, BLOCK_D))](index, table, out, dim, colours,
                                                                   P=p, GRID=grid, BLOCK_D=BLOCK_D)
    return out


@patch_bag.register_fake
def _(index, table, dtype):
    pp, _, dim = table.shape
    grid = SIZE // int(round(pp ** 0.5))
    return index.new_empty(index.shape[0], grid * grid, dim, dtype=dtype)


@torch.library.custom_op("token_world::patch_bag_grad", mutates_args=())
def patch_bag_grad(index: torch.Tensor, grad: torch.Tensor, pp: int, colours: int) -> torch.Tensor:
    """index [F, 256, 256], tokens' gradient [F, N, dim] -> M's gradient [pp, colours, dim] float32."""
    p = int(round(pp ** 0.5))
    grid = SIZE // p
    dim = grad.shape[-1]
    index, grad = index.contiguous(), grad.contiguous()
    n = grad.shape[0] * grad.shape[1]
    out = torch.zeros(pp, colours, dim, dtype=torch.float32, device=grad.device)
    chunks = triton.cdiv(n, TOKENS)
    _bag_backward[(pp, chunks, triton.cdiv(dim, BLOCK_D))](index, grad, out, n, dim, colours, TOKENS,
                                                           P=p, GRID=grid, C=max(16, triton.next_power_of_2(colours)),
                                                           BLOCK_D=BLOCK_D, TILE=TILE, EXACT=grad.dtype == torch.float32)
    return out


@patch_bag_grad.register_fake
def _(index, grad, pp, colours):
    return grad.new_empty(pp, colours, grad.shape[-1], dtype=torch.float32)


def _setup(ctx, inputs, output):
    index, table, _ = inputs
    ctx.save_for_backward(index)
    ctx.shape, ctx.dtype = table.shape, table.dtype


def _backward(ctx, grad):
    (index,) = ctx.saved_tensors
    pp, colours, _ = ctx.shape
    return None, patch_bag_grad(index, grad, pp, colours).to(ctx.dtype), None


patch_bag.register_autograd(_backward, setup_context=_setup)


def bag_table(conv_weight: torch.Tensor, colour_table: torch.Tensor) -> torch.Tensor:
    """The patch conv's weight [dim, colour_dim, P, P] and the colour table [colours, colour_dim] -> M
    [P*P, colours, dim]: each position's colour rows."""
    dim, e = conv_weight.shape[:2]
    return torch.einsum("dep,ke->pkd", conv_weight.reshape(dim, e, -1), colour_table)
