"""GELU(RMSNorm(x)) over narrow rows as one fused GPU kernel, forward and backward (2026-11-02).

The conv-ends pyramids (models/dynamics.py Dynamics._steps on the way in, Dynamics.pixel_groups on the way out)
put a per-pixel RMSNorm and a GELU between their maps: rows 16-64 wide, millions of them (12 rows of 64 frames:
12.6 M at the input's first level, 4.7 M at a head's last). Torch's fused RMSNorm is its layer-norm kernel,
made for wide rows: in a 12-row step it was the largest single kernel (136 ms forward, 52 backward, of a 1.46 s
step), the GELU a pass of its own. Compiling the pyramids fused them (input 128 -> 82 ms, heads 135 -> 92,
forward and backward) but the trainer's calls (token counts, sprite or not, grad or not, its thread count) ran
pixel_groups past dynamo's recompile limit, and a checkpointed call recomputed eagerly what it had saved
compiled (CheckpointError). So this is a plain autograd function: one memory pass forward (read x, write the
output), one backward (read x and the gradient, write x's; the weight's gradient as per-program partial sums).

The math is F.gelu(F.rms_norm(x, (w,), weight, eps)): the mean square, the norm and the exact (erf) GELU in
float32, the output in x's dtype; eps None is float32's machine epsilon (torch's default, also for bf16 input).
tests/test_norm_kernel.py checks it, values and gradients, against those ops.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

ELEMENTS = 8192      # values a program takes (rows of the padded width)
WARPS = 4


@triton.jit
def _norm_gelu_forward(x_ptr, w_ptr, y_ptr, rows, eps, W: tl.constexpr, WP: tl.constexpr, R: tl.constexpr):
    row = tl.program_id(0) * R + tl.arange(0, R)
    col = tl.arange(0, WP)
    inside = (row < rows)[:, None] & (col < W)[None, :]
    at = row[:, None] * W + col[None, :]
    x = tl.load(x_ptr + at, mask=inside, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + col, mask=col < W, other=0.0).to(tl.float32)
    r = 1.0 / tl.sqrt(tl.sum(x * x, 1) / W + eps)
    n = x * r[:, None] * w[None, :]
    tl.store(y_ptr + at, (0.5 * n * (1 + tl.erf(n * 0.7071067811865476))).to(y_ptr.dtype.element_ty), mask=inside)


@triton.jit
def _norm_gelu_backward(x_ptr, w_ptr, dy_ptr, dx_ptr, part_ptr, rows, eps, W: tl.constexpr, WP: tl.constexpr,
                        R: tl.constexpr):
    pid = tl.program_id(0)
    row = pid * R + tl.arange(0, R)
    col = tl.arange(0, WP)
    inside = (row < rows)[:, None] & (col < W)[None, :]
    at = row[:, None] * W + col[None, :]
    x = tl.load(x_ptr + at, mask=inside, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + at, mask=inside, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + col, mask=col < W, other=0.0).to(tl.float32)
    r = 1.0 / tl.sqrt(tl.sum(x * x, 1) / W + eps)
    xhat = x * r[:, None]
    n = xhat * w[None, :]
    dn = dy * (0.5 * (1 + tl.erf(n * 0.7071067811865476)) + n * 0.3989422804014327 * tl.exp(-0.5 * n * n))
    u = dn * w[None, :]
    dx = r[:, None] * (u - xhat * (tl.sum(u * xhat, 1) / W)[:, None])
    tl.store(dx_ptr + at, dx.to(dx_ptr.dtype.element_ty), mask=inside)
    tl.store(part_ptr + pid * WP + col, tl.sum(dn * xhat, 0))


def _shape(x: torch.Tensor):
    w = x.shape[-1]
    wp = triton.next_power_of_2(w)
    return x.numel() // w, w, wp, max(1, ELEMENTS // wp)


class NormGelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        x = x.contiguous()
        rows, w, wp, r = _shape(x)
        y = torch.empty_like(x)
        _norm_gelu_forward[(triton.cdiv(rows, r),)](x, weight, y, rows, eps, W=w, WP=wp, R=r, num_warps=WARPS)
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors
        rows, w, wp, r = _shape(x)
        programs = triton.cdiv(rows, r)
        dx = torch.empty_like(x)
        part = torch.empty(programs, wp, dtype=torch.float32, device=x.device)
        _norm_gelu_backward[(programs,)](x, weight, dy.contiguous(), dx, part, rows, ctx.eps, W=w, WP=wp, R=r,
                                         num_warps=WARPS)
        return dx, part.sum(0)[:w].to(weight.dtype), None


def norm_gelu(x: torch.Tensor, weight: torch.Tensor, eps: float | None) -> torch.Tensor:
    """GELU(RMSNorm(x)) over x's last dim with the norm's weight [w] (float32 under autocast, as the module
    keeps it) -> the same shape and dtype as x (the GPU's; models/dynamics.py's norm_gelu calls it)."""
    return NormGelu.apply(x, weight, torch.finfo(torch.float32).eps if eps is None else float(eps))
