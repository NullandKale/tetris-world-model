"""The pixel head's cross-entropy as one fused GPU kernel (2026-10-14).

Per pixel the head is small: a colour_dim vector against every colour's vector
plus a bias, a softmax over the colours, and the loss's gradient. Done as
separate PyTorch ops (a [pixels, 16] x [16, 58] matmul, a log-softmax over 58
values, a gather, the one-hot difference, two more matmuls), torch.compile
left it at about 50 GB/s and 41% of a 96 x 3 model's training step (profiled
2026-10-14). This kernel does it in one pass: each program walks a run of
pixels, keeps every logit in registers, writes the loss and the pixel
vectors' gradient, and adds its share of the colour table's and colour bias's
gradients once at the end. The same math as dynamics._ce_and_grad and the
matmuls around it (tests/test_head_kernel.py), the logits from tf32 matmuls and the softmax in float32.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

# Measured on the RTX 3090 (2026-10-14, 1M pixels: 2.4 ms against 11 ms for the separate ops): 32 pixels a step,
# 2 warps, 4,096 programs; tf32 matmuls (strict float32 is 2x slower; the separate ops ran them in bf16)
BLOCK = 32           # pixels per inner step (the matmuls' M)
WARPS = 2
PROGRAMS = 4096      # programs over all pixels: each adds its colour-table gradient once
PRECISION = "tf32"


@triton.jit
def _head_ce_kernel(pixels_ptr, table_ptr, bias_ptr, y_ptr, share_ptr, loss_ptr, g_pixels_ptr, g_table_ptr,
                    g_bias_ptr, n_pixels, colours, per_program,
                    E: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr, PRECISION: tl.constexpr):
    pid = tl.program_id(0)
    e = tl.arange(0, E)
    c = tl.arange(0, C)
    real = c < colours
    table = tl.load(table_ptr + c[:, None] * E + e[None, :], mask=real[:, None], other=0.0).to(tl.float32)  # [C, E]
    bias = tl.load(bias_ptr + c, mask=real, other=0.0).to(tl.float32)
    g_table = tl.zeros((C, E), dtype=tl.float32)
    g_bias = tl.zeros((C,), dtype=tl.float32)
    start = pid * per_program
    for offset in range(0, per_program, BLOCK):
        rows = start + offset + tl.arange(0, BLOCK)
        inside = (rows < n_pixels) & (rows < start + per_program)
        pix = tl.load(pixels_ptr + rows[:, None] * E + e[None, :], mask=inside[:, None], other=0.0).to(tl.float32)
        y = tl.load(y_ptr + rows, mask=inside, other=0)
        share = tl.load(share_ptr + rows, mask=inside, other=0.0)
        logits = tl.dot(pix, tl.trans(table), input_precision=PRECISION) + bias[None, :]           # [BLOCK, C]
        logits = tl.where(real[None, :], logits, float("-inf"))
        top = tl.max(logits, 1)
        p = tl.exp(logits - top[:, None])
        total = tl.sum(p, 1)
        lse = top + tl.log(total)
        hit = c[None, :] == y[:, None]
        loss = lse - tl.sum(tl.where(hit, logits, 0.0), 1)
        tl.store(loss_ptr + rows, loss, mask=inside)
        grad = (p / total[:, None] - hit.to(tl.float32)) * share[:, None]
        grad = tl.where(inside[:, None] & real[None, :], grad, 0.0)
        g_pix = tl.dot(grad, table, input_precision=PRECISION)                                       # [BLOCK, E]
        tl.store(g_pixels_ptr + rows[:, None] * E + e[None, :], g_pix.to(g_pixels_ptr.dtype.element_ty),
                 mask=inside[:, None])
        g_table += tl.dot(tl.trans(grad), pix, input_precision=PRECISION)                           # [C, E]
        g_bias += tl.sum(grad, 0)
    tl.atomic_add(g_table_ptr + c[:, None] * E + e[None, :], g_table, mask=real[:, None])
    tl.atomic_add(g_bias_ptr + c, g_bias, mask=real)


def head_ce(pixels: torch.Tensor, table: torch.Tensor, bias: torch.Tensor, y: torch.Tensor, share: torch.Tensor,
            precision: str = PRECISION, block: int = BLOCK, warps: int = WARPS, programs: int = PROGRAMS):
    """pixels [n, e] (any float dtype), table [colours, e], bias [colours], y [n] long, share [n] each pixel's
    weight -> per-pixel loss [n] float32, the gradient of the share-weighted summed loss for the pixels [n, e]
    in their dtype, and for the table [colours, e] and the bias [colours] in float32. precision: the matmuls' ("ieee": strict float32)."""
    n, e = pixels.shape
    colours = table.shape[0]
    if e < 16 or e & (e - 1):
        raise ValueError(f"colour_dim {e}: the kernel needs a power of two of at least 16")
    pixels, table, bias, y = pixels.contiguous(), table.contiguous(), bias.contiguous(), y.contiguous()
    share = share.float().contiguous()
    loss = torch.empty(n, dtype=torch.float32, device=pixels.device)
    g_pixels = torch.empty_like(pixels)
    g_table = torch.zeros(colours, e, dtype=torch.float32, device=pixels.device)
    g_bias = torch.zeros(colours, dtype=torch.float32, device=pixels.device)
    programs = max(1, min(programs, triton.cdiv(n, block)))
    per_program = triton.cdiv(triton.cdiv(n, programs), block) * block
    programs = triton.cdiv(n, per_program)
    _head_ce_kernel[(programs,)](pixels, table, bias, y, share, loss, g_pixels, g_table, g_bias, n, colours,
                                 per_program, E=e, C=max(16, triton.next_power_of_2(colours)), BLOCK=block,
                                 PRECISION=precision, num_warps=warps)
    return loss, g_pixels, g_table, g_bias
