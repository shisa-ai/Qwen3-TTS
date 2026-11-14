"""Faster 12 Hz codec decoder kernels (inference only, BF16 CUDA).

* SnakeBeta: one Triton pass, x + sin(x * e^a)^2 / (e^b + 1e-9) computed in FP32, one rounding
  (HF rounds after every BF16 op; this is slightly more accurate, not bit-identical).
* Dilated causal Conv1d (stride 1): cuDNN falls back to a slow legacy BF16 kernel for dilation > 1.
  Rewritten as an undilated conv over the d polyphase subsequences, batched:
  out[q*d + r] = sum_j w_j x[(q + j) * d + r]. Same arithmetic, different kernel.
"""
import types

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
if torch.version.hip is not None:
    from triton.language.extra import libdevice
else:
    from triton.language.extra.cuda import libdevice


@triton.jit
def _snake_kernel(x_ptr, y_ptr, a_ptr, b_ptr, T, C, BLOCK: tl.constexpr):
    row = tl.program_id(0)  # b * C + c
    c = row % C
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = offs < T
    a = tl.exp(tl.load(a_ptr + c).to(tl.float32))
    ib = 1.0 / (tl.exp(tl.load(b_ptr + c).to(tl.float32)) + 1e-9)
    base = row.to(tl.int64) * T
    x = tl.load(x_ptr + base + offs, mask=m).to(tl.float32)
    s = libdevice.sin(x * a)
    tl.store(y_ptr + base + offs, (x + ib * s * s).to(y_ptr.dtype.element_ty), mask=m)


def _snake_forward(self, x):
    if not x.is_cuda or not x.is_contiguous() or x.ndim != 3:
        return self._orig_forward(x)
    B, C, T = x.shape
    y = torch.empty_like(x)
    BLOCK = 2048
    _snake_kernel[(B * C, triton.cdiv(T, BLOCK))](x, y, self.alpha, self.beta, T, C, BLOCK=BLOCK, num_warps=4)
    return y


def _polyphase_forward(self, h):
    d = self.dilation
    conv = self.conv
    k = conv.kernel_size[0]
    B, C, T = h.shape
    P = (k - 1) * d
    Tp = T + P
    Q = -(-Tp // d)
    xp = F.pad(h, (P, Q * d - Tp))
    xr = xp.view(B, C, Q, d).permute(0, 3, 1, 2).reshape(B * d, C, Q)
    y = F.conv1d(xr, conv.weight, conv.bias)  # [B*d, Cout, Q-k+1]
    Qo = Q - k + 1
    y = y.view(B, d, -1, Qo).permute(0, 2, 3, 1).reshape(B, -1, Qo * d)
    return y[..., :T].contiguous()


def install_codec_fast(decoder, *, snake=True, dilated=True):
    n_snake = n_dil = 0
    for m in decoder.modules():
        name = type(m).__name__
        if snake and name == "SnakeBeta":
            m._orig_forward = m.forward
            m.forward = types.MethodType(_snake_forward, m)
            n_snake += 1
        elif (dilated and name == "Qwen3TTSTokenizerV2CausalConvNet" and m.dilation > 1 and m.stride == 1
              and m.conv.groups == 1 and m.conv.padding == (0,)):
            m._orig_forward = m.forward
            m.forward = types.MethodType(_polyphase_forward, m)
            n_dil += 1
    return dict(snake=n_snake, dilated=n_dil)
