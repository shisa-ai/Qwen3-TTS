"""Opt-in E4M3 scaled GEMM for transformer linears; no global modifications."""
from __future__ import annotations

import types
import torch
import torch.nn.functional as F

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


def quantize(x):
    scale = x.detach().abs().amax().float().clamp_min(1e-12) / FP8_MAX
    torch._assert_async(torch.isfinite(scale), "Nonfinite FP8 input scale")
    normalized = x.float() / scale
    return normalized.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn), scale


def scaled_linear(x, weight_fp8, weight_scale, bias=None):
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    rows = flat.shape[0]
    # cuBLAS FP8 GEMM requires aligned dimensions, including M on some builds.
    padded = F.pad(flat, (0, 0, 0, (-rows) % 16))
    activation, scale = quantize(padded)
    result = torch._scaled_mm(activation, weight_fp8.t(), scale, weight_scale,
                              out_dtype=torch.bfloat16, use_fast_accum=False)
    result = result[:rows].reshape(*shape[:-1], weight_fp8.shape[0])
    if bias is not None:
        result = result + bias
    return result.to(x.dtype)


class FP8Patch:
    """Retain original BF16 parameters for exact reversal and unsupported-input fallback.

    This deliberately does not claim reduced resident weight memory. Only attention
    and MLP projections in the talker and its code predictor are selected.
    """
    def __init__(self, model):
        self.entries = []
        self.skipped = []
        target = model.model.talker
        try:
            for name, module in target.named_modules():
                if not isinstance(module, torch.nn.Linear):
                    continue
                if not any(part in name.split('.') for part in ('self_attn', 'mlp')):
                    continue
                n, k = module.weight.shape
                if k % 16 or n % 16 or module.weight.dtype != torch.bfloat16 or not module.weight.is_cuda:
                    self.skipped.append(name)
                    continue
                weight, scale = quantize(module.weight)
                original = module.forward
                had_local = 'forward' in module.__dict__
                local = module.__dict__.get('forward')
                def forward(this, x, _w=weight, _s=scale, _original=original):
                    if (this.training or not x.is_cuda or x.dtype != torch.bfloat16
                            or x.shape[-1] != _w.shape[1] or x.numel() == 0
                            or torch.is_grad_enabled()):
                        return _original(x)
                    return scaled_linear(x, _w, _s, this.bias)
                self.entries.append((name, module, original, had_local, local, weight, scale))
                module.forward = types.MethodType(forward, module)
        except Exception:
            self.remove()
            raise

    def remove(self):
        for _, module, _, had_local, local, _, _ in self.entries:
            if had_local:
                module.forward = local
            else:
                del module.forward
        self.entries.clear()

    @property
    def summary(self):
        return {'layers': len(self.entries), 'names': [e[0] for e in self.entries],
                'skipped': self.skipped,
                'additional_weight_bytes': sum(e[5].numel() + e[6].numel()*4 for e in self.entries),
                'retains_bf16_weights': True, 'activation_scaling': 'dynamic per tensor FP32 amax',
                'weight_scaling': 'static per tensor FP32 amax', 'fp8_format': 'e4m3fn',
                'output_dtype': 'bfloat16', 'fast_accum': False}


def install_fp8(model):
    return FP8Patch(model)
