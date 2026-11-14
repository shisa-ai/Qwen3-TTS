"""Experimental instance-local predictor RMSNorm; install before CUDA graph capture.

No global patch or automatic installation. Uninstall/reinstall graph adapters when
changing this patch: already captured CUDA graphs retain their captured kernels.
"""
from __future__ import annotations
import types
import torch
import triton
import triton.language as tl


@triton.jit
def _norm(X, W, Y, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + col, col < N, 0).to(tl.float32)
    variance = tl.sum(x * x, 0) / N
    normalized = x * tl.rsqrt(variance + EPS)
    # Upstream rounds normalized activations to input dtype BEFORE weight multiply.
    rounded = normalized.to(Y.dtype.element_ty).to(tl.float32)
    w = tl.load(W + col, col < N, 0).to(tl.float32)
    tl.store(Y + row * N + col, rounded * w, col < N)


class FusedPredictorNorm:
    def __init__(self, predictor):
        if predictor.training or next(predictor.parameters()).device.type != 'cuda':
            raise ValueError('Requires an eval CUDA predictor')
        if hasattr(predictor, '_predictor_graph'):
            raise ValueError('Install fused norms before installing predictor graphs')
        if hasattr(predictor, '_fused_predictor_norm'):
            raise ValueError('Fused predictor norms already installed')
        self.predictor = predictor
        self.entries = []
        self.widths = {predictor.config.hidden_size, predictor.config.head_dim}
        for module in predictor.model.modules():
            if module.__class__.__name__ != 'Qwen3TTSRMSNorm':
                continue
            original = module.forward
            prior = module.__dict__.get('forward')
            width = module.weight.numel()
            def forward(m, x, original=original, width=width):
                if m.training or torch.is_grad_enabled() or torch.is_autocast_enabled('cuda'):
                    raise ValueError('Fused norms require eval, no_grad/inference_mode, no autocast')
                if x.device != m.weight.device or x.device.type != 'cuda':
                    raise ValueError('Input and weight must share a CUDA device')
                if x.dtype != m.weight.dtype or x.dtype not in (torch.bfloat16, torch.float32):
                    raise ValueError('Requires matching BF16 or FP32 input and weight')
                # Preserve upstream behavior for untargeted layouts and larger prefill.
                if (width not in self.widths or x.ndim not in (3, 4) or x.shape[0] != 1
                        or x.shape[1] not in (1, 2) or x.shape[-1] != width
                        or not x.is_contiguous() or not m.weight.is_contiguous()):
                    return original(x)
                y = torch.empty_like(x)
                _norm[(x.numel() // width,)](x, m.weight, y, width, m.variance_epsilon,
                                            triton.next_power_of_2(width), num_warps=4,
                                            enable_fp_fusion=False)
                return y
            module.forward = types.MethodType(forward, module)
            self.entries.append((module, original, prior))
        predictor._fused_predictor_norm = self

    def uninstall(self):
        if getattr(self.predictor, '_fused_predictor_norm', None) is not self:
            raise RuntimeError('Patch is not installed')
        if hasattr(self.predictor, '_predictor_graph'):
            raise ValueError('Uninstall predictor graphs before fused norms')
        for module, _, prior in self.entries:
            if prior is None:
                del module.forward
            else:
                module.forward = prior
        del self.predictor._fused_predictor_norm


def install_fused_predictor_norm(predictor):
    return FusedPredictorNorm(predictor)
