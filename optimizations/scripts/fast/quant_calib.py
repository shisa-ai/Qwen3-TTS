"""Calibration-based weight-only INT8 quantization (GPTQ/OBQ) for the gfx1151 engine.

RTN (round-to-nearest) per-channel INT8 is the default. GPTQ minimizes the layer output error
||WX - QX|| using a few calibration activations, propagating each column's rounding error to the
remaining columns via the inverse Hessian. Output is still INT8 + per-channel scales, so the fused
GEMV kernel is unchanged. At INT8 this cuts per-GEMM error ~1.5x vs RTN (more at lower bit widths).

Usage:
    from quant_calib import calibrate_gptq
    gw = calibrate_gptq(model)                 # captures activations, runs GPTQ per weight
    install_fast_tts(model, weight_bits=8, gptq_weights=gw)
"""
from __future__ import annotations
import torch
import _paths  # noqa: F401  (sys.path)
import talker_engine as te
from bench_common import TEXTS


def _gptq_dequant(W, X, bits=8, percdamp=0.01):
    """OBQ/GPTQ: return the dequantized weight (float) that minimizes ||(W-Q)X||. W [N,K], X [ns,K]."""
    N, K = W.shape
    W = W.float().clone()
    H = (X.T @ X) * (2.0 / X.shape[0])
    H.diagonal().add_(percdamp * torch.diag(H).mean().clamp(min=1e-6))
    Hinv = H.inverse()
    qmax = 2 ** (bits - 1) - 1
    scale = (W.abs().amax(dim=1) / qmax).clamp(min=1e-8)
    Q = torch.zeros_like(W)
    for k in range(K):
        w = W[:, k]
        q = torch.round(w / scale).clamp(-qmax, qmax) * scale
        Q[:, k] = q
        if k < K - 1:
            W[:, k + 1:] -= (w - q)[:, None] * (Hinv[k + 1:, k] / Hinv[k, k])[None, :]
    return Q


def _to_int8(Wdeq):
    scale = (Wdeq.abs().amax(dim=1) / 127.0).clamp(min=1e-8)
    q = torch.round(Wdeq / scale[:, None]).clamp(-127, 127).to(torch.int8).contiguous()
    return q, scale.contiguous()


def _repack_gu(q, scale, inter, hid, src_half=32, dst_half=4):
    """GPTQ is per-row independent (one Hessian for all rows), so the quantized gate/up rows can be
    re-interleaved from the BF16 calibration packing (src_half) to the quantized GEMV packing (dst_half)."""
    def repack(t, h):
        gate = t.reshape(-1, 2, src_half, h)[:, 0].reshape(inter, h)
        up = t.reshape(-1, 2, src_half, h)[:, 1].reshape(inter, h)
        return torch.stack((gate.reshape(-1, dst_half, h), up.reshape(-1, dst_half, h)), 1).reshape(-1, h)
    return repack(q, hid).contiguous(), repack(scale, 1).reshape(-1).contiguous()


@torch.inference_mode()
def calibrate_gptq(model, *, texts=("short", "medium", "numbers", "long"), max_new_tokens=40,
                   nsamples=256, bits=8, verbose=False):
    """Capture folded-GEMM calibration activations from a BF16 engine, then GPTQ-quantize each weight.
    Returns a gptq_weights dict for install_fast_tts(weight_bits=8, gptq_weights=...)."""
    h = te.install_talker_engine(model.model.talker)   # BF16
    eng = h.engine
    # map weight identity -> override key
    keymap = {}
    for li, L in enumerate(eng.layers):
        for name in ("qkv", "o", "gu", "down"):
            keymap[id(L[name])] = (li, name)
    keymap[id(eng.head)] = "head"
    for li, L in enumerate(eng.pred.pdl_w):
        for name in ("qkv", "o", "gu", "down"):
            keymap[id(L[name])] = ("pred", li, name)
    for i, w in enumerate(eng.pred.pdl_lm):
        keymap[id(w)] = ("pred", "lm", i)
    cap = {}

    def make_hook(orig):
        def hook(cfg, a, w, *args, **kw):
            if not torch.cuda.is_current_stream_capturing():
                key = keymap.get(id(w))
                if key is not None:
                    c = cap.setdefault(key, [])
                    if sum(t.shape[0] for t in c) < nsamples:
                        c.append(a.detach().reshape(-1, a.shape[-1]).float().clone())
            return orig(cfg, a, w, *args, **kw)
        return hook

    eng._gemm = make_hook(eng._gemm)
    eng.pred._gemm = make_hook(eng.pred._gemm)
    for tid in texts:
        torch.manual_seed(0)
        model.generate_custom_voice(text=TEXTS[tid], speaker="Ryan", language="English", max_new_tokens=max_new_tokens)
    if verbose:
        print(f"captured {len(cap)} weights", flush=True)
    gw = {}
    for key, chunks in cap.items():
        X = torch.cat(chunks, 0)
        # recover the BF16 folded weight from the keymap source
        W = _weight_for_key(eng, key).float()
        if X.shape[0] < 16:
            continue
        Wdeq = _gptq_dequant(W, X.to(W.device), bits=bits)
        q, s = _to_int8(Wdeq)
        is_gu = (key[0] == "pred" and key[2] == "gu") or (key[0] != "pred" and key != "head" and key[1] == "gu")
        if is_gu:
            inter = W.shape[0] // 2
            q, s = _repack_gu(q, s, inter, W.shape[1])
        gw[key] = (q, s)
        if verbose:
            print(f"  {key}: {tuple(W.shape)} ns={X.shape[0]}", flush=True)
    h.uninstall()
    return gw


def _weight_for_key(eng, key):
    if key == "head":
        return eng.head
    if key[0] == "pred":
        if key[1] == "lm":
            return eng.pred.pdl_lm[key[2]]
        return eng.pred.pdl_w[key[1]][key[2]]
    return eng.layers[key[0]][key[1]]