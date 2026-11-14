"""Calibration-based weight-only quantization (GPTQ/OBQ) for the gfx1151 engine.

Captures REAL decode activations by running the engine eagerly (graph replay would otherwise hide
the per-frame GEMM inputs), then GPTQ-quantizes each folded weight to minimize ||(W-Q)X||. Supports
INT8 (per-channel) and INT4 (per-group). Output feeds install_fast_tts(weight_bits=..., gptq_weights=...).

Usage:
    from quant_calib import calibrate_gptq
    gw = calibrate_gptq(model, bits=4, group_size=128)
    install_fast_tts(model, weight_bits=4, group_size=128, gptq_weights=gw)
"""
from __future__ import annotations
import torch
import _paths  # noqa: F401  (sys.path)
import talker_engine as te
from bench_common import TEXTS


def _gptq_dequant(W, X, bits=8, G=0, percdamp=0.05):
    """OBQ/GPTQ with per-channel (G=0) or per-group (G>0) scales. Returns dequantized W (float)."""
    N, K = W.shape
    W = W.float().clone()
    H = (X.T @ X) * (2.0 / X.shape[0])
    H.diagonal().add_(percdamp * torch.diag(H).mean().clamp(min=1e-6))
    Hinv = H.inverse()
    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))
    if G:
        scale = (W.reshape(N, K // G, G).abs().amax(dim=2) / qmax).clamp(min=1e-8)  # [N, K//G]
    else:
        scale = (W.abs().amax(dim=1) / qmax).clamp(min=1e-8)  # [N]
    Q = torch.zeros_like(W)
    for k in range(K):
        s = scale[:, k // G] if G else scale
        w = W[:, k]
        q = torch.round(w / s).clamp(qmin, qmax) * s
        Q[:, k] = q
        if k < K - 1:
            W[:, k + 1:] -= (w - q)[:, None] * (Hinv[k + 1:, k] / Hinv[k, k])[None, :]
    return Q


def _pack_int8(Wdeq):
    scale = (Wdeq.abs().amax(dim=1) / 127.0).clamp(min=1e-8)
    q = torch.round(Wdeq / scale[:, None]).clamp(-127, 127).to(torch.int8).contiguous()
    return q, scale.contiguous()


def _pack_int4_group(Wdeq, G):
    N, K = Wdeq.shape
    wr = Wdeq.reshape(N, K // G, G)
    scale = (wr.abs().amax(dim=2) / 7.0).clamp(min=1e-8)
    q = torch.round(wr / scale[:, :, None]).clamp(-8, 7).to(torch.int8).reshape(N, K)
    packed = (((q[:, 1::2] & 0xF) << 4) | (q[:, 0::2] & 0xF)).to(torch.uint8).contiguous()
    return packed, scale.contiguous()


def _repack_rows(t, inter, hid, src_half=32, dst_half=4):
    gate = t.reshape(-1, 2, src_half, hid)[:, 0].reshape(inter, hid)
    up = t.reshape(-1, 2, src_half, hid)[:, 1].reshape(inter, hid)
    return torch.stack((gate.reshape(-1, dst_half, hid), up.reshape(-1, dst_half, hid)), 1).reshape(-1, hid)


class _EagerGraph:
    def __init__(self, eng, key, pkey):
        self.eng, self.key, self.pkey = eng, key, pkey

    def replay(self):
        self.eng._frame(self.key, self.pkey)


@torch.inference_mode()
def calibrate_gptq(model, *, bits=8, group_size=0, texts=("short", "medium", "numbers", "long"),
                   max_new_tokens=80, nsamples=1024, percdamp=0.05, verbose=False):
    """Capture real decode activations (eager frames), GPTQ-quantize each folded weight. Returns a
    gptq_weights dict for install_fast_tts(weight_bits=bits, group_size=group_size, gptq_weights=...)."""
    if bits == 4 and not group_size:
        group_size = 128
    h = te.install_talker_engine(model.model.talker)   # BF16
    eng = h.engine
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
    eng._graph = lambda key, pkey: _EagerGraph(eng, key, pkey)   # force eager decode frames
    for tid in texts:
        torch.manual_seed(0)
        model.generate_custom_voice(text=TEXTS[tid], speaker="Ryan", language="English", max_new_tokens=max_new_tokens)
    if verbose:
        print(f"captured {len(cap)} weights", flush=True)
    gw = {}
    for key, chunks in cap.items():
        X = torch.cat(chunks, 0)
        X = X[X.abs().sum(1) > 1e-6]   # drop zero-padded prefill rows
        if X.shape[0] < 32:
            continue
        W = _weight_for_key(eng, key).float()
        Wdeq = _gptq_dequant(W, X.to(W.device), bits=bits, G=group_size, percdamp=percdamp)
        if bits == 4:
            q, s = _pack_int4_group(Wdeq, group_size)
        else:
            q, s = _pack_int8(Wdeq)
        is_gu = (key[0] == "pred" and key[2] == "gu") or (key[0] != "pred" and key != "head" and key[1] == "gu")
        if is_gu:
            inter = W.shape[0] // 2
            q = _repack_rows(q, inter, q.shape[1]).contiguous()
            s = _repack_rows(s.reshape(s.shape[0], -1), inter, s.shape[1] if s.ndim > 1 else 1)
            s = s.reshape(s.shape[0], -1).contiguous() if s.ndim > 1 else s.reshape(-1).contiguous()
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