"""Whole-frame CUDA-graph decode engine for Qwen3-TTS (batch one, BF16, CUDA).

One graph replay per codec frame runs, entirely on device:
  uniforms -> fused residual predictor (vendored PDL chain) -> codebook embedding sum
  -> 28 talker layers as a PDL GEMM chain (5 kernels/layer) -> final norm -> codec head
  -> talker sampler (repetition penalty, min-new-tokens, suppress, temperature, top-k,
     Gumbel-max) for the next frame's first code.
The host reads one int64 per frame (the sampled code) from pinned memory to stop at EOS.

Prefill runs through the unmodified HF talker; its KV cache is copied into the engine.
Numerics: BF16 storage with FP32 accumulation; RMSNorm weights are folded into the
following projections, so results are close to, not bit-identical with, HF.
Sampling is distributionally identical to HF multinomial sampling (Gumbel-max) but
consumes a different random stream.
"""
from __future__ import annotations

import math
import time

import torch
import triton
import triton.language as tl
from transformers.generation.utils import GenerateDecoderOnlyOutput

import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
for _p in (_os.path.dirname(_d), _d):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
from qwen_fused_predictor import _k, _mapped  # noqa: E402

_gemm_kernel = _k._cp_gemm_kernel
_gemm_stream_kernel = _k._cp_gemm_stream_kernel
_gemv_kernel = _k._cp_gemv_kernel
_gemv_w8_kernel = _k._cp_gemv_w8_kernel
_gemv_w8g_kernel = _k._cp_gemv_w8g_kernel
_gemv_w4g_kernel = _k._cp_gemv_w4g_kernel
_head_norm_rope = _k._head_norm_rope
_PDL_GU_BN = _k._PDL_GU_BN

# (BN, BK, NCH, BM, warps, MT, MSTAGES) register-panel configs, or ("s", BN, BK, SK, BM, warps, STAGES).
DEFAULT_CONFIGS = {
    "qkv": (64, 256, 2, 16, 4, 1, 1), "o": (32, 256, 1, 16, 4, 1, 2), "gu": (64, 256, 2, 16, 4, 1, 1),
    "down": (32, 128, 4, 16, 4, 1, 1), "lm": (64, 256, 2, 16, 8, 1, 1),
}

# Weight-only INT8 M=1 GEMV configs for bandwidth-bound GPUs (gfx1151): ~2x the BF16 GEMV.
QUANT_CONFIGS = {name: ("q", 8, 256, 4, 8) for name in ("qkv", "o", "gu", "down", "lm")}


@triton.jit
def _attn_kernel(
    qkv_ptr, kc_ptr, vc_ptr, qw_ptr, kw_ptr, cos_ptr, sin_ptr, state_ptr,
    ws_ptr, ml_ptr, cnt_ptr, out_ptr, eps, scale,
    H: tl.constexpr, KVH: tl.constexpr, HD: tl.constexpr, CAP: tl.constexpr,
    S: tl.constexpr, BK: tl.constexpr, PDL: tl.constexpr = True,
):
    """Split-KV decode attention for one new token with fused q/k RMSNorm, RoPE and cache write.

    grid (KVH, S). Program (g, s) handles both query heads of KV group g over key chunk s.
    The current token's key/value come from registers; split 0 stores them into the cache.
    The last arriving split of each group merges the partial softmax states in fixed order.
    """
    g = tl.program_id(0)
    s = tl.program_id(1)
    G: tl.constexpr = H // KVH
    HALF: tl.constexpr = HD // 2
    W: tl.constexpr = (H + 2 * KVH) * HD
    d = tl.arange(0, HALF)
    hq = tl.arange(0, G)
    p = tl.load(state_ptr + 1)  # write position; written >= 2 kernels earlier (PDL-safe)
    qw1 = tl.load(qw_ptr + d).to(tl.float32)
    qw2 = tl.load(qw_ptr + HALF + d).to(tl.float32)
    kw1 = tl.load(kw_ptr + d).to(tl.float32)
    kw2 = tl.load(kw_ptr + HALF + d).to(tl.float32)
    c1 = tl.load(cos_ptr + p * HD + d).to(tl.float32)
    c2 = tl.load(cos_ptr + p * HD + HALF + d).to(tl.float32)
    s1 = tl.load(sin_ptr + p * HD + d).to(tl.float32)
    s2 = tl.load(sin_ptr + p * HD + HALF + d).to(tl.float32)
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    # new key / value of this group
    kx1 = tl.load(qkv_ptr + (H + g) * HD + d).to(tl.float32)
    kx2 = tl.load(qkv_ptr + (H + g) * HD + HALF + d).to(tl.float32)
    vn1 = tl.load(qkv_ptr + (H + KVH + g) * HD + d).to(tl.float32)
    vn2 = tl.load(qkv_ptr + (H + KVH + g) * HD + HALF + d).to(tl.float32)
    kn1, kn2 = _head_norm_rope(kx1, kx2, kw1, kw2, c1, c2, s1, s2, eps, HD)
    base = (g * CAP).to(tl.int64) * HD
    if s == 0:
        dt = kc_ptr.dtype.element_ty
        tl.store(kc_ptr + base + p * HD + d, kn1.to(dt))
        tl.store(kc_ptr + base + p * HD + HALF + d, kn2.to(dt))
        tl.store(vc_ptr + base + p * HD + d, vn1.to(dt))
        tl.store(vc_ptr + base + p * HD + HALF + d, vn2.to(dt))
    # queries of the G heads of this group: [G, HALF] halves
    qb = qkv_ptr + (g * G + hq)[:, None] * HD + d[None, :]
    qx1 = tl.load(qb).to(tl.float32)
    qx2 = tl.load(qb + HALF).to(tl.float32)
    var = (tl.sum(qx1 * qx1, axis=1) + tl.sum(qx2 * qx2, axis=1)) / HD
    r = tl.rsqrt(var + eps)[:, None]
    n1 = _k._bf(qw1[None, :] * _k._bf(qx1 * r))
    n2 = _k._bf(qw2[None, :] * _k._bf(qx2 * r))
    q1 = _k._bf(_k._bf(n1 * c1[None, :]) + _k._bf(-n2 * s1[None, :])) * scale
    q2 = _k._bf(_k._bf(n2 * c2[None, :]) + _k._bf(n1 * s2[None, :])) * scale
    # chunk of keys 0..p handled by this split
    L = p + 1
    chunk = tl.cdiv(L, S)
    start = s * chunk
    end = tl.minimum(L, start + chunk)
    m_i = tl.full([G], -float("inf"), tl.float32)
    l_i = tl.zeros([G], tl.float32)
    a1 = tl.zeros([G, HALF], tl.float32)
    a2 = tl.zeros([G, HALF], tl.float32)
    j0 = tl.arange(0, BK)
    for b0 in range(start, end, BK):
        j = b0 + j0
        old = j < tl.minimum(end, p)
        kp = kc_ptr + base + j[:, None] * HD + d[None, :]
        vp = vc_ptr + base + j[:, None] * HD + d[None, :]
        k1 = tl.load(kp, mask=old[:, None], other=0.0).to(tl.float32)
        k2 = tl.load(kp + HALF, mask=old[:, None], other=0.0).to(tl.float32)
        v1 = tl.load(vp, mask=old[:, None], other=0.0).to(tl.float32)
        v2 = tl.load(vp + HALF, mask=old[:, None], other=0.0).to(tl.float32)
        cur = (j == p)[:, None]
        k1 = tl.where(cur, kn1[None, :], k1)
        k2 = tl.where(cur, kn2[None, :], k2)
        v1 = tl.where(cur, vn1[None, :], v1)
        v2 = tl.where(cur, vn2[None, :], v2)
        valid = j < end
        sc = tl.sum(q1[:, None, :] * k1[None, :, :], axis=2) + tl.sum(q2[:, None, :] * k2[None, :, :], axis=2)
        sc = tl.where(valid[None, :], sc, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(sc, axis=1))
        alpha = tl.exp(m_i - m_new)
        e = tl.exp(sc - m_new[:, None])
        l_i = l_i * alpha + tl.sum(e, axis=1)
        a1 = a1 * alpha[:, None] + tl.sum(e[:, :, None] * v1[None, :, :], axis=1)
        a2 = a2 * alpha[:, None] + tl.sum(e[:, :, None] * v2[None, :, :], axis=1)
        m_i = m_new
    wsb = ws_ptr + ((g * S + s) * G + hq)[:, None] * HD + d[None, :]
    tl.store(wsb, a1)
    tl.store(wsb + HALF, a2)
    tl.store(ml_ptr + ((g * S + s) * G + hq) * 2, m_i)
    tl.store(ml_ptr + ((g * S + s) * G + hq) * 2 + 1, l_i)
    tl.debug_barrier()
    arrived = tl.atomic_add(cnt_ptr + g, 1, sem="acq_rel", scope="gpu")
    if arrived == S - 1:
        M = tl.full([G], -float("inf"), tl.float32)
        for t in tl.static_range(S):
            M = tl.maximum(M, tl.load(ml_ptr + ((g * S + t) * G + hq) * 2, cache_modifier=".cg"))
        Lsum = tl.zeros([G], tl.float32)
        o1 = tl.zeros([G, HALF], tl.float32)
        o2 = tl.zeros([G, HALF], tl.float32)
        for t in tl.static_range(S):
            mt = tl.load(ml_ptr + ((g * S + t) * G + hq) * 2, cache_modifier=".cg")
            lt = tl.load(ml_ptr + ((g * S + t) * G + hq) * 2 + 1, cache_modifier=".cg")
            f = tl.where(lt > 0, tl.exp(mt - M), 0.0)
            Lsum += lt * f
            wb = ws_ptr + ((g * S + t) * G + hq)[:, None] * HD + d[None, :]
            o1 += tl.load(wb, cache_modifier=".cg") * f[:, None]
            o2 += tl.load(wb + HALF, cache_modifier=".cg") * f[:, None]
        ob = out_ptr + (g * G + hq)[:, None] * HD + d[None, :]
        dto = out_ptr.dtype.element_ty
        tl.store(ob, (o1 / Lsum[:, None]).to(dto))
        tl.store(ob + HALF, (o2 / Lsum[:, None]).to(dto))
        tl.atomic_xchg(cnt_ptr + g, 0)


@triton.jit
def _embed_sum_kernel(
    codes_ptr, first_ptr, tables_ptr, trail_ptr, pad_ptr, state_ptr, x_ptr, ss_ptr, codes_out_ptr,
    G: tl.constexpr, V: tl.constexpr, HID: tl.constexpr, BLOCK: tl.constexpr,
):
    """x = bf16(sum of the frame's 16 codebook embeddings) + text row; per-tile sums of squares.

    grid (HID // BLOCK,). Program 0 also records the frame's codes."""
    pid = tl.program_id(0)
    c = pid * BLOCK + tl.arange(0, BLOCK)
    step = tl.load(state_ptr)
    T = tl.load(state_ptr + 2)
    acc = tl.load(first_ptr + c).to(tl.float32)
    for i in tl.static_range(1, G):
        code = tl.load(codes_ptr + i)
        acc += tl.load(tables_ptr + ((i - 1) * V + code) * HID + c).to(tl.float32)
    e = acc.to(tl.bfloat16).to(tl.float32)
    t = tl.where(step < T, tl.load(trail_ptr + tl.minimum(step, T - 1) * HID + c).to(tl.float32),
                 tl.load(pad_ptr + c).to(tl.float32))
    x = (e + t).to(tl.bfloat16)
    tl.store(x_ptr + c, x)
    xf = x.to(tl.float32)
    tl.store(ss_ptr + pid, tl.sum(xf * xf, axis=0))
    if pid == 0:
        gi = tl.arange(0, G)
        tl.store(codes_out_ptr + step * G + gi, tl.load(codes_ptr + gi))


@triton.jit
def _final_norm_kernel(x_ptr, w_ptr, out_ptr, hist_ptr, state_ptr, eps, HID: tl.constexpr):
    """past_hidden = HF RMSNorm(x); also stored as hidden-state history row step + 1."""
    offs = tl.arange(0, HID)
    x = tl.load(x_ptr + offs).to(tl.float32)
    var = tl.sum(x * x, axis=0) / HID
    y = (x * tl.rsqrt(var + eps)).to(tl.bfloat16).to(tl.float32)
    o = (tl.load(w_ptr + offs).to(tl.float32) * y).to(tl.bfloat16)
    tl.store(out_ptr + offs, o)
    step = tl.load(state_ptr)
    tl.store(hist_ptr + (step + 1) * HID + offs, o)


@triton.jit
def _fkey(x):
    """Order-preserving int64 key of an FP32 value (larger value, larger key)."""
    b = x.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
    return tl.where(b >= 0x80000000, 0xFFFFFFFF - b, b | 0x80000000)


@triton.jit
def _talker_sample_kernel(
    logits_ptr, u_ptr, seen_ptr, suppress_ptr, state_ptr, emb_ptr, first_ptr, codes_out_ptr, tok_ptr,
    min_new, eos, penalty, temperature,
    V: tl.constexpr, BV: tl.constexpr, TOPK: tl.constexpr, HID: tl.constexpr, G: tl.constexpr,
    ADV: tl.constexpr, SAMPLE: tl.constexpr,
):
    """HF logits processors (repetition penalty, min new tokens, suppress, temperature, top-k)
    then Gumbel-max sampling (argmax when not sampling). Writes the code, its embedding, and
    advances the frame/position counters when ADV."""
    idx = tl.arange(0, BV)
    valid = idx < V
    step = tl.load(state_ptr)
    n = step + ADV  # tokens generated so far
    x = tl.load(logits_ptr + idx, mask=valid, other=0.0).to(tl.float32)
    seen = tl.load(seen_ptr + idx, mask=valid, other=0) != 0
    x = tl.where(seen, tl.where(x < 0, x * penalty, x / penalty), x)
    x = tl.where((idx == eos) & (n < min_new), -float("inf"), x)
    sup = tl.load(suppress_ptr + idx, mask=valid, other=1) != 0
    x = tl.where(sup | (idx >= V), -float("inf"), x)
    if SAMPLE:
        x = x / temperature
        if TOPK > 0:
            key = tl.where(valid, _fkey(x), 0)
            the = tl.zeros([], tl.int64)
            one = tl.full([], 1, tl.int64)
            for bit in tl.static_range(31, -1, -1):
                cand = the | (one << bit)
                cnt = tl.sum((key >= cand).to(tl.int32), axis=0)
                the = tl.where(cnt >= TOPK, cand, the)
            x = tl.where(key >= the, x, -float("inf"))
        u = tl.load(u_ptr + idx, mask=valid, other=0.5)
        u = tl.minimum(tl.maximum(u, 1e-7), 1.0 - 1e-7)
        score = x - tl.log(-tl.log(u))
    else:
        score = x
    score = tl.where(valid, score, -float("inf"))
    best = tl.max(score, 0)
    code = tl.min(tl.where(score == best, idx, BV), 0)
    code64 = code.to(tl.int64)
    tl.store(tok_ptr, code64)
    tl.store(codes_out_ptr + n * G, code64)
    tl.store(seen_ptr + code64, tl.full([], 1, tl.int8))
    c = tl.arange(0, HID)
    tl.store(first_ptr + c, tl.load(emb_ptr + code64 * HID + c))
    if ADV:
        tl.store(state_ptr, step + 1)
        tl.store(state_ptr + 1, tl.load(state_ptr + 1) + 1)


@triton.jit
def _prefill_kv_kernel(qkv_ptr, kc_ptr, vc_ptr, kw_ptr, cos_ptr, sin_ptr, eps,
                       H: tl.constexpr, KVH: tl.constexpr, HD: tl.constexpr, CAP: tl.constexpr):
    """grid (KVH, S): k RMSNorm + RoPE at position s and V copy into the cache."""
    g = tl.program_id(0)
    s = tl.program_id(1)
    HALF: tl.constexpr = HD // 2
    W: tl.constexpr = (H + 2 * KVH) * HD
    d = tl.arange(0, HALF)
    row = qkv_ptr + s.to(tl.int64) * W
    kx1 = tl.load(row + (H + g) * HD + d).to(tl.float32)
    kx2 = tl.load(row + (H + g) * HD + HALF + d).to(tl.float32)
    c1 = tl.load(cos_ptr + s * HD + d).to(tl.float32)
    c2 = tl.load(cos_ptr + s * HD + HALF + d).to(tl.float32)
    s1 = tl.load(sin_ptr + s * HD + d).to(tl.float32)
    s2 = tl.load(sin_ptr + s * HD + HALF + d).to(tl.float32)
    kw1 = tl.load(kw_ptr + d).to(tl.float32)
    kw2 = tl.load(kw_ptr + HALF + d).to(tl.float32)
    kn1, kn2 = _head_norm_rope(kx1, kx2, kw1, kw2, c1, c2, s1, s2, eps, HD)
    base = (g * CAP + s).to(tl.int64) * HD
    dt = kc_ptr.dtype.element_ty
    tl.store(kc_ptr + base + d, kn1.to(dt))
    tl.store(kc_ptr + base + HALF + d, kn2.to(dt))
    tl.store(vc_ptr + base + d, tl.load(row + (H + KVH + g) * HD + d))
    tl.store(vc_ptr + base + HALF + d, tl.load(row + (H + KVH + g) * HD + HALF + d))


@triton.jit
def _prefill_attn_kernel(qkv_ptr, kc_ptr, vc_ptr, qw_ptr, cos_ptr, sin_ptr, out_ptr, eps, scale,
                         H: tl.constexpr, KVH: tl.constexpr, HD: tl.constexpr, CAP: tl.constexpr,
                         BK: tl.constexpr):
    """grid (H, S): causal attention of query row s, head h, over cached keys 0..s."""
    h = tl.program_id(0)
    s = tl.program_id(1)
    g = h // (H // KVH)
    HALF: tl.constexpr = HD // 2
    W: tl.constexpr = (H + 2 * KVH) * HD
    d = tl.arange(0, HALF)
    row = qkv_ptr + s.to(tl.int64) * W
    qx1 = tl.load(row + h * HD + d).to(tl.float32)
    qx2 = tl.load(row + h * HD + HALF + d).to(tl.float32)
    c1 = tl.load(cos_ptr + s * HD + d).to(tl.float32)
    c2 = tl.load(cos_ptr + s * HD + HALF + d).to(tl.float32)
    s1 = tl.load(sin_ptr + s * HD + d).to(tl.float32)
    s2 = tl.load(sin_ptr + s * HD + HALF + d).to(tl.float32)
    qw1 = tl.load(qw_ptr + d).to(tl.float32)
    qw2 = tl.load(qw_ptr + HALF + d).to(tl.float32)
    q1, q2 = _head_norm_rope(qx1, qx2, qw1, qw2, c1, c2, s1, s2, eps, HD)
    q1 = q1 * scale
    q2 = q2 * scale
    base = (g * CAP).to(tl.int64) * HD
    m_i = -float("inf")
    l_i = 0.0
    a1 = tl.zeros([HALF], tl.float32)
    a2 = tl.zeros([HALF], tl.float32)
    j0 = tl.arange(0, BK)
    for b0 in range(0, s + 1, BK):
        j = b0 + j0
        ok = j <= s
        kp = kc_ptr + base + j[:, None] * HD + d[None, :]
        vp = vc_ptr + base + j[:, None] * HD + d[None, :]
        k1 = tl.load(kp, mask=ok[:, None], other=0.0).to(tl.float32)
        k2 = tl.load(kp + HALF, mask=ok[:, None], other=0.0).to(tl.float32)
        v1 = tl.load(vp, mask=ok[:, None], other=0.0).to(tl.float32)
        v2 = tl.load(vp + HALF, mask=ok[:, None], other=0.0).to(tl.float32)
        sc = tl.sum(k1 * q1[None, :], axis=1) + tl.sum(k2 * q2[None, :], axis=1)
        sc = tl.where(ok, sc, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(sc, axis=0))
        alpha = tl.exp(m_i - m_new)
        e = tl.exp(sc - m_new)
        l_i = l_i * alpha + tl.sum(e, axis=0)
        a1 = a1 * alpha + tl.sum(e[:, None] * v1, axis=0)
        a2 = a2 * alpha + tl.sum(e[:, None] * v2, axis=0)
        m_i = m_new
    ob = out_ptr + s.to(tl.int64) * (H * HD) + h * HD
    dto = out_ptr.dtype.element_ty
    tl.store(ob + d, (a1 / l_i).to(dto))
    tl.store(ob + HALF + d, (a2 / l_i).to(dto))


PREFILL_CONFIGS = {
    "qkv": ("s", 64, 128, 1, 64, 4, 3), "o": ("s", 64, 128, 1, 64, 4, 3), "gu": ("s", 64, 128, 1, 64, 4, 3),
    "down": ("s", 64, 128, 1, 64, 4, 3), "lm": ("s", 64, 128, 1, 16, 4, 3),
}


class TalkerEngine:
    def __init__(self, talker, *, capacity=4096, predictor_mode="pdl", attn_splits=8, attn_bk=32, attn_warps=4,
                 configs=None, pred_configs=None, pred_attn_o=False, own_prefill=True, prefill_max=256, weight_bits=16,
                 gptq_weights=None, group_size=0):
        self.talker = talker
        m = talker.model
        cfg = talker.config
        w0 = next(talker.parameters())
        self.device, dt = w0.device, w0.dtype
        if dt != torch.bfloat16 or self.device.type != "cuda":
            raise ValueError("TalkerEngine requires CUDA BF16")
        # PDL is Hopper-only; on ROCm the same chain runs as plain graph-ordered launches.
        self._pdl = _k._pdl_supported(self.device)
        if cfg.num_code_groups != 16 or cfg.hidden_act != "silu":
            raise ValueError("Unsupported talker config")
        a0 = m.layers[0].self_attn
        self.H, self.KVH, self.HD = cfg.num_attention_heads, cfg.num_key_value_heads, a0.head_dim
        self.hidden, self.vocab, self.G = cfg.hidden_size, cfg.vocab_size, cfg.num_code_groups
        self.eps = float(m.norm.variance_epsilon)
        self.capacity, self.S, self.abk, self.awarps = capacity, attn_splits, attn_bk, attn_warps
        self._w8 = weight_bits == 8
        self._w4 = weight_bits == 4
        self._quant = weight_bits in (4, 8)
        if weight_bits not in (4, 8, 16):
            raise ValueError("weight_bits must be 4, 8 or 16")
        self._group = group_size if self._quant else 0
        if self._w4 and not self._group:
            self._group = 128  # INT4 needs per-group scales
        if self._quant:
            # Quantized decode weights are INT8/INT4 tuples; the engine prefill kernels read BF16 weights,
            # so quantized mode uses the HF prefill (also faster than the engine prefill on gfx1151).
            own_prefill = False
        if self._w4:
            base_cfgs = {name: ("q4", 8, self._group, 4, 8) for name in QUANT_CONFIGS}
        elif self._w8 and self._group:
            base_cfgs = {name: ("qg", 8, self._group, 4, 8) for name in QUANT_CONFIGS}
        else:
            base_cfgs = QUANT_CONFIGS if self._w8 else DEFAULT_CONFIGS
        self.cfgs = dict(base_cfgs, **(configs or {}))
        f32 = lambda t: t.detach().float()
        bf = lambda t: t.to(torch.bfloat16).contiguous()
        fold = lambda w, ln: bf(f32(w) * f32(ln)[None, :])  # [N, K] weight with RMSNorm weight folded on K
        self._gptq = gptq_weights or {}
        if self._w4 and self._gptq:
            qw = lambda w, key: self._gptq.get(key, _k._quantize_w4_group(w, self._group))
        elif self._w4:
            qw = lambda w, key: _k._quantize_w4_group(w, self._group)
        elif self._w8 and self._gptq:
            qw = lambda w, key: self._gptq.get(key, _k._quantize_w8(w))
        elif self._w8 and self._group:
            qw = lambda w, key: _k._quantize_w8_group(w, self._group)
        else:
            qw = (lambda w, key: _k._quantize_w8(w)) if self._w8 else (lambda w, key: w)
        half = 4 if self._quant else _PDL_GU_BN // 2
        self.layers = []
        for li, layer in enumerate(m.layers):
            a, mlp = layer.self_attn, layer.mlp
            for p in (a.q_proj, a.k_proj, a.v_proj, a.o_proj, mlp.gate_proj, mlp.up_proj, mlp.down_proj):
                if p.bias is not None:
                    raise ValueError("bias-free projections required")
            qkv = torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight])
            gate = fold(mlp.gate_proj.weight, layer.post_attention_layernorm.weight)
            up = fold(mlp.up_proj.weight, layer.post_attention_layernorm.weight)
            gu = torch.stack((gate.reshape(-1, half, self.hidden), up.reshape(-1, half, self.hidden)), 1)
            self.layers.append(dict(
                qkv=qw(fold(qkv, layer.input_layernorm.weight), (li, "qkv")), o=qw(bf(a.o_proj.weight), (li, "o")),
                gu=qw(gu.reshape(-1, self.hidden).contiguous(), (li, "gu")), down=qw(bf(mlp.down_proj.weight), (li, "down")),
                qn=bf(a.q_norm.weight), kn=bf(a.k_norm.weight),
                eps1=float(layer.input_layernorm.variance_epsilon),
                eps2=float(layer.post_attention_layernorm.variance_epsilon),
                head_eps=float(a.q_norm.variance_epsilon)))
        self.scale = a0.scaling
        self.norm_w = bf(m.norm.weight)
        self.head = qw(fold(talker.codec_head.weight, m.norm.weight), "head")
        d0 = self.layers[0]["down"]
        self.inter = (d0[0] if self._quant else d0).shape[1]
        # RoPE tables exactly as HF builds them (text-only mRoPE: all three axes share positions).
        pos = torch.arange(capacity, device=self.device).view(1, 1, -1).expand(3, 1, -1)
        cos, sin = m.rotary_emb(torch.empty(1, device=self.device, dtype=dt), pos)
        from qwen_tts.core.models.modeling_qwen3_tts import apply_multimodal_rotary_pos_emb
        rs = a0.rope_scaling
        # Reduce the multimodal tables to plain ones by rotating unit vectors (exact selection).
        eye = torch.eye(self.HD, device=self.device, dtype=dt)
        self.cos = cos[0, 0].contiguous()
        self.sin = sin[0, 0].contiguous()
        probe = torch.randn(1, 1, 7, self.HD, device=self.device, dtype=dt)
        ref, _ = apply_multimodal_rotary_pos_emb(probe, probe, cos[:, :, :7], sin[:, :, :7],
                                                rs["mrope_section"], rs["interleaved"])
        def rot(x):
            return torch.cat((-x[..., self.HD // 2:], x[..., :self.HD // 2]), -1)
        plain = probe * self.cos[:7] + rot(probe) * self.sin[:7]
        if not torch.equal(ref, plain):
            raise ValueError("text-only mRoPE does not reduce to plain RoPE here")
        # Embedding tables
        self.emb0 = bf(talker.get_input_embeddings().weight)
        cp = talker.code_predictor
        self.tables = torch.stack([e.weight for e in cp.get_input_embeddings()]).to(torch.bfloat16).contiguous()
        # Predictor engine (vendored)
        if pred_attn_o:
            from pred_fused import FusedAttnOPredictor as _P
        else:
            _P = _k.FusedCodePredictor
        self.pred = _P(_mapped(cp), max_batch=1, mode=predictor_mode, weight_bits=weight_bits, gptq_weights=gptq_weights, group_size=group_size)
        self.pvocab = self.pred.vocab
        if self.pred._pdl_configs:
            p = self.pred
            p.pdl_ws = torch.zeros(64 * p._rows * max(p.pdl_qkv.shape[1], 2 * p.pdl_act.shape[1], p.vocab, p.hidden),
                                   device=self.device, dtype=torch.float32)
            if pred_configs:
                self.set_pred_configs(pred_configs)
            elif self._w4:
                self.set_pred_configs({name: ("q4", 8, self._group, 4, 8) for name in QUANT_CONFIGS})
            elif self._w8:
                pc = {name: ("qg", 8, self._group, 4, 8) for name in QUANT_CONFIGS} if self._group else QUANT_CONFIGS
                self.set_pred_configs(pc)
        # Buffers
        z = lambda *s, dtype=torch.bfloat16: torch.zeros(*s, device=self.device, dtype=dtype)
        L = len(self.layers)
        self.kc = z(L, self.KVH, capacity, self.HD)
        self.vc = z(L, self.KVH, capacity, self.HD)
        self.x = z(1, self.hidden)
        self.qkv = z(1, (self.H + 2 * self.KVH) * self.HD)
        self.att = z(1, self.H * self.HD)
        self.act = z(1, self.inter)
        self.logits = z(1, self.vocab)
        self.past = z(1, self.hidden)
        self.first = z(1, self.hidden)
        self.hist = z(capacity + 1, self.hidden)
        self.codes_out = z(capacity + 1, self.G, dtype=torch.long)
        self.trail = z(capacity, self.hidden)
        self.pad = z(self.hidden)
        self.state = z(4, dtype=torch.long)  # step, pos, T
        self.seen = z(self.vocab, dtype=torch.int8)
        self.suppress = z(self.vocab, dtype=torch.int8)
        self.uni = z(self.vocab + (self.G - 1) * self.pvocab, dtype=torch.float32)
        self.tok_host = torch.zeros(1, dtype=torch.long, pin_memory=True)
        self.tok = z(1, dtype=torch.long)
        widest = max(self.qkv.shape[1], 2 * self.inter, self.vocab, self.hidden)
        splits = max(self._split(c, k) for c, k in [(self.cfgs["qkv"], self.hidden), (self.cfgs["o"], self.H * self.HD),
                     (self.cfgs["gu"], self.hidden), (self.cfgs["down"], self.inter), (self.cfgs["lm"], self.hidden)])
        self.ws = z(64 * widest, dtype=torch.float32)  # any split-K up to 64
        self.cnt = z(16384, dtype=torch.int32)
        self.ss = [z(self.hidden // (8 if (torch.version.hip is not None or self._quant) else 16), 1, dtype=torch.float32) for _ in range(2)]
        # own prefill (bucketed graphs); prompts longer than pmax use the HF prefill
        self.pmax, self.pstep = prefill_max, 16
        P = self.pmax
        self.px = z(P, self.hidden)
        self.pqkv = z(P, self.qkv.shape[1])
        self.patt = z(P, self.H * self.HD)
        self.pact = z(P, self.inter)
        self.pss = [z(self.hidden // 16, P, dtype=torch.float32) for _ in range(2)]
        self.pws = z(1, dtype=torch.float32)  # split-K 1 prefill configs need no workspace
        self.pgraphs = {}
        self._neg1 = torch.full((4,), -1, device=self.device, dtype=torch.long)
        self.own_prefill = own_prefill
        self.aws = z(self.KVH * 64 * (self.H // self.KVH) * self.HD, dtype=torch.float32)  # splits <= 64
        self.aml = z(self.KVH * 64 * (self.H // self.KVH) * 2, dtype=torch.float32)
        self.acnt = z(self.KVH, dtype=torch.int32)
        self.graphs = {}
        self.stream = torch.cuda.current_stream(self.device)
        self.calls = self.fallback_calls = 0
        self.timings = {}
        self.codec = None
        self.audio = None

    def set_pred_configs(self, cfgs):
        base = dict(self.pred._pdl_configs[0][1])
        base.update(cfgs)
        self.pred._pdl_configs = ((1, base),)

    def pred_configs(self):
        return dict(self.pred._pdl_configs[0][1])

    @staticmethod
    def _split(cfg, k):
        if cfg[0] in ("g", "q", "qg", "q4"):
            return 1
        return cfg[3] if cfg[0] == "s" else k // (cfg[1] * cfg[2])

    def _gemm(self, cfg, a, w, out, ss_in, nt_in, ss_out, epi, eps=0.0, M=1, rmax=1, ro=0, ws=None):
        if cfg[0] in ("q", "qg", "q4"):
            w8, scale = w
            N, K = w8.shape[0], w8.shape[1] * (2 if cfg[0] == "q4" else 1)
            _, BN, BK, STAGES, warps = cfg
            kern = {"q": _gemv_w8_kernel, "qg": _gemv_w8g_kernel, "q4": _gemv_w4g_kernel}[cfg[0]]
            kern[(N // BN,)](
                a, 1, ro, w8, scale, self.ws if ws is None else ws, self.cnt, out, ss_in, ss_out, M, eps,
                K=K, N=N, HID=self.hidden, RMAX=rmax, BN=BN, BK=BK, EPI=epi,
                NT_IN=nt_in, STAGES=STAGES, num_warps=warps)
            return self.hidden // BN if epi == 2 else 0
        N, K = w.shape
        common = (a, 1, ro, w, self.ws if ws is None else ws, self.cnt, out, ss_in, ss_out, M, eps)
        lp = {"launch_pdl": True} if self._pdl else {}
        if cfg[0] == "g" and M == 1:
            _, BN, BK, STAGES, warps = cfg
            _gemv_kernel[(N // BN,)](
                *common, K=K, N=N, HID=self.hidden, RMAX=rmax, BN=BN, BK=BK, EPI=epi,
                NT_IN=nt_in, STAGES=STAGES, num_warps=warps)
            return self.hidden // BN if epi == 2 else 0
        if cfg[0] == "g":
            cfg = ("s", 64, 128, 1, 16, 4, 3)
        if cfg[0] == "s":
            _, BN, BK, SK, BM, warps, stages = cfg
            _gemm_stream_kernel[(N // BN, SK, triton.cdiv(M, BM))](
                *common, K=K, N=N, HID=self.hidden, RMAX=rmax, BN=BN, BK=BK, SK=SK, BM=BM, EPI=epi,
                NT_IN=nt_in, STAGES=stages, NLP=triton.next_power_of_2(K // SK // 64),
                num_warps=warps, PDL=self._pdl, **lp)
        else:
            BN, BK, NCH, BM, warps, MT, mstages = cfg
            _gemm_kernel[(N // BN, K // (BK * NCH), max(1, min(MT, triton.cdiv(M, BM))))](
                *common, K=K, N=N, HID=self.hidden, RMAX=rmax, BN=BN, BK=BK, NCH=NCH, BM=BM, EPI=epi,
                NT_IN=nt_in, MSTAGES=mstages, num_warps=warps, PDL=self._pdl, **lp)
        return self.hidden // BN if epi == 2 else 0

    def _sample(self, key, adv):
        sample, temperature, top_k, penalty, min_new, eos = key
        _talker_sample_kernel[(1,)](
            self.logits, self.uni, self.seen, self.suppress, self.state, self.emb0, self.first,
            self.codes_out, self.tok, min_new, eos, penalty, temperature,
            V=self.vocab, BV=triton.next_power_of_2(self.vocab), TOPK=top_k, HID=self.hidden, G=self.G,
            ADV=adv, SAMPLE=sample, num_warps=8)

    def _transformer(self):
        """Talker layers on self.x (with sums of squares in self.ss[0], NT tiles) -> logits."""
        ss_cur, ss_nxt = self.ss
        nt = self._nt0
        c = self.cfgs
        for li, Lw in enumerate(self.layers):
            self._gemm(c["qkv"], self.x, Lw["qkv"], self.qkv, ss_cur, nt, None, 0, Lw["eps1"])
            _attn_kernel[(self.KVH, self.S)](
                self.qkv, self.kc[li], self.vc[li], Lw["qn"], Lw["kn"], self.cos, self.sin, self.state,
                self.aws, self.aml, self.acnt, self.att, Lw["head_eps"], self.scale,
                H=self.H, KVH=self.KVH, HD=self.HD, CAP=self.capacity, S=self.S, BK=self.abk,
                num_warps=self.awarps, PDL=self._pdl, **({"launch_pdl": True} if self._pdl else {}))
            nt = self._gemm(c["o"], self.att, Lw["o"], self.x, None, 0, ss_nxt, 2)
            ss_cur, ss_nxt = ss_nxt, ss_cur
            self._gemm(c["gu"], self.x, Lw["gu"], self.act, ss_cur, nt, None, 1, Lw["eps2"])
            nt = self._gemm(c["down"], self.act, Lw["down"], self.x, None, 0, ss_nxt, 2)
            ss_cur, ss_nxt = ss_nxt, ss_cur
        _final_norm_kernel[(1,)](self.x, self.norm_w, self.past, self.hist, self.state, self.eps,
                                 HID=self.hidden, num_warps=8)
        self._gemm(c["lm"], self.x, self.head, self.logits, ss_cur, nt, None, 0, self.eps)

    def _frame(self, key, pkey):
        """One whole frame: predictor for the current first code, talker step, next first code."""
        psample, ptemp, ptopk = pkey
        if psample:
            self.uni.uniform_()
        else:
            self.uni.fill_(0.5)
        pu = self.uni[self.vocab:].view(1, self.G - 1, self.pvocab)
        codes = self.pred(self.tok, self.first, self.past, 1.0 / ptemp, ptopk, pu)
        blk = 256
        self._nt0 = self.hidden // blk
        _embed_sum_kernel[(self.hidden // blk,)](
            codes, self.first, self.tables, self.trail, self.pad, self.state, self.x, self.ss[0],
            self.codes_out, G=self.G, V=self.pvocab, HID=self.hidden, BLOCK=blk, num_warps=2)
        self._transformer()
        self._sample(key, 1)
        self.tok_host.copy_(self.tok, non_blocking=True)

    def _prefill_rows(self, M):
        """Talker transformer over px[:M] (positions 0..M-1), writing the KV cache; graph-capturable."""
        P, c = self.pmax, PREFILL_CONFIGS
        ss_cur, ss_nxt = self.pss
        torch.sum(self.px[:M].float().square(), dim=1, out=ss_cur[0, :M])
        nt = 1
        for li, Lw in enumerate(self.layers):
            self._gemm(c["qkv"], self.px, Lw["qkv"], self.pqkv, ss_cur, nt, None, 0, Lw["eps1"], M=M, rmax=P, ws=self.pws)
            _prefill_kv_kernel[(self.KVH, M)](self.pqkv, self.kc[li], self.vc[li], Lw["kn"], self.cos, self.sin,
                                              Lw["head_eps"], H=self.H, KVH=self.KVH, HD=self.HD, CAP=self.capacity,
                                              num_warps=1)
            _prefill_attn_kernel[(self.H, M)](self.pqkv, self.kc[li], self.vc[li], Lw["qn"], self.cos, self.sin,
                                              self.patt, Lw["head_eps"], self.scale, H=self.H, KVH=self.KVH,
                                              HD=self.HD, CAP=self.capacity, BK=32, num_warps=2)
            nt = self._gemm(c["o"], self.patt, Lw["o"], self.px, None, 0, ss_nxt, 2, M=M, rmax=P, ws=self.pws)
            ss_cur, ss_nxt = ss_nxt, ss_cur
            self._gemm(c["gu"], self.px, Lw["gu"], self.pact, ss_cur, nt, None, 1, Lw["eps2"], M=M, rmax=P, ws=self.pws)
            nt = self._gemm(c["down"], self.pact, Lw["down"], self.px, None, 0, ss_nxt, 2, M=M, rmax=P, ws=self.pws)
            ss_cur, ss_nxt = ss_nxt, ss_cur
        self._pss_final, self._pnt_final = ss_cur, nt

    def _prefill(self, x):
        """Own prefill for prompt x [1, S, H]; sets past, hist[0], logits and the KV cache rows [0, S)."""
        S = x.shape[1]
        b = -(-S // self.pstep) * self.pstep
        self.px[:S].copy_(x[0])
        self.px[S:b].zero_()
        if b not in self.pgraphs:
            st = torch.cuda.Stream(self.device)
            st.wait_stream(self.stream)
            with torch.cuda.stream(st):
                self._prefill_rows(b)
            self.stream.wait_stream(st)
            torch.cuda.synchronize(self.device)
            self.px[:S].copy_(x[0])
            self.px[S:b].zero_()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._prefill_rows(b)
            self.pgraphs[b] = (g, self._pss_final, self._pnt_final)
        g, ss, nt = self.pgraphs[b]
        g.replay()
        row = self.px[S - 1:S]
        _final_norm_kernel[(1,)](row, self.norm_w, self.past, self.hist, self._neg1, self.eps,
                                 HID=self.hidden, num_warps=8)
        self._gemm(PREFILL_CONFIGS["lm"], self.px, self.head, self.logits, ss, nt, None, 0, self.eps,
                   M=1, rmax=self.pmax, ro=S - 1, ws=self.pws)

    @torch.inference_mode()
    def prepare_prefill(self):
        """Capture every prefill bucket up front (writes scratch KV rows only)."""
        for b in range(self.pstep, self.pmax + 1, self.pstep):
            self._prefill(torch.zeros(1, b, self.hidden, device=self.device, dtype=torch.bfloat16))
        torch.cuda.synchronize(self.device)

    def _graph(self, key, pkey):
        gk = (key, pkey)
        if gk not in self.graphs:
            t = time.perf_counter()
            saved = (self.state.clone(), self.seen.clone(), self.codes_out[:4].clone(), self.hist[:4].clone(),
                     self.kc.clone(), self.vc.clone(), self.first.clone(), self.past.clone(), self.tok.clone(), self.logits.clone())
            s = torch.cuda.Stream(self.device)
            s.wait_stream(self.stream)
            with torch.cuda.stream(s), torch.random.fork_rng(devices=[self.device.index]):
                for _ in range(2):
                    self._frame(key, pkey)
            self.stream.wait_stream(s)
            torch.cuda.synchronize(self.device)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._frame(key, pkey)
            torch.cuda.synchronize(self.device)
            (self.state.copy_(saved[0]), self.seen.copy_(saved[1]), self.codes_out[:4].copy_(saved[2]),
             self.hist[:4].copy_(saved[3]), self.kc.copy_(saved[4]), self.vc.copy_(saved[5]),
             self.first.copy_(saved[6]), self.past.copy_(saved[7]), self.tok.copy_(saved[8]),
             self.logits.copy_(saved[9]))
            del saved
            self.graphs[gk] = g
            self.timings.setdefault("capture_s", []).append(time.perf_counter() - t)
        return self.graphs[gk]

    @torch.inference_mode()
    def run(self, kw, on_frame=None):
        """Batch-one generation with talker.generate keyword semantics."""
        t = self.talker
        x = kw["inputs_embeds"]
        S0 = x.shape[1]
        max_new = kw["max_new_tokens"]
        if S0 + max_new + 1 > self.capacity:
            max_new = self.capacity - S0 - 1  # bounded by engine capacity
            if max_new < 1:
                raise ValueError("prompt exceeds engine capacity")
        if self.own_prefill and S0 <= self.pmax:
            t.rope_deltas = torch.zeros(1, 1, device=self.device, dtype=torch.long)  # all-ones mask
            self._prefill(x)
        else:
            out = t(inputs_embeds=x, attention_mask=kw["attention_mask"], use_cache=True,
                    output_hidden_states=False, return_dict=True,
                    trailing_text_hidden=kw["trailing_text_hidden"], tts_pad_embed=kw["tts_pad_embed"])
            for li, layer in enumerate(out.past_key_values.layers):
                self.kc[li, :, :S0].copy_(layer.keys[0])
                self.vc[li, :, :S0].copy_(layer.values[0])
            self.past.copy_(out.past_hidden[:, -1])
            self.hist[0].copy_(out.past_hidden[0, -1])
            self.logits.copy_(out.logits[:, -1])
            del out
        tr = kw["trailing_text_hidden"][0]
        T = min(tr.shape[0], self.capacity)
        self.trail[:T].copy_(tr[:T])
        self.pad.copy_(kw["tts_pad_embed"].reshape(-1))
        self.state.copy_(torch.tensor([0, S0, T, 0], device=self.device))
        self.seen.zero_()
        self.suppress.zero_()
        if kw.get("suppress_tokens"):
            self.suppress[torch.tensor(kw["suppress_tokens"], device=self.device)] = 1
        sample = kw["do_sample"]
        key = (bool(sample), float(kw["temperature"]) if sample else 1.0,
               min(int(kw["top_k"]), self.vocab) if sample else 0,
               float(kw["repetition_penalty"]), int(kw["min_new_tokens"]), int(kw["eos_token_id"]))
        ps = kw["subtalker_dosample"]
        pkey = (bool(ps), float(kw["subtalker_temperature"]) if ps else 1.0,
                min(int(kw["subtalker_top_k"]), self.pvocab) if ps else 0)
        if not ps:
            pkey = (False, 1.0, 1)  # vendored predictor: identical Gumbel offsets + top-1
        g = self._graph(key, pkey)
        # first code from the prefill logits
        if key[0]:
            self.uni[:self.vocab].uniform_()
        self._sample(key, 0)
        self.tok_host.copy_(self.tok, non_blocking=True)
        self.stream.synchronize()
        eos = key[5]
        tokens = [int(self.tok_host[0])]
        n = 0
        codec = self.codec
        if codec is not None:
            codec.reset()
        while tokens[-1] != eos and n + 1 < max_new:
            g.replay()
            if codec is not None:
                codec.launch(n + 1)  # frames 0..n are queued; host launch overlaps the GPU frame
            self.stream.synchronize()
            if on_frame is not None:
                on_frame(self.codes_out[n:n + 1].clone())
            n += 1
            tokens.append(int(self.tok_host[0]))
        self.audio = codec.finish(n) if codec is not None else None
        return tokens, n

    def generate_output(self, kw):
        tokens, n = self.run(kw)
        H = self.hidden
        hist = self.hist[:n + 1].clone()
        codes = self.codes_out[:n].clone()
        hs = [((hist[0].view(1, 1, H),), None)]
        hs += [((hist[i + 1].view(1, 1, H),), codes[i:i + 1]) for i in range(n)]
        seq = torch.tensor([tokens], device=self.device)
        return GenerateDecoderOnlyOutput(sequences=seq, hidden_states=tuple(hs))


class EngineGenerate:
    """Replace talker.generate for supported batch-one requests; others use the original."""

    def __init__(self, talker, **kw):
        if hasattr(talker, "_talker_engine"):
            raise ValueError("already installed")
        self.engine = TalkerEngine(talker, **kw)
        self.talker = talker
        self.original = talker.generate
        talker.generate = self.generate
        talker._talker_engine = self

    def supported(self, kw):
        x, mask = kw.get("inputs_embeds"), kw.get("attention_mask")
        if x is None or x.ndim != 3 or x.shape[0] != 1 or mask is None or not bool((mask == 1).all()):
            return False
        if kw.get("subtalker_top_p", 1.0) != 1.0 or kw.get("top_p", 1.0) != 1.0:
            return False
        if not isinstance(kw.get("eos_token_id"), int):
            return False
        if self.talker.training:
            return False
        return True

    def generate(self, *args, **kw):
        if args or not self.supported(kw):
            self.engine.fallback_calls += 1
            return self.original(*args, **kw)
        self.engine.calls += 1
        return self.engine.generate_output(kw)

    def uninstall(self):
        self.talker.generate = self.original
        del self.talker._talker_engine


def install_talker_engine(talker, **kw):
    return EngineGenerate(talker, **kw)


def _gpu_tag():
    """Stable per-GPU tag for tuned-config filenames (ROCm: gcn arch; CUDA: device-name slug)."""
    import re
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    arch = getattr(props, "gcnArchName", "") or ""
    if arch:
        return arch.split(":")[0]
    return re.sub(r"[^a-z0-9]+", "-", torch.cuda.get_device_name().lower()).strip("-")


def load_tuned(size):
    """Engine kwargs tuned on this GPU type by tune.py. Prefers tuned-<size>-<gpu>.json, then the
    legacy tuned-<size>.json. Returns {} when no configs exist for this GPU (defaults are used)."""
    import json, os
    here = os.path.dirname(os.path.abspath(__file__))
    tag = _gpu_tag()
    t = None
    for fname in (f"tuned-{size}-{tag}.json", f"tuned-{size}.json"):
        path = os.path.join(here, "tuned", fname)
        if os.path.exists(path):
            t = json.load(open(path))
            break
    if t is None:
        return {}
    tup = lambda d: {k: tuple(v) for k, v in d.items()}
    return dict(configs=tup(t["configs"]), pred_configs=tup(t["pred_configs"]),
                attn_splits=t["attn_splits"], attn_bk=t["attn_bk"], attn_warps=t["attn_warps"])


def install_fast_tts(model, *, tuned="auto", codec_fast=True, codec_graphs=True, overlap=False, chunk=48, ctx=96,
                     **engine_kw):
    """Whole-frame graph engine for the talker, faster codec kernels, optional overlapped decode."""
    from codec_overlap import OverlapCodec
    if tuned == "auto":
        size = "1.7B" if model.model.talker.config.hidden_size == 2048 else "0.6B"
        lt = load_tuned(size)
        if engine_kw.get("weight_bits") in (4, 8):
            # Quantized GEMMs use QUANT_CONFIGS; only the tuned attention params apply.
            lt = {k: lt[k] for k in ("attn_splits", "attn_bk", "attn_warps") if k in lt}
        engine_kw = {**lt, **engine_kw}
    h = install_talker_engine(model.model.talker, **engine_kw)
    eng = h.engine
    if eng.own_prefill:
        eng.prepare_prefill()
    st = model.model.speech_tokenizer
    dec = st.model.decoder
    if codec_fast:
        from codec_fast import install_codec_fast
        h.codec_patches = install_codec_fast(dec)
    if codec_graphs:
        from codec_graphs import CodecGraphs
        h.codec_graphs = CodecGraphs(dec)
        h.codec_graphs.prepare_all()
    if overlap:
        eng.codec = OverlapCodec(eng, dec, chunk=chunk, ctx=ctx)
        up = int(dec.total_upsample)
        orig = st.decode
        sr = int(st.model.get_output_sample_rate())

        def decode(encoded):
            a, eng.audio = eng.audio, None
            if (a is not None and isinstance(encoded, list) and len(encoded) == 1
                    and a.numel() == encoded[0]["audio_codes"].shape[0] * up):
                return [a.to(torch.float32).cpu().numpy()], sr
            return orig(encoded)
        st.decode = decode
        h.decode_original = orig
    return h
