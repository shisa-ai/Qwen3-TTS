"""Predictor PDL chain with attention fused into the o-projection GEMM (4 kernels per layer).

The o-projection is split along K, and a K chunk of KC = NCH * BK columns is exactly
KC / head_dim heads. Each CTA (n tile, k chunk) computes attention for just its heads
(<= 16 keys) into an att scratch row, then multiplies its weight panel. CTAs of one k chunk
compute identical values. Split 0 of n tile 0 writes the new K/V cache rows.
"""
import torch
import triton
import triton.language as tl

from qwen_fused_predictor import _k

_bf = _k._bf
_head_norm_rope = _k._head_norm_rope


@triton.jit
def _attn_o_kernel(
    qkv_ptr, kc_ptr, vc_ptr, qw_ptr, kw_ptr, cos_ptr, sin_ptr, att_ptr,
    w_ptr, ws_ptr, cnt_ptr, x_ptr, ss_out_ptr, first, eps, scale,
    NQ: tl.constexpr, H: tl.constexpr, KVH: tl.constexpr, HD: tl.constexpr, MAXP: tl.constexpr,
    BKEY: tl.constexpr, N: tl.constexpr, HIDX: tl.constexpr, RMAX: tl.constexpr,
    BN: tl.constexpr, BK: tl.constexpr, NCH: tl.constexpr, BM: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    K: tl.constexpr = H * HD
    KC: tl.constexpr = BK * NCH
    SK: tl.constexpr = K // KC
    NH: tl.constexpr = KC // HD
    HALF: tl.constexpr = HD // 2
    W: tl.constexpr = (H + 2 * KVH) * HD
    n = pid_n * BN + tl.arange(0, BN)
    kb = pid_k * KC
    w = tl.load(w_ptr + n[:, None] * K + (kb + tl.arange(0, KC))[None, :])
    d = tl.arange(0, HALF)
    j = tl.arange(0, BKEY)
    qw1 = tl.load(qw_ptr + d).to(tl.float32)
    qw2 = tl.load(qw_ptr + HALF + d).to(tl.float32)
    kw1 = tl.load(kw_ptr + d).to(tl.float32)
    kw2 = tl.load(kw_ptr + HALF + d).to(tl.float32)
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    dtc = kc_ptr.dtype.element_ty
    for hh in tl.static_range(NH):
        h = pid_k * NH + hh
        kvh = h // (H // KVH)
        cache = (kvh * MAXP).to(tl.int64) * HD
        old = (j < first)[:, None]
        kp = kc_ptr + cache + j[:, None] * HD + d[None, :]
        vp = vc_ptr + cache + j[:, None] * HD + d[None, :]
        k1 = tl.load(kp, mask=old, other=0.0).to(tl.float32)
        k2 = tl.load(kp + HALF, mask=old, other=0.0).to(tl.float32)
        v1 = tl.load(vp, mask=old, other=0.0).to(tl.float32)
        v2 = tl.load(vp + HALF, mask=old, other=0.0).to(tl.float32)
        for qn in tl.static_range(NQ):
            on = first + qn
            rb = qkv_ptr + qn * W
            kx1 = tl.load(rb + (H + kvh) * HD + d).to(tl.float32)
            kx2 = tl.load(rb + (H + kvh) * HD + HALF + d).to(tl.float32)
            vn1 = tl.load(rb + (H + KVH + kvh) * HD + d).to(tl.float32)
            vn2 = tl.load(rb + (H + KVH + kvh) * HD + HALF + d).to(tl.float32)
            c1 = tl.load(cos_ptr + on * HD + d).to(tl.float32)
            c2 = tl.load(cos_ptr + on * HD + HALF + d).to(tl.float32)
            s1 = tl.load(sin_ptr + on * HD + d).to(tl.float32)
            s2 = tl.load(sin_ptr + on * HD + HALF + d).to(tl.float32)
            kn1, kn2 = _head_norm_rope(kx1, kx2, kw1, kw2, c1, c2, s1, s2, eps, HD)
            sel = (j == on)[:, None]
            k1 = tl.where(sel, kn1[None, :], k1)
            k2 = tl.where(sel, kn2[None, :], k2)
            v1 = tl.where(sel, vn1[None, :], v1)
            v2 = tl.where(sel, vn2[None, :], v2)
            if pid_n == 0:
                if h % (H // KVH) == 0:
                    tl.store(kc_ptr + cache + on * HD + d, kn1.to(dtc))
                    tl.store(kc_ptr + cache + on * HD + HALF + d, kn2.to(dtc))
                    tl.store(vc_ptr + cache + on * HD + d, vn1.to(dtc))
                    tl.store(vc_ptr + cache + on * HD + HALF + d, vn2.to(dtc))
        for qi in tl.static_range(NQ):
            p = first + qi
            qb = qkv_ptr + qi * W + h * HD
            qx1 = tl.load(qb + d).to(tl.float32)
            qx2 = tl.load(qb + HALF + d).to(tl.float32)
            c1 = tl.load(cos_ptr + p * HD + d).to(tl.float32)
            c2 = tl.load(cos_ptr + p * HD + HALF + d).to(tl.float32)
            s1 = tl.load(sin_ptr + p * HD + d).to(tl.float32)
            s2 = tl.load(sin_ptr + p * HD + HALF + d).to(tl.float32)
            q1, q2 = _head_norm_rope(qx1, qx2, qw1, qw2, c1, c2, s1, s2, eps, HD)
            km = j <= p
            s = (tl.sum(k1 * q1[None, :], axis=1) + tl.sum(k2 * q2[None, :], axis=1)) * scale
            s = tl.where(km, s, -float("inf"))
            e = tl.exp(s - tl.max(s, axis=0))
            e = tl.where(km, e, 0.0)
            pr = e / tl.sum(e, axis=0)
            ob = att_ptr + qi * K + h * HD
            dto = att_ptr.dtype.element_ty
            tl.store(ob + d, tl.sum(v1 * pr[:, None], axis=0).to(dto))
            tl.store(ob + HALF + d, tl.sum(v2 * pr[:, None], axis=0).to(dto))
    tl.debug_barrier()
    m = tl.arange(0, BM)
    rmask = m < NQ
    a = tl.load(att_ptr + m[:, None] * K + (kb + tl.arange(0, KC))[None, :], mask=rmask[:, None], other=0.0)
    acc = tl.dot(w, tl.trans(a), tl.zeros([BN, BM], dtype=tl.float32))
    if SK == 1:
        _k._cp_gemm_epilogue(acc, m, rmask, m, n, pid_n, x_ptr, None, ss_out_ptr, 0.0,
                             N, HIDX, RMAX, BN, 2, 0)
    else:
        tl.store(ws_ptr + pid_k * RMAX * N + m[None, :] * N + n[:, None], acc, mask=rmask[None, :])
        _k._cp_gemm_reduce(ws_ptr, cnt_ptr, pid_n, 0, 1, NQ, 1, 0, n, x_ptr, None, ss_out_ptr, 0.0,
                           N, HIDX, RMAX, BN, BM, SK, 2, 0)


class FusedAttnOPredictor(_k.FusedCodePredictor):
    """Vendored predictor with attention+o fused. Only the batch-one PDL chain is changed."""

    def _call_pdl(self, cfg, B, layer0_code, inp, inv_temperature, top_k, sample_uniforms):
        if B != 1 or cfg["o"][0] == "s":
            return super()._call_pdl(cfg, B, layer0_code, inp, inv_temperature, top_k, sample_uniforms)
        x = self.pdl_x
        if self.proj_wt is not None:
            torch.addmm(self.proj_b, inp, self.proj_wt, out=x[: 2 * B])
        else:
            x[: 2 * B].copy_(inp)
        ss_cur, ss_nxt = self.pdl_ss
        torch.sum(x[: 2 * B].float().square(), dim=1, out=ss_cur[0, : 2 * B])
        nt = 1
        codes = torch.empty(B, self.groups, device=x.device, dtype=torch.int64)
        codes[:, 0] = layer0_code.reshape(B)
        nq, first = 2, 0
        BN, BK, NCH, BM, warps = cfg["o"][:5]
        for step in range(1, self.groups):
            rows = B * nq
            for L, Wt in zip(self.layers, self.pdl_w):
                self._gemm(cfg["qkv"], x, 1, 0, Wt["qkv"], self.pdl_qkv, ss_cur, nt, None, rows, 0, L["eps1"])
                N = Wt["o"].shape[0]
                _attn_o_kernel[(N // BN, self.heads * self.head_dim // (BK * NCH))](
                    self.pdl_qkv, self.cache[L["index"], 0], self.cache[L["index"], 1], L["qn"], L["kn"],
                    self.cos, self.sin, self.pdl_att, Wt["o"], self.pdl_ws, self.pdl_cnt, x, ss_nxt,
                    first, L["head_eps"], self.scale,
                    NQ=nq, H=self.heads, KVH=self.kv_heads, HD=self.head_dim, MAXP=self.max_pos,
                    BKEY=self._attn_keys, N=N, HIDX=self.hidden, RMAX=self._rows, BN=BN, BK=BK, NCH=NCH, BM=BM,
                    num_warps=warps, launch_pdl=True)
                nt = self.hidden // BN
                ss_cur, ss_nxt = ss_nxt, ss_cur
                self._gemm(cfg["gu"], x, 1, 0, Wt["gu"], self.pdl_act, ss_cur, nt, None, rows, 1, L["eps2"])
                nt = self._gemm(cfg["down"], self.pdl_act, 1, 0, Wt["down"], x, None, 0, ss_nxt, rows, 2)
                ss_cur, ss_nxt = ss_nxt, ss_cur
            self._gemm(cfg["lm"], x, nq, nq - 1, self.pdl_lm[step - 1], self.pdl_logits, ss_cur, nt, None, B, 0,
                       self.eps)
            has_next = step < self.groups - 1
            us = sample_uniforms[:, step - 1]
            _k._cp_sample_kernel[(B,)](
                self.pdl_logits, us, us.stride(0), codes, step, self.groups,
                self.tables[step - 1] if has_next else self._dummy, x, inv_temperature,
                V=self.vocab, TOPK=top_k, HID=self.hidden, HAS_NEXT=has_next, ss_ptr=ss_nxt, PDL=True,
                num_warps=4, launch_pdl=True)
            ss_cur, ss_nxt = ss_nxt, ss_cur
            nt = 1
            first = step + 1
            nq = 1
        return codes
