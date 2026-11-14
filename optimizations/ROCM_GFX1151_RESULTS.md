# Qwen3-TTS inference optimization on AMD gfx1151 (ROCm/HIP)

This is the ROCm/HIP port of the CUDA optimization work recorded in
[`OPTIMIZATIONS.md`](../OPTIMIZATIONS.md). It runs the **same optimizations** on an AMD Ryzen AI MAX+ 395
(Strix Halo, Radeon 8060S, **gfx1151**, 40 CUs / 20 WGPs) and records where they land on this hardware.

Branch: `gfx1151`. The CUDA-specific surface (programmatic dependent launch, PTX prefetch, cuDNN) is
guarded so the CUDA path is unchanged; on ROCm the same fused kernels run as plain graph-ordered launches.

## TL;DR

Every CUDA-side optimization was ported and measured, then **weight-only INT8 quantization** was added as the
bandwidth lever. Best RTF on gfx1151: **0.6B 0.1607 (4.26× over HF)** and **1.7B 0.2477 (3.46×)** with the
INT8 whole-frame engine — intelligibility preserved (WER 0.53% / 1.06%). Without quantization the best were
0.6B 0.3444 (base stack, ~2×) and 1.7B 0.6875 (BF16 engine, 1.25×). gfx1151 is **bandwidth-bound**
(~226 GB/s vs the H20's ~4 TB/s), so halving the weight bytes with INT8 nearly halves the frame time; the
H20's 40×/29× is still out of reach, but quantization closes much of the gap. The BF16 whole-frame engine —
the CUDA crown jewel — does not win here without quantization, because hipBLASLt streams BF16 weights faster
than the vendored fused Triton chain.

## Hardware and software

- **GPU:** AMD Radeon 8060S (gfx1151, RDNA 3.5), 40 CUs, 2 MB L2, unified LPDDR5X.
- **Measured DRAM bandwidth:** ~226 GB/s read-only, ~202 GB/s copy (read+write). This is the dominant
  constraint: the H20 used for the CUDA work has ~4 TB/s, ~18× more.
- **PyTorch** 2.13.0+rocm10.0.0, **Triton** 3.8.0 (AMD backend), **Transformers** 4.57.3, BF16, batch one.
- Models: `Qwen/Qwen3-TTS-12Hz-{0.6B,1.7B}-CustomVoice`. RTF and suite methodology are identical to
  `OPTIMIZATIONS.md` (2 voices × 4 texts × 3 seeds, mean of per-condition median RTFs, full
  `generate_custom_voice` including codec decode).

## The key difference: bandwidth-bound, not latency-bound

On the H20 the workload was kernel-latency bound, so fusion and graphs were the whole game. On gfx1151
the weight-read floor is much higher. Per generated 80 ms frame the talker reads ~0.88 GB and the
15-step predictor re-reads its ~0.16 GB of weights 15× (~2.36 GB), ~3.2 GB/frame total. At ~226 GB/s
that is a **~14–15 ms/frame floor (RTF ≈ 0.18–0.19)** — no kernel work can beat it without reading
fewer bytes (weight-only quantization). The CUDA engine's 2.55 ms/frame is not reachable here.

## Port notes (what changed for ROCm)

- `vendor/qwen_fused_predictor/kernels.py`: PDL intrinsics (`gdc_wait`/`gdc_launch_dependents`) and the
  PTX L2 prefetch are behind a `PDL` constexpr; `_pdl_supported()` returns False on HIP. The fused GEMM
  chain runs on ROCm with `PDL=False` (plain launches inside HIP graphs). libdevice import is
  backend-aware. The chain's sum-of-squares bookkeeping no longer depends on PDL.
- `scripts/fast/talker_engine.py`: same PDL gating for the talker attention + GEMM chain. HIP graph
  capture, pinned-memory EOS readback, on-device sampler, RoPE reduction and prefill graphs are unchanged.
- `scripts/fast/codec_fast.py`: backend-aware libdevice. Polyphase dilated-conv rewrite + fused SnakeBeta
  and bucketed codec graphs are unchanged (MIOpen replaces cuDNN).
- Tuned configs are GPU-aware: `tuned/tuned-<size>-gfx1151.json` (H20 references kept as `-h20`).
  `tune.py` works unchanged on ROCm (HIP graphs, per-GPU cache key).

## Results

Full per-config comparison (mean of per-condition median RTFs, 24 samples each, all reached EOS):

| Configuration | 0.6B RTF ↓ | 0.6B speedup | 1.7B RTF ↓ | 1.7B speedup |
|---|---:|---:|---:|---:|
| HF `generate` + SDPA (reference) | 0.6845 | 1.0× | 0.8581 | 1.0× |
| Base stack: predictor/talker HIP graphs, fused norms, hipBLASLt fused predictor, talker fusion, frame loop | 0.3444 | 1.99× | 0.8525 | 1.01× |
| Whole-frame engine, un-tuned H200 configs | ~0.70 | ~1.0× | — | — |
| Whole-frame engine, gfx1151-tuned BF16 + fast codec | 0.4094 | 1.67× | 0.6875 | 1.25× |
| **Whole-frame engine, weight-only INT8 + fast codec** | **0.1607** | **4.26×** | **0.2477** | **3.46×** |

**Best configuration: the INT8 whole-frame engine for both sizes.** Without quantization the best flips by
size (0.6B → base stack, 1.7B → BF16 engine); quantization makes the engine the clear winner everywhere.

This is the headline difference from the CUDA work. On the H20 the whole-frame engine was the clear winner
(40×/29× over HF) because the workload was latency-bound and fusion cut kernel count. On gfx1151 the gains
are far smaller (2×/1.25×) and the *best stack flips by size*, because the part is bandwidth-bound:

- The base stack runs its GEMMs through **hipBLASLt (~213 GB/s)**; the whole-frame engine runs the vendored
  **fused Triton GEMM chain (~85 GB/s)**. For the small 0.6B talker, hipBLASLt's faster weight streaming wins
  despite the base stack launching more kernels, so base < engine. For the larger 1.7B talker the engine's
  whole-frame graph + on-device sampling + norm folding recover enough to edge out the base stack.
- Per-frame (tuned engine): 0.6B talker **8.9 ms** + predictor **23.0 ms** ≈ 33 ms; 1.7B talker **30.1 ms** +
  predictor **23.5 ms** ≈ 54 ms. The un-tuned H200 register-panel GEMM configs spill on RDNA (~55 ms/frame
  for 0.6B); per-GPU tuning is essential and selects the streaming GEMM variant.
- The 1.7B base stack barely beats HF because its large talker is already bandwidth-bound under hipBLASLt,
  so graph capture (launch-overhead removal) buys little.

The absolute ceiling here is the weight-read floor (RTF ≈ 0.19, above); neither stack approaches it because
the fused GEMM is below hipBLASLt bandwidth and the predictor re-reads its weights 15× per frame.

## GEMV investigation (why the engine does not reach the bandwidth floor)

The fused chain GEMM reaches only ~85 GB/s cold on gfx1151, while hipBLASLt GEMV reaches ~213 GB/s for the
same shapes. A pipelined FMA GEMV (fine N tiling BN=8, BK=256, `num_stages=4`, `num_warps=2`) matches
hipBLASLt at ~200 GB/s **in isolation**. But integrating it into the fused chain is *slower* than the tuned
streaming kernel: the fine N tiling that saturates bandwidth also multiplies the epilogue's per-tile
sum-of-squares reduction (BN=8 → hidden/8 tiles vs hidden/64 for the streaming kernel), and that reduction
dominates. Measured full-frame: tuned streaming ~32 ms vs best GEMV variant ~39 ms. The GEMV kernel is kept
as a selectable config variant (`("g", BN, BK, STAGES, warps)`) for future work. Reaching the floor needs a
redesigned fusion (decoupled sum-of-squares reduction) or weight-only FP8/INT8 — the latter halves the bytes
and, unlike on the latency-bound H20, would pay off directly here.

## Weight-only INT8 quantization (the bandwidth lever)

Because gfx1151 is bandwidth-bound, halving the weight bytes is the direct lever (the CUDA work rejected FP8
on the latency-bound H20, where quant/dequant overhead dominated). Implemented as **weight-only INT8** with
per-output-row symmetric scales and **fused in-register dequant** — activations stay BF16, no separate
dequant kernel or materialized BF16 weights.

- A new M=1 GEMV (`_cp_gemv_w8_kernel`, config variant `("q", BN, BK, STAGES, warps)`) loads INT8 weights,
  converts to FP32 in registers, accumulates, and applies the per-row scale before the chain epilogue. With
  `BN=8, BK=256, num_stages=4, num_warps=8` it runs **2.02× the BF16 GEMV** (20.8 vs 42.1 µs for an 8 MB
  weight; 404 GB/s BF16-equivalent). `num_warps=8` is essential to hide the int8→fp32 conversion latency.
- The kernel loops over M rows so it also covers the predictor's M=2 first step (re-reads weights once more,
  1 of 15 steps). Gate/up is re-packed for BN=8 tiles; sum-of-squares buffers use hidden/8 tiles.
- Enabled with `weight_bits=8` on `TalkerEngine` / `install_fast_tts`. Quantized mode uses the HF prefill
  (the engine prefill is slower than HF eager on gfx1151 anyway). The BF16 path is unchanged.
- **FP8 E4M3 was tried and rejected**: the fp8→fp32 conversion is slower than BF16 on RDNA (0.84×) and less
  accurate (2.7% vs INT8 0.8% per-weight rel-err).

Per-frame (0.6B): talker **3.35 ms** (BF16 8.9), predictor **8.8 ms** (BF16 23.0), frame **12.16 ms** (BF16 33).
The predictor benefits most because it re-reads its weights 15× per frame. The INT8 frame even beats the BF16
weight-read floor, since the floor itself halves with the bytes.

### Improving quantization quality (AutoRound / SmoothQuant / SpinQuant / GPTQ)

Assessed and prototyped for the weight-only W8A16 setup (INT8 weights, BF16 activations):

| Method | Fit | Measured outcome |
|---|---|---|
| **Per-group scales** (`group_size`) | reliable, no calibration | rel-err 3.16% → **2.53%** (G128) / 2.65% (G256); RTF ~unchanged (G256 0.1624, WER 0.89%). Grouped-dequant GEMV applies the scale per K-chunk. |
| **GPTQ / AutoRound** (`quant_calib.py`, `gptq_weights`) | weight-only, kernel unchanged | ~1.5× lower per-GEMM error *in-sample*, but **overfits** with limited calibration: decode runs via graph replay, so few real activation rows are captured (mostly zero-padded prefill). Full-engine error was worse than RTN without proper eager calibration. Needs a dedicated calibration pass to generalize. |
| **SmoothQuant** | W8A8-oriented | migrates activation outliers into weights so *activations* can be quantized; we keep activations BF16, so its core benefit does not apply (reduces to per-channel scaling already done). |
| **SpinQuant** | learned rotations | needs an online Hadamard rotation of activations (changes the GEMM structure); overkill for INT8, valuable at INT4/INT2. |

**Conclusion:** at INT8, RTN per-channel is already near-optimal in practice — WER is 0.53%/1.06%, so the
modest numeric-error reductions above do not change intelligibility. These methods matter at **INT4**, where
RTN error is ~18% and GPTQ/AutoRound-style rounding is needed to keep quality usable. Per-group scales and
GPTQ are kept as optional modes (`group_size`, `gptq_weights`); the default remains per-channel RTN (fastest).

## Quality checks

Same principles as the CUDA work: compare against an FP32 reference and sampled behavior, not greedy token
parity. All checks passed; the port introduces no quality regression.

| Check | Method | Result |
|---|---|---|
| Talker numerics (0.6B) | engine incremental hidden states vs FP32, teacher-forced | rel-err mean **0.0128** (p95 0.0159), KL 4.59e-4 |
| Talker numerics (1.7B) | same | engine rel-err **0.0120** (p95 0.0153), KL 3.98e-4 — **no worse than HF BF16** (0.0126, KL 4.87e-4) |
| Prefill (both sizes) | engine prefill vs HF on all 4 prompts | argmax identical on every prompt; top-5 overlap **5/5**; logits rel ~1–1.7% |
| Codec (0.6B) | fast kernels vs FP32 decode, 220 frames | fast **36.6 dB** vs stock 35.7 dB (more accurate); fast vs stock 39.7 dB |
| Intelligibility | Whisper large-v3-turbo WER, 24-sample suites | BF16 fast **0.35%** (0.6B), **0.53%** (1.7B); **INT8 0.53% (0.6B), 1.06% (1.7B)**; HF 0.89% (1.7B). The 0.6B HF suite pooled 41% only because one sample hit the model's documented laugh-loop ("Hey, hey, hey…", WER 28.6); its other 23 files are ≤6%. |
| INT8 numerics (0.6B) | engine hidden states vs FP32 | rel-err **0.0316** (p95 0.0464), KL 1.45e-3 (HF BF16 0.0140, KL 4.06e-4) |
| INT8 numerics (1.7B) | same | rel-err **0.0405** (p95 0.0493), KL 1.37e-3 (HF BF16 0.0130, KL 3.85e-4) |

INT8 raises the numeric error vs FP32 (3–4% vs BF16's ~1.3%) as expected, but **WER stays near zero** — the
sampling distribution and intelligibility are preserved. This is the doc's "validate distributions, not
tokens" principle: the quantized engine is accepted on WER + sampled behavior, not bit parity.

Notes specific to gfx1151:
- **Engine prefill is slower than HF eager here** (0.6B ~13 vs 12 ms; 1.7B ~50 vs 35 ms), unlike the H20
  where it was a large win. The prefill stream-GEMM configs and per-head/row attention kernel are not tuned
  for the prefill M shapes on RDNA. It is a one-time fixed cost dwarfed by the frame loop, so overall RTF
  still improves; `own_prefill=False` (HF prefill) is available if short-request latency matters.
- **Codec win is smaller** (1.26× vs the H20's 1.53×): MIOpen's composable-kernel grouped conv is already
  reasonable, so the polyphase rewrite removes less overhead than it did from cuDNN's legacy dilated kernel.
  The fused SnakeBeta pass and bucketed graphs still help and are more accurate than stock BF16.

## Remaining opportunities (gfx1151-specific)

- **Done: weight-only INT8** (above) — 4.26×/3.46× over HF, the best config. FP8 tried, rejected (slow + less
  accurate on RDNA).
- **Finer quantization granularity** (per-group scales, e.g. group 128) could cut the 3–4% numeric error
  toward BF16 at a small bandwidth cost; WER is already fine so this is optional.
- **INT4 weight-only** would halve the bytes again (floor ~0.08 RTF) if accuracy holds; untested.
- **hipBLASLt GEMMs inside the BF16 engine** (hybrid) would make the BF16 engine win without quantization;
  superseded by INT8 for this bandwidth-bound part.
- **Predictor weight reuse**: the 15 residual steps re-read the same weights 15×/frame; a persistent
  megakernel could cut this further (the 2 MB L2 cannot hold them).

## Reproduce

```bash
# suites (w8 = INT8 quantized engine, the best config)
python optimizations/scripts/fast/suite.py --size 0.6B --stack hf   --label rocm-hf
python optimizations/scripts/fast/suite.py --size 0.6B --stack base --label rocm-base
python optimizations/scripts/fast/suite.py --size 0.6B --stack fast --label rocm-w8 \
    --engine-kw '{"weight_bits": 8}'
# in Python: install_fast_tts(model, weight_bits=8)  # tuned attention auto-loaded
# retune BF16 attention/GEMM configs on this GPU
python optimizations/scripts/fast/tune.py 0.6B
# quality
python optimizations/scripts/fast/test_numerics.py 1.7B
python optimizations/scripts/fast/test_prefill.py 0.6B
python optimizations/scripts/fast/test_codec_fast.py
python optimizations/scripts/fast/asr_wer.py optimizations/results/fast/0.6b-rocm-w8.manifest.json
```