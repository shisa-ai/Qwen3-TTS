# Qwen3-TTS inference optimization on AMD gfx1151 (ROCm/HIP)

This is the ROCm/HIP port of the CUDA optimization work recorded in
[`OPTIMIZATIONS.md`](../OPTIMIZATIONS.md). It runs the **same optimizations** on an AMD Ryzen AI MAX+ 395
(Strix Halo, Radeon 8060S, **gfx1151**, 40 CUs / 20 WGPs) and records where they land on this hardware.

Branch: `gfx1151`. The CUDA-specific surface (programmatic dependent launch, PTX prefetch, cuDNN) is
guarded so the CUDA path is unchanged; on ROCm the same fused kernels run as plain graph-ordered launches.

## TL;DR

Every CUDA-side optimization was ported and measured. Best RTF on gfx1151: **0.6B 0.3444 (~2× over HF)** with
the base graph stack, **1.7B 0.6875 (1.25×)** with the whole-frame engine. Quality is preserved (numerics no
worse than HF BF16 vs FP32; WER 0.35%/0.53%). The gains are much smaller than the H20's 40×/29× because
gfx1151 is **bandwidth-bound** (~226 GB/s vs ~4 TB/s): the weight-read floor alone is RTF ≈ 0.19, and the
vendored fused GEMM streams weights slower than hipBLASLt. The whole-frame engine — the CUDA crown jewel —
does not win here for 0.6B; faster GEMMs (hipBLASLt) beat fewer kernels.

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
| Base stack: predictor/talker HIP graphs, fused norms, hipBLASLt fused predictor, talker fusion, frame loop | **0.3444** | **1.99×** | 0.8525 | 1.01× |
| Whole-frame engine, un-tuned H200 configs | ~0.70 | ~1.0× | — | — |
| Whole-frame engine, gfx1151-tuned configs + fast codec | 0.4094 | 1.67× | **0.6875** | **1.25×** |

**Best configuration per size: 0.6B → base stack (0.3444, ~2×); 1.7B → whole-frame engine (0.6875, 1.25×).**

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

## Quality checks

Same principles as the CUDA work: compare against an FP32 reference and sampled behavior, not greedy token
parity. All checks passed; the port introduces no quality regression.

| Check | Method | Result |
|---|---|---|
| Talker numerics (0.6B) | engine incremental hidden states vs FP32, teacher-forced | rel-err mean **0.0128** (p95 0.0159), KL 4.59e-4 |
| Talker numerics (1.7B) | same | engine rel-err **0.0120** (p95 0.0153), KL 3.98e-4 — **no worse than HF BF16** (0.0126, KL 4.87e-4) |
| Prefill (both sizes) | engine prefill vs HF on all 4 prompts | argmax identical on every prompt; top-5 overlap **5/5**; logits rel ~1–1.7% |
| Codec (0.6B) | fast kernels vs FP32 decode, 220 frames | fast **36.6 dB** vs stock 35.7 dB (more accurate); fast vs stock 39.7 dB |
| Intelligibility | Whisper large-v3-turbo WER, 24-sample suites | fast **0.35%** (0.6B), **0.53%** (1.7B); HF 0.89% (1.7B). The 0.6B HF suite pooled 41% only because one sample hit the model's documented laugh-loop ("Hey, hey, hey…", WER 28.6); its other 23 files are ≤6%. Fast max per-file WER 5.6% ("9"/"nine"). |

Notes specific to gfx1151:
- **Engine prefill is slower than HF eager here** (0.6B ~13 vs 12 ms; 1.7B ~50 vs 35 ms), unlike the H20
  where it was a large win. The prefill stream-GEMM configs and per-head/row attention kernel are not tuned
  for the prefill M shapes on RDNA. It is a one-time fixed cost dwarfed by the frame loop, so overall RTF
  still improves; `own_prefill=False` (HF prefill) is available if short-request latency matters.
- **Codec win is smaller** (1.26× vs the H20's 1.53×): MIOpen's composable-kernel grouped conv is already
  reasonable, so the polyphase rewrite removes less overhead than it did from cuDNN's legacy dilated kernel.
  The fused SnakeBeta pass and bucketed graphs still help and are more accurate than stock BF16.

## Remaining opportunities (gfx1151-specific)

- **Weight-only FP8/INT8** is the biggest lever: the part is bandwidth-bound, so halving weight bytes nearly
  halves the floor. On the latency-bound H20 this was rejected (quant/dequant overhead dominated); here it
  would pay off directly. Untested.
- **hipBLASLt GEMMs inside the whole-frame engine** (hybrid): keep the engine's graph + fused attention +
  on-device sampling, but replace the slow fused Triton GEMM chain with hipBLASLt. Would likely make the
  engine win for both sizes. Requires decoupling the norm-fold/residual/sum-of-squares epilogues.
- **Decoupled sum-of-squares reduction** so the fast BN=8 GEMV can be used in the chain without the epilogue
  reduction overhead that currently negates it (see above).
- **Predictor weight reuse**: the 15 residual steps re-read the same ~0.16 GB 15×/frame. A persistent
  megakernel or larger-cache strategy could cut this; the 2 MB L2 cannot hold it.

## Reproduce

```bash
# suites (base = best 0.6B, fast = best 1.7B)
python optimizations/scripts/fast/suite.py --size 0.6B --stack hf   --label rocm-hf
python optimizations/scripts/fast/suite.py --size 0.6B --stack base --label rocm-base
python optimizations/scripts/fast/suite.py --size 1.7B --stack fast --label rocm-fast
# retune on this GPU (writes tuned-<size>-gfx1151.json)
python optimizations/scripts/fast/tune.py 0.6B
# quality
python optimizations/scripts/fast/test_numerics.py 1.7B
python optimizations/scripts/fast/test_prefill.py 0.6B
python optimizations/scripts/fast/test_codec_fast.py
python optimizations/scripts/fast/asr_wer.py optimizations/results/fast/0.6b-rocm-fast.manifest.json
```