# Qwen3-TTS inference optimization log

This is the complete record of the Qwen3-TTS CustomVoice inference optimization work done in this fork (2026-11-14 → 15): what was changed, how much each change bought, how quality was checked, and what did not work. It is written to be reused when optimizing other autoregressive TTS models.

Detailed per-phase reports live in [`optimizations/`](optimizations/):
- [OPTIMIZATION_RESULTS.md](optimizations/OPTIMIZATION_RESULTS.md): phase 1, graphs and fused norms.
- [GAP_CLOSING_RESULTS.md](optimizations/GAP_CLOSING_RESULTS.md): phase 2, fused kernels and frame loop.
- [FAST_ENGINE_RESULTS.md](optimizations/FAST_ENGINE_RESULTS.md): phase 3, the whole-frame graph engine.

The current runtime is [`optimizations/scripts/fast/`](optimizations/scripts/fast/).

## TL;DR

| Configuration (cumulative) | 0.6B RTF ↓ | 1.7B RTF ↓ | 0.6B speedup | 1.7B speedup |
|---|---:|---:|---:|---:|
| Original package: HF `generate` + Flash Attention 2 | 1.4246 | 1.3154 | 1.0× | 1.0× |
| SDPA attention | 1.0548 | 1.0211 | 1.35× | 1.29× |
| + predictor and talker CUDA graphs | 0.2240 | 0.2230 | 6.4× | 5.9× |
| + fused predictor RMSNorm, drop unused predictor hidden states | 0.1842 | 0.1945 | 7.7× | 6.8× |
| + vendored fused predictor kernels, talker norm/QKV packing, config cache | 0.1027 | 0.1086 | 13.9× | 12.1× |
| + explicit Python frame loop | 0.0891 | 0.1016 | 16.0× | 12.9× |
| + **whole-frame CUDA graph engine** (talker PDL GEMM chain, on-device sampling) | 0.0568 | 0.0711 | 25.1× | 18.5× |
| + GEMM/attention configs tuned for the H20, faster codec kernels | 0.0457 | 0.0598 | 31.2× | 22.0× |
| + **engine prefill graphs, bucketed codec graphs (current)** | **0.0351** | **0.0447** | **40.6×** | **29.4×** |
| *Reference: vLLM-Omni 0.32.0 deployment (historical)* | *0.0753* | *0.0844* | *18.9×* | *15.6×* |

The current engine generates about **28× (0.6B) / 22× (1.7B) faster than real time**, single stream, including waveform decoding. Quality checks (FP32-reference numerics, Whisper WER, and sampled-behavior rates against unmodified HF) found no regression.

## Setup and measurement method

- **Models:** `Qwen/Qwen3-TTS-12Hz-{0.6B,1.7B}-CustomVoice`, BF16, batch one.
- **Hardware and software:** NVIDIA H20 (78 SMs, ~4 TB/s HBM, comparatively low tensor FLOPs), PyTorch 2.15.0+cu132, Triton 3.9.0, Transformers 4.57.3, flash-attn 2.8.3.
- **RTF** = wall time of a full `generate_custom_voice` call, including codec decoding, divided by output audio duration. Lower is better.
- **Suite:** 2 voices (Ryan, Aiden) × 4 texts (short, medium, long, numbers) × 3 seeds after one warmup per condition. The headline is the **mean of per-condition median RTFs**. Sampled output lengths differ between runs, so these are end-to-end throughput numbers, not identical-token speedups.
- **Model structure, for orientation:** each 80 ms codec frame needs one talker step (28-layer transformer) and 15 sequential steps of the code predictor (5-layer transformer, one step per residual codebook). The codec decoder then turns 16 codebooks × frames into 24 kHz audio.

### The key observation

At batch one, almost everything here is **kernel-latency bound, not bandwidth bound**. The 0.6B and 1.7B talkers took nearly the same time per step (3.0 vs 3.5 ms) even though their weights differ 3×. The weight-read floor per step was only about 0.2 / 0.7 ms. Fewer, fused, dependency-chained kernels were therefore the main lever throughout, not FLOP reduction.

## Phase 1: graphs and fused normalization

| Change | 0.6B | 1.7B | Notes |
|---|---:|---:|---|
| Flash Attention 2 → SDPA | 1.425 → 1.055 | 1.315 → 1.021 | FA2 is slower for these tiny decode shapes |
| + predictor CUDA graph only | 0.397 | 0.394 | Captures the fixed 15-step codebook loop with a static cache |
| + talker CUDA graph only | 0.834 | 0.884 | Captures only the transformer decode step |
| + both graphs | **0.224** | **0.223** | 4.7× / 4.6× over SDPA |
| Fused Triton RMSNorm in the predictor (fixed-work replay) | 11.86 → 8.58 ms | 12.01 → 8.70 ms | FP32 accumulation, upstream rounding preserved |
| Stop computing unused predictor hidden states | 8.46 → 8.26 ms | 8.57 → 8.32 ms | Fingerprint-verified internal call change |

Batching (phase 1, predictor graph only, no talker graph) scaled to 35 / 38 audio-seconds per wall-second at batch 32. All later work is batch one.

## Phase 2: fused kernels and an explicit frame loop

| Change | 0.6B | 1.7B | Notes |
|---|---:|---:|---|
| Previous accepted stack | 0.1842 | 0.1946 | |
| + vendored fused residual predictor (Apache-2.0, PDL GEMM chain), talker fused RMSNorm and packed QKV/gate-up, cached immutable GenerationConfig defaults | 0.1027 | 0.1086 | Config cache alone saved ~52 µs CPU per call |
| + explicit Python frame loop (separate graphs, synchronous EOS) | 0.0891 | 0.1016 | Fixed-work gain only 1.006× / 1.024× |
| Exact-shape codec graphs, direct frame streaming | — | — | Streaming first audio **56–59 ms** (4-frame start, then 25-frame chunks, 588-frame context) |

**Lesson from this phase:** the frame loop cut CUDA syncs from 446 to 228, but kernel counts barely moved (127k → 125k for 72 frames). Removing Python checks alone is not enough; the kernels themselves had to go.

## Phase 3: the whole-frame CUDA graph engine

Baseline profile before phase 3, per frame: talker graph 3.02 / 3.48 ms, predictor graph 2.40 / 2.22 ms, about 1 ms of eager glue between them, **6.38 / 6.64 ms in total, ~1,700 kernels**.

### 3a. One CUDA graph per frame

| | 0.6B | 1.7B |
|---|---:|---:|
| Frame time, previous stack | 6.38 ms | 6.64 ms |
| Frame time, engine v1 (untuned) | **3.30 ms** | **4.14 ms** |
| Suite RTF | 0.0926 → **0.0568** | 0.1002 → **0.0711** |

What runs inside each replay, all on the GPU:
1. Uniforms (one RNG kernel).
2. The vendored fused predictor.
3. A Triton kernel summing the 16 codebook embeddings plus the text row.
4. 28 talker layers.
5. Final norm and codec head.
6. The talker sampler, which picks the next frame's first code.

The host reads one int64 per frame from pinned memory to test for EOS, and the kernel count drops to ~560 per frame.

- **Talker as a PDL GEMM chain.** It reuses the vendored predictor's programmatic-dependent-launch GEMMs, which fetch weights before waiting on the previous kernel. Each layer is 5 kernels instead of ~25:
  1. qkv, with the RMSNorm weight folded into the weights and the row scale applied in the epilogue
  2. split-KV decode attention, with qk-norm, RoPE, the cache write and the final merge fused into one kernel
  3. o-projection, with an in-place residual add and the sum of squares for the next norm
  4. gate/up with SiLU
  5. down projection with residual add
- **RoPE simplification.** Text-only multimodal RoPE has identical positions on all three axes, so it reduces exactly to plain RoPE. This is asserted at install.
- **On-device talker sampler** in one Triton kernel. It applies repetition penalty, min-new-tokens and suppress tokens, then temperature, then exact top-k (bisection on FP32 order keys, ties kept), then Gumbel-max sampling. The distribution is identical to HF multinomial sampling, but the random stream differs.

### 3b. Tuning for the GPU actually used

The vendored GEMM configs were tuned on an H200 (132 SMs). `tune.py` first times every candidate config per GEMM shape in a PDL chain over 24 distinct weight copies, so nothing is served from L2. It then runs coordinate descent on the top candidates, scoring each by the real captured talker or predictor graph.

| Graph | 0.6B before → after | 1.7B before → after |
|---|---:|---:|
| Talker transformer (incl. attention splits/block/warps) | 1.073 → **0.795 ms** | 1.918 → **1.494 ms** |
| Predictor (15 steps) | 2.201 → **1.731 ms** | 2.197 → **1.736 ms** |

How to run the tuner on another GPU: see [README → Tuning on a new GPU](README.md#tuning-on-a-new-gpu).

Even after tuning, single-row GEMMs on the 1.7B shapes reach only ~2–2.6 TB/s of the H20's ~4 TB/s.

### 3c. Codec decoder kernels

Profiling one 256-frame decode (53 ms) showed **43% of the time in four calls** to cuDNN's legacy `implicit_convolve_sgemm`. Those are the dilated (d=3, 9) k7 convolutions at 192 and 384 channels, which took 5.1–6.5 ms each against 1.2–1.3 ms for their undilated siblings. `cudnn.benchmark` does not help.

| Change | Decode, 233 frames | SNR vs FP32 decode |
|---|---:|---:|
| Stock decoder (BF16) | 51.6 ms | 33.8 dB |
| Dilated conv → undilated conv over the d polyphase subsequences, plus SnakeBeta as one FP32 Triton pass | **33.8 ms** | **35.0 dB** (more accurate) |

The polyphase rewrite uses `out[q·d + r] = Σ_j w_j · x[(q + j)·d + r]`, so the d phases become a batch of d sequences through an undilated conv. It is the same arithmetic on a fast kernel.

### 3d. Fixed per-request costs

Timeline of a short request (0.6B, 53 frames):

| Stage | Before | After |
|---|---:|---:|
| HF prefill | 18.2 ms | — |
| Engine prefill (graph per 16-token bucket) | — | **~1.7–2.8 ms** (1.7B: 3.6–5.7 ms) |
| Frame loop | 137 ms | 136 ms |
| Codec decode + numpy | 20.4 ms | **10.5 ms** (bucketed graphs) |
| **Total** | **178 ms** | **150 ms** |

- **Engine prefill.** It uses the same PDL GEMM chain with M = prompt length, plus two small Triton kernels: per-position k-norm, RoPE and cache write, then causal attention per head and row. The prompt is right-padded to a 16-token bucket and replayed as a graph. Padding the end of a causal prompt cannot change earlier positions, and decode overwrites the padded KV slots before reading them.
- **Codec graphs.** The decoder is causal (left-padded convolutions, 72-frame sliding-window attention), so right-padding codes to a 32-frame bucket and replaying that bucket's graph is exact. Upstream's 300-frame / 25-context chunking is kept. All buckets share one memory pool.

Suite result after 3b–3d: **0.0351 (0.6B) / 0.0447 (1.7B)**, every request reaching EOS. The final per-frame times are about 2.55 / 3.25 ms.

## Quality checks

Two principles shaped these checks:
- **Don't use greedy token parity as the test.** Any rounding change flips near-ties in the predictor within a few frames. This happened here, and it is expected.
- **Compare against an FP32 reference and against sampled-behavior statistics**, not against HF BF16 bit patterns.

| Check | Method | Result |
|---|---|---|
| Talker numerics | Teacher-forced on the engine's own codes; hidden states and codec-head distribution (T = 0.9) vs the same model in FP32 (`test_numerics.py`, 1.7B) | Engine rel-err **1.20%** (p95 1.47%), KL 4.19e-4; HF BF16 rel-err 1.26% (p95 1.59%), KL 4.22e-4. **No worse than HF BF16.** |
| Teacher-forced logits (engine v1, 0.6B) | HF parallel forward over the engine's codes | argmax agreement 100%, cosine ≥ 0.9996 |
| Prefill | Engine prefill vs HF on all four prompts (`test_prefill.py`) | Same top token on every prompt; top-5 overlap 4–5 of 5; ~1% logit relative error |
| Codec | Fast kernels and bucket graphs vs FP32 decode, and vs eager on the same shape | 35.0 dB vs stock 33.8 dB; graph = eager bit-exact; padding noise floor 39–50 dB |
| Intelligibility | Whisper large-v3-turbo WER on 24-sample suites | **0.0%** (0.6B), **1.1%** (1.7B); remaining errors are "9"/"nine" normalization |
| Sampled behavior vs unmodified HF | 120 samples (2 voices × 2 texts × 30 seeds), then 100 seeds of the worst condition (Ryan, medium) | Laugh openings: **HF 8/100, engine 9/100**; pooled WER 1.914 vs 1.915. The behavior is the model's own. |
| EOS / length | All suites | All requests ended with EOS; per-text durations match HF |

**Watch out:** this checkpoint sometimes opens "Good morning…" in the Ryan voice with a laugh, and Whisper then transcribes "ha ha…" until it hits its token limit (WER ≈ 24). A single such sample looks like a catastrophic regression until you measure the same rate in the reference implementation.

## What didn't work (or isn't worth it yet)

| Attempt | Result | Why / lesson |
|---|---|---|
| FP8 E4M3 scaled matmul (weights and activations), no graphs | 44–53% of BF16 throughput | Quant/dequant kernel overhead dominates in a latency-bound regime. A fused weight-only FP8 GEMM inside the graphs is still untested. |
| `torch.compile` of talker decode | 3–9% slower than graphs + fused norms | Numerics were fine; Inductor kernels don't beat hand-chained PDL kernels at M=1 |
| Disabling cuDNN SDPA | More repeatable, but slower (0.264 vs 0.224) | Nondeterminism is acceptable when semantics are correct |
| Position-readback-only wrapper | No change | Host overhead was not the bottleneck |
| Compute-only unified frame capture (phase 2) | Negligible gain, unresolved discrepancy | Superseded by the phase-3 engine, which captures sampling and history state properly |
| Explicit Python frame loop alone | 1.006× / 1.024× on fixed work | Removing syncs without removing kernels buys little |
| Streaming 1-frame start, then 25-frame chunks | 53 ms first audio, but 270–330 ms simulated playback stalls | First audio is not uninterrupted playback. Use a ramped schedule (4/25 worked). |
| Overlapped codec stream in phase-2 streaming | Did not consistently improve | Eager codec launches contend with generation |
| Overlapped chunked codec decode during offline generation (`codec_overlap.py`) | Quality OK at chunk 48 / context 96 (~39–40 dB, the BF16 noise floor), but ~3× decode work | Not worth it once codec decode is a cheap graph. 25-frame context gives only ~21 dB; 72 frames ~34 dB. |
| `cudnn.benchmark = True` for the codec | No change | cuDNN has no fast BF16 dilated kernel to pick; the polyphase rewrite fixed it |
| Predictor attention fused into the o-projection (`pred_fused.py`) | 1.98–2.5 ms vs 1.76 ms | Moves a serial attention chain onto every o-tile's critical path; a separate kernel parallelizes over heads |
| Greedy token parity as a correctness test | Diverges within the first frames | Use FP32-reference error and sampled-rate statistics instead |
| 0.6B predictor in PDL mode (phase 2) | Rejected then on one numerical guard | Now used, accepted on sampled quality evidence (0.0% WER) |
| Matching vLLM-Omni's architecture | Abandoned as a goal | Its offline number overlaps the codec with generation and uses 72-frame context, so it isn't equal work. Optimizing our own stack paid off more. |

## Transferable playbook for other AR TTS models

1. **Profile kernels per generated frame before anything else.** Count kernels and compare frame time against the weight-bandwidth floor (bytes per step ÷ HBM bandwidth). If two model sizes take the same time, you're latency bound, and fusion and graphs beat everything else.
2. **Capture whole frames, not pieces.** Put sampling, embedding lookups, counters and history writes on the GPU so one graph replay covers a frame. Read back only the EOS signal, through pinned memory.
3. **Chain kernels with PDL and fold norms into weights.** Fold the RMSNorm weight into the next projection, apply the row scale in the epilogue, and fuse residual add plus sum of squares into the producing GEMM. That gets each transformer layer to about 5 dependent kernels.
4. **Retune on the target GPU.** Vendored "tuned" configs come from someone else's SM count. Score candidates in a graph over distinct weights so L2 doesn't flatter them.
5. **Don't forget the fixed costs.** Prefill and the vocoder/codec were ~25% of a short request once decode got fast. Check vocoders for cuDNN fallback kernels: dilated or grouped BF16 convs are common offenders.
6. **Use causality as a tool.** In a causal model, right-padding to a bucket is exact, which makes variable-length prefill and decode graph-friendly.
7. **Validate distributions, not tokens.** Compare error against an FP32 reference alongside the reference BF16 path, run ASR WER, and compare failure-mode rates against the unmodified implementation over 100+ seeds before calling anything a regression.

## Remaining opportunities

- **Engine-native streaming API.** The frame loop already has an `on_frame` hook and the codec bucket graphs are ready.
- **Predictor**, now about 65% of the frame. Options are a persistent megakernel or weight-only FP8 (about half the bytes).
- **1.7B talker GEMMs** reach only about 2–2.6 TB/s; an FMA-based GEMV for M=1 would avoid tensor-core padding to M=16.
- **Codec:** about 30% of the remaining decode time is NCHW↔NHWC conversions and permute copies.
- **Batching or continuous batching** for server throughput. Everything above is batch one.

## Files

| Path | Contents |
|---|---|
| `optimizations/scripts/fast/talker_engine.py` | Engine, prefill, sampler, `install_fast_tts(model)` |
| `optimizations/scripts/fast/codec_fast.py`, `codec_graphs.py` | Codec polyphase conv, fused Snake, bucketed graphs |
| `optimizations/scripts/fast/tune.py`, `tuned/` | Per-GPU tuner and H20 configs |
| `optimizations/scripts/fast/run_fast.py` | One-off generation CLI |
| `optimizations/scripts/fast/suite.py`, `asr_wer.py` | Benchmark suite (`--stack base/engine/fast/hf`) and Whisper WER |
| `optimizations/scripts/fast/test_*.py` | Numerics, prefill, codec, overlap and predictor-fusion checks |
| `optimizations/results/fast/` | All phase-3 result JSON and ASR outputs |
| `optimizations/scripts/`, `optimizations/vendor/` | Phase 1–2 adapters and vendored fused predictor kernels (Apache-2.0) |

Boundaries for the current engine:
- Batch one, CustomVoice, BF16, CUDA sm90 (PDL).
- Requests with top-p below 1.0 or a non-all-ones attention mask fall back to the original `generate`.
- KV capacity is 4096 positions; prompts over 256 tokens use the HF prefill.
- Single stream, not thread-safe.
