# Qwen3-TTS optimization results

## Recommendation

Use opt-in predictor and talker CUDA graphs with SDPA for single-stream inference. Fused predictor RMSNorm provides a further measured benefit on H20. Removing unused internal predictor hidden states adds a smaller benefit. These are experimental adapters, not production-quality or perceptual-quality certifications.

Keep the default SDPA backend: disabling cuDNN improved repeatability but reduced combined-graph performance in this suite. Numerical nondeterminism is acceptable when cache, mask and sampling semantics remain correct.

Do not enable the tested FP8 or compiled-decode variants for performance: neither improved the measured end-to-end workloads. Streaming works, but one-frame first audio followed by large chunks causes playback starvation; first-audio latency is not uninterrupted-playback latency.

## Environment and method

Tests used Qwen/Qwen3-TTS-12Hz-{0.6B,1.7B}-CustomVoice, BF16, NVIDIA H20-3e, PyTorch 2.15.0+cu132 and Transformers 4.57.3 in `/root/miniforge3/envs/qwen3-tts`. GPU 0 was used for early talker tests before being reserved for separate vLLM-Omni work. Subsequent work used only GPUs 1, 2 and 3.

RTF (real-time factor) is generation wall time divided by generated audio duration; lower is better. Full-call measurements include waveform decoding, use synchronized boundaries, and exclude model loading and graph capture from warmed timing. Unless a section says otherwise, results use three measured calls after warmup. Generated lengths can differ between sampled configurations and even between reseeded requests. Ratios of normalized timings are not identical-output latency comparisons.

No listening, transcription-accuracy or speaker-similarity acceptance study was performed. Finite audio, valid tokens, EOS checks and numerical comparisons do not establish perceptual equivalence.

## Graph ablations

Mean per-cell RTF over two English voices (`ryan`, `aiden`) and four texts in `scripts/bench_common.py`:

| Configuration | 0.6B | 1.7B |
|---|---:|---:|
| Original Flash Attention 2 | 1.425 | 1.315 |
| SDPA baseline | 1.055 | 1.021 |
| SDPA + predictor graph | 0.397 | 0.394 |
| SDPA + talker graph | 0.834 | 0.884 |
| SDPA + both graphs, reviewed | **0.224** | **0.223** |
| Both graphs, cuDNN SDPA disabled | 0.264 | 0.276 |

Both graphs improve RTF by 4.71× and 4.58× against matched SDPA baselines, respectively. Against the original Flash Attention baseline, the ratios are 6.36× and 5.90×; those comparisons include an attention-backend change.

The predictor captures its fixed codebook loop with a static cache. The talker captures only its transformer decode step. Prefill, outer generation, sampling/embedding work and host-side checks are not all eliminated. The talker adapter supports serial batch-one use, has a bounded prompt-plus-generation cache, and enforces device/stream ownership.

Evidence: [raw graph results](results/gpu3-optimization/), [audio samples](samples/gpu3-optimization/), `scripts/test_predictor_graph.py`, `scripts/test_talker_graph.py`, and `scripts/test_predictor_graph_reseed.py`. Tests covered greedy parity, reset/poisoned cache, retained output ownership, padding and fully masked rows, capacity boundaries, and stream rejection. Sampled repeatability differences also occurred upstream; backend controls implicated numerical attention differences rather than predictor RNG offsets.

## Fused normalization and unused outputs

The custom Triton RMSNorm preserves FP32 accumulation and upstream rounding before multiplying by the norm weight. It patches predictor instances only and falls back on untargeted layouts.

| Fixed-work predictor measurement | 0.6B before → after | 1.7B before → after |
|---|---:|---:|
| Fused RMSNorm, graph replay | 11.863 → 8.583 ms | 12.006 → 8.702 ms |
| Remove unused hidden states, over fused norms | 8.458 → 8.255 ms | 8.571 → 8.324 ms |

These are separate experiments, not an additive timing decomposition. Fused normalization also reduced short/medium sampled end-to-end RTF by 17–24% in its repeated experiment; fixed-work measurements provide stronger isolated evidence because sampled durations differ.

The fast-decode adapter changes only a fingerprint-verified internal call to stop requesting unused predictor hidden states. Direct predictor callers retain their output contract. Host synchronization and mask staging remain unchanged.

Evidence: [fused-norm report](results/kernel-profile/fused-norm/REPORT.md), [fast-decode report](results/fast-decode/README.md). No sm120 hardware was available: source portability and toolchain support do not establish RTX PRO 6000/5090 correctness or speed. Sampling fusion and normalization/rotary/cache-store fusion remain potential follow-up experiments, not delivered speedups.

## Batching

Sampled medium text, `ryan`, native list API; aggregate generated audio seconds per wall second:

| Batch | 0.6B baseline | 0.6B predictor graph | 1.7B baseline | 1.7B predictor graph |
|---:|---:|---:|---:|---:|
| 1 | 1.01 | 2.49 | 1.01 | 2.57 |
| 8 | 6.94 | 13.55 | 5.97 | 13.94 |
| 16 | 12.17 | 21.50 | 11.75 | 22.05 |
| 32 | 14.59 | 35.33 | 22.68 | 38.27 |

These runs do not use the batch-one talker graph or fused normalization. Batch-32 predictor-graph wall times were 7.995 s and 6.254 s. All sampled configurations and timed calls reached EOS without truncation or OOM. Mixed-length batch-eight throughput was lower: 8.33 and 7.10 audio seconds/wall second with the predictor graph. Padding and varying completion lengths prevent uniformly linear scaling.

Batched greedy token discrepancies were investigated with teacher-forced logits. Tested FP32 static/dynamic outputs matched; BF16 differences changed small-margin top-token decisions. Mask/reset checks passed. Earlier greedy runs that hit generation caps are preserved separately and are not valid throughput comparisons.

Evidence: [sampled summary](results/batching/summary-sampled.json), `results/batching/*-sampled.json`, [samples](samples/batching/), and `scripts/batching_diagnostic.py`.

## Streaming and codec overlap

The adapter delivers owned, newly decoded CPU audio through callbacks while generation proceeds. It retains conservative architecture-derived left context (588 frames); shorter context is explicitly approximate. Chunked BF16 decoding is not generally identical to whole decoding.

Matched short/medium runs with graphs and fused normalization:

| Schedule | Median first-audio range | Median simulated playback-stall range |
|---|---:|---:|
| Serial 12-frame chunks | 199.8–229.1 ms | 0 |
| Serial one-frame start, then 25 frames | 52.8–55.9 ms | 268.6–330.1 ms |
| Overlap one-frame start, then 25 frames | 58.3–59.2 ms | 321.8–344.0 ms |

A one-frame chunk contains 80 ms of audio; it does not cover the delay until the next 25 frames arrive. Serial 12 remains the default. Overlap uses a bounded queue and a separate codec stream, with events and lifetime protection, but did not consistently improve these requests. Overlapped audio matched serial decoding on identical tokens in the measured runs. Cancellation, callback/decoder errors and subsequent reuse were tested. Playback stalls are simulated from callback timestamps, not measured with an audio device.

Evidence and reproduction harnesses: [streaming report](results/streaming/REPORT.md), [overlap report](results/streaming/overlap/REPORT.md), [initial samples](samples/streaming/); overlap WAVs are stored beside its metrics. A smaller startup chunk schedule and codec execution optimization remain follow-up work.

## Rejected performance candidates

### FP8

The tested E4M3 scaled-matrix-multiplication patch quantizes selected linear weights and activations, with BF16 output. It does not quantize codec, embeddings or output heads. It retains BF16 weights for reversal/fallback, so it does not demonstrate resident weight-memory savings.

Across both sizes at batches one and sixteen, its aggregate throughput was only 44–53% of matching BF16 throughput. These tests used no graphs. Local numerical checks passed, but that does not establish speech quality. Reject this implementation for performance; this does not rule out a fused or graph-captured FP8 implementation.

Evidence: [FP8 report](results/fp8/REPORT.md), [raw results](results/fp8/), [samples](samples/fp8/).

### torch.compile

The follow-up used identical inputs and restored pre-decode caches, with FP32 arithmetic on the same BF16-rounded tensors as reference. Default compiled hidden states were closer to FP32 in relative L2 error at every tested position. Precision-cast emulation reproduced eager outputs in all tested comparisons, supporting fusion/rounding differences rather than a cache defect.

Despite credible numerics on tested inputs, compiled decode was slower end to end than graphs plus fused normalization on matched 24/72-token workloads: about 8–9% for 0.6B and 3% for 1.7B. Fixed-length tests are not natural EOS/quality benchmarks. Keep the compiled variant disabled by default.

Evidence: [compile-reference artifacts](results/compile-reference/) and `scripts/test_qwen_compile_reference.py`.

## Reproduction and installation

Run from `/root/tts`. Select only an available GPU among physical GPUs 1, 2 and 3; GPU 0 is reserved. For example, the following repeats a graph ablation on GPU 3 under a new label (the harness refuses to overwrite existing JSON):

```bash
CUDA_VISIBLE_DEVICES=GPU-88395248-f4ce-ee5f-0cc5-1bdd951de1e8 \
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
/root/miniforge3/envs/qwen3-tts/bin/python scripts/bench_qwen_optimization.py \
  --size 0.6B --label local-both-graphs --attn sdpa \
  --installer qwen_predictor_graph:install_predictor_graph \
  --installer qwen_talker_graph:install_talker_graph
```

Use `--size 1.7B` for the other model. Omit installers for SDPA baseline; use one installer for an individual graph. `--disable-cudnn-sdpa` selects the explicit repeatability control. Specialized tests and exact commands are in each experiment report; some tests intentionally require their assigned GPU UUID.

For application integration, add `scripts` to the Python import path, load the model with BF16/SDPA, then install in this order:

```python
from qwen_fused_norm import install_fused_predictor_norm
from qwen_predictor_graph import install_predictor_graph
from qwen_talker_graph import install_talker_graph
from qwen_fast_decode import install_fast_decode

norm = install_fused_predictor_norm(model.model.talker.code_predictor)
predictor = install_predictor_graph(model.model.talker.code_predictor)
talker = install_talker_graph(model.model.talker, max_cache_len=1024)
fast = install_fast_decode(model.model.talker)
```

Install before graph capture; changing kernels after capture requires recapture. Use serial inference on the owning CUDA device/stream, no autocast, and do not mutate weights or device/dtype. Cache capacity includes prompt positions. Uninstall in reverse order under `torch.inference_mode()` after inference caches have been initialized. The adapters remain optional; upstream package source and unrelated benchmark results were not changed.

## Comparison boundaries and remaining work

The user-supplied vLLM-Omni probe reported one warm request with 6.1 s audio in 0.51 s and first nonempty audio at 17 ms. This is an external provisional target, not an independently verified matched benchmark in this report. Do not compare it to suite averages as a controlled speedup.

There is no single full-matrix benchmark of every accepted optimization combined, no perceptual quality certification, no sm120 hardware validation, and no validated batch-multiple talker graph. Residual host synchronization/mask work, faster codec execution, a startup schedule without playback starvation, and additional fused kernels remain opportunities rather than completed optimizations.
