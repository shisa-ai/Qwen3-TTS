# Fast engine: whole-frame CUDA graph decode

2026-11-14, H20, BF16, batch one, CustomVoice. Follow-up to [GAP_CLOSING_RESULTS.md](GAP_CLOSING_RESULTS.md). Code: [`scripts/fast/`](scripts/fast/).

## Outcome

| Suite (Ryan/Aiden × 4 texts × 3 seeds, incl. codec) | 0.6B RTF ↓ | 1.7B RTF ↓ |
|---|---:|---:|
| Previous accepted stack (fused predictor + talker fusion + graphs + frame loop), re-measured | 0.0926 | 0.1002 |
| Whole-frame graph engine (first version) | 0.0568 | 0.0711 |
| + H20-tuned GEMM/attention configs, faster codec kernels | 0.0457 | 0.0598 |
| **+ Engine prefill graphs + bucketed codec graphs (current)** | **0.0351** | **0.0447** |
| Speedup vs previous accepted stack | **2.64×** | **2.24×** |

Mean of condition-level median RTFs; pooled RTF 0.0347 / 0.0439. Every request reached EOS. Steady-state per-frame graph time went from 6.38 / 6.64 ms to about **2.55 / 3.25 ms**. Historical vLLM-Omni was 0.0753 / 0.0844.

## What changed

The previous stack ran a talker graph and a predictor graph per frame, with roughly 1 ms of eager Python, sampling and embedding work between them, about 1,700 kernels per frame. The talker graph was kernel-latency bound: 3.0 ms for 0.6B and 3.5 ms for 1.7B, where weight reads take only about 0.2 / 0.7 ms.

1. **One CUDA graph per frame** (`talker_engine.py`). It runs predictor → codebook-embedding sum → 28 talker layers → final norm → codec head → talker sampler for the next frame, all on device. The host reads one int64 per frame from pinned memory to check for EOS. About 560 kernels per frame.
2. **Talker as a PDL GEMM chain**, reusing the vendored predictor's GEMM kernels and epilogues. Each layer is 5 kernels: qkv (RMSNorm folded into the weights), split-KV decode attention (qk-norm, RoPE and cache write fused), o with in-place residual add and sum of squares, gate/up with SiLU, and down. Text-only mRoPE is checked at install to reduce exactly to plain RoPE.
3. **On-device talker sampler**: repetition penalty, min-new-tokens, suppress tokens, temperature, exact top-k (bisection on FP32 order keys, ties kept), then Gumbel-max. It matches HF's sampling distribution, but draws a different random stream.
4. **Per-GPU tuning** (`tune.py`, results in `scripts/fast/tuned/`). Talker graph: 1.07 → 0.79 ms (0.6B), 1.92 → 1.49 ms (1.7B). Predictor: 2.20 → 1.73 ms. The vendored configs were tuned for H200 (132 SMs); the H20 has 78.
5. **Engine prefill** using the same chain with M = prompt length, plus two Triton prefill-attention kernels. One graph per 16-token bucket; right-padding a causal prompt leaves earlier positions unchanged. HF ~18 ms → **1.7–2.8 ms (0.6B) / 3.6–5.7 ms (1.7B)**.
6. **Codec** (`codec_fast.py`, `codec_graphs.py`). cuDNN falls back to a slow legacy BF16 kernel for the dilated k7 convolutions (5× slower than undilated ones). These are rewritten as undilated convolutions over the d polyphase subsequences, which is the same arithmetic. SnakeBeta is fused into one FP32 Triton pass. Decode for 233 frames: 51.6 → 33.8 ms. Decode then replays bucketed graphs (32-frame buckets; right-padding is exact for this causal decoder), keeping upstream's 300/25 chunking.

## Correctness evidence

- **Talker numerics vs FP32**, teacher-forced on the same codes (`test_numerics.py`, 1.7B): engine hidden-state relative error 1.20% (p95 1.47%) and KL 4.19e-4, against HF BF16's own 1.26% (p95 1.59%) and KL 4.22e-4. The engine is no less accurate than HF BF16.
- **Prefill vs HF** (`test_prefill.py`): last-token logits argmax identical on every prompt, top-5 overlap 4–5 of 5, relative error ~1%.
- **Codec vs FP32 decode** (`test_codec_fast.py`): fast kernels 35.0 dB SNR, stock BF16 33.8 dB. The fast path is closer to FP32. Bucket padding and graph replay are bit-identical to eager on the same shape.
- **ASR (Whisper large-v3-turbo) on the 24-sample suites:** pooled WER 0.0% (0.6B) and 0.9–1.1% (1.7B); the remaining errors are "9"/"nine" normalization.
- **Sampling-rate check against unmodified HF (1.7B):** 120 samples (2 voices × short/medium × 30 seeds) and 100 Ryan-medium seeds. The Ryan voice occasionally opens "Good morning…" with laughter, and Whisper then loops on it. On Ryan-medium, HF did this on **8/100** seeds and the engine on **9/100**; pooled WER 1.914 vs 1.915. The behavior is inherent to the model, not caused by the engine.
- Greedy decoding diverges from HF within the first frames at near-ties in the predictor, as any rounding change does. Use sampled-distribution checks, not token identity.

## Rejected or deferred

- **Fused predictor attention+o** (`pred_fused.py`): correct, but slower (1.98–2.5 ms vs 1.76 ms), because attention moves onto each o-tile's critical path. Kept for reference only.
- **Overlapped chunked codec decode** (`codec_overlap.py`): chunk 48 with 96 frames of context reaches the codec's BF16 noise floor (~39–40 dB), but costs about 3× the decode work and competes with the talker. Off by default (`overlap=False`).
- Not done yet: engine-native streaming API (the frame loop already has an `on_frame` hook), FP8 weights, a predictor megakernel, and `cudnn` layout cleanup in the codec (~30% of the remaining decode time is NCHW↔NHWC and permute copies).

## Boundaries

Batch one, CustomVoice, BF16, CUDA sm90 (PDL). Supported requests have an all-ones attention mask, top-p 1.0 for both samplers and an integer EOS; anything else falls back to the original `generate`. KV capacity is 4096 positions; `max_new_tokens` is clamped to fit. Prompts over 256 tokens use the HF prefill. The engine is single-stream and not thread-safe. Tuned configs are for H20; run `tune.py` on other GPUs. The first request with a new sampling configuration captures its frame graph (~4 s).

## Reproduce

```bash
# one-off generation
CUDA_VISIBLE_DEVICES=0 python optimizations/scripts/fast/run_fast.py --size 0.6B --text "Hello." --output out.wav --repeat 3
# suite (results/fast/<size>-<label>.json, WAVs in samples/fast/), then ASR
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python optimizations/scripts/fast/suite.py --size 1.7B --stack fast --label mytest
python optimizations/scripts/fast/asr_wer.py optimizations/results/fast/1.7b-mytest.manifest.json
# stacks: base (previous accepted), engine (talker only), fast (engine + codec), hf (unmodified)
# retune on a new GPU (~30 min/size; writes scripts/fast/tuned/tuned-<size>.json)
CUDA_VISIBLE_DEVICES=0 python optimizations/scripts/fast/tune.py 1.7B
```

Result JSON from this pass is in `results/fast/`; manifests reference the original `/root/tts` paths as provenance. Audio is not bundled.
