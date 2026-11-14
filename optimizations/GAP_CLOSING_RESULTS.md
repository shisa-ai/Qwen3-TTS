# Qwen3-TTS gap-closing results

2026-11-14. Follow-up to [OPTIMIZATION_RESULTS.md](OPTIMIZATION_RESULTS.md).

## Outcome

The final custom offline stack improves mean RTF by **2.068× / 1.914×** over our previous complete accepted stack for 0.6B / 1.7B. It **has not beaten the historical vLLM-Omni measurements**. All additions remain opt-in; the live server still uses vLLM-Omni and was not changed.

RTF here means wall time divided by output audio duration: lower is better. Full-suite figures are means of eight condition-level median RTFs: Ryan/Aiden, four shared texts, three measured repetitions after warmup. Sampled output lengths differ, so these are not identical-output speedups.

| Model | Previous accepted stack | Fusion + config cache | Final + explicit loop | Final speed vs previous | Historical vLLM | Final/vLLM RTF |
|---|---:|---:|---:|---:|---:|---:|
| 0.6B | 0.18423 | 0.10265 | **0.08909** | **2.068×** | 0.07534 | 1.183× |
| 1.7B | 0.19455 | 0.10859 | **0.10162** | **1.914×** | 0.08438 | 1.204× |

Final throughput expressed as inverse mean RTF: **11.22× / 9.84× real time**. vLLM comparison is historical, not a contemporaneous controlled test. GPU 0 and server settings were untouched. Final offline runs used GPU 1; final streaming runs used GPU 2. GPU 3 was excluded after the user reserved it for miotts.

Evidence: [offline audit](results/gap/final/audit-loop-v2.json), [0.6B](results/gap/final/0.6b-loop-v2.json), [1.7B](results/gap/final/1.7b-loop-v2.json). Each size passed 24 timed requests with EOS, finite audio, no truncation, no timed captures, and no default-path fallback. Source snapshots and commands are recorded with results.

## Kept and rejected work

| Candidate | Disposition | Evidence / limitation |
|---|---|---|
| Fused residual predictor | Kept experimentally | Independent controls preserved; strict fallback for unsupported top-p, batches, or hidden-state requests. Apache-2.0 source/provenance under `vendor/qwen_fused_predictor/`. |
| Predictor PDL on 0.6B | Not validated for acceptance | One unchanged numerical guard failed; diagnostic supports propagated rounding but does not erase failure. Use explicit `mode="nonpdl"` for 0.6B. |
| Predictor non-PDL on 0.6B; PDL on 1.7B | Passed bounded deep tests | All-step cache/causality, sampler/RNG, fallback and ownership checks. No perceptual certification or sm120 hardware test. |
| Talker RMSNorm and packed QKV/gate-up | Kept experimentally | Isolated replay improvements; transactional installation and snapshot readiness fixes; BF16/SDPA and bias-free projection restrictions. |
| Immutable GenerationConfig defaults cache | Kept | About 52 microseconds less CPU validation work per call; no meaningful isolated predictor speedup established. Live configuration validation remains. |
| Position-readback-only wrapper | Rejected for performance | Fixed-work timings essentially unchanged. |
| Explicit frame loop | Kept experimentally | Separate engine graphs, synchronous EOS. Independent controls and strict fallback preserved. See fixed-work qualification below. |
| Compute-only unified frame capture | Rejected | Negligible gain and unresolved correctness/state-sensitive discrepancy. Fully captured sampler/history was not delivered. |
| Exact-shape codec graphs | Kept experimentally | Known prepared shapes accelerate small windows; uncached shapes run eagerly. Bounded cache and ownership/failure guards. |
| Direct frame streaming | Kept experimentally | Avoids retaining layer histories, but existing graphs still compute and clone layer outputs. Synchronous callbacks provide backpressure. |
| 4 initial frames / 25 subsequent frames | Explicit tested schedule | No simulated stalls in bounded tests; not a universal playback guarantee. |
| 72-frame codec context | Approximate only | Nonzero error versus whole decoding; not substituted for conservative 588-context default. |
| Stateful codec and new side-stream overlap | Deferred | Stateful port needs attention caches, convolution histories and transposed-convolution boundary handling. No new overlap benefit claimed. |

Predictor details: [deep validation](results/gap/fused-predictor/DEEP_VALIDATION.md), [0.6B diagnosis](results/gap/fused-predictor/DIAGNOSIS.md), [non-PDL option](results/gap/fused-predictor/NONPDL.md).

## Fixed-work profile: do not overattribute the loop gain

Final GPU 1 measurement: direct talker generation including prefill, excluding tokenization and codec, fixed 72-frame workload, capture excluded.

| Model | Without loop | With loop | Speedup |
|---|---:|---:|---:|
| 0.6B | 517.26 ms | 514.38 ms | 1.006× |
| 1.7B | 537.08 ms | 524.30 ms | 1.024× |

Earlier bounded tests showed larger loop gains; this final result is smaller. Do not attribute the sampled full-suite improvement entirely to host-overhead removal.

Instrumented profiles reduced CUDA runtime synchronization calls from 446 to 228 for each size. Kernel counts changed from 127,162 to 124,754 (0.6B), and 115,498 to 113,090 (1.7B). Substantial kernel work remains. Profiler durations and CUDA event spans must not be subtracted from uninstrumented wall time to infer pure host/component costs. Traces are under `results/gap/final/`.

## Final streaming results

Post-guard source, GPU 2, cuDNN SDPA disabled for deterministic comparison, context 588, schedule 4/25, seed 123. Means of three timed repetitions after warmup. Shapes were prepared using known benchmark lengths; capture/setup excluded.

| Model/text | Hook + eager codec RTF | Direct + graph codec RTF | Direct + graph first audio |
|---|---:|---:|---:|
| 0.6B short | 0.13195 | **0.10354** | **55.98 ms** |
| 0.6B medium | 0.12987 | **0.10959** | **59.07 ms** |
| 1.7B short | 0.14299 | **0.11922** | **59.27 ms** |
| 1.7B medium | 0.13335 | **0.11017** | **58.10 ms** |

Each size passed 24 timed requests across hook/direct and eager/graph variants. Waveforms were exactly equal in these observed deterministic comparisons; direct frames matched the offline EOS reference. No timed captures, codec misses, or simulated playback starvation occurred.

A separate default-cuDNN smoke measured first audio around 53 ms and RTF around 0.095–0.099, with some eager shape misses. **Its EOS/truncation status is unverified** because the streaming result does not expose the terminal token. Do not treat it as an equal-work speedup or certified complete-output result.

Evidence: [final streaming audit](results/gap/frame-streaming/final-measurement-audit.json), per-size `*-final-deterministic.json` and `*-final-default-cudnn.json` in that directory. Final run recorded waveform hashes rather than new WAVs; earlier integration WAVs remain under `samples/gap/frame-streaming/`.

Codec graphs match the same-window eager decoder, not whole-sequence decoding. Even 588-context chunking showed nonzero whole-decode error. Shorter context and crossfades change waveform semantics. No listening evaluation was performed.

Callback lifecycle review found and fixed recursive generation, fallback after partial delivery, interrupted-cache cleanup, and callback stream changes. Final tests passed 45 streaming and 103 offline regression checks per model. Slow callbacks still stall synthesis; no delivery worker, continuous batching, or production concurrency scheduler was added.

## Reproduction

Existing environment: `/root/miniforge3/envs/qwen3-tts/bin/python`. Use a new output label; runners preserve existing artifacts.

```bash
CUDA_VISIBLE_DEVICES=GPU-74eb9316-76d6-7587-c3de-2153fc1ea3dc \
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
/root/miniforge3/envs/qwen3-tts/bin/python scripts/bench_qwen_gap.py \
  --size 0.6B --stack combined --predictor-mode nonpdl \
  --frame-loop --profile-frames 72 --label loop-v3
```

For 1.7B use `--size 1.7B --predictor-mode pdl`. Exact streaming commands and source hashes are embedded in the final result JSON files; use those instead of assuming the offline runner measures streaming.

Implementation entry points:
- `scripts/qwen_fused_predictor.py`
- `scripts/qwen_talker_fusion.py`
- `scripts/qwen_frame_loop.py`
- `scripts/qwen_codec_graph.py`
- `scripts/qwen_frame_streaming.py`

The benchmark runner establishes installation order. Uninstall dependent graph/loop adapters before changing the modules they captured. These are single-instance experimental inference adapters, not thread-safe serving engines.

## Community-source audit and remaining work

Examined `andimarafioti/faster-qwen3-tts` at `b35fdc7d71eb3e4d19e63f4b4a330a1aef31d682` and `dffdeeq/Qwen3-TTS-streaming` at `13ef823ffe76aec4ac7e35d3fec28ecfc8d7427a`. Their headline CUDA-graph optimization was already covered. The concrete streaming-loop integration gap led to the direct-frame interface above.

Further candidates, not delivered or measured here: eliminate unnecessary talker hidden-state computation/copies through a distinct output-minimal graph; move mask updates inside capture; evaluate compilation specifically at the codec boundary; full talker attention/cache fusion; production shape-cache policy; asynchronous EOS and fully captured sampling/history. Voice-conditioning caches matter for Base-model cloning, not the current CustomVoice workload.

Do not copy fixed predictor settings, altered top-p filtering, silent cache-limit stopping, zero-token padding, or short codec context and claim an equivalent speedup. Their inverse-RTF convention, hardware, voice-cloning workload and chunk sizes also prevent direct comparison with our measurements.

The remaining measured offline gap to historical vLLM is about 18–20% higher RTF. Closing it is still an objective, not an achieved result. Perceptual quality, long/adversarial utterances, production concurrency, unseen-shape latency, and non-H20 hardware remain unvalidated.
