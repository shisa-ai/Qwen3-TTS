# Experimental optimized CustomVoice runtime

> **Newest: fast engine** ([`scripts/fast/`](scripts/fast/), report: [FAST_ENGINE_RESULTS.md](FAST_ENGINE_RESULTS.md)). One CUDA graph per codec frame, faster codec kernels, and engine prefill: mean RTF **0.0351 (0.6B) / 0.0447 (1.7B)**, 2.64× / 2.24× faster than the stack documented below, with WER and sampling behavior matching unmodified HF. Offline only for now.
>
> ```bash
> CUDA_VISIBLE_DEVICES=0 python optimizations/scripts/fast/run_fast.py --size 0.6B --speaker ryan \
>   --text "Hello from the fast engine." --output out.wav
> ```
>
> In Python: `sys.path.insert(0, "optimizations/scripts/fast"); from talker_engine import install_fast_tts; install_fast_tts(model)`, then call `model.generate_custom_voice(...)` as usual. Unsupported requests fall back to the original path.

This directory packages the completed H20 optimization pass from 2026-11-14. It is self-contained within this checkout; it does not import `/root/tts` files, vLLM, or a server. Newer work by other agents is deliberately excluded. `SNAPSHOT.json` records the frozen benchmark source hashes. The runtime adapters and vendored kernels are unchanged from that snapshot; the benchmark runner has portable device/path handling.

## Install

Use Linux, an NVIDIA CUDA GPU, Python 3.12, and a CUDA-enabled PyTorch installation compatible with your driver. The measured environment used **PyTorch 2.15.0, Triton 3.9.0, Transformers 4.57.3**, BF16, and H20 GPUs. These exact Torch/Triton builds may require a suitable package index; other combinations have not been validated. Install Torch and matching torchaudio first using the appropriate PyTorch distribution for your machine, then:

```bash
git clone https://github.com/shisa-ai/Qwen3-TTS.git
cd Qwen3-TTS
python -m pip install -e .
python -m pip install triton
```

Do not upgrade Transformers independently: adapters check upstream generation semantics and deliberately reject incompatible source changes. Model weights download from Hugging Face on first use. Flash Attention is not required; the optimized path uses SDPA. No vLLM installation is required.

## Generate audio

From the repository root, select an available GPU (the index below is only an example):

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
python optimizations/scripts/run_optimized.py \
  --size 0.6B --speaker ryan --text "Hello from the optimized runtime." \
  --output output.wav
```

Use `--size 1.7B` for the larger model. The runner chooses the validated predictor mode: `nonpdl` for 0.6B, `pdl` for 1.7B. PDL means programmatic dependent launch; the fast kernel configuration was tested on H20, not sm120 hardware. Unsupported hardware/version combinations can fail rather than silently provide equivalent speed.

Independent sampling controls:

```bash
CUDA_VISIBLE_DEVICES=0 python optimizations/scripts/run_optimized.py \
  --text "Separate sampling controls remain available." --output controlled.wav \
  --temperature 0.7 --subtalker-temperature 0.9 --subtalker-top-p 0.85
```

Nondefault subtalker top-p uses strict fallback and may be slower. `--greedy` disables both samplers. Do not use zero temperature as a synonym for greedy decoding.

For callback-driven streaming into an incrementally written WAV:

```bash
CUDA_VISIBLE_DEVICES=0 python optimizations/scripts/run_optimized.py \
  --text "This file is written as audio chunks arrive." --output streamed.wav --streaming
```

Streaming uses four initial frames, then 25-frame chunks, with conservative 588-frame context. This command writes chunks synchronously; it does not play them or run a network server. Slow callbacks stall generation. Streaming and offline waveforms need not match because codec windows differ.

First use includes graph capture and kernel compilation. Warm the desired model and sampling settings before measuring steady-state latency. The CLI does not overwrite existing output. A failed streaming call can leave a partial WAV; do not treat it as a complete utterance.

## Codec graphs and Python integration

`run_optimized.py` shows the tested installation order and reverse-order cleanup. To prepare codec graphs, use `CodecGraphs` from `scripts/qwen_codec_graph.py` inside `torch.inference_mode()` on the owning CUDA stream:

```python
codec = CodecGraphs(model.model.speech_tokenizer.model.decoder)
codec.prepare(codes)  # real contiguous CUDA int64 tensor [1, 16, frames]
with codec.installed():
    # Call direct streaming or ordinary generation here.
    # Prepared frame lengths replay; unseen lengths decode eagerly.
    ...
codec.reset()
```

Import helpers by adding this directory's `scripts` directory to `sys.path`, as with the command-line runner. Preparation must happen outside requests; do not assume the next sampled request will have the same shapes. No padding or shorter-context approximation is applied. `max_graphs` and `max_frames` bound cache preparation. The CLI intentionally uses eager codec execution rather than pretending it can predict future window lengths.

## Boundaries

- CustomVoice, batch one, BF16/SDPA; no training, autocast, concurrent callers, or model mutation after capture.
- The talker cache defaults to 1024 positions, including prompt positions. Overflow raises; it does not silently truncate. `--cache-length` changes the allocation, with larger sizes unbenchmarked.
- Independent temperature/top-k/top-p controls remain in the underlying API. Unsupported fast-path options fall back or are rejected explicitly.
- Numerical rounding changes and default cuDNN SDPA nondeterminism can change sampled output. `--disable-cudnn-sdpa` selects the slower diagnostic backend; it is not a universal determinism guarantee.
- No perceptual quality certification, production scheduler, continuous batching, or sm120 validation. Existing server deployment is not included or modified.
- Rejected experiments (`qwen_fp8`, `qwen_compiled_decode`, `qwen_frame_decode`, `qwen_frame_capture`) are preserved for research, never installed by the runner. Unified capture has unresolved correctness findings.

## Benchmarks, evidence, and tests

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
python optimizations/scripts/bench_qwen_gap.py --size 0.6B \
  --stack combined --predictor-mode nonpdl --frame-loop --label checkout-test
```

For 1.7B use `--size 1.7B --predictor-mode pdl`. Select exactly one GPU. Results and audio go under this directory's `results/` and `samples/`; use a fresh label. The benchmark includes warmup, EOS checks, sampling smoke tests and source snapshots.

[Gap-closing report](GAP_CLOSING_RESULTS.md) and [earlier report](OPTIMIZATION_RESULTS.md) retain the historical methodology. Selected compact result JSON/Markdown is included under `results/`. Original absolute paths, commands, source hashes and sample references in those records are historical provenance, not portable commands. Large traces, audio and model weights are not bundled; some historical report links therefore refer to omitted artifacts.

`historical_tests/` preserves the original GPU-specific validation harnesses. These are not auto-run tests: inspect their device guards, output paths and imports before use. Do not run them on shared hardware blindly. Portable packaging checks:

```bash
python optimizations/scripts/run_optimized.py --help
python optimizations/scripts/test_snapshot.py
```

The package was checked for imports, syntax and snapshot integrity after copying. A fresh-environment GPU end-to-end run was not performed during packaging because GPUs were reserved for ongoing work. The original frozen implementation's GPU evidence is included.

Vendored predictor kernels retain their Apache-2.0 license and provenance in `vendor/qwen_fused_predictor/`. No external community source was copied during packaging.
