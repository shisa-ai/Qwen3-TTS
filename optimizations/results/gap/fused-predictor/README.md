# Fused predictor experiment

**Follow-up:** explicit `mode="nonpdl"` passes the unchanged 0.6B deep guard
with a measured 2.944x fixed-input predictor speedup. See `NONPDL.md`. The
original default-PDL failure remains recorded; model defaults were not changed.

**Acceptance update:** deeper incremental tests found one 0.6B per-layer
relative-error guard violation; 1.7B passed. Do not treat the initial tests below
as final acceptance. See `DEEP_VALIDATION.md` and the `*-deep.json` artifacts.

An opt-in port reduces fixed-input predictor latency on GPU2 (H20), with
independent subtalker temperature/top-k and greedy controls. Keep it experimental:
weight folding changes rounding, and listening evaluation is still required.

| Model | Strict graph + fused norm, no hidden states | Fused predictor | Fixed-input speedup | Strict mean RTF | Fused mean RTF |
|---|---:|---:|---:|---:|---:|
| 0.6B | 7.435 ms | 2.351 ms | 3.162x | 0.2037 | 0.1192 |
| 1.7B | 6.713 ms | 2.352 ms | 2.855x | 0.2202 | 0.1340 |

Fixed-input timings are medians of 20 synchronized calls, excluding capture.
End-to-end figures use SDPA, talker graph, fused predictor norms, and audited
unused-hidden-state removal for both variants; three repetitions each of shared
short/medium text, Ryan, talker temperature 0.7 and subtalker temperature 0.9.
Mean-RTF ratios are 1.708x / 1.643x, but sampled durations differ: these are not
fixed-output speedups. See JSON for every duration and capture/setup time.

## Integration

```python
from qwen_fused_predictor import install_fused_predictor
handle = install_fused_predictor(model.model.talker.code_predictor)
# Install the existing talker graph and fast-decode adapter as usual.
# Do not also install the strict predictor graph.
# ... generate_custom_voice(..., temperature=0.7, subtalker_temperature=0.9)
handle.uninstall()
```

The existing fast-decode adapter must be active for internal generation to use
fusion: otherwise requested predictor hidden states correctly take the strict
fallback. The handle exposes `_predictor_graph` compatibility for that adapter.
Uninstall fast decode before uninstalling the predictor.

Batch one, CUDA BF16 only. Hidden-state requests, batches larger than one, and
sampled `top_p < 1` explicitly execute the strict predictor graph. Temperatures
and top-k have bounded graph variants (eight by default). Greedy mode ignores
sampling controls. One owning CUDA stream; no concurrent calls, weight mutation,
or device/dtype changes. Original HF transformer weights remain untouched.

## Validation

Both scripts exited 0. Eight real talker input frames per model were tested.
Greedy token agreement with strict graphs was 87.5% / 95.0%; this is not a claim
of bitwise equivalence. For four real inputs, all five layers were compared on
identical teacher-forced rows against HF BF16 and an FP32 layer copy. The maximum
PDL-path RMSE divided by reference RMS was 0.003191 / 0.003682, below the explicit
0.04 guard. The FP32 reference uses the same BF16-derived RoPE tables; it is not
an independently recomputed high-precision RoPE reference.

Checks passed for finite outputs, poisoned cache reset, independent output
storage, owning-stream rejection, greedy replay, sampled temperature/top-k
variants, and strict top-p fallback. End-to-end runs retain the HF talker loop,
including EOS handling; all measured audio durations are below the 180-frame
cap. No dedicated forced-EOS test or listening test was performed. Layer tests
cover the initial two-row prefill; incremental attention numerics are exercised
by generation but not independently compared at every cache position.

Programmatic dependent launch (PDL) was enabled on the tested H20. Kernels use
Hopper-or-newer support detection and upstream shape checks. No sm120 testing,
batched fused-path validation, kernel-count profiling, or full benchmark-suite
comparison was performed. Non-PDL layer math was checked, but non-PDL complete
execution on other hardware remains untested.

## Reproduction

From `/root/tts`, for each `SIZE` in `0.6B` and `1.7B`:

```sh
CUDA_VISIBLE_DEVICES=GPU-3e620c1c-c243-7353-580e-cde9e5eb4b8a \
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
/root/miniforge3/envs/qwen3-tts/bin/python scripts/test_qwen_fused_predictor.py --size "$SIZE"
```

`0.6B.json`, `1.7B.json`, and matching logs contain the evidence. Syntax checks:

```sh
/root/miniforge3/envs/qwen3-tts/bin/python -m py_compile \
  scripts/qwen_fused_predictor.py scripts/test_qwen_fused_predictor.py \
  vendor/qwen_fused_predictor/kernels.py
```

License and source provenance: `vendor/qwen_fused_predictor/PROVENANCE.md`.
GPU0, server configuration, upstream repositories, and other adapters were not edited.
