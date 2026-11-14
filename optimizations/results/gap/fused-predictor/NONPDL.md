# Explicit non-PDL candidate for 0.6B

**Retain `mode="nonpdl"` as the validated experimental 0.6B option.** It passes
the unchanged 0.04 deep guard and retains a useful speed advantage. This does
not change the original PDL failure or silently switch either model's default.

```python
handle = install_fused_predictor(predictor, mode="nonpdl")
```

`mode="pdl"` remains the default, including for 1.7B. The non-PDL choice is
explicit for every model; no model-size inference changes execution policy.
It disables the programmatic dependent launch chain but retains the upstream
cuBLAS packed projections, folded input norms, and fused attention kernels.
Its final RMSNorm is explicit before the LM projection, unlike PDL's folded
final norm. GEMM accumulation/cast ordering also differs. Thus this is not a
fully unfolded implementation, nor a claim of mathematical superiority for
all inputs. No explicit-normalization transformer variant was retained:
the first bounded candidate already passed with useful performance.

## GPU2 results

- Full deep checks, 0.6B non-PDL: **exit 0**, no guard failures.
- Maximum relative logits error vs FP32: **0.019855**.
- Maximum per-layer relative output error: **0.037670**, below 0.04.
- Maximum historical KV relative error: **0.014605**.
- Complete numerical/semantic checks include all 15 steps, repeated frames,
  poisoned future slots, prefill causality, sampling, RNG, strict fallbacks,
  output ownership, and uninstall restoration as described in DEEP_VALIDATION.
- Default 1.7B PDL deep checks rerun after the mode addition: **exit 0**.
- Python syntax compilation: **exit 0**.

Matched 0.6B benchmark (capture excluded):

| Metric | Strict graph + fused norm, no hidden states | Non-PDL fused |
|---|---:|---:|
| Fixed-input median predictor latency | 7.438 ms | 2.526 ms |
| Mean short/medium end-to-end RTF | 0.2040 | 0.1222 |

Fixed-input speedup is **2.944x**. Mean-RTF ratio is **1.670x**, but sampled
output lengths differ; this is not a fixed-output end-to-end comparison.
The previous PDL predictor measurement was 2.351 ms in a separate run.
The non-PDL candidate therefore gives up some speed while passing this bounded
numerical guard. Greedy agreement is 80% and is not used as an acceptance gate.
Listening evaluation and wider numerical coverage remain necessary.

## Reproduce

Append `--deep` for the deep check; omit it for the fixed-work and end-to-end
benchmark:

```sh
CUDA_VISIBLE_DEVICES=GPU-3e620c1c-c243-7353-580e-cde9e5eb4b8a \
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
/root/miniforge3/envs/qwen3-tts/bin/python scripts/test_qwen_fused_predictor.py \
  --size 0.6B --mode nonpdl --deep
```

Artifacts: `0.6B-nonpdl-deep.json`, `0.6B-nonpdl.json`, matching logs, and the
rerun `1.7B-deep.json`/log. Original failing 0.6B PDL artifacts remain preserved,
including `0.6B-deep-original-failure.*`; they are not relabeled passing.
