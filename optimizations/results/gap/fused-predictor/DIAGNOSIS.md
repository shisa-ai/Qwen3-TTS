# Diagnosis of the isolated 0.6B guard failure

The evidence points to **propagated BF16/folded-computation error amplified by
layer 4**, rather than an isolated faulty PDL layer kernel. The original 0.04
acceptance guard still fails; no threshold or production implementation changed.

Original failing JSON/log were preserved as `0.6B-deep-original-failure.*`.
The targeted diagnostic completed with exit 0; that means diagnostic execution
completed, **not** that the original acceptance guard passed.

## Controlled local comparisons

At real-input frame 3, predictor step index 13, layer index 4, each row below
uses identical source activation/history for its HF FP32, HF BF16, PDL, and
non-PDL evaluations. BF16 execution necessarily rounds the FP32 source and
cache to BF16; the FP32 comparator retains FP32 values. Accumulated sources
are promoted without inventing precision. Error is RMSE/reference RMS.

| Source activation and KV history | FP32 layer output vs original full-FP32 output | HF BF16 vs same-source FP32 | PDL vs same-source FP32 | Non-PDL vs same-source FP32 | PDL vs original full-FP32 output |
|---|---:|---:|---:|---:|---:|
| Original FP32 local source | 0 | 0.009874 | 0.010085 | 0.009588 | 0.010085 |
| Accumulated BF16 source | 0.031392 | 0.007414 | 0.005641 | 0.006543 | 0.032928 |
| Accumulated fused source | **0.042702** | 0.006587 | 0.006328 | 0.006973 | **0.043819** |

The fused-source activation already differs from the original FP32 activation
by 0.019143 at this layer's input. Running the layer **in FP32** on that
accumulated source/history produces 0.042702 global error, already above the
0.04 guard. The actual PDL result is 0.043819. On the clean FP32 local source,
all three reduced-precision implementations have about 0.01 local error.

This separates propagation from the local kernel: switching only this layer
to FP32 does not remove the violation. PDL and non-PDL local errors are of
similar magnitude; PDL is not systematically worse than HF BF16 here. The
experiment does not prove every preceding kernel correct, but it provides no
specific defect to repair at the failing layer.

## Projection folding isolation

On identical FP32 layer input, QKV projection relative errors versus unfused
FP32 normalization/projection were:

- FP32-folded weights, FP32 multiplication: **1.286685e-7**.
- BF16-folded weights and BF16 multiplication: **0.002038**.
- Unfolded projection with HF normalization cast ordering and BF16 GEMM:
  **0.002184**.

Thus folding is algebraically consistent in FP32; its BF16 error is comparable
to the HF cast-ordered path in this local case. This check isolates QKV, not
every gate/up projection or all previous-layer rounding contributions.

## Decision

No minimal precision-preserving fix is justified by this evidence. A change
to this layer alone would target the symptom rather than the propagated error.
No production code was changed and no new performance claim is made.

Recommend **strict predictor fallback for 0.6B when the 0.04 guard is a hard
acceptance requirement**. Otherwise, retaining 0.6B fusion must be an explicit
experimental acceptance of the recorded failure, with listening and broader
numerical coverage still outstanding. 1.7B retains its prior passing deep-test
result, not a newly measured result. The original 0.6B deep test remains failed.

Reproduction (GPU2 only):

```sh
CUDA_VISIBLE_DEVICES=GPU-3e620c1c-c243-7353-580e-cde9e5eb4b8a \
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
/root/miniforge3/envs/qwen3-tts/bin/python scripts/test_qwen_fused_predictor.py \
  --size 0.6B --deep --diagnose
```

Artifacts: `0.6B-diagnosis.json`, `0.6B-diagnosis.log`, and diagnostic progress
JSON. The diagnostic mode stops at the targeted point and does not overwrite
the final deep-test artifacts.
