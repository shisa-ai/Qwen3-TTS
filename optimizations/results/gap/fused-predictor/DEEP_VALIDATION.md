# Incremental teacher-forced validation

**Do not mark the candidate fully accepted: 0.6B exceeds the unchanged 0.04
per-layer relative-error guard at one tested point.** The 1.7B deep check passes.
No implementation defect was established; the failing result remains recorded
and causes exit status 1 rather than being hidden by a wider threshold.

The added `--deep` mode in `scripts/test_qwen_fused_predictor.py` collected real
predictor inputs and evaluated four distinct frames in order `[0,1,3,2,0]`.
Every frame uses all 15 predictor steps and a common token history selected
from HF FP32 logits. Comparators are HF FP32 dynamic cache, HF BF16 dynamic
cache, HF BF16 static cache, and the actual fused PDL kernels. FP32 uses a full
FP32 predictor copy including projection and rotary evaluation.

Normalized errors below are RMSE divided by reference RMS, not percentages.

| Metric versus FP32 | 0.6B fused | 0.6B BF16 | 1.7B fused | 1.7B BF16 |
|---|---:|---:|---:|---:|
| Maximum logits error | 0.017989 | 0.019978 | 0.019394 | 0.035806 |
| Mean logits error | 0.009713 | 0.009231 | 0.007906 | 0.009206 |
| Maximum per-layer output error | **0.043819** | 0.040692 | 0.038630 | 0.045005 |
| Maximum all-historical-slot KV error | 0.015356 | 0.015227 | 0.014326 | 0.018216 |
| Maximum newly written KV error | 0.018019 | 0.019336 | 0.019894 | 0.025040 |

The failing 0.6B point is real-input frame 3, codebook step index 13, layer
index 4. Fused error is 0.043819; BF16 at the **same point** is 0.033834.
The final logits at that point remain 0.014427 fused versus 0.018775 BF16.
Thus a local residual-output discrepancy exists beyond the BF16 baseline at
that point, even though the output logits are closer to FP32. It needs an
explicit numerical acceptance decision or further attribution, not a silent
threshold adjustment.

No monotonic stepwise accumulation is evident in this bounded sample. Mean
fused/BF16 logits errors at step indices 0, 7, 14 are:

- 0.6B: 0.012115/0.012953, 0.007825/0.006430, 0.007749/0.007843.
- 1.7B: 0.007245/0.007510, 0.006558/0.008979, 0.008887/0.013034.

Per-step FP32 top-two logit margins and fused maximum absolute errors are in
the JSON. Margins range from 0.005079 to 10.362720 for 0.6B, and 0.006449 to
15.113073 for 1.7B. Small margins explain why exact greedy-token parity is not
a sufficient numerical criterion.

## Semantic checks passed on both sizes

- BF16 dynamic/static logits match exactly at all tested steps.
- Per-layer outputs and newly written/all historical K/V slots are recorded
  against FP32 at every step. Existing fused historical slots remain bitwise
  unchanged when adding the next position.
- Future cache slots poisoned with +123 and -321 produce identical logits
  and valid K/V. A repeated frame after other frames reproduces its results.
- Changing only prefill row 1 leaves every layer's row-0 output and row-0 K/V
  unchanged, demonstrating causal isolation of that future row.
- Fixed-logit, fixed-uniform fused sampling matches a PyTorch Gumbel reference
  for temperatures 0.5/1.5, k=0/1/50/vocabulary, cutoff ties and signed zeros.
- Actual fused execution receives inverse temperatures 2.0 and 2/3.
- CUDA RNG state advances; reseeding reproduces outputs; first-use capture
  preserves replay-equivalent random draws; alternating graph variants reseed
  consistently.
- Top-p fallback exactly matches direct strict generation with the same RNG.
  Hidden-state and batch-two fallback outputs also match strict generation,
  and retain independent output/hidden-state storage after subsequent calls.
- Uninstall restores the exact preexisting instance `generate`, or restores
  class lookup when no instance method existed; markers are removed.
- Installing fast decode with the fused predictor, then uninstalling fast
  decode before the predictor, restores its state successfully.

## Commands and artifacts

Run from `/root/tts` with each size:

```sh
CUDA_VISIBLE_DEVICES=GPU-3e620c1c-c243-7353-580e-cde9e5eb4b8a \
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
/root/miniforge3/envs/qwen3-tts/bin/python scripts/test_qwen_fused_predictor.py \
  --size 0.6B --deep
```

Final exact outcomes: **0.6B exit 1**, **1.7B exit 0**. Syntax compilation exits
0. Numerical guard failures are accumulated to finish independent checks;
they still cause a final assertion. `measurements_complete=true` means all
measurements ran; `complete=false` on 0.6B means acceptance did not pass.

Artifacts: `0.6B-deep.json`, `1.7B-deep.json` and matching logs. This follow-up
changed only the owned test and report artifacts, not production kernels or
adapter logic. GPU2 only. Listening, broader input coverage and sm120 remain
untested.
