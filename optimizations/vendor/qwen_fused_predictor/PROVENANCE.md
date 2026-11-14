# Fused predictor provenance

`kernels.py` is derived from
`vllm-omni/vllm_omni/model_executor/models/qwen3_tts/fused_code_predictor.py`
at local checkout commit `5917730a4b02aedb44d374173cdc1b10484ccd52`.
The copied source SHA-256 is
`d033b8ca98a358993c6449ed252e0cbadd17776620e239d8e9b2653b6e39744e`.
This hash identifies the actual local source, independently of checkout changes.

The source carries Apache-2.0 and vLLM-Omni contributor copyright notices,
retained in the copy. `LICENSE` is the upstream Apache-2.0 license.

Local changes:
- Import Triton directly, without importing vLLM or its platform registry.
- Detect CUDA programmatic dependent launch using PyTorch and Triton.
- Add explicit `pdl`/`nonpdl` execution selection; default PDL behavior remains
  unchanged. Non-PDL uses the existing cuBLAS path even on supported hardware.
- Apply sampling temperature in FP32, matching the HF score precision. Find
  the top-k cutoff on unscaled BF16 logits; positive temperature preserves
  their ordering. This avoids introducing additional BF16 temperature ties.

The adapter in `scripts/qwen_fused_predictor.py` builds a separate structural
view of HF weights, concatenating Q/K/V and gate/up projections. It does not
modify the original model weights. The source's folded normalization and
embedding/projection tables change floating-point rounding.
