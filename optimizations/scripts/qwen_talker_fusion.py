"""Opt-in inference-only talker fusion. Install before TalkerGraph; uninstall it first.

Packed projections support bias-free BF16 SDPA inference only. Standalone dynamic
caches are supported; other attention backends and FP32 are rejected.
Projection snapshots require frozen weights/device/dtype until uninstall. No shared
scratch buffers are used; graph stream/cache ownership remains TalkerGraph's job.
Original parameter modules and state_dict keys are retained. Packed weights are
additional, unregistered inference snapshots, released on uninstall.
"""
from __future__ import annotations
import types
import torch
import torch.nn.functional as F
import triton
from qwen_fused_norm import _norm
from qwen_tts.core.models.modeling_qwen3_tts import (
    apply_multimodal_rotary_pos_emb, eager_attention_forward, ALL_ATTENTION_FUNCTIONS,
)


class TalkerFusion:
    def __init__(self, talker, *, norms=True, projections=True):
        self.entries = []
        self.snapshots = []
        try:
            self._install(talker, norms=norms, projections=projections)
        except BaseException:
            for module, previous, _ in reversed(self.entries):
                if previous is None:
                    del module.forward
                else:
                    module.forward = previous
            self.entries.clear()
            self.snapshots.clear()
            raise

    def _install(self, talker, *, norms, projections):
        self.model = talker.model
        if self.model.__class__.__name__ != 'Qwen3TTSTalkerModel':
            raise ValueError('Expected Qwen3TTSTalkerModel')
        if hasattr(self.model, '_talker_graph') or hasattr(self.model, '_talker_fusion'):
            raise ValueError('Install fusion once, before talker graph')
        self.device = next(self.model.parameters()).device
        if self.model.training or self.device.type != 'cuda':
            raise ValueError('Requires eval CUDA talker')
        dtype = next(self.model.parameters()).dtype
        if dtype != torch.bfloat16 or any(p.device != self.device or p.dtype != dtype for p in self.model.parameters()):
            raise ValueError('Requires uniform CUDA BF16 weights')
        if self.model.config._attn_implementation != 'sdpa':
            raise ValueError('Only validated SDPA attention is supported')
        if projections and any(m.bias is not None for layer in self.model.layers for m in
                (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj,
                 layer.mlp.gate_proj, layer.mlp.up_proj)):
            raise ValueError('Packed projections currently require bias-free weights')
        self.owner = torch.cuda.current_stream(self.device)
        self.entries = []
        self.snapshots = []
        def patch(module, forward):
            previous = module.__dict__.get('forward')
            installed = types.MethodType(forward, module)
            module.forward = installed
            self.entries.append((module, previous, installed))
        if norms:
            for module in self.model.modules():
                if module.__class__.__name__ != 'Qwen3TTSRMSNorm':
                    continue
                original = module.forward
                def forward(m, x, original=original):
                    self._check(m, x)
                    width = m.weight.numel()
                    if not x.is_contiguous() or not m.weight.is_contiguous():
                        return original(x)
                    y = torch.empty_like(x)
                    _norm[(x.numel() // width,)](x, m.weight, y, width, m.variance_epsilon,
                        triton.next_power_of_2(width), num_warps=4, enable_fp_fusion=False)
                    return y
                patch(module, forward)
        if projections:
            for layer in self.model.layers:
                attn = layer.self_attn
                weights, bias, widths = self._pack([attn.q_proj, attn.k_proj, attn.v_proj])
                def attention(m, hidden_states, position_embeddings, attention_mask,
                              past_key_values=None, cache_position=None,
                              weights=weights, bias=bias, widths=widths, **kwargs):
                    self._check(m.q_proj, hidden_states)
                    shape = (*hidden_states.shape[:-1], -1, m.head_dim)
                    q, k, v = F.linear(hidden_states, weights, bias).split(widths, dim=-1)
                    q = m.q_norm(q.reshape(shape)).transpose(1, 2)
                    k = m.k_norm(k.reshape(shape)).transpose(1, 2)
                    v = v.reshape(shape).transpose(1, 2)
                    cos, sin = position_embeddings
                    q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin,
                        m.rope_scaling['mrope_section'], m.rope_scaling['interleaved'])
                    if past_key_values is not None:
                        k, v = past_key_values.update(k, v, m.layer_idx,
                            {'sin': sin, 'cos': cos, 'cache_position': cache_position})
                    interface = eager_attention_forward if m.config._attn_implementation == 'eager' else ALL_ATTENTION_FUNCTIONS[m.config._attn_implementation]
                    output, scores = interface(m, q, k, v, attention_mask,
                        dropout=0.0, scaling=m.scaling, sliding_window=m.sliding_window, **kwargs)
                    return m.o_proj(output.reshape(*hidden_states.shape[:-1], -1).contiguous()), scores
                patch(attn, attention)
                mlp = layer.mlp
                weights, bias, widths = self._pack([mlp.gate_proj, mlp.up_proj])
                def feedforward(m, x, weights=weights, bias=bias, widths=widths):
                    self._check(m.gate_proj, x)
                    gate, up = F.linear(x, weights, bias).split(widths, dim=-1)
                    return m.down_proj(m.act_fn(gate) * up)
                patch(mlp, feedforward)
        # Complete asynchronous snapshot copies before publishing installation.
        # This permits standalone use and graph warmup on other streams safely.
        self.owner.synchronize()
        self.model._talker_fusion = self

    def _pack(self, modules):
        weight = torch.cat([m.weight.detach() for m in modules], dim=0).contiguous()
        bias = None
        if any(m.bias is not None for m in modules):
            bias = torch.cat([m.bias.detach() if m.bias is not None else m.weight.new_zeros(m.out_features) for m in modules])
        self.snapshots.append((weight, bias))
        return weight, bias, tuple(m.out_features for m in modules)

    def _check(self, module, x):
        if self.model.config._attn_implementation != 'sdpa':
            raise ValueError('Only validated SDPA attention is supported')
        if self.model.training or torch.is_grad_enabled() or torch.is_autocast_enabled('cuda'):
            raise ValueError('Fusion requires eval, inference/no_grad, no autocast')
        if x.device != self.device or module.weight.device != self.device or x.dtype != module.weight.dtype or x.dtype != torch.bfloat16:
            raise ValueError('Fusion requires matching CUDA BF16 inputs and frozen weights')

    def uninstall(self):
        if torch.cuda.current_stream(self.device) != self.owner:
            raise ValueError('Uninstall requires owning stream')
        if hasattr(self.model, '_talker_graph'):
            raise ValueError('Uninstall talker graph before fusion')
        if getattr(self.model, '_talker_fusion', None) is not self or any(m.forward != installed for m, _, installed in self.entries):
            raise RuntimeError('Fusion ownership changed')
        for module, previous, _ in self.entries:
            if previous is None:
                del module.forward
            else:
                module.forward = previous
        self.entries.clear()
        self.snapshots.clear()
        del self.model._talker_fusion


def install_talker_fusion(talker, *, norms=True, projections=True):
    return TalkerFusion(talker, norms=norms, projections=projections)
