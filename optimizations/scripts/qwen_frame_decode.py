"""Opt-in request-owned position fast path; HF generation/sampling stays intact.

This intermediate does not capture a whole frame. It removes the strict talker
transformer's per-token position readback only inside the audited HF generate
call. Direct forwards retain the strict adapter. Serial inference only.
"""
from __future__ import annotations

import hashlib
import inspect
import torch
from transformers.modeling_outputs import BaseModelOutputWithPast


class FrameDecode:
    def __init__(self, talker):
        if hasattr(talker, '_frame_decode'):
            raise ValueError('Frame decode already installed')
        from transformers.generation.utils import GenerationMixin
        expected = {
            'generate': 'ab717bcccff3c98cb56f2a12919e39cec16b7b40ae0421c8f5051e3bced6edd4',
            '_sample': '676dcf123496b831d13024402537120a5a5dcb16133a342705892ac8bd2291d6',
            'prepare_inputs_for_generation': 'fe8e0716319dff2fe17060e55a333ceca665fbcde78d0ced4c4b3912fa52e0c1',
            '_update_model_kwargs_for_generation': '8dd8daed97a17b3ff7b2ff7e73e3a7139dfdc011852d1970ad3e4b6a51ab142b',
        }
        if talker.__class__.__name__ != 'Qwen3TTSTalkerForConditionalGeneration':
            raise ValueError('Expected pinned Qwen3-TTS talker')
        for name, digest in expected.items():
            if hashlib.sha256(inspect.getsource(getattr(GenerationMixin, name)).encode()).hexdigest() != digest:
                raise ValueError('Unsupported transformers generation source')
        for name in ('generate', '_sample', 'prepare_inputs_for_generation'):
            if getattr(talker, name).__func__ is not getattr(GenerationMixin, name):
                raise ValueError('Custom generation methods require strict execution')
        if hashlib.sha256(inspect.getsource(talker._update_model_kwargs_for_generation).encode()).hexdigest() != 'e71e0e65d3c82421e39f0bf59e4d1a1f6aaf6de5373449fb0a4665c5d2025427':
            raise ValueError('Unsupported talker generation update')
        self.talker = talker
        self.graph = talker.model._talker_graph
        self.original_generate = talker.generate
        self.prior_generate = talker.__dict__.get('generate')
        self.original_forward = talker.model.forward
        self.signature = inspect.signature(self.graph.original)
        self.active = False
        self.fast_calls = 0
        self.fallback_calls = 0
        self.generate_wrapper = self.generate
        self.forward_wrapper = self.forward
        talker.generate = self.generate_wrapper
        talker.model.forward = self.forward_wrapper
        talker._frame_decode = self

    def generate(self, *args, **kwargs):
        if self.active:
            raise RuntimeError('Frame decode is serial and non-reentrant')
        x = kwargs.get('inputs_embeds')
        # Only the normal embedding-only, batch-one HF generation contract.
        # Any external hook/custom position/cache machinery stays strict.
        allowed = {'inputs_embeds', 'attention_mask', 'trailing_text_hidden',
                   'tts_pad_embed', 'do_sample', 'top_k', 'top_p', 'temperature',
                   'subtalker_dosample', 'subtalker_top_k', 'subtalker_top_p',
                   'subtalker_temperature', 'repetition_penalty', 'max_new_tokens',
                   'min_new_tokens', 'eos_token_id', 'pad_token_id', 'bos_token_id',
                   'suppress_tokens', 'output_hidden_states', 'return_dict_in_generate'}
        config = self.talker.generation_config
        # A changed config can select a different cache/decoding algorithm.
        from transformers import GenerationConfig
        defaults = GenerationConfig().to_dict()
        permitted = allowed | {'_from_model_config', 'transformers_version'}
        changed = {k for k, v in config.to_dict().items() if v != defaults.get(k)}
        supported = (not any(m._forward_hooks or m._forward_pre_hooks for m in self.talker.modules()) and
                     not args and x is not None and x.ndim == 3 and
                     x.shape[0] == 1 and x.shape[1] > 1 and
                     not (set(kwargs) - allowed) and not (changed - permitted))
        if not supported:
            self.fallback_calls += 1
            return self.original_generate(*args, **kwargs)
        self.graph._check_owner()
        if torch.is_grad_enabled() or self.talker.training:
            raise ValueError('Frame decode requires eval and no_grad/inference_mode')
        self.active = True
        self.next_position = 0
        self.graph.reset()
        try:
            return self.original_generate(*args, **kwargs)
        finally:
            self.active = False

    def forward(self, *args, **kwargs):
        if not self.active:
            return self.original_forward(*args, **kwargs)
        g = self.graph
        g._check_owner()
        a = dict(self.signature.bind(*args, **kwargs).arguments)
        x = a.get('inputs_embeds')
        if self.next_position == 0:
            out = self.original_forward(*args, **kwargs)
            self.next_position = x.shape[1]
            return out
        start = self.next_position
        end = start + 1
        if end > g.capacity:
            raise ValueError('TalkerGraph cache capacity exceeded')
        if (x.shape != (1, 1, g.model.config.hidden_size) or
                a.get('past_key_values') is not g.cache or not g.active or
                g.cache.length != start):
            raise ValueError('Frame decode requires its active contiguous request cache')
        # Only HF owns cache_position under this restricted request contract.
        # Position IDs retain upstream rope_deltas/mRoPE and padding semantics.
        positions = a.get('position_ids')
        if positions is None:
            positions = torch.full((3, 1, 1), start, device=x.device, dtype=torch.long)
        elif positions.ndim == 2:
            positions = positions.unsqueeze(0).expand(3, -1, -1)
        if positions.shape == (4, 1, 1):
            positions = positions[1:]
        if positions.shape != (3, 1, 1):
            raise ValueError('Unsupported rotary position shape')
        padding = a.get('attention_mask')
        if padding is not None and padding.shape != (1, end):
            raise ValueError('Expected full-length padding mask')
        if g.graph is None:
            # First decode uses all strict checks and initializes/captures storage.
            out = self.original_forward(*args, **kwargs)
        else:
            if x.dtype != g.embeds.dtype or x.device != g.embeds.device:
                raise ValueError('Model/input dtype and device cannot change after capture')
            g.embeds.copy_(x)
            g.positions.copy_(positions)
            g.position.fill_(start)
            g.mask.fill_(float('-inf'))
            g.mask[..., :end].zero_()
            if padding is not None:
                g.mask[..., :end].masked_fill_(padding[:, None, None, :] == 0, float('-inf'))
            g.graph.replay()
            g.replays += 1
            g.cache.length = end
            hidden = a.get('output_hidden_states')
            if hidden is None:
                hidden = g.model.config.output_hidden_states
            out = BaseModelOutputWithPast(
                last_hidden_state=g.output.last_hidden_state.clone(),
                past_key_values=g.cache,
                hidden_states=tuple(t.clone() for t in g.output.hidden_states) if hidden else None)
            self.fast_calls += 1
        self.next_position = end
        return out

    def uninstall(self):
        self.graph._check_owner()
        if self.active:
            raise RuntimeError('Cannot uninstall during generation')
        if (self.talker.generate is not self.generate_wrapper or
                self.talker.model.forward is not self.forward_wrapper):
            raise RuntimeError('Uninstall frame decode before replacing its wrappers')
        if self.prior_generate is None:
            del self.talker.generate
        else:
            self.talker.generate = self.prior_generate
        self.talker.model.forward = self.original_forward
        del self.talker._frame_decode


def install_frame_decode(talker):
    return FrameDecode(talker)
