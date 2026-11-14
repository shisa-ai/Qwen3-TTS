"""Opt-in, single-stream CUDA graph for Qwen3-TTS's fixed code predictor.

Call ``install_predictor_graph(model.model.talker.code_predictor)`` after loading
an eval model with SDPA or eager attention. The first call captures a graph for
its batch size, sampling settings and hidden-state setting; later calls replay
it once. ``handle.uninstall()`` restores the original bound generate method.
No upstream source or global class is modified.

This is a narrow inference API: two prompt embeddings, all remaining code
groups, no EOS/stopping criteria, padding, beams or custom logits processors.
Outputs own their sequence/hidden-state storage. Past KV state is deliberately
not returned: each frame starts a fresh cache. Do not use concurrently, change
weights/device/dtype after capture, or replay from different CUDA streams.
"""
from __future__ import annotations

import math
import time
import types

import torch
from transformers import GenerationConfig
from transformers.cache_utils import StaticCache
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.generation.logits_process import (
    TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper,
)


class _FixedGraph:
    def __init__(self, predictor, embeds, sample, temperature, top_k, top_p, hidden):
        self.predictor = predictor
        self.inputs = torch.empty_like(embeds)
        self.inputs.copy_(embeds)
        self.steps = predictor.config.num_code_groups - 1
        self.cache = StaticCache(config=predictor.config, max_cache_len=self.steps + 1)
        self.positions = [torch.arange(2, device=embeds.device)] + [
            torch.tensor([i + 1], device=embeds.device) for i in range(1, self.steps)
        ]
        keys = torch.arange(self.steps + 1, device=embeds.device)
        self.masks = [
            torch.zeros((1, 1, len(pos), self.steps + 1), device=embeds.device, dtype=embeds.dtype)
            .masked_fill(keys[None, :] > pos[:, None], torch.finfo(embeds.dtype).min)
            for pos in self.positions
        ]
        self.sample, self.hidden = sample, hidden
        self.warpers = []
        if sample:
            if temperature != 1.0:
                self.warpers.append(TemperatureLogitsWarper(temperature))
            if top_k:
                self.warpers.append(TopKLogitsWarper(top_k))
            if top_p < 1.0:
                self.warpers.append(TopPLogitsWarper(top_p))
        self.stream = torch.cuda.current_stream(embeds.device)
        start = time.perf_counter()
        warmup = torch.cuda.Stream(device=embeds.device)
        warmup.wait_stream(self.stream)
        # Initialize library workspaces and cache allocations outside capture.
        with torch.cuda.stream(warmup):
            for _ in range(3):
                self._run()
        self.stream.wait_stream(warmup)
        torch.cuda.synchronize(embeds.device)
        self.graph = torch.cuda.CUDAGraph()
        capture_start = time.perf_counter()
        with torch.cuda.graph(self.graph):
            self._run()
        torch.cuda.synchronize(embeds.device)
        self.capture_seconds = time.perf_counter() - capture_start
        self.setup_seconds = time.perf_counter() - start

    def _run(self):
        # Zeroing is part of the graph, not a Python-side per-frame operation.
        self.cache.reset()
        tokens, hidden = [], []
        x = self.inputs
        for step in range(self.steps):
            if step:
                x = self.predictor.get_input_embeddings()[step - 1](tokens[-1])
            output = self.predictor.model(
                inputs_embeds=self.predictor.small_to_mtp_projection(x),
                past_key_values=self.cache, use_cache=True,
                cache_position=self.positions[step],
                position_ids=self.positions[step].unsqueeze(0),
                attention_mask={"full_attention": self.masks[step]},
                output_hidden_states=self.hidden, output_attentions=False,
                return_dict=True,
            )
            scores = self.predictor.lm_head[step](output.last_hidden_state[:, -1:])[:, -1].float()
            if self.sample:
                for warper in self.warpers:
                    scores = warper(None, scores)
                token = torch.multinomial(scores.softmax(-1), num_samples=1)
            else:
                token = scores.argmax(-1, keepdim=True)
            tokens.append(token)
            if self.hidden:
                hidden.append(output.hidden_states)
        self.sequences = torch.cat(tokens, dim=1)
        self.hidden_states = tuple(hidden) if self.hidden else None

    def replay(self, embeds):
        if torch.cuda.current_stream(embeds.device) != self.stream:
            raise ValueError("Predictor graph must be replayed on its capture caller's CUDA stream")
        self.inputs.copy_(embeds)
        self.graph.replay()
        sequences = self.sequences.clone()
        hidden = None if self.hidden_states is None else tuple(
            tuple(t.clone() for t in step) for step in self.hidden_states
        )
        return sequences, hidden


class PredictorGraph:
    """Instance patch handle; ``graphs`` exposes capture/setup timing by variant."""

    def __init__(self, predictor, max_graphs=8):
        if predictor.training:
            raise ValueError("Predictor must be in eval mode")
        if predictor.config._attn_implementation not in ("sdpa", "eager"):
            raise ValueError("Predictor graph requires sdpa or eager attention")
        if any(t != "full_attention" for t in predictor.config.layer_types):
            raise ValueError("Only full-attention predictor layers are supported")
        if getattr(predictor.model.rotary_emb, "rope_type", "default") != "default":
            raise ValueError("Only default rotary embeddings are supported")
        if not isinstance(max_graphs, int) or max_graphs < 1:
            raise ValueError("max_graphs must be a positive integer")
        self.predictor = predictor
        self.original_generate = predictor.generate
        self.graphs = {}
        self.max_graphs = max_graphs
        self._prior_instance_generate = predictor.__dict__.get("generate")
        # Only library defaults are immutable. Serialize the live config on every
        # call below so replacement and nested mutations remain visible.
        self._generation_defaults = GenerationConfig().to_dict()

    def _default_generation_settings(self):
        return self._generation_defaults

    def uninstall(self):
        if self.predictor.__dict__.get("_predictor_graph") is not self:
            raise RuntimeError("This predictor graph is not installed")
        if self._prior_instance_generate is None:
            del self.predictor.generate
        else:
            self.predictor.generate = self._prior_instance_generate
        del self.predictor._predictor_graph

    @torch.inference_mode()
    def generate(self, *args, **kwargs):
        if torch.is_autocast_enabled('cuda'):
            raise ValueError("Predictor graph does not support CUDA autocast")
        if args:
            raise ValueError("Pass only keyword inputs_embeds to predictor graph generate")
        allowed = {"inputs_embeds", "max_new_tokens", "do_sample", "temperature", "top_k", "top_p",
                   "output_hidden_states", "return_dict_in_generate"}
        unknown = set(kwargs) - allowed
        if unknown:
            raise ValueError(f"Unsupported predictor graph kwargs: {sorted(unknown)}")
        # Reject consequential non-default generation settings even when inherited.
        config = self.predictor.generation_config
        defaults = self._default_generation_settings()
        supported = allowed | {"bos_token_id", "pad_token_id", "_from_model_config", "transformers_version"}
        for name, value in config.to_dict().items():
            if name not in supported and value != defaults.get(name):
                raise ValueError(f"Unsupported predictor generation_config setting: {name}={value!r}")
        if self.predictor.training:
            raise ValueError("Predictor must remain in eval mode")
        embeds = kwargs.get("inputs_embeds")
        weight = self.predictor.get_input_embeddings()[0].weight
        if (not isinstance(embeds, torch.Tensor) or embeds.ndim != 3 or
                embeds.shape[1:] != (2, weight.shape[1]) or embeds.shape[0] < 1 or
                embeds.device.type != "cuda" or embeds.device != weight.device or embeds.dtype != weight.dtype):
            raise ValueError("inputs_embeds must be CUDA [batch, 2, embedding_dim] with the model device/dtype")
        steps = self.predictor.config.num_code_groups - 1
        if kwargs.get("max_new_tokens", config.max_new_tokens) != steps:
            raise ValueError(f"max_new_tokens must be exactly {steps}")
        def setting(name):
            return kwargs[name] if name in kwargs else getattr(config, name)
        sample, temperature, top_k, top_p = (setting(n) for n in ("do_sample", "temperature", "top_k", "top_p"))
        hidden, return_dict = (setting(n) for n in ("output_hidden_states", "return_dict_in_generate"))
        if not all(isinstance(v, bool) for v in (sample, hidden, return_dict)):
            raise ValueError("do_sample/output_hidden_states/return_dict_in_generate must be booleans")
        if sample:
            if not isinstance(temperature, (float, int)) or not math.isfinite(temperature) or temperature <= 0:
                raise ValueError("temperature must be finite and positive")
            if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0:
                raise ValueError("top_k must be a nonnegative integer")
            if not isinstance(top_p, (float, int)) or not math.isfinite(top_p) or not 0 < top_p <= 1:
                raise ValueError("top_p must be in (0, 1]")
        else:
            # HF ignores sampling warpers in greedy mode.
            temperature, top_k, top_p = 1.0, 0, 1.0
        key = (tuple(embeds.shape), embeds.device, embeds.dtype, sample, float(temperature), top_k, float(top_p), hidden)
        if key not in self.graphs:
            if len(self.graphs) >= self.max_graphs:
                raise ValueError("Predictor graph variant limit reached; uninstall/reinstall to release variants")
            with torch.cuda.device(embeds.device):
                # Capture/warmup must not consume the caller's random draws.
                with torch.random.fork_rng(devices=[embeds.device.index]):
                    self.graphs[key] = _FixedGraph(self.predictor, embeds, sample, float(temperature), top_k, float(top_p), hidden)
        sequences, states = self.graphs[key].replay(embeds)
        if return_dict:
            return GenerateDecoderOnlyOutput(sequences=sequences, hidden_states=states)
        return sequences


def install_predictor_graph(code_predictor, *, max_graphs=8):
    """Patch this predictor only; return a handle with ``uninstall()``.

    Captures lazily. Supports standard temperature -> top-k -> top-p warpers,
    using PyTorch's graph-aware CUDA random generator (not bitwise HF sampling).
    The returned output has sequences and optional hidden_states, but no cache.
    """
    if hasattr(code_predictor, "_predictor_graph"):
        raise ValueError("Predictor graph is already installed")
    handle = PredictorGraph(code_predictor, max_graphs=max_graphs)
    def generate(_instance, *args, **kwargs):
        return handle.generate(*args, **kwargs)
    code_predictor.generate = types.MethodType(generate, code_predictor)
    code_predictor._predictor_graph = handle
    return handle
