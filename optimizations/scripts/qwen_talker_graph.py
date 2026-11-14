"""Opt-in, single-request CUDA graph for Qwen3-TTS talker.model (not predictor)."""
from __future__ import annotations

import inspect
import time
import torch
from transformers.cache_utils import DynamicCache, StaticCache
from transformers.modeling_outputs import BaseModelOutputWithPast


class _DecodeCache(StaticCache):
    # Generation must retain its ordinary 2-D padding mask outside the transformer.
    is_compileable = False

    def get_seq_length(self, layer_idx=0):
        return self.length


class TalkerGraph:
    """Instance adapter. Serial use only; reset invalidates the active request cache.

    Prefill runs upstream with DynamicCache, then copies into fixed storage. Decode
    uses upstream forward with an explicit additive causal/padding mask. Returned
    tensors are cloned because generation retains hidden states across replays.
    """

    def __init__(self, talker, max_cache_len=1024):
        self.model = talker.model
        c = self.model.config
        if self.model.__class__.__name__ != "Qwen3TTSTalkerModel":
            raise ValueError("Expected Qwen3-TTS talker.model")
        if c._attn_implementation != "sdpa" or c.sliding_window is not None:
            raise ValueError("TalkerGraph requires SDPA and full attention")
        if getattr(self.model.rotary_emb, "rope_type", "default") != "default":
            raise ValueError("TalkerGraph requires default rotary embeddings")
        if max_cache_len < 2 or next(self.model.parameters()).device.type != "cuda":
            raise ValueError("TalkerGraph requires CUDA and max_cache_len >= 2")
        if hasattr(self.model, "_talker_graph"):
            raise ValueError("TalkerGraph is already installed")
        self.device = next(self.model.parameters()).device
        if self.device.index != torch.cuda.current_device() or any(
            p.device != self.device for p in self.model.parameters()
        ):
            raise ValueError("TalkerGraph requires all model parameters on the current CUDA device")
        self.owner_stream = torch.cuda.current_stream(self.device)
        self.original = self.model.forward
        self.signature = inspect.signature(self.original)
        self.capacity = max_cache_len
        self.cache = _DecodeCache(config=c, max_cache_len=max_cache_len)
        self.cache.length = 0
        self.active = False
        self.graph = None
        self.capture_seconds = []
        self.replays = 0
        self.model.forward = self.forward
        self.model._talker_graph = self

    def _check_owner(self):
        if torch.cuda.current_device() != self.device.index:
            raise ValueError("TalkerGraph requires its owning CUDA device to be current")
        if torch.cuda.current_stream(self.device) != self.owner_stream:
            raise ValueError("TalkerGraph requires its owning CUDA stream")
        if next(self.model.parameters()).device != self.device:
            raise ValueError("TalkerGraph model device cannot change after installation")

    def reset(self):
        self._check_owner()
        self.cache.reset()
        self.cache.length = 0
        self.active = False

    def uninstall(self):
        self._check_owner()
        self.model.forward = self.original
        del self.model._talker_graph
        self.reset()

    def _run(self):
        return self.original(inputs_embeds=self.embeds, position_ids=self.positions,
                             attention_mask=self.mask, past_key_values=self.cache,
                             cache_position=self.position, use_cache=True,
                             output_hidden_states=True, output_attentions=False,
                             return_dict=True)

    def _capture(self):
        torch.cuda.synchronize()
        start = time.perf_counter()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._run()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = self._run()
        torch.cuda.synchronize()
        self.capture_seconds.append(time.perf_counter() - start)

    def forward(self, *args, **kwargs):
        self._check_owner()
        bound = self.signature.bind(*args, **kwargs)
        a = dict(bound.arguments)
        extras = a.pop("flash_attn_kwargs", {})
        if set(extras) - {"return_dict"} or extras.get("return_dict", True) is False:
            raise ValueError(f"Unsupported talker transformer arguments: {extras}")
        if self.model.training or torch.is_grad_enabled() or torch.is_autocast_enabled():
            raise ValueError("TalkerGraph requires eval(), no_grad/inference_mode and no autocast")
        output_attentions = a.get("output_attentions")
        use_cache = a.get("use_cache")
        if output_attentions is None:
            output_attentions = self.model.config.output_attentions
        if use_cache is None:
            use_cache = self.model.config.use_cache
        if output_attentions or not use_cache:
            raise ValueError("TalkerGraph requires use_cache and no attention outputs")
        x = a.get("inputs_embeds")
        if x is None or a.get("input_ids") is not None or x.shape[0] != 1:
            raise ValueError("TalkerGraph requires batch-one inputs_embeds")
        for name in ("inputs_embeds", "position_ids", "cache_position", "attention_mask"):
            value = a.get(name)
            if value is not None and value.device != self.device:
                raise ValueError(f"TalkerGraph {name} must be on the owning CUDA device")
        cp = a.get("cache_position")
        past = a.get("past_key_values")
        if cp is None:
            raise ValueError("TalkerGraph requires explicit cache_position")
        start = int(cp[0].item())
        end = start + x.shape[1]
        if start < 0 or end > self.capacity:
            raise ValueError("TalkerGraph cache capacity exceeded")
        if not torch.equal(cp, torch.arange(start, end, device=cp.device)):
            raise ValueError("TalkerGraph requires contiguous cache positions")
        if start == 0:
            if past is not None and (not isinstance(past, DynamicCache) or past.get_seq_length() != 0):
                raise ValueError("Prefill requires an empty DynamicCache")
            self.reset()
            out = self.original(*args, **kwargs)
            for dst, src in zip(self.cache.layers, out.past_key_values.layers):
                if not dst.is_initialized:
                    dst.lazy_initialization(src.keys)
                dst.keys[:, :, :end].copy_(src.keys)
                dst.values[:, :, :end].copy_(src.values)
            self.cache.length = end
            self.active = True
            out.past_key_values = self.cache
            return out
        if x.shape[1] != 1 or past is not self.cache or not self.active or start != self.cache.length:
            raise ValueError("Decode requires the active serial cache and one contiguous token")
        positions = a.get("position_ids")
        if positions is None:
            positions = cp.view(1, 1, 1).expand(3, 1, 1)
        elif positions.ndim == 2:
            positions = positions.unsqueeze(0).expand(3, -1, -1)
        if positions.shape == (4, 1, 1):
            positions = positions[1:]
        if positions.shape != (3, 1, 1):
            raise ValueError("Unsupported rotary position shape")
        padding = a.get("attention_mask")
        if padding is not None and (padding.ndim != 2 or padding.shape != (1, end)):
            raise ValueError("Decode requires a 2-D full-length padding mask")
        if self.graph is None:
            self.embeds = torch.empty_like(x)
            self.positions = torch.empty_like(positions)
            self.position = torch.empty_like(cp)
            self.mask = torch.empty((1, 1, 1, self.capacity), device=x.device, dtype=x.dtype)
        if x.dtype != self.embeds.dtype or x.device != self.embeds.device:
            raise ValueError("Model/input dtype and device cannot change after capture")
        self.embeds.copy_(x)
        self.positions.copy_(positions)
        self.position.copy_(cp)
        self.mask.fill_(float("-inf"))
        self.mask[..., :end].zero_()
        if padding is not None:
            self.mask[..., :end].masked_fill_(padding[:, None, None, :] == 0, float("-inf"))
        if self.graph is None:
            self._capture()
        self.graph.replay()
        self.replays += 1
        self.cache.length = end
        hidden = a.get("output_hidden_states")
        if hidden is None:
            hidden = self.model.config.output_hidden_states
        return BaseModelOutputWithPast(
            last_hidden_state=self.output.last_hidden_state.clone(), past_key_values=self.cache,
            hidden_states=tuple(t.clone() for t in self.output.hidden_states) if hidden else None)


def install_talker_graph(talker, max_cache_len=1024):
    """Install on one talker instance; return its reset/uninstall controller."""
    return TalkerGraph(talker, max_cache_len)
