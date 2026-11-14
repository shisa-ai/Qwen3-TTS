"""Opt-in BF16 fused predictor, batch one, with strict graph fallback.

Weight folding changes rounding; sampled sequences need not match HF. No
hidden states are fabricated: requests for them use the strict implementation.
Weights must remain unchanged while installed. One caller CUDA stream only.
"""
from pathlib import Path
import importlib.util
import math
import types
import time

import torch
from transformers.generation.utils import GenerateDecoderOnlyOutput
from qwen_predictor_graph import PredictorGraph

_spec = importlib.util.spec_from_file_location("_qwen_fused_kernels", Path(__file__).resolve().parents[1] / "vendor/qwen_fused_predictor/kernels.py")
_k = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_k)


@torch.inference_mode()
def _mapped(p):
    """Create a read-only structural view; never patch HF modules."""
    ns = types.SimpleNamespace
    cfg, m = p.config, p.model
    layers = []
    for l in m.layers:
        a = l.self_attn
        for proj in (a.q_proj, a.k_proj, a.v_proj, a.o_proj, l.mlp.gate_proj, l.mlp.up_proj, l.mlp.down_proj):
            if proj.bias is not None:
                raise ValueError("Fused predictor requires bias-free transformer projections")
        att = ns(num_heads=cfg.num_attention_heads, num_kv_heads=cfg.num_key_value_heads,
                 head_dim=a.head_dim, scaling=a.scaling, max_seq=cfg.num_code_groups,
                 qkv_proj=ns(weight=torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]), bias=None),
                 q_norm=a.q_norm, k_norm=a.k_norm, o_proj=a.o_proj)
        mlp = ns(gate_up_proj=ns(weight=torch.cat([l.mlp.gate_proj.weight, l.mlp.up_proj.weight])), down_proj=l.mlp.down_proj)
        layers.append(ns(self_attn=att, mlp=mlp, input_layernorm=l.input_layernorm,
                         post_attention_layernorm=l.post_attention_layernorm))
    w = next(m.parameters())
    pos = torch.arange(cfg.num_code_groups, device=w.device).unsqueeze(0)
    cos, sin = m.rotary_emb(torch.empty(1, cfg.num_code_groups, cfg.hidden_size, device=w.device, dtype=w.dtype), pos)
    view = ns(layers=layers, norm=m.norm, codec_embedding=m.codec_embedding,
              rotary_emb=ns(cos_cached=cos[0], sin_cached=sin[0]), parameters=m.parameters)
    return ns(model=view, config=cfg, small_to_mtp_projection=p.small_to_mtp_projection, lm_head=p.lm_head)


class _Graph:
    def __init__(self, engine, embeds, sample, temperature, top_k):
        self.engine = engine
        self.inputs = embeds.clone()
        self.stream = torch.cuda.current_stream(embeds.device)
        self.uniforms = torch.empty(embeds.shape[0], engine.groups - 1, engine.vocab, device=embeds.device)
        self.code = torch.zeros(embeds.shape[0], device=embeds.device, dtype=torch.long)
        self.sample, self.temperature, self.top_k = sample, temperature, top_k
        start = time.perf_counter()
        warmup = torch.cuda.Stream(device=embeds.device)
        warmup.wait_stream(self.stream)
        with torch.cuda.stream(warmup):
            for _ in range(3):
                self.run()
        self.stream.wait_stream(warmup)
        torch.cuda.synchronize(embeds.device)
        self.graph = torch.cuda.CUDAGraph()
        capture = time.perf_counter()
        with torch.cuda.graph(self.graph):
            self.run()
        torch.cuda.synchronize(embeds.device)
        self.capture_seconds = time.perf_counter() - capture
        self.setup_seconds = time.perf_counter() - start

    def run(self):
        if self.sample:
            self.uniforms.uniform_().clamp_(1e-7, 1 - 1e-7)
        else:
            self.uniforms.fill_(0.5)  # identical Gumbel offsets give argmax
        # Every used cache position is overwritten before read; zeroing also
        # makes reset explicit and bounds poisoned-cache testing.
        self.engine.cache.zero_()
        self.output = self.engine(self.code, self.inputs[:, 1], self.inputs[:, 0],
                                  1.0 / self.temperature, self.top_k, self.uniforms)[:, 1:]

    def replay(self, embeds):
        if torch.cuda.current_stream(embeds.device) != self.stream:
            raise ValueError("Fused predictor requires its owning CUDA stream")
        self.inputs.copy_(embeds)
        self.graph.replay()
        return self.output.clone()


class FusedPredictor:
    def __init__(self, predictor, max_graphs, mode="pdl"):
        self.strict = PredictorGraph(predictor, max_graphs=max_graphs)
        w = next(predictor.parameters())
        if w.device.type != "cuda" or w.dtype != torch.bfloat16:
            raise ValueError("Fused predictor requires CUDA BF16")
        cfg = predictor.config
        if cfg.num_code_groups != 16 or cfg.vocab_size != 2048 or cfg.hidden_size != 1024:
            raise ValueError("Only the validated 16-group, 2048-vocabulary, 1024-hidden predictor is supported")
        if cfg.hidden_act != "silu":
            raise ValueError("Fused predictor requires SiLU activation")
        self.predictor, self.max_graphs = predictor, max_graphs
        self.prior = predictor.__dict__.get("generate")
        self.stream = torch.cuda.current_stream(w.device)
        self.mode = mode
        self.engine = _k.FusedCodePredictor(_mapped(predictor), max_batch=1, mode=mode)
        self.graphs = {}
        self.fallback_calls = 0

    @torch.inference_mode()
    def generate(self, *args, **kw):
        if args:
            raise ValueError("Pass keyword inputs_embeds only")
        p, cfg = self.predictor, self.predictor.generation_config
        allowed = {"inputs_embeds", "max_new_tokens", "do_sample", "temperature", "top_k", "top_p", "output_hidden_states", "return_dict_in_generate"}
        if set(kw) - allowed:
            raise ValueError(f"Unsupported predictor kwargs: {sorted(set(kw) - allowed)}")
        defaults = self.strict._default_generation_settings()
        supported = allowed | {"bos_token_id", "pad_token_id", "_from_model_config", "transformers_version"}
        for name, value in cfg.to_dict().items():
            if name not in supported and value != defaults.get(name):
                raise ValueError(f"Unsupported predictor generation_config setting: {name}={value!r}")
        if p.training or torch.is_autocast_enabled("cuda"):
            raise ValueError("Fused predictor requires eval mode without autocast")
        x = kw.get("inputs_embeds")
        w = p.get_input_embeddings()[0].weight
        if not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[1:] != (2, w.shape[1]) or x.shape[0] < 1 or x.device != w.device or x.dtype != w.dtype:
            raise ValueError("inputs_embeds must have model device/dtype and [batch, 2, embedding_dim] shape")
        if torch.cuda.current_stream(x.device) != self.stream:
            raise ValueError("Fused predictor requires its owning CUDA stream")
        get = lambda n: kw.get(n, getattr(cfg, n))
        if get("max_new_tokens") != p.config.num_code_groups - 1:
            raise ValueError("max_new_tokens must equal remaining code groups")
        sample, hidden, ret = (get(n) for n in ("do_sample", "output_hidden_states", "return_dict_in_generate"))
        if not all(isinstance(v, bool) for v in (sample, hidden, ret)):
            raise ValueError("Sampling/output flags must be booleans")
        t, k, prob = (get(n) for n in ("temperature", "top_k", "top_p"))
        if sample:
            if not isinstance(t, (int, float)) or not math.isfinite(t) or t <= 0:
                raise ValueError("temperature must be finite and positive")
            if not isinstance(k, int) or isinstance(k, bool) or k < 0:
                raise ValueError("top_k must be a nonnegative integer")
            if not isinstance(prob, (int, float)) or not math.isfinite(prob) or not 0 < prob <= 1:
                raise ValueError("top_p must be in (0, 1]")
        else:
            t, k, prob = 1.0, 0, 1.0
        if hidden or x.shape[0] != 1 or prob != 1.0:
            self.fallback_calls += 1
            return self.strict.generate(**kw)
        key = (sample, float(t), min(k, self.engine.vocab))
        if key not in self.graphs:
            if len(self.graphs) >= self.max_graphs:
                raise ValueError("Fused predictor graph variant limit reached")
            with torch.cuda.device(x.device), torch.random.fork_rng(devices=[x.device.index]):
                self.graphs[key] = _Graph(self.engine, x, *key)
        out = self.graphs[key].replay(x)
        return GenerateDecoderOnlyOutput(sequences=out) if ret else out

    def uninstall(self):
        if getattr(self.predictor, "_fused_predictor", None) is not self:
            raise RuntimeError("This fused predictor is not installed")
        if self.prior is None:
            del self.predictor.generate
        else:
            self.predictor.generate = self.prior
        del self.predictor._fused_predictor
        del self.predictor._predictor_graph


def install_fused_predictor(code_predictor, *, max_graphs=8, mode="pdl"):
    """Install batch-one fusion; top-p/hidden-state/batched calls use strict graphs."""
    if hasattr(code_predictor, "_fused_predictor") or hasattr(code_predictor, "_predictor_graph"):
        raise ValueError("Install fused predictor instead of predictor graph, not on top")
    handle = FusedPredictor(code_predictor, max_graphs, mode=mode)
    code_predictor.generate = types.MethodType(lambda _, *a, **kw: handle.generate(*a, **kw), code_predictor)
    code_predictor._fused_predictor = handle
    # Compatibility marker for the audited no-hidden-state talker adapter.
    code_predictor._predictor_graph = handle
    return handle
