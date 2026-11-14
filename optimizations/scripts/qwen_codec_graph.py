"""Opt-in exact-shape codec graphs; prepare outside requests, eager on cache misses.

Single device/stream/thread, eval and inference mode only. Weights, architecture
and backend settings must remain unchanged for the adapter lifetime. No padding,
short-context approximation, persistent decoder state or lazy capture is added.
"""
from contextlib import contextmanager
import threading
import torch


class CodecGraphs:
    def __init__(self, decoder, *, max_graphs=16, max_frames=613):
        if type(max_graphs) is not int or max_graphs < 1 or type(max_frames) is not int or max_frames < 1:
            raise ValueError('Positive integer graph and frame limits required')
        if decoder.__class__.__name__ != 'Qwen3TTSTokenizerV2Decoder':
            raise ValueError('Requires Qwen3TTSTokenizerV2Decoder')
        self.decoder = decoder
        self.eager = decoder.forward
        self.device = next(decoder.parameters()).device
        if self.device.type != 'cuda':
            raise ValueError('CUDA decoder required')
        self.stream = torch.cuda.current_stream(self.device)
        self.thread = threading.get_ident()
        self.max_graphs, self.max_frames = max_graphs, max_frames
        self.graphs = {}
        self.captures = self.hits = self.misses = 0
        self.active = False
        self._guard()

    def _guard(self):
        if threading.get_ident() != self.thread or torch.cuda.current_device() != self.device.index or torch.cuda.current_stream(self.device) != self.stream:
            raise RuntimeError('Codec requires its original thread, device and stream')
        if torch.is_autocast_enabled('cuda'):
            raise RuntimeError('CUDA autocast is unsupported')
        rotary = self.decoder.pre_transformer.rotary_emb
        scaling = getattr(self.decoder.config, 'rope_scaling', None)
        if getattr(rotary, 'rope_type', 'default') != 'default' or (scaling and scaling.get('rope_type', scaling.get('type', 'default')) != 'default'):
            raise RuntimeError('Only default codec rotary configuration is supported')
        if self.decoder.training or not torch.is_inference_mode_enabled():
            raise RuntimeError('Codec requires eval and inference mode')
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Nested codec capture is unsupported')
        if next(self.decoder.parameters()).device != self.device:
            raise RuntimeError('Decoder device changed')

    def _validate(self, codes):
        self._guard()
        if codes.dtype != torch.long or codes.device != self.device or codes.ndim != 3 or codes.shape[:2] != (1, self.decoder.config.num_quantizers) or codes.shape[-1] < 1 or not codes.is_contiguous():
            raise ValueError('Expected contiguous CUDA int64 codes [1, groups, positive frames]')

    def prepare(self, codes):
        self._validate(codes)
        n = codes.shape[-1]
        if n in self.graphs:
            return
        if self.active:
            raise RuntimeError('Prepare graphs outside installed requests')
        if n > self.max_frames or len(self.graphs) >= self.max_graphs:
            raise ValueError('Codec graph capacity exceeded')
        inp = codes.clone()
        warm = torch.cuda.Stream(device=self.device)
        warm.wait_stream(self.stream)
        try:
            with torch.cuda.stream(warm):
                for _ in range(3):
                    self.eager(inp)
            self.stream.wait_stream(warm)
            self.stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=warm):
                out = self.eager(inp)
        finally:
            # Retain temporary inputs until even a failed warmup/capture has drained.
            warm.synchronize()
            self.stream.wait_stream(warm)
        self.graphs[n] = (graph, inp, out)
        self.captures += 1

    def __call__(self, codes):
        self._validate(codes)
        entry = self.graphs.get(codes.shape[-1])
        if entry is None:
            self.misses += 1
            return self.eager(codes)
        graph, inp, out = entry
        inp.copy_(codes)
        graph.replay()
        self.hits += 1
        return out.clone()

    def reset(self):
        self._guard()
        if self.active:
            raise RuntimeError('Reset outside installed requests')
        self.stream.synchronize()
        self.graphs.clear()

    @contextmanager
    def installed(self):
        self._guard()
        if self.active or self.decoder.forward != self.eager:
            raise RuntimeError('Decoder already wrapped')
        had = 'forward' in self.decoder.__dict__
        prior = self.decoder.__dict__.get('forward')
        self.active = True
        self.decoder.forward = self
        try:
            yield self
        finally:
            if had:
                self.decoder.forward = prior
            else:
                del self.decoder.forward
            self.active = False
