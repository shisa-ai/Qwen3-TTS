"""Bucketed CUDA graphs for the causal 12 Hz codec decoder.

decoder.forward(codes[1, Q, n]) is replaced by: copy codes into a bucket-sized buffer (bucket
>= n, multiple of `step`), zero the padded tail, replay that bucket's graph, return the first
n * upsample samples. The decoder is causal, so right padding cannot change those samples
(beyond BF16 kernel-shape noise). Upstream chunked_decode (300-frame chunks, 25 context) is
left intact and simply calls the graphed forward per chunk. All buckets share one memory pool.
"""
import types
import torch


class CodecGraphs:
    def __init__(self, decoder, *, step=32, max_frames=352):
        self.decoder = decoder
        self.eager = decoder.forward
        self.step, self.max_frames = step, max_frames
        self.up = int(decoder.total_upsample)
        self.Q = decoder.config.num_quantizers
        self.device = next(decoder.parameters()).device
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs = {}
        self.hits = self.misses = 0
        decoder.forward = types.MethodType(lambda _m, codes: self(codes), decoder)

    def bucket(self, n):
        return -(-n // self.step) * self.step

    @torch.inference_mode()
    def prepare(self, b):
        if b in self.graphs:
            return
        inp = torch.zeros(1, self.Q, b, dtype=torch.long, device=self.device)
        s = torch.cuda.Stream(self.device)
        s.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(s):
            for _ in range(2):
                self.eager(inp)
        torch.cuda.current_stream(self.device).wait_stream(s)
        torch.cuda.synchronize(self.device)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            out = self.eager(inp)
        torch.cuda.synchronize(self.device)
        self.graphs[b] = (g, inp, out)

    def prepare_all(self):
        for b in range(self.step, self.max_frames + 1, self.step):
            self.prepare(b)

    def __call__(self, codes):
        n = codes.shape[-1]
        if (codes.ndim != 3 or codes.shape[:2] != (1, self.Q) or n > self.max_frames
                or torch.cuda.is_current_stream_capturing()):
            self.misses += 1
            return self.eager(codes)
        b = self.bucket(n)
        if b not in self.graphs:
            self.prepare(b)
        g, inp, out = self.graphs[b]
        inp[..., :n].copy_(codes)
        inp[..., n:].zero_()
        g.replay()
        self.hits += 1
        return out[..., :n * self.up].clone()

    def uninstall(self):
        self.decoder.forward = self.eager
