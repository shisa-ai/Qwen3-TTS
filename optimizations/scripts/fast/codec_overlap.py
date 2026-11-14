"""Decode codec chunks on a side stream while the talker engine generates.

The 12 Hz decoder is causal (left-padded convolutions, sliding-window attention), so a
chunk [s, e) decoded with `ctx` frames of left context matches whole-sequence decoding
up to the truncated (decaying) dependency beyond ctx and BF16 kernel-shape noise.
Upstream offline decoding itself uses 300-frame chunks with 25 frames of context.

Chunks are launched right after the next frame graph is queued, so host-side launch work
overlaps the GPU frame; the side stream waits on an event recorded after the frames it
reads. finish() decodes the tail and joins the streams.
"""
import torch


class OverlapCodec:
    def __init__(self, engine, decoder, *, chunk=48, ctx=96):
        self.engine, self.decoder = engine, decoder
        self.chunk, self.ctx = chunk, ctx
        self.up = int(decoder.total_upsample)
        self.main = engine.stream
        self.side = torch.cuda.Stream(engine.device)
        self.reset()

    def reset(self):
        self.done = 0
        self.pieces = []

    def _decode(self, s, e):
        ev = torch.cuda.Event()
        ev.record(self.main)
        a = max(0, s - self.ctx)
        with torch.cuda.stream(self.side):
            self.side.wait_event(ev)
            codes = self.engine.codes_out[a:e].t().unsqueeze(0).clamp(min=0)
            wav = self.decoder(codes)[..., (s - a) * self.up:]
            self.pieces.append(wav)
        self.done = e

    def launch(self, upto):
        """Frames [0, upto) are complete or queued on the main stream."""
        if upto - self.done >= self.chunk:
            self._decode(self.done, upto)

    def finish(self, n):
        if n > self.done:
            self._decode(self.done, n)
        self.main.wait_stream(self.side)
        for p in self.pieces:
            p.record_stream(self.main)
        if not self.pieces:
            return torch.zeros(0, device=self.engine.device)
        out = torch.cat(self.pieces, dim=-1).reshape(-1)
        self.pieces = []
        return out
