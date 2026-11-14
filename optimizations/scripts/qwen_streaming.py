"""Experimental, serial CustomVoice streaming; no upstream or graph patches persist.

Usage: stream_custom_voice(model, text="Hello", speaker="Ryan",
                          on_audio=lambda chunk: sink(chunk.samples))
Callbacks receive owned CPU float32 tensors, sample offsets, and sample rate.
They run synchronously on the caller thread: slow consumers provide backpressure.
Do not reenter/use the model concurrently, change its device/stream, or retain
unbounded output in the consumer. Install optional predictor/talker graphs first
and call from their owning device and stream. Graph reset/uninstall must also
run under torch.inference_mode() after this adapter initializes inference caches.
No worker is created. overlap=True uses a separate codec CUDA stream with a
bounded in-flight queue. Callbacks still run on the caller thread, polled between
talker steps; finish drains the queue. Cancellation synchronizes outstanding
codec work before releasing its tensors, without delivering queued audio.

The hook reads talker output.hidden_states[1]: all code groups for the PREVIOUS
sampled first-codebook token. Prefill has None; the final sampled token has no
complete frame, matching upstream extraction even at max_new_tokens. A normal
HF streamer sees only first-codebook tokens and is insufficient. The outer
Qwen generate ignores arbitrary kwargs (including streamer). We temporarily
wrap talker.generate to disable retained hidden states, then unwind before the
outer offline stacking/decode. Public CustomVoice prompt/default handling stays
unchanged. This integration is tied to the inspected upstream implementation.

Codec windows are decoded directly, not through tokenizer.decode (which itself
chunks at 300/25). Full causal context is retained by default using a conservative
architecture-derived bound. Shorter context requires explicit approximate=True;
25 frames is NOT a whole-decode equivalence guarantee. Position resets, attention
backends and floating-point kernels still require GPU waveform comparisons.
Codec memory is bounded by context + chunk frames; generation KV and token history
remain bounded by max_new_tokens and prompt length, not constant in duration.
Cancellation drops the un-emitted tail and raises StreamingCancelled. Callback,
codec and generation errors propagate; already emitted audio cannot be retracted.
"""
from __future__ import annotations

from dataclasses import dataclass
import threading
import torch


class StreamingCancelled(RuntimeError):
    """The cancellation predicate requested termination."""


class _GenerationComplete(BaseException):
    pass


@dataclass(frozen=True)
class AudioChunk:
    samples: torch.Tensor
    sample_rate: int
    start_sample: int


@dataclass(frozen=True)
class StreamResult:
    frames: int
    samples: int
    sample_rate: int
    context_frames: int
    approximate: bool


def codec_context_frames(config):
    """Conservative left dependency for the inspected 12 Hz decoder.

    Reverse the earliest output sample through final k7 convolution, each
    (k7 dilation 1,3,9) residual stack and k=2s transposed convolution,
    initial k7 convolution, then each ConvNeXt k7 and k=s upsampler.
    Add stacked attention windows and pre-convolution k3. Using W rather
    than W-1 per attention layer deliberately covers window conventions.
    All convolutions pad left or crop right: no future-frame holdback.
    """
    w = config.sliding_window
    if not isinstance(w, int) or w < 1 or any(
        t != "sliding_attention" for t in config.layer_types
    ):
        raise ValueError("Finite sliding attention is required for bounded codec context")
    i = -6
    for s in reversed(config.upsample_rates):
        i -= 6 * (1 + 3 + 9)
        i = (i - s) // s
    i -= 6
    for s in reversed(config.upsampling_ratios):
        i = (i - 6) // s
    return -i + config.num_hidden_layers * w + 2


class ChunkedCodec:
    """Bounded frame buffer. decode accepts [1, groups, frames], returns [1,1,N]."""

    def __init__(self, decode, *, groups, hop, sample_rate, chunk_frames,
                 context_frames, on_audio, cancelled=lambda: False, first_chunk_frames=None):
        for name, value, minimum in (("groups", groups, 1), ("hop", hop, 1),
                                     ("sample_rate", sample_rate, 1),
                                     ("chunk_frames", chunk_frames, 1),
                                     ("context_frames", context_frames, 0)):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if first_chunk_frames is not None and (type(first_chunk_frames) is not int or first_chunk_frames < 1):
            raise ValueError("first_chunk_frames must be a positive integer")
        self.first_chunk_frames = first_chunk_frames or chunk_frames
        self.decode, self.groups, self.hop = decode, groups, hop
        self.sample_rate, self.chunk_frames = sample_rate, chunk_frames
        self.context_frames, self.on_audio = context_frames, on_audio
        self.cancelled = cancelled
        self.buffer = []
        self.pending = self.frames = self.samples = 0
        self.closed = False

    def check_cancelled(self):
        if self.cancelled():
            raise StreamingCancelled("Streaming cancelled")

    def push(self, frame):
        self.check_cancelled()
        if self.closed:
            raise RuntimeError("Codec stream is closed")
        if frame.shape != (1, self.groups) or frame.dtype != torch.long:
            raise ValueError("Expected one complete int64 codec frame [1, groups]")
        self.buffer.append(frame.detach().clone())
        self.pending += 1
        self.frames += 1
        target = self.first_chunk_frames if self.frames == self.pending else self.chunk_frames
        if self.pending == target:
            self._emit()

    def _emit(self):
        self.check_cancelled()
        if not self.pending:
            return
        codes = torch.stack(self.buffer, dim=-1)
        context = len(self.buffer) - self.pending
        wav = self.decode(codes)
        expected = len(self.buffer) * self.hop
        if wav.shape != (1, 1, expected):
            raise RuntimeError(f"Unexpected codec shape {tuple(wav.shape)}; expected (1, 1, {expected})")
        samples = wav[0, 0, context * self.hop:].detach().to(device="cpu", dtype=torch.float32).clone()
        self.check_cancelled()
        self.on_audio(AudioChunk(samples, self.sample_rate, self.samples))
        self.samples += samples.numel()
        self.pending = 0
        self.buffer = self.buffer[-self.context_frames:] if self.context_frames else []

    def poll(self):
        self.check_cancelled()

    def close(self):
        self.buffer.clear()

    def finish(self):
        if not self.closed:
            self._emit()
            self.closed = True
            self.buffer.clear()


class OverlappedCodec(ChunkedCodec):
    """Same-thread enqueue/poll, with explicit events and bounded tensor lifetimes.

    Decoder is used only on this stream. Input stacks are made on the owner
    stream, transferred by wait_event and record_stream. Each queue entry owns
    input, output and pinned CPU storage until D2H completion. No graph work or
    callbacks run inside the codec stream context. close must run on all exits.
    """

    def __init__(self, *args, device, max_inflight=2, **kwargs):
        super().__init__(*args, **kwargs)
        if type(max_inflight) is not int or max_inflight < 1:
            raise ValueError("max_inflight must be a positive integer")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("Overlap requires CUDA")
        self.owner_stream = torch.cuda.current_stream(self.device)
        self.stream = torch.cuda.Stream(device=self.device)
        self.stream.wait_stream(self.owner_stream)
        self.max_inflight = max_inflight
        self.queue = []
        self.scheduled_samples = 0
        self.peak_inflight = 0

    def _deliver(self, block=False):
        self.check_cancelled()
        while self.queue:
            done, codes, gpu, cpu, offset = self.queue[0]
            if block:
                done.synchronize()
            elif not done.query():
                break
            self.check_cancelled()
            self.queue.pop(0)
            self.on_audio(AudioChunk(cpu, self.sample_rate, offset))
            self.samples += cpu.numel()
            if block:
                break

    def poll(self):
        self._deliver()

    def _emit(self):
        self.check_cancelled()
        if not self.pending:
            return
        if torch.cuda.current_stream(self.device) != self.owner_stream:
            raise RuntimeError("Codec enqueue must use its owner stream")
        self._deliver()
        if len(self.queue) >= self.max_inflight:
            self._deliver(block=True)
        codes = torch.stack(self.buffer, dim=-1)
        context = len(self.buffer) - self.pending
        ready = torch.cuda.Event()
        ready.record(self.owner_stream)
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(ready)
            codes.record_stream(self.stream)
            wav = self.decode(codes)
            expected = len(self.buffer) * self.hop
            if wav.shape != (1, 1, expected):
                raise RuntimeError(f"Unexpected codec shape {tuple(wav.shape)}")
            gpu = wav[0, 0, context * self.hop:].detach().float().contiguous()
            cpu = torch.empty(gpu.shape, dtype=torch.float32, device="cpu", pin_memory=True)
            cpu.copy_(gpu, non_blocking=True)
            done = torch.cuda.Event()
            done.record(self.stream)
        self.queue.append((done, codes, gpu, cpu, self.scheduled_samples))
        self.peak_inflight = max(self.peak_inflight, len(self.queue))
        self.scheduled_samples += cpu.numel()
        self.pending = 0
        self.buffer = self.buffer[-self.context_frames:] if self.context_frames else []

    def finish(self):
        if not self.closed:
            self._emit()
            while self.queue:
                self._deliver(block=True)
            self.closed = True
            self.buffer.clear()

    def close(self):
        # Includes kernels launched before an exception prevented queue insertion.
        self.stream.synchronize()
        self.queue.clear()
        super().close()


@torch.inference_mode()
def stream_custom_voice(model, *, text, speaker, on_audio, language=None,
                        instruct=None, non_streaming_mode=True, chunk_frames=12,
                        context_frames=None, approximate=False, cancelled=lambda: False,
                        first_chunk_frames=None, overlap=False, max_inflight=2,
                        **generation_kwargs):
    """Generate batch-one audio via callbacks; return counts, not an audio list.

    Only upstream's explicit sampling/max_new_tokens options are accepted.
    Defaults remain serial with 12-frame chunks. For a one-frame start followed
    by 25-frame chunks, pass first_chunk_frames=1, chunk_frames=25. This lowers
    first delivery latency but supplies only one frame of playback reserve.
    overlap=True uses a separate codec stream, at most max_inflight=2 decoded
    windows, and event polling between generation steps. It does not promise
    uninterrupted playback or faster generation. Conservative context is unchanged.
    CPU accounting tests and GPU1 serial checks are recorded under
    results/streaming/. These are numerical checks, not listening validation.
    """
    if not isinstance(text, str) or not isinstance(speaker, str):
        raise ValueError("Streaming requires scalar text and speaker")
    if language is not None and not isinstance(language, str):
        raise ValueError("Streaming requires scalar language")
    if instruct is not None and not isinstance(instruct, str):
        raise ValueError("Streaming requires scalar instruct")
    allowed = {"do_sample", "top_k", "top_p", "temperature", "repetition_penalty",
               "subtalker_dosample", "subtalker_top_k", "subtalker_top_p",
               "subtalker_temperature", "max_new_tokens"}
    if set(generation_kwargs) - allowed:
        raise ValueError(f"Unsupported streaming options: {sorted(set(generation_kwargs) - allowed)}")
    core = model.model
    talker = core.talker
    tokenizer = core.speech_tokenizer.model
    if core.tts_model_type != "custom_voice" or tokenizer.get_model_type() != "qwen3_tts_tokenizer_12hz":
        raise ValueError("Requires CustomVoice with the 12 Hz codec")
    if core.training or tokenizer.training:
        raise ValueError("Streaming requires eval models")
    config = talker.generation_config
    if (config.num_beams != 1 or config.num_return_sequences != 1 or
            getattr(config, "penalty_alpha", None) is not None):
        raise ValueError("Only single-sequence sampling/greedy generation is supported")
    decoder = tokenizer.decoder
    if decoder.__class__.__name__ != "Qwen3TTSTokenizerV2Decoder":
        raise ValueError("Unsupported codec decoder architecture")
    if getattr(decoder.pre_transformer.rotary_emb, "rope_type", "default") != "default":
        raise ValueError("Only default codec rotary embeddings are supported")
    required = codec_context_frames(decoder.config)
    context = required if context_frames is None else context_frames
    if context < required and not approximate:
        raise ValueError(f"context_frames must be >= {required}, or set approximate=True")
    device = next(decoder.parameters()).device
    talker_device = next(talker.parameters(), torch.empty(0)).device
    if device.type == "cuda" and talker_device != device:
        raise ValueError("Streaming requires talker and codec on the same CUDA device")
    owner_stream = torch.cuda.current_stream(device) if device.type == "cuda" else None
    if owner_stream is not None and torch.cuda.current_device() != device.index:
        raise ValueError("Streaming requires its CUDA device to be current")
    owner = threading.get_ident()
    # No implicit device/stream migration: existing graph guards remain active.
    if hasattr(talker, "_streaming_active"):
        raise RuntimeError("A streaming request is already active")
    codec_class = OverlappedCodec if overlap else ChunkedCodec
    codec_options = dict(device=device, max_inflight=max_inflight) if overlap else {}
    codec = codec_class(lambda codes: decoder(codes.to(device)),
                         groups=decoder.config.num_quantizers,
                         hop=int(decoder.total_upsample),
                         sample_rate=int(tokenizer.get_output_sample_rate()),
                         chunk_frames=chunk_frames, context_frames=context,
                         on_audio=on_audio, cancelled=cancelled,
                         first_chunk_frames=first_chunk_frames, **codec_options)
    talker._streaming_active = True
    original = talker.generate
    had_override = "generate" in talker.__dict__
    prior = talker.__dict__.get("generate")
    hook = None
    pre_hook = None

    def check_owner():
        if threading.get_ident() != owner:
            raise RuntimeError("Streaming model used from another thread")
        if owner_stream is not None and (torch.cuda.current_device() != device.index or
                                          torch.cuda.current_stream(device) != owner_stream):
            raise RuntimeError("Streaming requires its original CUDA device and stream")
        codec.check_cancelled()

    def before_forward(module, args):
        check_owner()
        codec.poll()

    def after_forward(module, args, output):
        check_owner()
        frame = output.hidden_states[1]
        if frame is not None:
            if frame.shape[0] != 1:
                raise ValueError("Streaming requires batch one")
            if int(frame[0, 0]) != talker.config.codec_eos_token_id:
                codec.push(frame)

    def generate(*args, **kwargs):
        check_owner()
        kwargs.update(output_hidden_states=False, return_dict_in_generate=False)
        original(*args, **kwargs)
        codec.finish()
        # Avoid offline aggregation, which expects retained hidden states.
        raise _GenerationComplete()

    try:
        check_owner()
        pre_hook = talker.register_forward_pre_hook(before_forward)
        hook = talker.register_forward_hook(after_forward)
        talker.generate = generate
        try:
            model.generate_custom_voice(text=text, speaker=speaker, language=language,
                                        instruct=instruct, non_streaming_mode=non_streaming_mode,
                                        **generation_kwargs)
        except _GenerationComplete:
            return StreamResult(codec.frames, codec.samples, codec.sample_rate,
                                context, context < required)
        raise RuntimeError("Upstream did not invoke the streaming talker")
    finally:
        if hook is not None:
            hook.remove()
        if pre_hook is not None:
            pre_hook.remove()
        if had_override:
            talker.generate = prior
        elif "generate" in talker.__dict__:
            del talker.generate
        del talker._streaming_active
        codec.close()
