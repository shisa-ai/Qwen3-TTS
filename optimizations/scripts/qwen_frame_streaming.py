"""Direct complete-frame streaming using the reviewed explicit loop.

Callbacks are synchronous and own their frame storage. No worker or unbounded
queue is created. Layer outputs are still computed by the existing talker graph,
but are not retained across steps. Offline full-output behavior is unchanged.
"""
import torch
from qwen_streaming import (ChunkedCodec, StreamResult, StreamingCancelled,
                            codec_context_frames, _GenerationComplete)


@torch.inference_mode()
def stream_frames(loop, kwargs, on_frame, *, cancelled=lambda: False):
    """Emit complete codec frames without returning layer/token history.

    Unsupported generation settings raise before starting; callers may select a
    separate strict streaming entrypoint before any frames have been emitted.
    """
    if loop.active:
        raise RuntimeError('Frame loop is already active')
    import threading
    owner=threading.get_ident()
    loop.tg._check_owner()
    # Reuse exactly the public request checks without duplicating evolving rules.
    original_run = loop.run
    def run(kw):
        def poll():
            if threading.get_ident()!=owner:
                raise RuntimeError('Frame streaming requires its owning thread')
            loop.tg._check_owner()
            if cancelled():
                raise StreamingCancelled('Streaming cancelled')
        poll()
        def emit(frame):
            on_frame(frame)
            poll()
        return original_run(kw,on_frame=emit,poll=poll,retain_history=False)
    if not loop.supported((),kwargs):
        raise ValueError('Unsupported direct-frame streaming options; use strict hook streaming')
    loop.run = run
    try:
        return loop.generate(**kwargs)
    except BaseException:
        with torch.cuda.device(loop.tg.device), torch.cuda.stream(loop.tg.owner_stream):
            loop.tg.reset()
        raise
    finally:
        loop.run = original_run


@torch.inference_mode()
def stream_custom_voice(model, *, text, speaker, on_audio, language='English',
                        instruct=None, chunk_frames=25, first_chunk_frames=4,
                        context_frames=588, cancelled=lambda: False,
                        on_frame=None, **generation_kwargs):
    from qwen_streaming import stream_custom_voice as strict_stream
    allowed={'do_sample','top_k','top_p','temperature','repetition_penalty',
             'subtalker_dosample','subtalker_top_k','subtalker_top_p',
             'subtalker_temperature','max_new_tokens'}
    if set(generation_kwargs)-allowed:
        raise ValueError('Unsupported streaming generation options')
    if model.model.tts_model_type!='custom_voice':
        raise ValueError('Direct streaming requires CustomVoice')
    t=model.model.talker
    loop=t._frame_loop
    if hasattr(t,'_streaming_active'):
        raise RuntimeError('A streaming request is already active')
    tokenizer=model.model.speech_tokenizer.model
    if tokenizer.get_model_type() != 'qwen3_tts_tokenizer_12hz':
        raise ValueError('Requires CustomVoice with the 12 Hz codec')
    if model.model.training or tokenizer.training:
        raise ValueError('Streaming requires eval models')
    decoder=tokenizer.decoder
    if decoder.__class__.__name__ != 'Qwen3TTSTokenizerV2Decoder':
        raise ValueError('Unsupported codec decoder architecture')
    if getattr(decoder.pre_transformer.rotary_emb,'rope_type','default') != 'default':
        raise ValueError('Only default codec rotary embeddings are supported')
    if decoder.training or next(decoder.parameters()).device != next(t.parameters()).device:
        raise ValueError('Streaming requires eval codec on talker device')
    import threading
    owner=threading.get_ident()
    def check_owner():
        if threading.get_ident()!=owner:
            raise RuntimeError('Streaming requires its owning thread')
        loop.tg._check_owner()
    check_owner()
    def audio(chunk):
        check_owner()
        on_audio(chunk)
        check_owner()
    required=codec_context_frames(decoder.config)
    if context_frames < required:
        raise ValueError('Direct streaming requires full conservative codec context')
    if not isinstance(text,str) or not isinstance(speaker,str):
        raise ValueError('Direct streaming requires scalar CustomVoice inputs')
    codec=ChunkedCodec(decoder,groups=decoder.config.num_quantizers,
        hop=int(decoder.total_upsample),sample_rate=int(tokenizer.get_output_sample_rate()),
        chunk_frames=chunk_frames,first_chunk_frames=first_chunk_frames,
        context_frames=context_frames,on_audio=audio,cancelled=cancelled)
    original=t.generate
    had='generate' in t.__dict__
    prior=t.__dict__.get('generate')
    t._streaming_active=True
    class Unsupported(Exception):
        pass
    entered=False
    def generate(*args,**kw):
        nonlocal entered
        if entered:
            raise RuntimeError('Streaming generation is non-reentrant, including final audio delivery')
        entered=True
        check_owner()
        if args or not loop.supported(args,kw):
            raise Unsupported()
        def push(frame):
            check_owner()
            if on_frame is not None:
                on_frame(frame.clone())
                check_owner()
            codec.push(frame)
            check_owner()
        stream_frames(loop,kw,push,cancelled=cancelled)
        codec.finish()
        raise _GenerationComplete()
    fallback=False
    try:
        t.generate=generate
        try:
            model.generate_custom_voice(text=text,speaker=speaker,language=language,
                instruct=instruct,**generation_kwargs)
        except _GenerationComplete:
            return StreamResult(codec.frames,codec.samples,codec.sample_rate,context_frames,False)
        except Unsupported:
            fallback=True
        else:
            raise RuntimeError('Upstream did not invoke direct streaming')
    except BaseException:
        with torch.cuda.device(loop.tg.device), torch.cuda.stream(loop.tg.owner_stream):
            loop.tg.reset()
        raise
    finally:
        if had:
            t.generate=prior
        else:
            del t.generate
        del t._streaming_active
        codec.close()
    if fallback:
        if on_frame is not None:
            raise ValueError('Frame callback requires supported explicit-loop options')
        def fallback_audio(chunk):
            check_owner()
            saved=t.generate
            def reject(*a,**kw):
                raise RuntimeError('Streaming generation is non-reentrant during callbacks')
            t.generate=reject
            try:
                on_audio(chunk)
                check_owner()
            finally:
                t.generate=saved
        try:
            return strict_stream(model,text=text,speaker=speaker,language=language,instruct=instruct,
                on_audio=fallback_audio,chunk_frames=chunk_frames,first_chunk_frames=first_chunk_frames,
                context_frames=context_frames,cancelled=cancelled,**generation_kwargs)
        except BaseException:
            with torch.cuda.device(loop.tg.device), torch.cuda.stream(loop.tg.owner_stream):
                loop.tg.reset()
            raise
