"""GPU2-only codec correctness, fixed-input timings and streaming schedule experiment."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

GPU = 'GPU-3e620c1c-c243-7353-580e-cde9e5eb4b8a'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--size', required=True, choices=['0.6B', '1.7B'])
    a = p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == GPU
    assert all(os.environ.get(x) == '8' for x in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'))
    import torch
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel
    from qwen_codec_graph import CodecGraphs
    from qwen_streaming import ChunkedCodec, stream_custom_voice, codec_context_frames
    from qwen_fused_norm import install_fused_predictor_norm
    from qwen_fused_predictor import install_fused_predictor
    from qwen_talker_fusion import install_talker_fusion
    from qwen_talker_graph import install_talker_graph
    from qwen_fast_decode import install_fast_decode
    from bench_common import TEXTS
    torch.set_num_threads(8)
    torch.backends.cuda.enable_cudnn_sdp(False)
    root = Path(__file__).resolve().parent.parent
    dest = root / 'results/gap/codec'
    audio = root / 'samples/gap/codec' / (a.size + '-v2')
    dest.mkdir(parents=True, exist_ok=True)
    audio.mkdir(parents=True, exist_ok=True)
    result = dict(size=a.size, gpu=GPU, cudnn_sdpa=False, checks=[], fixed=[], streaming=[], complete=False)
    def save():
        (dest / (a.size + '-v2.json')).write_text(json.dumps(result, indent=2))
    def check(name, condition=True):
        assert condition, name
        result['checks'].append(name)
        save()
    def timed(fn):
        torch.cuda.synchronize()
        start = time.perf_counter()
        value = fn()
        torch.cuda.synchronize()
        return value, time.perf_counter() - start
    def errors(x, y):
        d = (x.float() - y.float()).cpu()
        return dict(max_abs=d.abs().max().item(), rmse=d.square().mean().sqrt().item())
    model = Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-' + a.size + '-CustomVoice',
        device_map='cuda:0', dtype=torch.bfloat16, attn_implementation='sdpa')
    talker = model.model.talker
    decoder = model.model.speech_tokenizer.model.decoder
    install_fused_predictor_norm(talker.code_predictor)
    install_fused_predictor(talker.code_predictor, mode='nonpdl' if a.size == '0.6B' else 'pdl')
    install_talker_fusion(talker)
    tg = install_talker_graph(talker)
    install_fast_decode(talker)
    hop = int(decoder.total_upsample)
    sr = int(model.model.speech_tokenizer.model.get_output_sample_rate())
    context = codec_context_frames(decoder.config)
    result.update(context_frames=context, hop=hop, sample_rate=sr)
    settings = dict(speaker='ryan', language='English', do_sample=True, subtalker_dosample=True, max_new_tokens=1000,
                    temperature=0.9, top_k=50, top_p=1.0, subtalker_temperature=0.9, subtalker_top_k=50, subtalker_top_p=1.0)
    result.update(seed=42, generation=settings)
    original_generate = talker.generate
    eos_observations = []
    def observe_eos(*args, **kwargs):
        output = original_generate(*args, **kwargs)
        sequences = output.sequences if hasattr(output, 'sequences') else output
        eos_observations.append(int(sequences[0, -1]) == talker.config.codec_eos_token_id)
        assert eos_observations[-1], 'Request did not reach EOS'
        return output
    talker.generate = observe_eos
    with torch.inference_mode():
        codes_by_text = {}
        for text_id in ('short', 'medium'):
            frames = []
            def hook(m, inp, out):
                frame = out.hidden_states[1]
                if frame is not None:
                    frames.append(frame.clone())
            h = talker.register_forward_hook(hook)
            tg.reset()
            torch.manual_seed(42)
            try:
                model.generate_custom_voice(text=TEXTS[text_id], **settings)
            finally:
                h.remove()
            codes_by_text[text_id] = torch.stack(frames, -1).contiguous()
            check(text_id + ' EOS before limit', len(frames) < 999)
        codes = codes_by_text['medium']
        graph = CodecGraphs(decoder, max_graphs=16)
        for n in (1, 12, 25, 97, 613):
            x = codes.repeat(1, 1, (n // codes.shape[-1]) + 1)[..., :n].contiguous()
            ref, _ = timed(lambda: decoder(x))
            _, setup = timed(lambda: graph.prepare(x))
            got = graph(x)
            e = errors(ref, got)
            check('graph same-input waveform ' + str(n), e['max_abs'] <= 1e-5)
            snapshot = got.clone()
            graph(torch.zeros_like(x))
            check('owned output ' + str(n), torch.equal(snapshot, got))
            times = {}
            for name, fn in [('eager', lambda: decoder(x)), ('graph', lambda: graph(x))]:
                fn()
                times[name] = [timed(fn)[1] for _ in range(5)]
            result['fixed'].append(dict(frames=n, setup_s=setup, error=e, times=times,
                speedup=statistics.median(times['eager']) / statistics.median(times['graph'])))
            save()
        captures = graph.captures
        graph(codes[..., :2].contiguous())
        check('uncached shape falls back without capture', graph.captures == captures and graph.misses == 1)
        for label, fn in [('dtype', lambda: graph(codes.float())),
                          ('contiguous batch2', lambda: graph(codes.repeat(2, 1, 1))),
                          ('noncontiguous batch1', lambda: graph(codes[..., ::2])),
                          ('wrong groups', lambda: graph(codes[:, :-1].contiguous())),
                          ('empty frames', lambda: graph(codes[..., :0].contiguous())),
                          ('CPU device', lambda: graph(codes.cpu()))]:
            try:
                fn()
            except ValueError:
                check('reject ' + label)
            else:
                raise AssertionError(label)
        stream = torch.cuda.Stream()
        try:
            with torch.cuda.stream(stream):
                graph(codes)
        except RuntimeError:
            check('reject foreign stream')
        else:
            raise AssertionError('foreign stream')
        def rejects(name, fn):
            try:
                fn()
            except RuntimeError:
                check(name)
            else:
                raise AssertionError(name)
        with torch.inference_mode(False):
            rejects('reject inference disabled', lambda: graph(codes))
        with torch.autocast('cuda'):
            rejects('reject autocast replay', lambda: graph(codes))
            rejects('reject autocast prepare', lambda: graph.prepare(codes))
            rejects('reject autocast constructor', lambda: CodecGraphs(decoder))
        rotary = decoder.pre_transformer.rotary_emb
        prior_rope = rotary.rope_type
        rotary.rope_type = 'dynamic'
        try:
            rejects('reject nondefault rotary before warmup', lambda: CodecGraphs(decoder))
            rejects('reject changed rotary', lambda: graph.prepare(codes))
        finally:
            rotary.rope_type = prior_rope
        import threading
        thread_errors = []
        def foreign_thread():
            with torch.inference_mode():
                try:
                    graph(codes)
                except RuntimeError as exc:
                    thread_errors.append(str(exc))
        thread = threading.Thread(target=foreign_thread)
        thread.start()
        thread.join()
        check('reject foreign thread', len(thread_errors) == 1 and 'original thread' in thread_errors[0])
        # Only GPU2 is exposed. Mock current-device metadata rather than touching another GPU.
        from unittest.mock import patch
        with patch.object(torch.cuda, 'current_device', return_value=1):
            rejects('reject foreign device metadata', lambda: graph(codes))
        with patch.object(torch.cuda, 'is_current_stream_capturing', return_value=True):
            rejects('reject nested capture state', lambda: graph(codes))
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            nested_adapter = CodecGraphs(decoder)
        outer = torch.cuda.CUDAGraph()
        nested_rejected = False
        with torch.cuda.graph(outer, stream=capture_stream):
            marker = codes + 0
            try:
                nested_adapter(codes)
            except RuntimeError as exc:
                nested_rejected = 'Nested codec capture' in str(exc)
        torch.cuda.current_stream().wait_stream(capture_stream)
        check('reject real nested capture', nested_rejected)
        with graph.installed():
            def nested_install():
                with graph.installed():
                    pass
            rejects('reject nested install', nested_install)
        original_forward = decoder.forward
        decoder.forward = original_forward
        preexisting = CodecGraphs(decoder)
        with preexisting.installed():
            pass
        check('restore preexisting forward identity', decoder.__dict__['forward'] is original_forward)
        del decoder.forward
        # Inject Python failures after queued work, separately during warmup and capture.
        for phase in ('warmup', 'capture'):
            eager = graph.eager
            ref = graph(codes[..., :12].contiguous())
            def fail(x):
                value = eager(x)
                if (phase == 'capture') == torch.cuda.is_current_stream_capturing():
                    raise LookupError('injected ' + phase)
                return value
            graph.eager = fail
            before = graph.captures
            try:
                graph.prepare(codes[..., :3].contiguous())
            except LookupError:
                check('injected ' + phase + ' failure')
            else:
                raise AssertionError(phase)
            finally:
                graph.eager = eager
            check(phase + ' preserves existing graphs', graph.captures == before and 3 not in graph.graphs and torch.equal(ref, graph(codes[..., :12].contiguous())))
            check(phase + ' eager remains usable', torch.equal(eager(codes[..., :2].contiguous()), graph(codes[..., :2].contiguous())))
            graph.prepare(codes[..., :3].contiguous())
            check(phase + ' subsequent prepare usable', torch.equal(eager(codes[..., :3].contiguous()), graph(codes[..., :3].contiguous())))
            del graph.graphs[3]
        retained = graph(codes[..., :12].contiguous())
        snapshot = retained.clone()
        graph.reset()
        check('reset and owned output', not graph.graphs and torch.equal(retained, snapshot))
        limited = CodecGraphs(decoder, max_graphs=1, max_frames=25)
        limited.prepare(codes[..., :1].contiguous())
        for n in (26, 2):
            if n == 26:
                limited.reset()
            else:
                limited.prepare(codes[..., :1].contiguous())
            try:
                limited.prepare(codes[..., :n].contiguous())
            except ValueError:
                check('capacity reject ' + str(n))
            else:
                raise AssertionError('capacity')
        limited.reset()
        try:
            with graph.installed():
                raise LookupError('consumer failure')
        except LookupError:
            pass
        check('exception restores decoder', decoder.forward == graph.eager and not graph.active)
        decoder.train()
        try:
            graph(codes)
        except RuntimeError:
            check('reject training')
        else:
            raise AssertionError('training')
        finally:
            decoder.eval()
        # Fixed real-code history repeated long enough to exceed conservative context.
        long_codes = codes.repeat(1, 1, 640 // codes.shape[-1] + 1)[..., :640].contiguous()
        whole = decoder(long_codes).cpu()[0, 0]
        result['window_errors'] = []
        for ctx in (context, 72):
            chunks = []
            codec = ChunkedCodec(decoder, groups=codes.shape[1], hop=hop, sample_rate=sr,
                chunk_frames=25, first_chunk_frames=4, context_frames=ctx, on_audio=chunks.append)
            for frame in long_codes.unbind(-1):
                codec.push(frame)
            codec.finish()
            wav = torch.cat([c.samples for c in chunks])
            check('window alignment ' + str(ctx), wav.numel() == whole.numel() and all(c.start_sample == sum(z.samples.numel() for z in chunks[:i]) for i, c in enumerate(chunks)))
            boundary = torch.cat([(wav - whole)[max(0,c.start_sample-128):c.start_sample+128] for c in chunks[1:]])
            result['window_errors'].append(dict(context=ctx, approximate=ctx<context,
                error=errors(wav,whole), boundary_rmse=boundary.square().mean().sqrt().item(), boundary_max=boundary.abs().max().item()))
            save()
        # Capture each schedule's exact warm shapes; reset to bound graph storage.
        for text_id, real_codes in codes_by_text.items():
            for first, chunk, ctx in ((12,12,context),(1,25,context),(4,25,context),(4,25,72)):
                graph.reset()
                shapes = set()
                end = 0
                while end < real_codes.shape[-1]:
                    start = end
                    end = min(end + (first if end == 0 else chunk), real_codes.shape[-1])
                    shapes.add(end - max(0, start-ctx))
                setup_start = time.perf_counter()
                for n in sorted(shapes)[:graph.max_graphs]:
                    graph.prepare(real_codes[..., :n].contiguous())
                torch.cuda.synchronize()
                setup_s = time.perf_counter()-setup_start
                for variant in ('eager','graph'):
                    outputs = []
                    for repeat in range(2):
                        tg.reset()
                        torch.manual_seed(42)
                        chunks, arrivals = [], []
                        def receive(c):
                            arrivals.append(time.perf_counter()-start_time)
                            chunks.append(c)
                        torch.cuda.synchronize()
                        hits_before, misses_before = graph.hits, graph.misses
                        start_time = time.perf_counter()
                        kwargs = dict(text=TEXTS[text_id], on_audio=receive, first_chunk_frames=first,
                            chunk_frames=chunk, context_frames=ctx, approximate=ctx<context, **settings)
                        if variant == 'graph':
                            with graph.installed():
                                r = stream_custom_voice(model, **kwargs)
                        else:
                            r = stream_custom_voice(model, **kwargs)
                        torch.cuda.synchronize()
                        total = time.perf_counter()-start_time
                        wav = torch.cat([c.samples for c in chunks])
                        check('stream finite/aligned/EOS ' + str((text_id,first,ctx,variant,repeat)),
                            torch.isfinite(wav).all().item() and wav.numel()==r.frames*hop and r.frames==real_codes.shape[-1])
                        # Playback starts on first delivery; gaps accumulate after reserve exhaustion.
                        finish = arrivals[0]
                        starvation = 0.
                        for arrival,c in zip(arrivals,chunks):
                            starvation += max(0.,arrival-finish)
                            finish = max(finish,arrival)+c.samples.numel()/sr
                        outputs.append(wav)
                        row = dict(text=text_id, first=first, chunk=chunk, context=ctx, approximate=ctx<context,
                            variant=variant, repeat=repeat, frames=r.frames, first_audio_s=arrivals[0], total_s=total,
                            rtf=total/(wav.numel()/sr), starvation_s=starvation, arrivals_s=arrivals,
                            setup_s=setup_s, captures=len(graph.graphs), hits=graph.hits-hits_before, misses=graph.misses-misses_before)
                        reference = eager_outputs[repeat] if variant == 'graph' else outputs[0]
                        row['waveform_error'] = errors(wav, reference)
                        row['waveform_exact'] = torch.equal(wav, reference)
                        check('per-repeat waveform tolerance ' + str((text_id,first,ctx,variant,repeat)), row['waveform_error']['max_abs'] <= 1e-5)
                        result['streaming'].append(row)
                        save()
                    if variant == 'eager':
                        eager_outputs = outputs
                        eager_wav = outputs[-1]
                    else:
                        check('stream graph reference ' + str((text_id,first,ctx)), errors(outputs[-1],eager_wav)['max_abs'] <= 1e-5)
                        filename = audio / f'{text_id}-{first}-{chunk}-{ctx}.wav'
                        sf.write(filename, outputs[-1].numpy(), sr, subtype='FLOAT')
                        reopened, rate = sf.read(filename)
                        check('reopen ' + filename.name, rate==sr and len(reopened)==outputs[-1].numel())
                check('no request-time capture', len(graph.graphs) == min(len(shapes),graph.max_graphs))
        graph.reset()
    result['eos_observations'] = eos_observations
    check('all observed requests reached EOS', all(eos_observations))
    result['complete'] = True
    save()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
