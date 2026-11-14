from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time

GPU = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--size', required=True, choices=['0.6B', '1.7B'])
    p.add_argument('--label', default='accepted-stack')
    p.add_argument('--runs', type=int, default=3)
    p.add_argument('--stack', choices=['baseline', 'combined'], default='baseline')
    p.add_argument('--frame-loop', action='store_true')
    p.add_argument('--profile-frames', type=int, default=0)
    p.add_argument('--predictor-mode', choices=['pdl', 'nonpdl'], default='pdl')
    p.add_argument('--installer', action='append', default=[], metavar='MODULE:FUNCTION:TARGET')
    args = p.parse_args()
    if args.runs < 1 or Path(args.label).name != args.label:
        p.error('positive runs and single-component label required')
    if any(os.environ.get(k) != '8' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS')):
        raise RuntimeError('Set OMP_NUM_THREADS=8 and MKL_NUM_THREADS=8')
    if args.stack == 'combined' and args.size == '0.6B' and args.predictor_mode != 'nonpdl':
        p.error('0.6B combined fusion requires explicit --predictor-mode nonpdl')
    if args.frame_loop and args.stack != 'combined':
        p.error('frame loop requires combined stack')
    output_group = 'final' if args.frame_loop else args.stack
    tracked = list((ROOT / 'scripts').glob('qwen_*.py')) + list((ROOT / 'vendor/qwen_fused_predictor').glob('*.py')) + [Path(__file__), ROOT / 'scripts/bench_common.py']
    source_bytes = {str(path): path.read_bytes() for path in tracked}
    source_start = {path: hashlib.sha256(data).hexdigest() for path, data in source_bytes.items()}
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel
    from bench_common import TEXTS, system_info
    from qwen_fused_norm import install_fused_predictor_norm
    from qwen_predictor_graph import install_predictor_graph
    from qwen_talker_graph import install_talker_graph
    from qwen_fast_decode import install_fast_decode

    if args.stack == 'combined':
        from qwen_fused_predictor import install_fused_predictor
        from qwen_talker_fusion import install_talker_fusion
    if args.frame_loop:
        from qwen_frame_loop import install_frame_loop
    assert all(Path(path).read_bytes() == data for path, data in source_bytes.items()), 'sources changed during imports; retry with stable files'
    torch.set_num_threads(8)
    assert torch.cuda.device_count() == 1, 'Select exactly one GPU with CUDA_VISIBLE_DEVICES'
    torch.cuda.set_device(0)
    out = ROOT / f'results/gap/{output_group}' / f'{args.size.lower()}-{args.label}.json'
    if out.exists() or out.with_suffix('.partial.json').exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sample_dir = ROOT / f'samples/gap/{output_group}' / f'{args.size.lower()}-{args.label}'
    sample_dir.mkdir(parents=True, exist_ok=False)
    result = {'meta': system_info() | vars(args), 'rows': [], 'smoke': [], 'complete': False}
    meta = result['meta']
    meta['source_sha256_start'] = source_start
    if args.stack == 'combined':
        snapshot = out.parent / f'{args.size.lower()}-{args.label}-sources'
        snapshot.mkdir(exist_ok=False)
        for path, data in source_bytes.items():
            (snapshot / Path(path).name).write_bytes(data)
        meta['source_snapshot'] = str(snapshot.relative_to(ROOT))
    meta.update(command=f'CUDA_VISIBLE_DEVICES={GPU} OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 ' + shlex.join([sys.executable, *sys.argv]),
                gpu_uuid=GPU, torch=torch.__version__, cuda=torch.version.cuda,
                cudnn_sdpa_enabled=torch.backends.cuda.cudnn_sdp_enabled(), threads=torch.get_num_threads(),
                texts=TEXTS, timing='synchronized full generate_custom_voice, including codec; no streaming',
                seeds=list(range(args.runs)), warmup_seed=1234)
    meta['gpu_metadata'] = subprocess.check_output(['nvidia-smi', '-i', GPU, '--query-gpu=uuid,name,driver_version,memory.total,power.limit,clocks.sm,clocks.mem', '--format=csv'], text=True)
    def save():
        out.with_suffix('.partial.json').write_text(json.dumps(result, indent=2))
    save()
    start = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(f'Qwen/Qwen3-TTS-12Hz-{args.size}-CustomVoice', device_map='cuda:0', dtype=torch.bfloat16, attn_implementation='sdpa')
    torch.cuda.synchronize()
    meta['load_s'] = time.perf_counter() - start
    meta['model'] = model.model.config._name_or_path
    meta['resolved_defaults'] = model._merge_generate_kwargs()
    meta['model_generate_defaults'] = model.generate_defaults
    talker = model.model.talker
    predictor = talker.code_predictor
    handles, timings = [], []
    def install(fn, target, **kwargs):
        start = time.perf_counter()
        handle = fn(target, **kwargs)
        torch.cuda.synchronize()
        timings.append({'installer': f'{fn.__module__}:{fn.__name__}', 'kwargs': kwargs, 'seconds': time.perf_counter() - start})
        handles.append(handle)
        return handle
    install(install_fused_predictor_norm, predictor)
    pg = install(install_fused_predictor, predictor, mode=args.predictor_mode) if args.stack == 'combined' else install(install_predictor_graph, predictor)
    meta['effective_predictor_mode'] = getattr(pg, 'mode', 'strict')
    if args.stack == 'combined':
        install(install_talker_fusion, talker)
    tg = install(install_talker_graph, talker)
    install(install_fast_decode, talker)
    loop = install(install_frame_loop, talker) if args.frame_loop else None
    for spec in args.installer:
        module, function, target = spec.split(':')
        install(getattr(importlib.import_module(module), function), {'model': model, 'talker': talker, 'predictor': predictor}[target])
    meta['install_timings'] = timings
    paths = {Path(__file__), ROOT / 'scripts/bench_common.py'}
    for obj in [model.__class__, model.model.__class__, talker.__class__, *[type(h) for h in handles]]:
        paths.add(Path(inspect.getfile(obj)))
    meta['source_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}
    meta['model_config_sha256'] = hashlib.sha256(model.model.config.to_json_string(use_diff=False).encode()).hexdigest()
    observed = {}
    original_generate = talker.generate
    def observe(*a, **kw):
        observed['kwargs'] = kw
        output = original_generate(*a, **kw)
        observed['output'] = output
        observed['samplers'] = {k: kw.get(k) for k in ['do_sample', 'temperature', 'top_k', 'top_p', 'subtalker_dosample', 'subtalker_temperature', 'subtalker_top_k', 'subtalker_top_p', 'repetition_penalty', 'max_new_tokens', 'eos_token_id', 'min_new_tokens', 'suppress_tokens']}
        return output
    talker.generate = observe
    def captures():
        def graphs(handle):
            return [{'key': str(k), 'capture_s': g.capture_seconds, 'setup_s': g.setup_seconds} for k, g in handle.graphs.items()]
        return {'talker_capture_s': list(tg.capture_seconds), 'predictor': graphs(pg),
                'strict_fallback': graphs(pg.strict) if hasattr(pg, 'strict') else []}
    @torch.inference_mode()
    def request(voice, tid, seed, overrides=None):
        tg.reset()
        torch.manual_seed(seed)
        before = tg.replays
        loop_before = (loop.calls, loop.fallback_calls) if loop else (0, 0)
        fallback_before = getattr(pg, 'fallback_calls', 0)
        torch.cuda.synchronize()
        start = time.perf_counter()
        wavs, sr = model.generate_custom_voice(text=TEXTS[tid], speaker=voice, language='English', **(overrides or {}))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        raw = observed.pop('output')
        seq = raw.sequences[0].tolist()
        frames = [h[-1] for h in raw.hidden_states if h[-1] is not None]
        eos_id = talker.config.codec_eos_token_id
        first_codes = [int(f[0, 0]) for f in frames]
        effective = first_codes.index(eos_id) if eos_id in first_codes else len(first_codes)
        audio = wavs[0]
        assert len(audio) > 0 and np.isfinite(audio).all()
        eos = seq[-1] == eos_id
        row = {'seed': seed, 'gen_s': elapsed, 'audio_s': len(audio)/sr, 'rtf': elapsed/(len(audio)/sr),
               'samples': len(audio), 'sample_rate': sr, 'finite': True, 'frame_count': effective,
               'raw_codec_frames': len(frames), 'sequence_tokens': len(seq), 'last_token': seq[-1],
               'eos_token': eos_id, 'eos': eos, 'truncated': not eos, 'decode_replays': tg.replays - before,
               'fallback_calls': getattr(pg, 'fallback_calls', 0) - fallback_before,
               'loop_calls': loop.calls - loop_before[0] if loop else 0,
               'loop_fallback_calls': loop.fallback_calls - loop_before[1] if loop else 0,
               'samplers': observed['samplers'], 'ttfa_s': None}
        return row, audio
    for voice in ['ryan', 'aiden']:
        for tid in TEXTS:
            warm, _ = request(voice, tid, 1234)
            capture_before = captures()
            torch.cuda.reset_peak_memory_stats()
            calls = []
            for seed in range(args.runs):
                call, audio = request(voice, tid, seed)
                path = sample_dir / f'{voice}_{tid}_seed{seed}.wav'
                sf.write(path, audio, call['sample_rate'])
                call['sample'] = str(path.relative_to(ROOT))
                calls.append(call)
            assert captures() == capture_before, 'capture occurred during timed repetitions'
            row = {'voice': voice, 'text_id': tid, 'warmup': warm, 'calls': calls,
                   'gen_s': statistics.median(c['gen_s'] for c in calls),
                   'rtf': statistics.median(c['rtf'] for c in calls),
                   'peak_gpu_mb': torch.cuda.max_memory_allocated()/2**20}
            result['rows'].append(row)
            meta['captures_after_suite'] = captures()
            save()
            print(json.dumps({'size': args.size, 'voice': voice, 'text': tid, 'rtf': row['rtf'], 'all_eos': all(c['eos'] for c in calls)}), flush=True)
    original_predict = predictor.generate
    predictor_settings = []
    def observe_predict(*a, **kw):
        if not predictor_settings:
            predictor_settings.append({k: kw.get(k) for k in ['do_sample', 'temperature', 'top_k', 'top_p']})
        return original_predict(*a, **kw)
    predictor.generate = observe_predict
    for overrides in [{'temperature': 0.65}, {'subtalker_temperature': 1.15}, {'temperature': 0.7, 'subtalker_temperature': 1.1, 'subtalker_top_p': 0.85}]:
        predictor_settings.clear()
        call, audio = request('ryan', 'short', 321, overrides)
        expected = model._merge_generate_kwargs(**overrides)
        assert call['samplers']['temperature'] == expected['temperature']
        if loop and call['loop_calls']:
            key = (expected['subtalker_dosample'], expected['subtalker_temperature'], expected['subtalker_top_k'])
            assert key in pg.graphs
            assert call['loop_fallback_calls'] == 0
            predictor_settings.append({'temperature': key[1], 'top_k': key[2], 'top_p': 1.0,
                                       'evidence': 'explicit loop selected fused graph key; predictor.generate bypassed'})
        assert predictor_settings[0]['temperature'] == expected['subtalker_temperature']
        assert predictor_settings[0]['top_p'] == expected['subtalker_top_p']
        assert call['eos']
        if args.stack == 'combined':
            assert (call['fallback_calls'] > 0) == (expected['subtalker_top_p'] != 1.0)
        path = sample_dir / f'smoke_{len(result["smoke"])}.wav'
        sf.write(path, audio, call['sample_rate'])
        result['smoke'].append({'overrides': overrides, 'predictor_observed': predictor_settings[0], 'call': call, 'sample': str(path.relative_to(ROOT)), 'passed': True})
        save()
    predictor.generate = original_predict
    talker.generate = original_generate
    if args.profile_frames:
        kw = dict(observed['kwargs'])
        kw.update(max_new_tokens=args.profile_frames + 1, min_new_tokens=args.profile_frames + 1,
                  temperature=0.9, subtalker_temperature=0.9, subtalker_top_p=1.0)
        result['fixed_work'] = {}
        for name, generate in [('without_loop', loop.original), ('with_loop', loop.generate)]:
            @torch.inference_mode()
            def fixed():
                tg.reset()
                torch.manual_seed(0)
                with torch.inference_mode():
                    return generate(**kw)
            fixed()
            before_capture = captures()
            measurements = []
            for _ in range(3):
                torch.cuda.synchronize()
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start = time.perf_counter()
                a.record()
                raw = fixed()
                b.record()
                b.synchronize()
                frames = sum(h[-1] is not None for h in raw.hidden_states)
                assert frames == args.profile_frames
                measurements.append({'wall_ms': (time.perf_counter()-start)*1000, 'cuda_event_span_ms': a.elapsed_time(b), 'frames': frames})
            assert before_capture == captures()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                fixed()
                torch.cuda.synchronize()
            trace = out.parent / f'{args.size.lower()}-{args.label}-{name}-trace.json'
            prof.export_chrome_trace(str(trace))
            events = json.loads(trace.read_text())['traceEvents']
            kernels = [e for e in events if e.get('cat') == 'kernel' and 'dur' in e]
            syncs = [e for e in events if e.get('cat') == 'cuda_runtime' and 'Synchronize' in e.get('name', '')]
            result['fixed_work'][name] = {'measurements': measurements, 'trace': str(trace.relative_to(ROOT)),
                'kernel_count': len(kernels), 'summed_kernel_ms': sum(e['dur'] for e in kernels)/1000,
                'cuda_runtime_sync_count': len(syncs), 'cuda_runtime_sync_ms': sum(e.get('dur',0) for e in syncs)/1000,
                'scope': 'Direct talker.generate including prefill and 72 materialized frames; excludes tokenizer and codec. CUDA event span includes host scheduling gaps; kernel durations from one instrumented profiler run, not additive wall-time decomposition.'}
            save()
    meta['captures_final'] = captures()
    meta['source_sha256_end'] = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in source_start}
    meta['sources_changed_during_run'] = [path for path, digest in source_start.items() if meta['source_sha256_end'][path] != digest]
    meta['source_provenance'] = 'Candidate Python modules imported before installation; start snapshots identify loaded source. End hashes identify concurrent disk changes, not loaded code.'
    result['summary'] = {'mean_row_median_rtf': statistics.mean(r['rtf'] for r in result['rows']),
                         'all_eos': all(c['eos'] for r in result['rows'] for c in r['calls'])}
    assert result['summary']['all_eos']
    assert all(c['fallback_calls'] == 0 for r in result['rows'] for c in r['calls'])
    if loop:
        assert all(c['loop_calls'] == 1 and c['loop_fallback_calls'] == 0 for r in result['rows'] for c in r['calls'])
    assert not meta['sources_changed_during_run'], 'source stability required for final acceptance'
    result['complete'] = True
    save()
    out.with_suffix('.partial.json').replace(out)
    print(out, result['summary'], flush=True)


if __name__ == '__main__':
    main()
