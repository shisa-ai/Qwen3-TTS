"""Numerical checks and isolated-GPU sampled FP8/BF16 CustomVoice benchmark."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

GPU = 'GPU-3e620c1c-c243-7353-580e-cde9e5eb4b8a'
if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU:
    raise RuntimeError(f'CUDA_VISIBLE_DEVICES must be {GPU}')
if any(os.environ.get(k) != '8' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS')):
    raise RuntimeError('OMP_NUM_THREADS and MKL_NUM_THREADS must be 8')

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from qwen_fp8 import install_fp8, quantize, scaled_linear, FP8_MAX

ROOT = Path(__file__).resolve().parent.parent


def errors(actual, reference):
    a, r = actual.float(), reference.float()
    return dict(relative_l2=((a-r).norm()/r.norm().clamp_min(1e-12)).item(),
                max_absolute=(a-r).abs().max().item(), finite=bool(torch.isfinite(a).all()))


def check(x, w, bias=None):
    q, s = quantize(w)
    a, sa = quantize(x)
    y = scaled_linear(x, q, s, bias)
    bf16 = F.linear(x, w, bias)
    fp32 = F.linear(x.float(), w.float(), None if bias is None else bias.float())
    return dict(shape=list(x.shape), weight_shape=list(w.shape),
                vs_bf16=errors(y, bf16), vs_fp32=errors(y, fp32),
                bf16_vs_fp32=errors(bf16, fp32),
                input_nonfinite=int((~torch.isfinite(x)).sum()),
                activation_scale=sa.item(), weight_scale=s.item(),
                activation_clipped=int((x.float().abs()/sa > FP8_MAX).sum()),
                weight_clipped=int((w.float().abs()/s > FP8_MAX).sum()),
                activation_at_max=int((a.float().abs() == FP8_MAX).sum()),
                weight_at_max=int((q.float().abs() == FP8_MAX).sum()))


@torch.inference_mode()
def unit_tests():
    torch.manual_seed(42)
    rows = []
    for m, k, n in [(1, 128, 256), (16, 1024, 1024), (17, 256, 128)]:
        x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
        w = torch.randn(n, k, device='cuda', dtype=torch.bfloat16) / k**0.5
        bias = torch.randn(n, device='cuda', dtype=torch.bfloat16)
        row = check(x, w, bias)
        assert row['vs_fp32']['finite'] and row['vs_fp32']['relative_l2'] < .08
        assert row['activation_clipped'] == row['weight_clipped'] == 0
        rows.append(row)
    x = torch.zeros(1, 128, device='cuda', dtype=torch.bfloat16)
    w = torch.zeros(256, 128, device='cuda', dtype=torch.bfloat16)
    assert scaled_linear(x, *quantize(w)).count_nonzero() == 0
    # Synthetic instance tests: unsupported weight shape and output head untouched.
    from types import SimpleNamespace
    talker = torch.nn.Module()
    talker.self_attn = torch.nn.ModuleDict({'aligned': torch.nn.Linear(128, 256),
                                          'unaligned': torch.nn.Linear(127, 255)})
    talker.head = torch.nn.Linear(128, 256)
    talker = talker.cuda().bfloat16().eval()
    model = SimpleNamespace(model=SimpleNamespace(talker=talker))
    original = talker.self_attn['aligned'].forward
    head = talker.head.forward
    patch = install_fp8(model)
    assert len(patch.entries) == 1 and len(patch.skipped) == 1
    assert talker.head.forward == head
    assert talker.self_attn['aligned'](x).dtype == torch.bfloat16
    talker.self_attn['unaligned'](x[:, :127])
    patch.remove()
    assert talker.self_attn['aligned'].forward == original
    assert 'forward' not in talker.self_attn['aligned'].__dict__
    return rows


def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(path)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', choices=['0.6B', '1.7B'])
    parser.add_argument('--mode', choices=['baseline', 'fp8'])
    parser.add_argument('--unit', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(8)
    assert torch.cuda.device_count() == 1
    output_dir = ROOT / 'results/fp8'
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.unit:
        report = dict(passed=True, cases=unit_tests(), torch=torch.__version__)
        save(output_dir / 'unit.json', report)
        print(json.dumps(report), flush=True)
        return
    if not args.size or not args.mode:
        parser.error('--size and --mode required without --unit')
    from qwen_tts import Qwen3TTSModel
    from bench_common import TEXTS
    out = output_dir / f'{args.size}-{args.mode}.json'
    if out.exists():
        raise FileExistsError(out)
    sample_dir = ROOT / 'samples/fp8' / f'{args.size}-{args.mode}'
    sample_dir.mkdir(parents=True, exist_ok=True)
    report = dict(complete=False, meta=dict(command=sys.argv, executable=sys.executable,
        gpu_uuid=GPU, gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
        cuda=torch.version.cuda, threads=torch.get_num_threads(), size=args.size, mode=args.mode,
        attention='sdpa', graphs=False, cudnn_sdpa=torch.backends.cuda.cudnn_sdp_enabled(),
        seed=42, warmups=1, timed_runs=3, text=TEXTS['medium'], voice='ryan',
        dtype='bfloat16', non_streaming_mode=True), rows=[])
    start = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(f'Qwen/Qwen3-TTS-12Hz-{args.size}-CustomVoice',
        local_files_only=True, device_map='cuda:0', dtype=torch.bfloat16, attn_implementation='sdpa')
    torch.cuda.synchronize()
    report['meta']['load_s'] = time.perf_counter()-start
    settings = model._merge_generate_kwargs()
    report['meta']['sampling'] = settings
    assert settings['do_sample'] and settings['subtalker_dosample']
    talker = model.model.talker
    observed = {}
    original_generate = talker.generate
    def observe(*a, **kw):
        result = original_generate(*a, **kw)
        observed['sequences'] = result.sequences.detach()
        return result
    talker.generate = observe

    def call(batch, cap=None):
        torch.manual_seed(42)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        wavs, sr = model.generate_custom_voice(text=[TEXTS['medium']]*batch,
            language=['English']*batch, speaker=['ryan']*batch, non_streaming_mode=True,
            **({} if cap is None else {'max_new_tokens': cap}))
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        peak, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
        seq = observed['sequences'].cpu()
        eos = (seq == talker.config.codec_eos_token_id).any(-1).tolist()
        durations = [len(w)/sr for w in wavs]
        assert len(wavs) == batch and all(len(w) and np.isfinite(w).all() for w in wavs)
        return dict(wall_s=elapsed, audio_s=durations, aggregate_audio_s=sum(durations),
            aggregate_audio_s_per_wall_s=sum(durations)/elapsed, peak_allocated_bytes=peak,
            peak_reserved_bytes=reserved, eos_per_item=eos, cap_hit=not all(eos),
            generated_sequence_length=seq.shape[-1], sample_rate=sr,
            waveform_peak=[float(np.abs(w).max()) for w in wavs]), wavs, sr

    patch = None
    try:
        if args.mode == 'fp8':
            # Real activations from BF16 sampled generation, not random-only testing.
            validation, hooks = {}, []
            for name, module in talker.named_modules():
                if isinstance(module, torch.nn.Linear) and any(p in name.split('.') for p in ('self_attn', 'mlp')):
                    if any(d % 16 for d in module.weight.shape):
                        continue
                    def validate(mod, inputs, _name=name):
                        if _name not in validation:
                            validation[_name] = check(inputs[0].reshape(-1, mod.in_features)[:128], mod.weight, mod.bias)
                    hooks.append(module.register_forward_pre_hook(validate))
            start = time.perf_counter()
            try:
                call(16, cap=16)
            finally:
                for hook in hooks:
                    hook.remove()
            report['validation'] = validation
            report['meta']['validation_s'] = time.perf_counter()-start
            # Local-output threshold is a rejection gate, not an audio quality claim.
            assert validation and all(v['vs_fp32']['finite'] and v['vs_fp32']['relative_l2'] < .1
                and v['input_nonfinite'] == v['activation_clipped'] == v['weight_clipped'] == 0
                for v in validation.values()), 'FP8 realistic activation validation failed'
            torch.cuda.empty_cache()
            before = torch.cuda.memory_allocated()
            start = time.perf_counter()
            patch = install_fp8(model)
            torch.cuda.synchronize()
            report['meta']['conversion_s'] = time.perf_counter()-start
            report['meta']['conversion_allocated_bytes'] = torch.cuda.memory_allocated()-before
            report['patch'] = patch.summary
        for batch in (1, 16):
            row = dict(batch=batch, runs=[])
            report['rows'].append(row)
            row['warmup'], _, _ = call(batch)
            save(out, report)
            assert not row['warmup']['cap_hit'], 'Warmup hit generation cap'
            for run in range(3):
                record, wavs, sr = call(batch)
                row['runs'].append(record)
                if run == 0:
                    for i, wav in enumerate(wavs):
                        sf.write(sample_dir / f'b{batch}-item{i}.wav', wav, sr)
                save(out, report)
                assert not record['cap_hit'], 'Timed generation hit cap'
            row['median_wall_s'] = statistics.median(r['wall_s'] for r in row['runs'])
            row['median_aggregate_audio_s_per_wall_s'] = statistics.median(r['aggregate_audio_s_per_wall_s'] for r in row['runs'])
            save(out, report)
            print(json.dumps(row), flush=True)
        report['complete'] = True
    except Exception as exc:
        import traceback
        report['error'] = repr(exc)
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        if patch:
            patch.remove()
        talker.generate = original_generate
        save(out, report)
    print(f'COMPLETE {out}', flush=True)


if __name__ == '__main__':
    main()
