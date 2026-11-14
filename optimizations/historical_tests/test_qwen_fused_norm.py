"""GPU2-only numerical and performance experiment for predictor fused RMSNorm."""
import os
GPU = 'GPU-3e620c1c-c243-7353-580e-cde9e5eb4b8a'
if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU or any(os.environ.get(k) != '8' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS')):
    raise RuntimeError('Set GPU2 UUID visibility and OMP/MKL threads=8 before running')
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
import numpy as np
import soundfile as sf
import torch
import triton
from qwen_tts import Qwen3TTSModel
from bench_common import TEXTS
from qwen_fused_norm import install_fused_predictor_norm
from qwen_predictor_graph import install_predictor_graph
from qwen_talker_graph import install_talker_graph

ROOT = Path(__file__).resolve().parent.parent / 'results/kernel-profile/fused-norm'


def errors(x, ref):
    d = x.float() - ref.float()
    return {'max_abs': d.abs().max().item(), 'relative_l2': (d.norm() / ref.float().norm().clamp_min(1e-30)).item(),
            'max_relative_floor_1e-6': (d.abs() / ref.float().abs().clamp_min(1e-6)).max().item(),
            'finite': bool(torch.isfinite(x).all())}


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--size', choices=['0.6B', '1.7B'], required=True)
    args = p.parse_args()
    out = ROOT / args.size
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    assert torch.cuda.device_count() == 1
    assert str(torch.cuda.get_device_properties(0).uuid) == GPU.removeprefix('GPU-')
    result = {'size': args.size, 'torch': torch.__version__, 'triton': triton.__version__, 'cuda': torch.version.cuda,
              'gpu': str(torch.cuda.get_device_properties(0)), 'threads': 8,
              'command': f'CUDA_VISIBLE_DEVICES={GPU} OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 {sys.executable} scripts/test_qwen_fused_norm.py --size {args.size}',
              'sdpa': {k: getattr(torch.backends.cuda, k + '_sdp_enabled')() for k in ['flash','math','mem_efficient','cudnn']},
              'numerics': [], 'variants': {}, 'complete': False}
    def save():
        (out / 'summary.json').write_text(json.dumps(result, indent=2))
    save()
    model = Qwen3TTSModel.from_pretrained(f'Qwen/Qwen3-TTS-12Hz-{args.size}-CustomVoice', device_map='cuda:0', dtype=torch.bfloat16, attn_implementation='sdpa')
    predictor = model.model.talker.code_predictor
    defaults = model._merge_generate_kwargs()
    result['sampling_defaults'] = defaults
    actual = {}
    hooks = []
    for name, module in predictor.model.named_modules():
        if module.__class__.__name__ == 'Qwen3TTSRMSNorm':
            def hook(m, inputs, name=name):
                if name not in actual:
                    actual[name] = inputs[0].detach().clone()
            hooks.append(module.register_forward_pre_hook(hook))
    captured = {}
    original_generate = predictor.generate
    def capture(*a, **kw):
        if not captured:
            captured.update({k: v.detach().clone() if isinstance(v, torch.Tensor) else v for k,v in kw.items()})
        return original_generate(*a, **kw)
    predictor.generate = capture
    torch.manual_seed(0)
    model.generate_custom_voice(text=TEXTS['short'], speaker='ryan', language='English')
    predictor.generate = original_generate
    for h in hooks:
        h.remove()
    result['predictor_config'] = predictor.config.to_dict()
    result['captured_generation_kwargs'] = {k: (list(v.shape) if isinstance(v, torch.Tensor) else v) for k,v in captured.items()}
    embeds = captured['inputs_embeds']
    # Same actual initial predictor input through all transformer layers, no sampling.
    def logits():
        x = predictor.small_to_mtp_projection(embeds)
        hidden = predictor.model(inputs_embeds=x, use_cache=False, return_dict=True).last_hidden_state
        return predictor.lm_head[0](hidden[:, -1]).float()
    upstream_logits = logits()
    handle = install_fused_predictor_norm(predictor)
    result['patched_modules'] = len(handle.entries)
    result['widths'] = sorted(handle.widths)
    for name, module in predictor.model.named_modules():
        if name not in actual:
            continue
        original = next(orig for m,orig,_ in handle.entries if m is module)
        cases = {'actual': actual[name]}
        for scale in [0.0, 1e-8, 1.0, 1e4]:
            cases[f'scale_{scale}'] = (torch.randn_like(actual[name].float()) * scale).to(actual[name].dtype)
        cases['noncontiguous_fallback'] = torch.randn((*actual[name].shape, 2), device='cuda', dtype=torch.bfloat16)[..., 0]
        cases['longer_fallback'] = torch.randn((1, 3, *actual[name].shape[2:]), device='cuda', dtype=torch.bfloat16)
        for case,x in cases.items():
            y = module(x)
            baseline = original(x)
            z = x.float()
            ref = z * torch.rsqrt(z.square().mean(-1, keepdim=True) + module.variance_epsilon)
            ref = ref * module.weight.float()  # FP32 reference intentionally omits BF16 intermediate rounding.
            row = {'module': name, 'shape': list(x.shape), 'case': case,
                   'vs_upstream': errors(y, baseline), 'vs_fp32': errors(y,ref), 'upstream_vs_fp32': errors(baseline,ref)}
            assert row['vs_upstream']['finite']
            assert row['vs_upstream']['relative_l2'] < 0.01
            if 'fallback' in case:
                assert torch.equal(y, baseline)
            result['numerics'].append(row)
    fused_logits = logits()
    result['logits'] = errors(fused_logits, upstream_logits)
    result['logits']['description'] = 'First predictor code-group logits, actual two-token input, propagated through all predictor layers; not sampled equality or full teacher-forced decode.'
    assert result['logits']['finite'] and result['logits']['relative_l2'] < 0.05
    result['guards'] = []
    m = handle.entries[0][0]
    x = next(iter(actual.values()))
    def rejected(label, fn):
        try:
            fn()
        except ValueError:
            result['guards'].append(label)
        else:
            raise AssertionError('Expected rejection: ' + label)
    rejected('duplicate_install', lambda: install_fused_predictor_norm(predictor))
    rejected('dtype_mismatch', lambda: m(x.float()))
    rejected('cpu_input', lambda: m(x.cpu()))
    m.train()
    rejected('training', lambda: m(x))
    m.eval()
    with torch.enable_grad():
        rejected('grad_enabled', lambda: m(x))
    with torch.autocast('cuda', dtype=torch.bfloat16):
        rejected('autocast', lambda: m(x))
    handle.uninstall()
    assert torch.equal(logits(), upstream_logits)
    result['uninstall_restores_logits'] = True
    save()
    # Separate graph captures for every variant; never reuse a captured old kernel.
    for variant in ['baseline', 'fused']:
        norm = install_fused_predictor_norm(predictor) if variant == 'fused' else None
        graph = install_predictor_graph(predictor)
        start = time.perf_counter()
        predictor.generate(**captured)
        torch.cuda.synchronize()
        cold = time.perf_counter() - start
        fixed = next(iter(graph.graphs.values()))
        for _ in range(10):
            fixed.replay(embeds)
        torch.cuda.synchronize()
        gpu_ms, wall_ms = [], []
        for _ in range(7):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start = time.perf_counter()
            a.record()
            for _ in range(100):
                fixed.replay(embeds)
            b.record()
            b.synchronize()
            gpu_ms.append(a.elapsed_time(b) / 100)
            wall_ms.append((time.perf_counter() - start) * 1000 / 100)
        v = {'cold_call_s': cold, 'capture_s': fixed.capture_seconds, 'setup_s': fixed.setup_seconds,
             'fixed_replay_gpu_ms': gpu_ms, 'fixed_replay_wall_ms': wall_ms,
             'median_fixed_gpu_ms': statistics.median(gpu_ms), 'median_fixed_wall_ms': statistics.median(wall_ms), 'e2e': []}
        result['variants'][variant] = v
        talker = install_talker_graph(model.model.talker)
        for tid in ['short', 'medium']:
            for _ in range(2):
                torch.manual_seed(0)
                model.generate_custom_voice(text=TEXTS[tid], speaker='ryan', language='English')
            for seed in [0, 1, 2]:
                torch.manual_seed(seed)
                before = talker.replays
                torch.cuda.synchronize()
                start = time.perf_counter()
                wavs, sr = model.generate_custom_voice(text=TEXTS[tid], speaker='ryan', language='English')
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                wav = wavs[0]
                steps = talker.replays - before
                # Actual configured limit is recorded rather than inventing an EOS assertion.
                generation_config = model.model.generation_config.to_dict()
                sample = out / f'{variant}-{tid}-seed{seed}.wav'
                sf.write(sample, wav, sr)
                row = {'text': tid, 'seed': seed, 'seconds': elapsed, 'audio_s': len(wav)/sr,
                       'rtf': elapsed/(len(wav)/sr), 'decode_replays': steps, 'finite': bool(np.isfinite(wav).all()),
                       'nonempty': bool(len(wav)), 'peak_abs': float(np.max(np.abs(wav))),
                       'sample': str(sample.relative_to(ROOT)), 'sample_rate': sr,
                       'below_adapter_cache_capacity': steps < talker.capacity - 1,
                       'generation_cap': defaults['max_new_tokens'],
                       'hit_generation_cap': steps + 1 >= defaults['max_new_tokens']}
                assert not row['hit_generation_cap']
                assert row['finite'] and row['nonempty'] and row['below_adapter_cache_capacity']
                v['e2e'].append(row)
                save()
                print(args.size, variant, json.dumps(row), flush=True)
        v['talker_capture_s'] = talker.capture_seconds
        v['generation_config'] = generation_config
        talker.uninstall()
        graph.uninstall()
        del fixed, graph, talker
        if norm:
            norm.uninstall()
        save()
    result['complete'] = True
    save()

if __name__ == '__main__':
    main()
