"""GPU2 regression and matched timing for the request-owned intermediate."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

GPU = 'GPU-3e620c1c-c243-7353-580e-cde9e5eb4b8a'


def processor_contract_tests():
    import torch
    from transformers.generation.logits_process import (
        RepetitionPenaltyLogitsProcessor, SuppressTokensLogitsProcessor,
        MinNewTokensLengthLogitsProcessor, TemperatureLogitsWarper,
        TopKLogitsWarper, TopPLogitsWarper)
    history = torch.tensor([[1, 1, 2]])
    logits = torch.tensor([[2., -3., 4., 1., 0.]])
    penalized = RepetitionPenaltyLogitsProcessor(2.0)(history, logits.clone())
    torch.testing.assert_close(penalized, torch.tensor([[2., -6., 2., 1., 0.]]))
    suppressed = SuppressTokensLogitsProcessor([0, 4], device='cpu')(history, logits.clone())
    assert torch.isneginf(suppressed[0, [0, 4]]).all()
    minimum = MinNewTokensLengthLogitsProcessor(3, 2, 4, device='cpu')
    assert torch.isneginf(minimum(history, logits.clone())[0, 4])
    assert minimum(torch.tensor([[1, 1, 2, 2, 2]]), logits.clone())[0, 4] == 0
    torch.testing.assert_close(TemperatureLogitsWarper(0.5)(history, logits.clone()), logits * 2)
    assert torch.isfinite(TopKLogitsWarper(2)(history, logits.clone())).sum() == 2
    assert torch.isfinite(TopPLogitsWarper(0.5)(history, logits.clone())).sum() == 1
    assert logits.argmax(-1).item() == 2
    return ['negative and duplicate repetition history', 'suppression', 'minimum EOS boundary',
            'temperature', 'top-k', 'top-p', 'greedy argmax']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--size', required=True, choices=['0.6B', '1.7B'])
    args = p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == GPU
    assert all(os.environ.get(k) == '8' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'))
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel
    from qwen_fused_norm import install_fused_predictor_norm
    from qwen_fused_predictor import install_fused_predictor
    from qwen_talker_fusion import install_talker_fusion
    from qwen_talker_graph import install_talker_graph
    from qwen_fast_decode import install_fast_decode
    from qwen_frame_decode import install_frame_decode
    from bench_common import TEXTS
    torch.set_num_threads(8)
    root = Path(__file__).resolve().parent.parent
    out = root / 'results/gap/frame-decode' / (args.size + '-v1.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    audio_dir = root / 'samples/gap/frame-decode' / args.size
    audio_dir.mkdir(parents=True, exist_ok=True)
    result = {'size': args.size, 'gpu': GPU, 'rows': [], 'checks': processor_contract_tests(), 'complete': False}
    def save():
        out.write_text(json.dumps(result, indent=2))
    def check(name, ok=True):
        assert ok, name
        result['checks'].append(name)
        save()
    model = Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-' + args.size + '-CustomVoice',
        device_map='cuda:0', dtype=torch.bfloat16, attn_implementation='sdpa')
    talker = model.model.talker
    install_fused_predictor_norm(talker.code_predictor)
    pg = install_fused_predictor(talker.code_predictor, mode='nonpdl' if args.size == '0.6B' else 'pdl')
    install_talker_fusion(talker)
    tg = install_talker_graph(talker)
    install_fast_decode(talker)
    # Save exact wrapper inputs once; replay with original HF generate for matched work.
    original = talker.generate
    captured = {}
    def observe(*a, **kw):
        captured.update(kw)
        return original(*a, **kw)
    talker.generate = observe
    with torch.inference_mode():
        model.generate_custom_voice(text=TEXTS['short'], speaker='ryan', language='English', do_sample=False, subtalker_dosample=False)
    del talker.generate
    check('restored original generate', talker.generate == original)
    base = dict(captured)
    base.update(do_sample=False, subtalker_dosample=False, max_new_tokens=24, min_new_tokens=24)
    @torch.inference_mode()
    def raw(kw):
        tg.reset()
        torch.manual_seed(123)
        torch.cuda.synchronize()
        start = time.perf_counter()
        value = talker.generate(**kw)
        torch.cuda.synchronize()
        return value, time.perf_counter() - start
    ref, _ = raw(base)
    baseline = [raw(base)[1] for _ in range(3)]
    fd = install_frame_decode(talker)
    got, _ = raw(base)
    check('fast path exercised', fd.fast_calls > 0 and fd.fallback_calls == 0)
    check('greedy fixed-work sequences exact', torch.equal(ref.sequences, got.sequences))
    check('real hidden-state count', len(ref.hidden_states) == len(got.hidden_states))
    for a, b in zip(ref.hidden_states, got.hidden_states):
        if a[-1] is not None:
            assert torch.equal(a[-1], b[-1])
        for x, y in zip(a[0], b[0]):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
    check('all retained real hidden states exact')
    candidate = [raw(base)[1] for _ in range(3)]
    check('retained outputs survive replay', torch.equal(ref.sequences, got.sequences))
    result['fixed_work'] = {'tokens': 24, 'baseline_s': baseline, 'candidate_s': candidate,
        'speedup': statistics.median(baseline) / statistics.median(candidate)}
    # Unsupported hooks remain in HF on the strict graph path.
    before = fd.fallback_calls
    raw(base | {'output_scores': True})
    check('unsupported output option strict fallback', fd.fallback_calls == before + 1)
    with torch.inference_mode(), torch.cuda.stream(torch.cuda.Stream()):
        try:
            talker.generate(**base)
        except ValueError:
            check('cross-stream rejected')
        else:
            raise AssertionError('cross-stream accepted')
    old_capacity = tg.capacity
    tg.capacity = base['inputs_embeds'].shape[1] + 2
    try:
        raw(base)
    except ValueError as e:
        check('cache capacity raises', 'capacity' in str(e))
    else:
        raise AssertionError('capacity accepted')
    finally:
        tg.capacity = old_capacity
    raw(base)
    check('request resets after failure', not fd.active)
    fd.uninstall()
    check('uninstall restores forward', talker.model.forward == tg.forward)
    check('uninstall restores generate', talker.generate == original)
    # Same HF processors/samplers, different position plumbing. Check controls
    # against strict execution with identical RNG states, not just distribution.
    for label, options in [
        ('sampled independent temperatures', dict(do_sample=True, temperature=0.7, subtalker_dosample=True, subtalker_temperature=1.1)),
        ('top-p strict predictor fallback', dict(do_sample=True, top_p=0.85, top_k=12, subtalker_dosample=True, subtalker_top_p=0.85)),
        ('repetition and suppression', dict(repetition_penalty=1.2, suppress_tokens=base['suppress_tokens'] + [1, 2, 3])),
    ]:
        settings = base | options
        reference, _ = raw(settings)
        handle = install_frame_decode(talker)
        actual, _ = raw(settings)
        check(label + ' exact seeded sequence', torch.equal(reference.sequences, actual.sequences))
        handle.uninstall()
    # Matched seeded end-to-end runs, including codec; warmup excluded.
    for text_id in ['short', 'medium']:
        outputs = {}
        for variant in ['baseline', 'candidate']:
            handle = install_frame_decode(talker) if variant == 'candidate' else None
            effective_generate = talker.generate
            observed = {}
            def record(*a, **kw):
                value = effective_generate(*a, **kw)
                observed['value'] = value
                return value
            talker.generate = record
            runs = []
            for seed in [1234, 0, 1, 2]:
                with torch.inference_mode():
                    tg.reset()
                torch.manual_seed(seed)
                before_capture = list(tg.capture_seconds)
                with torch.inference_mode():
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    waves, sr = model.generate_custom_voice(text=TEXTS[text_id], speaker='ryan', language='English')
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                raw_output = observed['value']
                check(f'{text_id}/{variant}/{seed} EOS', raw_output.sequences[0, -1].item() == talker.config.codec_eos_token_id)
                check(f'{text_id}/{variant}/{seed} finite audio', np.isfinite(waves[0]).all() and len(waves[0]) > 0)
                check(f'{text_id}/{variant}/{seed} no capture', before_capture == tg.capture_seconds)
                if seed != 1234:
                    path = audio_dir / f'{text_id}-{variant}-{seed}.wav'
                    sf.write(path, waves[0], sr)
                    runs.append({'seed': seed, 'seconds': elapsed, 'audio_seconds': len(waves[0])/sr,
                        'rtf': elapsed/(len(waves[0])/sr), 'tokens': raw_output.sequences.shape[1], 'audio': str(path)})
            outputs[variant] = runs
            talker.generate = effective_generate
            if handle:
                check(f'{text_id} fast path exercised', handle.fast_calls > 0 and handle.fallback_calls == 0)
                handle.uninstall()
            else:
                del talker.generate
        result['rows'].append({'text': text_id, **outputs})
        save()
    result['complete'] = True
    save()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
