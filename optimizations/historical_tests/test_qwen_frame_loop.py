"""GPU2 regression and matched timing for the request-owned intermediate."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

GPU = os.environ.get('FRAME_TEST_GPU', 'GPU-3e620c1c-c243-7353-580e-cde9e5eb4b8a')


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
    # Direct candidate processor composition against independently assembled HF math.
    from qwen_frame_loop import processors
    kw = dict(repetition_penalty=2., min_new_tokens=4, eos_token_id=4,
              suppress_tokens=[0], do_sample=True, temperature=0.7, top_k=3, top_p=0.85)
    expected = RepetitionPenaltyLogitsProcessor(2.)(history, logits.clone())
    expected = MinNewTokensLengthLogitsProcessor(0, 4, 4, device='cpu')(history, expected)
    expected = SuppressTokensLogitsProcessor([0], device='cpu')(history, expected)
    for transform in [TemperatureLogitsWarper(.7), TopKLogitsWarper(3), TopPLogitsWarper(.85)]:
        expected = transform(history, expected)
    torch.testing.assert_close(processors(kw, 'cpu')(history, logits.clone()), expected)
    return ['candidate processor composition', 'negative and duplicate repetition history', 'suppression', 'minimum EOS boundary',
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
    from qwen_frame_loop import install_frame_loop as install_frame_decode
    from bench_common import TEXTS
    torch.set_num_threads(8)
    deterministic_probe = os.environ.get('FRAME_LOOP_NO_CUDNN') == '1'
    if deterministic_probe:
        torch.backends.cuda.enable_cudnn_sdp(False)
    root = Path(__file__).resolve().parent.parent
    out = root / 'results/gap/frame-decode' / (args.size + os.environ.get('FRAME_TEST_SUFFIX', '-loop-no-cudnn-v2' if deterministic_probe else '-loop-review-v2') + '.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    audio_dir = root / 'samples/gap/frame-decode' / (args.size + os.environ.get('FRAME_TEST_SUFFIX', '-loop-no-cudnn-v2' if deterministic_probe else '-loop-review-v2'))
    audio_dir.mkdir(parents=True, exist_ok=True)
    result = {'size': args.size, 'gpu': GPU, 'warmup_seed': 42, 'rows': [], 'checks': processor_contract_tests(), 'complete': False}
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
    reference_steps = []
    def reference_hook(module, inputs, output):
        if output.logits.shape[1] == 1:
            pos = tg.cache.length - 1
            reference_steps.append((output.logits[:, -1].float().clone(),
                tg.cache.layers[0].keys[:, :, pos:pos+1].clone()))
    hook = talker.register_forward_hook(reference_hook)
    ref, _ = raw(base)
    hook.remove()
    baseline = [raw(base)[1] for _ in range(3)]
    fd = install_frame_decode(talker)
    def compare_step(step, logits, cache, codes, past_hidden):
        expected_logits, expected_key = reference_steps[step]
        torch.testing.assert_close(logits, expected_logits, atol=0, rtol=0)
        pos = cache.length - 1
        torch.testing.assert_close(cache.layers[0].keys[:, :, pos:pos+1], expected_key, atol=0, rtol=0)
    fd.step_observer = compare_step
    got, _ = raw(base)
    fd.step_observer = None
    check('matched-history step logits and layer-zero cache exact')
    check('fast path exercised', fd.calls > 0 and fd.fallback_calls == 0)
    check('greedy fixed-work sequences exact', torch.equal(ref.sequences, got.sequences))
    check('real hidden-state count', len(ref.hidden_states) == len(got.hidden_states))
    for a, b in zip(ref.hidden_states, got.hidden_states):
        if a[-1] is not None:
            assert torch.equal(a[-1], b[-1])
        for x, y in zip(a[0], b[0]):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
    check('all retained real hidden states exact')
    import copy
    config = talker.code_predictor.generation_config
    for label, mutate in [
        ('mutated', lambda c: setattr(c, 'repetition_penalty', 1.2)),
        ('nested', lambda c: setattr(c, 'bad_words_ids', [[1]])),
        ('unknown', lambda c: setattr(c, 'unknown_review_setting', {'nested': [1]})),
    ]:
        replacement = copy.deepcopy(config)
        mutate(replacement)
        talker.code_predictor.generation_config = replacement
        count, length = fd.calls, tg.cache.length
        try:
            raw(base)
        except ValueError:
            check('predictor config ' + label + ' rejected before loop', fd.calls == count and not fd.active)
        else:
            raise AssertionError(label)
        finally:
            talker.code_predictor.generation_config = config
    talker.code_predictor.training = True
    try:
        raw(base)
    except ValueError:
        check('predictor-only training rejected')
    else:
        raise AssertionError('predictor training')
    finally:
        talker.code_predictor.training = False
    for name in ['trailing_text_hidden', 'tts_pad_embed']:
        for kind in ['dtype', 'device', 'layout']:
            value = base[name]
            bad = value.float() if kind == 'dtype' else value.cpu() if kind == 'device' else value.repeat_interleave(2, -1)[..., ::2]
            for state in ['cold', 'warm']:
                saved_graph = tg.graph
                if state == 'cold':
                    tg.graph = None
                try:
                    raw(base | {name: bad})
                except ValueError:
                    check(name + '/' + kind + '/' + state + ' rejected')
                else:
                    raise AssertionError('bad embedding accepted')
                finally:
                    tg.graph = saved_graph
    snapshot = tuple((tuple(x.clone() for x in h[0]), None if h[1] is None else h[1].clone()) for h in got.hidden_states)
    candidate = [raw(base)[1] for _ in range(3)]
    assert len(got.hidden_states) == len(snapshot)
    for original_state, saved in zip(got.hidden_states, snapshot):
        assert len(original_state[0]) == len(saved[0])
        for actual_tensor, saved_tensor in zip(original_state[0], saved[0]):
            torch.testing.assert_close(actual_tensor, saved_tensor, atol=0, rtol=0)
        if saved[1] is not None:
            assert torch.equal(original_state[1], saved[1])
    check('every retained layer and codec frame survives later calls and resets')
    check('retained outputs survive replay', torch.equal(ref.sequences, got.sequences))
    result['fixed_work'] = {'tokens': 24, 'baseline_s': baseline, 'candidate_s': candidate,
        'speedup': statistics.median(baseline) / statistics.median(candidate)}
    # Force EOS to prove the materialized-frame off-by-one contract.
    boundary = base | {'suppress_tokens': [i for i in range(talker.config.vocab_size) if i != talker.config.codec_eos_token_id],
                       'min_new_tokens': 0, 'max_new_tokens': 4}
    ended, _ = raw(boundary)
    check('EOS first selection has only prefill hidden states', ended.sequences.shape[1] == 1 and len(ended.hidden_states) == 1)
    capped, _ = raw(base | {'max_new_tokens': 1, 'min_new_tokens': 1})
    check('maximum one preserves prefill-only alignment', capped.sequences.shape[1] == 1 and len(capped.hidden_states) == 1)
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
        ('talker top-p fast path', dict(do_sample=True, top_p=0.85, top_k=12, subtalker_dosample=True, subtalker_top_p=1.0)),
        ('repetition and suppression', dict(repetition_penalty=1.2, suppress_tokens=base['suppress_tokens'] + [1, 2, 3])),
    ]:
        settings = base | options
        reference, _ = raw(settings)
        handle = install_frame_decode(talker)
        actual, _ = raw(settings)
        check(label + ' exact seeded sequence', torch.equal(reference.sequences, actual.sequences))
        assert len(reference.hidden_states) == len(actual.hidden_states)
        for ref_state, actual_state in zip(reference.hidden_states, actual.hidden_states):
            assert len(ref_state[0]) == len(actual_state[0])
            if ref_state[1] is not None:
                assert torch.equal(ref_state[1], actual_state[1])
            for a, b in zip(ref_state[0], actual_state[0]):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
        check(label + ' residual codes and states exact')
        check(label + ' dispatch', (handle.fallback_calls == 1 and handle.calls == 0) if options.get('subtalker_top_p', 1.) != 1. else (handle.calls == 1 and handle.fallback_calls == 0))
        handle.uninstall()
    # Nonzero minimum: allow only EOS and a fixed ordinary token, then force EOS
    # once the minimum processor stops masking it through a large head bias hook
    # is unsupported; instead compare normal greedy termination/alignment to HF.
    minimum_settings = base | {'min_new_tokens': 2, 'max_new_tokens': 8}
    reference_min, _ = raw(minimum_settings)
    minimum_handle = install_frame_decode(talker)
    candidate_min, _ = raw(minimum_settings)
    check('nonzero minimum sequences and frame count match HF', torch.equal(reference_min.sequences, candidate_min.sequences)
          and len(reference_min.hidden_states) == len(candidate_min.hidden_states))
    minimum_handle.uninstall()
    divergence = []
    long_settings = dict(captured)
    long_settings.update(max_new_tokens=768, do_sample=True, subtalker_dosample=True)
    for variant in ['baseline-a', 'baseline-b', 'candidate-a', 'candidate-b']:
        handle = install_frame_decode(talker) if variant.startswith('candidate') else None
        try:
            value, elapsed = raw(long_settings)
            divergence.append({'variant': variant, 'sequence': value.sequences[0].tolist(),
                'codec_frames': [h[1][0].tolist() for h in value.hidden_states if h[1] is not None],
                'rng_end': __import__('hashlib').sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest()})
        finally:
            if handle:
                handle.uninstall()
    result['same_seed_long_probe'] = divergence
    save()
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
            for seed in [42, 0, 1, 2]:
                with torch.inference_mode():
                    tg.reset()
                torch.manual_seed(seed)
                before_capture = list(tg.capture_seconds)
                with torch.inference_mode():
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    try:
                        waves, sr = model.generate_custom_voice(text=TEXTS[text_id], speaker='ryan', language='English')
                    except ValueError as error:
                        if 'capacity' not in str(error):
                            raise
                        result.setdefault('capacity_failures', []).append({'text': text_id, 'variant': variant, 'seed': seed, 'error': str(error)})
                        save()
                        continue
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                raw_output = observed['value']
                check(f'{text_id}/{variant}/{seed} EOS', raw_output.sequences[0, -1].item() == talker.config.codec_eos_token_id)
                check(f'{text_id}/{variant}/{seed} finite audio', np.isfinite(waves[0]).all() and len(waves[0]) > 0)
                check(f'{text_id}/{variant}/{seed} no capture', before_capture == tg.capture_seconds)
                if seed != 42:
                    path = audio_dir / f'{text_id}-{variant}-{seed}.wav'
                    sf.write(path, waves[0], sr)
                    runs.append({'seed': seed, 'seconds': elapsed, 'audio_seconds': len(waves[0])/sr,
                        'rtf': elapsed/(len(waves[0])/sr), 'tokens': raw_output.sequences.shape[1], 'audio': str(path)})
            outputs[variant] = runs
            talker.generate = effective_generate
            if handle:
                check(f'{text_id} fast path exercised', handle.calls > 0 and handle.fallback_calls == 0)
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
