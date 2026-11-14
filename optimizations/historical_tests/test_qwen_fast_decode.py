"""GPU3-only correctness and three-run benchmark for bounded fast decode."""
import os
GPU = 'GPU-88395248-f4ce-ee5f-0cc5-1bdd951de1e8'
if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU or any(os.environ.get(k) != '8' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS')):
    raise RuntimeError('Set GPU3 UUID visibility and OMP/MKL threads=8')
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
import numpy as np
import torch
from qwen_tts import Qwen3TTSModel
from bench_common import TEXTS
from qwen_fast_decode import install_fast_decode
from qwen_fused_norm import install_fused_predictor_norm
from qwen_predictor_graph import install_predictor_graph
from qwen_talker_graph import install_talker_graph


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', choices=['0.6B', '1.7B'], required=True)
    args = parser.parse_args()
    out = Path('results/fast-decode') / args.size
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    assert torch.cuda.device_count() == 1
    assert str(torch.cuda.get_device_properties(0).uuid) == GPU.removeprefix('GPU-')
    result = {'size': args.size, 'gpu': GPU, 'torch': torch.__version__, 'threads': 8,
              'command': f'CUDA_VISIBLE_DEVICES={GPU} OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 {sys.executable} scripts/test_qwen_fast_decode.py --size {args.size}',
              'variants': {}, 'checks': {}, 'complete': False}
    def save():
        (out / 'summary.json').write_text(json.dumps(result, indent=2))
    save()
    model = Qwen3TTSModel.from_pretrained(f'Qwen/Qwen3-TTS-12Hz-{args.size}-CustomVoice', device_map='cuda:0', dtype=torch.bfloat16, attn_implementation='sdpa')
    talker = model.model.talker
    predictor = talker.code_predictor
    norm = install_fused_predictor_norm(predictor)
    pg = install_predictor_graph(predictor)
    tg = install_talker_graph(talker)
    captured = {}
    original = predictor.generate
    def capture(*a, **kw):
        if not captured:
            captured.update({k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in kw.items()})
        return original(*a, **kw)
    predictor.generate = capture
    def request(text='short', greedy=False):
        return model.generate_custom_voice(text=TEXTS[text], speaker='ryan', language='English',
                                           **({'do_sample': False, 'subtalker_dosample': False} if greedy else {}))
    torch.manual_seed(0)
    request()
    predictor.generate = original
    result['initial_captures'] = {'talker_s': list(tg.capture_seconds), 'predictor_s': [g.capture_seconds for g in pg.graphs.values()]}
    embeds = captured['inputs_embeds']
    # Same input, full transformer propagation, changing only hidden-state collection.
    def logits(hidden):
        x = predictor.model(inputs_embeds=predictor.small_to_mtp_projection(embeds), use_cache=False,
                            output_hidden_states=hidden, return_dict=True).last_hidden_state
        return predictor.lm_head[0](x[:, -1]).float()
    a, b = logits(True), logits(False)
    assert torch.equal(a, b) and torch.isfinite(b).all()
    result['checks']['same_input_logits_exact'] = True
    kw = dict(captured, do_sample=False)
    ref = predictor.generate(**kw)
    alt = predictor.generate(**dict(kw, output_hidden_states=False))
    assert torch.equal(ref.sequences, alt.sequences) and ref.hidden_states and alt.hidden_states is None
    saved = ref.sequences.clone()
    predictor.generate(**dict(kw, inputs_embeds=embeds + 0.01))
    assert torch.equal(ref.sequences, saved)
    assert torch.equal(predictor.generate(**kw).sequences, saved)
    result['checks']['greedy_tokens_reset_owned_storage'] = True
    # Capture greedy full requests with the exact hook consumed by streaming.
    def traced():
        frames, scores = [], []
        def hook(module, inputs, output):
            if output.hidden_states[1] is not None:
                frames.append(output.hidden_states[1].clone())
            if len(scores) < 3:
                scores.append(output.logits.clone())
        h = talker.register_forward_hook(hook)
        try:
            tg.reset()
            torch.manual_seed(0)
            wavs, sr = request(greedy=True)
        finally:
            h.remove()
        return torch.stack(frames), scores, wavs[0], sr
    baseline = traced()
    fast = install_fast_decode(talker)
    try:
        try:
            install_fast_decode(talker)
        except ValueError:
            result['checks']['duplicate_rejected'] = True
        else:
            raise AssertionError('Duplicate accepted')
        assert predictor.generate(**kw).hidden_states is not None
        try:
            predictor.generate(**dict(kw, num_beams=2))
        except ValueError:
            result['checks']['external_strict_kwargs_preserved'] = True
        else:
            raise AssertionError('Invalid external call accepted')
        stream = torch.cuda.Stream()
        try:
            with torch.cuda.stream(stream):
                predictor.generate(**kw)
        except ValueError:
            result['checks']['external_stream_guard_preserved'] = True
        else:
            raise AssertionError('Wrong stream accepted')
        for repeat in range(2):
            actual = traced()
            assert torch.equal(actual[0], baseline[0])
            assert all(torch.equal(x, y) for x, y in zip(actual[1], baseline[1]))
            assert np.array_equal(actual[2], baseline[2])
        result['checks']['full_greedy_repeat_frames_logits_waveform_exact'] = True
        result['checks']['streaming_frame_hook_preserved'] = True
        result['checks']['external_hidden_states_preserved'] = True
    finally:
        fast.uninstall()
    assert talker.forward == fast.original
    result['checks']['uninstall_restored'] = True
    save()
    # Benchmark matched sampling requests; EOS inspected directly, not inferred from length.
    for variant in ['baseline', 'fast']:
        fast = install_fast_decode(talker) if variant == 'fast' else None
        v = {'e2e': []}
        result['variants'][variant] = v
        fixedkw = dict(captured, output_hidden_states=variant == 'baseline')
        keys = set(pg.graphs)
        torch.cuda.synchronize()
        start = time.perf_counter()
        predictor.generate(**fixedkw)
        torch.cuda.synchronize()
        v['first_fixed_call_s'] = time.perf_counter() - start
        v['new_captures'] = [{'capture_s': g.capture_seconds, 'setup_s': g.setup_seconds} for k, g in pg.graphs.items() if k not in keys]
        for _ in range(10):
            predictor.generate(**fixedkw)
        wall, gpu = [], []
        for _ in range(3):
            torch.cuda.synchronize()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start = time.perf_counter()
            a.record()
            for _ in range(100):
                predictor.generate(**fixedkw)
            b.record()
            b.synchronize()
            wall.append((time.perf_counter() - start) * 10)
            gpu.append(a.elapsed_time(b) / 100)
        v.update(fixed_generate_wall_ms=wall, fixed_generate_gpu_ms=gpu,
                 median_fixed_wall_ms=statistics.median(wall))
        generated = {}
        orig_generate = talker.generate
        def observe(*a, **kw):
            output = orig_generate(*a, **kw)
            generated['sequences'] = output.sequences
            return output
        talker.generate = observe
        for text in ['short', 'medium']:
            torch.manual_seed(0)
            request(text)
            for seed in range(3):
                tg.reset()
                torch.manual_seed(seed)
                before = tg.replays
                torch.cuda.synchronize()
                start = time.perf_counter()
                wavs, sr = request(text)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                wav = wavs[0]
                last = generated['sequences'][0, -1].item()
                eos = talker.config.codec_eos_token_id
                row = {'text': text, 'seed': seed, 'seconds': elapsed, 'audio_s': len(wav) / sr,
                       'rtf': elapsed / (len(wav) / sr), 'decode_replays': tg.replays - before,
                       'last_token': last, 'eos_token': eos, 'eos': last == eos,
                       'finite': bool(np.isfinite(wav).all()), 'samples': len(wav), 'sample_rate': sr}
                assert row['finite'] and row['samples'] > 0 and row['eos']
                v['e2e'].append(row)
                save()
                print(args.size, variant, row, flush=True)
        talker.generate = orig_generate
        if fast:
            fast.uninstall()
    result['captures'] = {'talker_s': tg.capture_seconds,
                          'predictor': [{'key': str(k), 'capture_s': g.capture_seconds, 'setup_s': g.setup_seconds} for k, g in pg.graphs.items()]}
    tg.reset()
    tg.uninstall()
    pg.uninstall()
    norm.uninstall()
    result['complete'] = True
    save()


if __name__ == '__main__':
    main()
