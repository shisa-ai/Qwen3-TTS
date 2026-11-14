"""GPU2 numerical, cache, ownership and performance checks for talker fusion."""
import os
GPU = 'GPU-3e620c1c-c243-7353-580e-cde9e5eb4b8a'
assert os.environ.get('CUDA_VISIBLE_DEVICES') == GPU
assert all(os.environ.get(k) == '8' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'))
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import torch
import numpy as np
from qwen_tts import Qwen3TTSModel
from bench_common import TEXTS
from qwen_fused_norm import install_fused_predictor_norm
from qwen_predictor_graph import install_predictor_graph
from qwen_talker_graph import install_talker_graph
from qwen_fast_decode import install_fast_decode
from qwen_talker_fusion import install_talker_fusion


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', required=True, choices=['0.6B', '1.7B'])
    args = parser.parse_args()
    out = Path('results/gap/talker-fusion') / (args.size + '-review-gpu2')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    result = {'size': args.size, 'gpu': GPU, 'torch': torch.__version__, 'variants': {}, 'complete': False,
              'hashes': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['scripts/qwen_talker_fusion.py', 'scripts/test_qwen_talker_fusion.py']}}
    def save():
        (out / 'summary.json').write_text(json.dumps(result, indent=2))
    save()
    model = Qwen3TTSModel.from_pretrained(f'Qwen/Qwen3-TTS-12Hz-{args.size}-CustomVoice', device_map='cuda:0', dtype=torch.bfloat16, attn_implementation='sdpa')
    talker = model.model.talker
    norm = install_fused_predictor_norm(talker.code_predictor)
    pg = install_predictor_graph(talker.code_predictor)
    fast = install_fast_decode(talker)
    records = []
    def record(module, a, kw):
        if len(records) < 12:
            records.append({k: v.clone() if torch.is_tensor(v) else v for k, v in kw.items() if k != 'past_key_values'})
    hook = talker.model.register_forward_pre_hook(record, with_kwargs=True)
    def request(text):
        return model.generate_custom_voice(text=TEXTS[text], speaker='ryan', language='English', max_new_tokens=256)
    request('short')
    hook.remove()
    def teacher():
        cache = None
        outputs = []
        for kw in records:
            kw = {k: v.to(next(talker.model.parameters()).dtype) if torch.is_tensor(v) and v.is_floating_point() else v for k,v in kw.items()}
            output = talker.model(**dict(kw, past_key_values=cache))
            cache = output.past_key_values
            outputs.append(talker.codec_head(output.last_hidden_state[:, -1]).float().clone())
        return outputs
    reference = teacher()
    # Transactional install: injected second-pack failure must restore all forwards.
    from qwen_talker_fusion import TalkerFusion
    identities = {id(mod): mod.__dict__.get('forward') for mod in talker.model.modules()}
    state_keys = tuple(talker.model.state_dict())
    original_pack = TalkerFusion._pack
    pack_calls = [0]
    def fail_pack(self, modules):
        pack_calls[0] += 1
        if pack_calls[0] == 2:
            raise RuntimeError('injected pack failure')
        return original_pack(self, modules)
    TalkerFusion._pack = fail_pack
    try:
        try: install_talker_fusion(talker)
        except RuntimeError as exc: assert str(exc) == 'injected pack failure'
        else: raise AssertionError('Expected injected failure')
    finally: TalkerFusion._pack = original_pack
    assert not hasattr(talker.model, '_talker_fusion')
    assert all(mod.__dict__.get('forward') is identities[id(mod)] for mod in talker.model.modules())
    assert tuple(talker.model.state_dict()) == state_keys
    result['transaction_rollback'] = True
    # A second module instance must remain untouched by instance-local patches.
    import copy
    isolated = copy.deepcopy(talker.model.layers[0])
    isolated_forward = isolated.self_attn.forward
    result['second_instance_class_forward'] = isolated_forward.__func__ is talker.model.layers[0].self_attn.forward.__func__
    talker.model.float(); talker.codec_head.float()
    try: install_talker_fusion(talker)
    except ValueError: result['fp32_explicitly_rejected'] = True
    else: raise AssertionError('FP32 accepted')
    precision_reference = teacher()
    talker.model.bfloat16(); talker.codec_head.bfloat16()
    attention_backend = talker.model.config._attn_implementation
    talker.model.config._attn_implementation = 'eager'
    try: install_talker_fusion(talker)
    except ValueError: result['alternate_attention_rejected'] = True
    else: raise AssertionError('Unvalidated attention accepted')
    finally: talker.model.config._attn_implementation = attention_backend
    projection = talker.model.layers[0].self_attn.q_proj
    projection.bias = torch.nn.Parameter(projection.weight.new_zeros(projection.out_features))
    try: install_talker_fusion(talker)
    except ValueError: result['mixed_biases_rejected'] = True
    else: raise AssertionError('Mixed biases accepted')
    finally: projection.bias = None
    # Padded synthetic prefill and decode, with nondefault mRoPE positions.
    def padded():
        torch.manual_seed(71)
        cache = None
        outputs = []
        for start, length in [(0, 5), (5, 1), (6, 1)]:
            cp = torch.arange(start, start + length, device='cuda')
            mask = torch.ones(1, start + length, device='cuda', dtype=torch.long)
            mask[:, :2] = 0
            output = talker.model(inputs_embeds=torch.randn(1, length, talker.config.hidden_size, device='cuda', dtype=torch.bfloat16),
                cache_position=cp, position_ids=cp.view(1, 1, -1).expand(3, 1, -1) + torch.tensor([7, 13, 21], device='cuda').view(3,1,1),
                attention_mask=mask, past_key_values=cache, use_cache=True, output_hidden_states=True)
            cache = output.past_key_values
            outputs.append(output.last_hidden_state.clone())
        return outputs
    padding_ref = padded()
    for variant in ['baseline', 'norms', 'projections', 'combined']:
        fusion = None if variant == 'baseline' else install_talker_fusion(talker, norms=variant != 'projections', projections=variant != 'norms')
        if fusion:
            # Standalone dynamic-cache path, before graph capture.
            standalone = teacher()
            standalone_padding = padded()
            for a,b in zip(padding_ref,standalone_padding):
                assert float((a-b).square().mean().sqrt()) < .15
            # A newly created caller stream observes initialized packed snapshots.
            other = torch.cuda.Stream()
            other.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(other):
                cross_stream = teacher()
            torch.cuda.current_stream().wait_stream(other)
            for a,b in zip(standalone,cross_stream):
                torch.testing.assert_close(a,b,atol=.15,rtol=.03)
        tg = install_talker_graph(talker)
        v = {'e2e': []}
        result['variants'][variant] = v
        actual = teacher()
        saved_outputs = [a.clone() for a in actual]
        assert isolated.self_attn.forward == isolated_forward
        errors = [float((a-b).abs().max()) for a,b in zip(reference,actual)]
        v['teacher_max_logit_error'] = max(errors)
        v['teacher_argmax_matches'] = [bool(a.argmax() == b.argmax()) for a,b in zip(reference,actual)]
        v['fp32_reference_errors'] = []
        if variant == 'baseline':
            graph_reference = [x.clone() for x in actual]
        v['versus_baseline_graph_max_logit_error'] = max(float((a-b).abs().max()) for a,b in zip(graph_reference,actual))
        for a,b,fp in zip(reference,actual,precision_reference):
            base_rmse = float((a-fp).square().mean().sqrt())
            fused_rmse = float((b-fp).square().mean().sqrt())
            v['fp32_reference_errors'].append({'baseline_rmse':base_rmse,'variant_rmse':fused_rmse,
                'baseline_argmax_fp32':bool(a.argmax()==fp.argmax()),'variant_argmax_fp32':bool(b.argmax()==fp.argmax())})
            assert fused_rmse <= base_rmse * 1.5 + .01, (base_rmse, fused_rmse)
        save()
        for a,b in zip(padding_ref,padded()):
            assert float((a-b).square().mean().sqrt()) < .15
        tg.reset()
        again = teacher()
        assert all(torch.equal(a,b) for a,b in zip(actual,saved_outputs))
        v['output_storage_owned'] = True
        for a,b in zip(actual,again):
            torch.testing.assert_close(a,b,atol=.15,rtol=.03)
        v['padding_reset_checked'] = True
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            try:
                tg.reset()
            except ValueError:
                v['wrong_stream_rejected'] = True
            else:
                raise AssertionError('Wrong stream accepted')
        if fusion:
            try:
                fusion.uninstall()
            except ValueError:
                v['uninstall_before_graph_rejected'] = True
            else:
                raise AssertionError('Unsafe uninstall')
        times = []
        for _ in range(10): tg.graph.replay()
        for _ in range(3):
            a,b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(100): tg.graph.replay()
            b.record(); b.synchronize()
            times.append(a.elapsed_time(b)/100)
        v['fixed_graph_ms'] = times
        v['fixed_graph_median_ms'] = statistics.median(times)
        generated = {}
        original = talker.generate
        def observe(*a, **kw):
            output = original(*a, **kw)
            generated['sequences'] = output.sequences
            return output
        talker.generate = observe
        for text in ['short','medium']:
            torch.manual_seed(0); request(text)
            for seed in range(3):
                torch.manual_seed(seed)
                before = tg.replays
                torch.cuda.synchronize(); start = time.perf_counter()
                wavs,sr = request(text)
                torch.cuda.synchronize(); elapsed = time.perf_counter()-start
                seq = generated['sequences']
                row = {'text':text,'seed':seed,'seconds':elapsed,'audio_s':len(wavs[0])/sr,
                    'rtf':elapsed/(len(wavs[0])/sr),'frames':tg.replays-before,
                    'eos':bool(seq[0,-1] == talker.config.codec_eos_token_id),
                    'finite':bool(np.isfinite(wavs[0]).all()), 'sample_rate':sr}
                assert row['finite'] and row['eos']
                v['e2e'].append(row); save(); print(variant,row,flush=True)
        talker.generate = original
        v['capture_seconds'] = tg.capture_seconds
        tg.uninstall()
        if fusion: fusion.uninstall()
        assert all(mod.__dict__.get('forward') is identities[id(mod)] for mod in talker.model.modules() if mod is not talker.model)
        assert tuple(talker.model.state_dict()) == state_keys
        v['uninstall_identity_state_checked'] = True
        save()
    fast.uninstall(); pg.uninstall(); norm.uninstall()
    result['complete'] = True
    save()

if __name__ == '__main__': main()
