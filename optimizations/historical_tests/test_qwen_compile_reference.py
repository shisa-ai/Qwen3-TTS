"""Fixed-state talker compile diagnosis; never installs a production adapter."""
import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from qwen_tts import Qwen3TTSModel
from qwen_talker_graph import _DecodeCache

GPU = 'GPU-3e620c1c-c243-7353-580e-cde9e5eb4b8a'


def metrics(x, ref):
    x, ref = x.float().flatten(), ref.float().flatten()
    diff = x - ref
    return dict(max_abs=diff.abs().max().item(), rmse=diff.square().mean().sqrt().item(),
                relative_l2=(diff.norm() / ref.norm()).item(), cosine=F.cosine_similarity(x, ref, dim=0).item())


def timing(fn, n=100):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(3):
        start = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000 / n)
    return dict(median_ms=statistics.median(samples), samples_ms=samples)


class Runner:
    def __init__(self, model, snapshot, capacity):
        self.model = model
        dtype = next(model.parameters()).dtype
        self.cache = _DecodeCache(config=model.config, max_cache_len=capacity)
        self.cache.length = snapshot['position']
        self.sources = []
        for layer, (k, v) in zip(self.cache.layers, snapshot['cache']):
            k, v = k.cuda().to(dtype), v.cuda().to(dtype)
            layer.lazy_initialization(k)
            self.sources.append((k, v))
        self.x = snapshot['x'].cuda().to(dtype)
        self.positions = snapshot['positions'].cuda()
        self.cp = torch.tensor([snapshot['position']], device='cuda')
        self.mask = torch.full((1, 1, 1, capacity), float('-inf'), device='cuda', dtype=dtype)
        self.mask[..., :snapshot['position'] + 1] = 0
        padding = snapshot['mask'].cuda()
        self.mask[..., :padding.shape[-1]].masked_fill_(padding[:, None, None, :] == 0, float('-inf'))
        self.restore()

    def restore(self):
        for layer, (k, v) in zip(self.cache.layers, self.sources):
            layer.keys.zero_()
            layer.values.zero_()
            layer.keys[..., :k.shape[-2], :].copy_(k)
            layer.values[..., :v.shape[-2], :].copy_(v)

    def run(self):
        return self.model(inputs_embeds=self.x, position_ids=self.positions, attention_mask=self.mask,
                          past_key_values=self.cache, cache_position=self.cp, use_cache=True,
                          output_hidden_states=True, output_attentions=False, return_dict=True).last_hidden_state


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--size', choices=['0.6B', '1.7B'], required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--end-to-end', action='store_true')
    args = p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == GPU
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = Qwen3TTSModel.from_pretrained(f'Qwen/Qwen3-TTS-12Hz-{args.size}-CustomVoice',
        device_map='cuda:0', dtype=torch.bfloat16, attn_implementation='sdpa')
    talker = model.model.talker
    if args.end_to_end:
        end_to_end(model, args)
        return
    snapshots = []
    ordinal = 0
    original = talker.model.forward

    def record(*a, **kw):
        nonlocal ordinal
        if kw['inputs_embeds'].shape[1] == 1:
            if ordinal in (0, 7, 23, 47):
                snapshots.append(dict(ordinal=ordinal, position=int(kw['cache_position'][0]),
                    x=kw['inputs_embeds'].detach().cpu().clone(),
                    positions=kw['position_ids'].detach().cpu().clone(),
                    mask=kw['attention_mask'].detach().cpu().clone(),
                    cache=[(l.keys.detach().cpu().clone(), l.values.detach().cpu().clone())
                           for l in kw['past_key_values'].layers]))
            ordinal += 1
        return original(*a, **kw)

    talker.model.forward = record
    torch.manual_seed(0)
    model.generate_custom_voice(text='This is a reference test of speech generation, with enough words to inspect several real decoding positions.',
        speaker='ryan', language='English', do_sample=False, subtalker_dosample=False,
        min_new_tokens=56, max_new_tokens=56)
    talker.model.forward = original
    reference = copy.deepcopy(talker.model).float().eval()
    head = talker.codec_head.weight.detach().float()
    result = dict(size=args.size, torch=torch.__version__, gpu=GPU, threads=torch.get_num_threads(),
        reference='FP32 arithmetic on BF16-rounded weights, identical BF16 input and pre-decode cache promoted to FP32; TF32 disabled',
        capacity=256, rows=[])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for snapshot in snapshots:
            row = dict(ordinal=snapshot['ordinal'], position=snapshot['position'], variants={})
            fp = Runner(reference, snapshot, 256)
            with sdpa_kernel(SDPBackend.MATH):
                ref = fp.run().clone()
            ref_logits = F.linear(ref, head)
            top = ref_logits.flatten().topk(2)
            row['reference_top2'] = top.indices.tolist()
            row['reference_margin'] = (top.values[0] - top.values[1]).item()
            bf = Runner(talker.model, snapshot, 256)
            eager = None
            for name, options in [('eager', None), ('compiled', {}),
                                  ('compiled_emulate_casts', {'emulate_precision_casts': True})]:
                bf.restore()
                fn = bf.run if options is None else torch.compile(bf.run, fullgraph=True, options=options)
                start = time.perf_counter()
                value = fn().clone()
                torch.cuda.synchronize()
                setup = time.perf_counter() - start
                # Repeated calls overwrite only the current slot: prove idempotence before timing.
                repeated = fn().clone()
                assert torch.equal(value, repeated), 'Repeated-position mutation changed output'
                for layer, (k, v) in zip(bf.cache.layers, bf.sources):
                    assert torch.equal(layer.keys[..., :k.shape[-2], :], k)
                    assert torch.equal(layer.values[..., :v.shape[-2], :], v)
                if eager is None:
                    eager = value.clone()
                logits = talker.codec_head(value).float()
                projected = F.linear(value.float(), head)
                predicted = logits.flatten().topk(2)
                row['variants'][name] = dict(hidden=metrics(value, ref), against_eager=metrics(value, eager),
                    logits=metrics(logits, ref_logits), fp32_head_logits=metrics(projected, ref_logits),
                    top2=predicted.indices.tolist(), margin=(predicted.values[0]-predicted.values[1]).item(),
                    setup_s=setup, timing=timing(fn))
                if name == 'eager':
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            fn()
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        graph_value = fn()
                    row['graph_timing'] = timing(graph.replay)
                    assert torch.equal(graph_value, eager)
                print(json.dumps(dict(position=row['position'], variant=name, **row['variants'][name])), flush=True)
            # Isolate backend variation from compiler fusion.
            bf.restore()
            with sdpa_kernel(SDPBackend.MATH):
                math_value = bf.run().clone()
            row['eager_math_backend'] = dict(hidden=metrics(math_value, ref), against_default=metrics(math_value, eager))
            fp.restore()
            fp_default = fp.run().clone()
            row['fp32_default_vs_math'] = metrics(fp_default, ref)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                bf.run()
            row['attention_ops'] = [e.key for e in prof.key_averages() if 'attention' in e.key]
            result['rows'].append(row)
            output.write_text(json.dumps(result, indent=2))
            del fp, bf
    assert len(snapshots) == 4
    result['complete'] = True
    output.write_text(json.dumps(result, indent=2))


def end_to_end(model, args):
    from qwen_fused_norm import install_fused_predictor_norm
    from qwen_predictor_graph import install_predictor_graph
    from qwen_talker_graph import install_talker_graph
    from qwen_compiled_decode import install_compiled_talker
    talker = model.model.talker
    original_generate = talker.generate
    def fixed_generate(*a, **kw):
        kw['min_new_tokens'] = kw['max_new_tokens']
        return original_generate(*a, **kw)
    # The outer API ignores min_new_tokens; enforce it at the actual generation boundary.
    talker.generate = fixed_generate
    install_fused_predictor_norm(talker.code_predictor)
    install_predictor_graph(talker.code_predictor)
    rows = []
    for label, installer in [('graphs_fusednorm', install_talker_graph), ('compiled_fusednorm', install_compiled_talker)]:
        adapter = installer(talker, 256)
        for text, limit in [('Hello.', 24), ('This is a medium length speech generation test with a few more words to speak clearly.', 72)]:
            times, lengths = [], []
            for run in range(4):
                torch.manual_seed(0)
                torch.cuda.synchronize()
                start = time.perf_counter()
                before_replays = adapter.replays
                audio, sr = model.generate_custom_voice(text=text, speaker='ryan', language='English',
                    do_sample=False, subtalker_dosample=False, min_new_tokens=limit, max_new_tokens=limit)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                assert adapter.replays - before_replays == limit - 1
                assert all(torch.isfinite(torch.as_tensor(w)).all() for w in audio)
                if run:
                    times.append(elapsed)
                    lengths.append(len(audio[0]) / sr)
                else:
                    cold = elapsed
            rows.append(dict(variant=label, text=text, forced_tokens=limit, cold_s=cold, seconds=times,
                             median_s=statistics.median(times), audio_seconds=lengths))
            print(json.dumps(rows[-1]), flush=True)
        # Verify owner stream rejection and invalidation without a second CUDA device.
        with torch.cuda.stream(torch.cuda.Stream()):
            try:
                adapter.reset()
                raise AssertionError('Cross-stream reset accepted')
            except ValueError as exc:
                assert 'owning CUDA stream' in str(exc)
        adapter.reset()
        assert not adapter.active and adapter.cache.length == 0
        adapter.uninstall()
    Path(args.output).write_text(json.dumps(dict(size=args.size, complete=True, rows=rows,
        caveat='Matched forced token counts, not natural EOS or equal generated content; cold first call excluded'), indent=2))


if __name__ == '__main__':
    main()
