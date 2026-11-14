"""Real-input correctness and latency test; restricted to physical GPU2.

CUDA_VISIBLE_DEVICES=GPU-3e620c1c-c243-7353-580e-cde9e5eb4b8a \
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python scripts/test_predictor_graph.py --size 0.6B
"""
import argparse
import json
import os
import statistics
import time

GPU = "GPU-3e620c1c-c243-7353-580e-cde9e5eb4b8a"
assert os.environ.get("CUDA_VISIBLE_DEVICES") in (GPU, "GPU-88395248-f4ce-ee5f-0cc5-1bdd951de1e8")
assert os.environ.get("OMP_NUM_THREADS") == os.environ.get("MKL_NUM_THREADS") == "8"

import numpy as np
import torch
from qwen_tts import Qwen3TTSModel
from qwen_predictor_graph import install_predictor_graph


def timed(fn, runs):
    values = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - start) * 1000)
    return statistics.median(values)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=["0.6B", "1.7B"], default="0.6B")
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()
    torch.manual_seed(42)
    model = Qwen3TTSModel.from_pretrained(
        f"Qwen/Qwen3-TTS-12Hz-{args.size}-CustomVoice", local_files_only=True,
        device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    predictor = model.model.talker.code_predictor
    original = predictor.generate
    inputs = []
    def collect(*args, **kwargs):
        if len(inputs) < 12:
            inputs.append(kwargs["inputs_embeds"].clone())
        return original(*args, **kwargs)
    predictor.generate = collect
    def speech():
        return model.generate_custom_voice(text="Hello! This is a short test.", language="English",
                                          speaker="ryan", do_sample=False, subtalker_dosample=False,
                                          max_new_tokens=32)
    speech()
    predictor.generate = original
    assert len(inputs) >= 8
    print("Collected", len(inputs), "real inputs; config", predictor.generation_config, flush=True)
    options = dict(max_new_tokens=predictor.config.num_code_groups - 1, do_sample=False,
                   output_hidden_states=True, return_dict_in_generate=True)
    refs = [original(inputs_embeds=x, **options) for x in inputs]
    handle = install_predictor_graph(predictor)
    with torch.autocast('cuda', dtype=torch.float16):
        try:
            predictor.generate(inputs_embeds=inputs[0], **options)
        except ValueError as exc:
            assert 'autocast' in str(exc)
        else:
            raise AssertionError('CUDA autocast was accepted')
    for i, x in enumerate(inputs):
        out = predictor.generate(inputs_embeds=x, **options)
        assert torch.equal(out.sequences, refs[i].sequences), (i, out.sequences, refs[i].sequences)
        assert len(out.hidden_states) == len(refs[i].hidden_states)
        for got, expected in zip(out.hidden_states, refs[i].hidden_states):
            assert [t.shape for t in got] == [t.shape for t in expected]
    saved = predictor.generate(inputs_embeds=inputs[0], **options).sequences
    for i in [7, 0, 3, 0, 11, 0]:
        out = predictor.generate(inputs_embeds=inputs[i], **options)
        assert torch.equal(out.sequences, refs[i].sequences)
    assert torch.equal(saved, refs[0].sequences), "output storage alias"
    for graph in handle.graphs.values():
        for layer in graph.cache.layers:
            layer.keys.fill_(123)
            layer.values.fill_(-123)
    assert torch.equal(predictor.generate(inputs_embeds=inputs[0], **options).sequences, refs[0].sequences)
    no_hidden = predictor.generate(inputs_embeds=inputs[0], **(options | dict(output_hidden_states=False)))
    assert no_hidden.hidden_states is None
    assert torch.equal(no_hidden.sequences, refs[0].sequences)
    print("Greedy real-input parity, poisoned-cache reset, replay and output ownership passed", flush=True)
    for bad in [{"repetition_penalty": 1.1}, {"top_p": 0.0, "do_sample": True},
                {"max_new_tokens": 2}, {"attention_mask": torch.ones(1, 2, device="cuda")},
                {"output_scores": True}, {"eos_token_id": 10}]:
        try:
            predictor.generate(inputs_embeds=inputs[0], **(options | bad))
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted unsupported option {bad}")
    predictor.generation_config.repetition_penalty = 1.1
    try:
        predictor.generate(inputs_embeds=inputs[0], **options)
    except ValueError:
        pass
    else:
        raise AssertionError("Inherited repetition_penalty ignored")
    predictor.generation_config.repetition_penalty = 1.0
    sample = options | dict(do_sample=True, temperature=0.9, top_k=50, top_p=0.9)
    # Use broad sampling to test RNG even when ordinary voice logits are peaked.
    random_options = sample | dict(temperature=2.0, top_k=0, top_p=1.0)
    samples = [predictor.generate(inputs_embeds=inputs[0], **random_options).sequences for _ in range(12)]
    assert all(s.shape == refs[0].sequences.shape and bool(((s >= 0) & (s < predictor.config.vocab_size)).all()) for s in samples)
    unique = len({tuple(s.flatten().tolist()) for s in samples})
    assert unique > 1, "replayed RNG repeated the same result"
    torch.manual_seed(123)
    first = predictor.generate(inputs_embeds=inputs[0], **sample).sequences
    torch.manual_seed(123)
    assert torch.equal(first, predictor.generate(inputs_embeds=inputs[0], **sample).sequences)
    # top-k ties survive filtering, so sampling is not necessarily argmax.
    one = predictor.generate(inputs_embeds=inputs[0], **(sample | dict(top_k=1, top_p=0.5)))
    assert one.sequences.shape == refs[0].sequences.shape
    tensor = predictor.generate(inputs_embeds=inputs[0], **(options | dict(return_dict_in_generate=False)))
    assert torch.equal(tensor, refs[0].sequences)
    baseline = timed(lambda: original(inputs_embeds=inputs[0], **options), args.runs)
    graph = timed(lambda: predictor.generate(inputs_embeds=inputs[0], **options), args.runs)
    sample_baseline = timed(lambda: original(inputs_embeds=inputs[0], **sample), args.runs)
    sample_graph = timed(lambda: predictor.generate(inputs_embeds=inputs[0], **sample), args.runs)
    torch.cuda.synchronize()
    start = time.perf_counter()
    wavs, sr = speech()
    torch.cuda.synchronize()
    smoke_s = time.perf_counter() - start
    assert len(wavs[0]) > 0 and np.isfinite(wavs[0]).all()
    report = dict(size=args.size, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  real_inputs=len(inputs), runs=args.runs, sampled_unique=unique,
                  upstream_greedy_ms=baseline, graph_greedy_ms=graph, greedy_speedup=baseline / graph,
                  upstream_sample_ms=sample_baseline, graph_sample_ms=sample_graph,
                  sample_speedup=sample_baseline / sample_graph,
                  captures=[dict(sample=k[3], top_k=k[5], top_p=k[6], hidden=k[7],
                                 capture_s=g.capture_seconds, setup_including_warmup_s=g.setup_seconds)
                            for k, g in handle.graphs.items()],
                  smoke_s=smoke_s, smoke_samples=len(wavs[0]), sample_rate=sr,
                  smoke_audio_s=len(wavs[0]) / sr)
    handle.uninstall()
    assert predictor.generate == original
    assert torch.equal(predictor.generate(inputs_embeds=inputs[0], **options).sequences, refs[0].sequences)
    print(json.dumps(report, indent=2), flush=True)
    print("PASS", flush=True)


if __name__ == "__main__":
    main()
