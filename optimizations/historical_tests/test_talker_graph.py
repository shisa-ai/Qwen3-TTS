"""GPU3 integration check: greedy codec/waveform equality and isolated decode timing."""
import argparse
import json
import os
import time
import numpy as np
import torch
from qwen_tts import Qwen3TTSModel
from qwen_talker_graph import install_talker_graph


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", default="0.6B", choices=["0.6B", "1.7B"])
    p.add_argument("--compile", action="store_true")
    args = p.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "GPU-88395248-f4ce-ee5f-0cc5-1bdd951de1e8"
    model = Qwen3TTSModel.from_pretrained(f"Qwen/Qwen3-TTS-12Hz-{args.size}-CustomVoice",
        device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
    talker = model.model.talker
    codes = []
    times = []
    def record(module, inputs, output):
        if output.hidden_states[1] is not None:
            codes.append(output.hidden_states[1].detach().clone())
    hook = talker.register_forward_hook(record)
    texts = ["Hello.", "This is a longer test of speech generation.", "Hello."]
    def generate(text):
        codes.clear()
        torch.manual_seed(0)
        torch.cuda.synchronize()
        start = time.perf_counter()
        wavs, sr = model.generate_custom_voice(text=text, speaker="ryan", language="English",
            do_sample=False, subtalker_dosample=False, max_new_tokens=96)
        torch.cuda.synchronize()
        return torch.cat(codes).cpu(), wavs[0], time.perf_counter() - start
    original = talker.model.forward
    def timed(*a, **kw):
        if kw["inputs_embeds"].shape[1] != 1:
            return original(*a, **kw)
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = original(*a, **kw)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
        return out
    talker.model.forward = timed
    baselines = [generate(t) for t in texts]
    baseline_ms = float(np.median(times))
    talker.model.forward = original
    graph = install_talker_graph(talker, 256)
    rows = []
    for i, text in enumerate(texts):
        if i == 2:
            graph.reset()
        actual = generate(text)
        expected = baselines[i]
        equal = torch.equal(actual[0], expected[0])
        row = dict(text=text, codec_shape=list(actual[0].shape), codec_equal=equal,
                   waveform_equal=np.array_equal(actual[1], expected[1]))
        rows.append(row)
        print(json.dumps(row), flush=True)
        assert equal, "Greedy codec mismatch"
        assert row["waveform_equal"], "Greedy waveform mismatch"
    with torch.inference_mode():
        # Fixed-position microbenchmark excludes capture, outer predictor and input staging.
        def measure(fn, n=100):
            for _ in range(5): fn()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(n): fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - start) * 1000 / n
        static_ms = measure(graph._run)
        replay_ms = measure(graph.graph.replay)
        compile_result = None
        if args.compile:
            try:
                start = time.perf_counter()
                compiled = torch.compile(graph._run, fullgraph=True, mode="default")
                eager = graph._run().last_hidden_state.clone()
                result = compiled().last_hidden_state.clone()
                torch.cuda.synchronize()
                setup = time.perf_counter() - start
                compile_result = dict(setup_s=setup, ms=measure(compiled),
                    max_abs_error=float((eager-result).abs().max()),
                    close=bool(torch.allclose(eager, result, atol=0.05, rtol=0.05)))
            except Exception as e:
                compile_result = dict(error=repr(e))
        print(json.dumps(dict(size=args.size, rows=rows, upstream_decode_median_ms=baseline_ms,
            static_decode_ms=static_ms, graph_replay_ms=replay_ms,
            capture_s=graph.capture_seconds, replays=graph.replays, compile=compile_result)), flush=True)
        # Direct transformer checks exercise padding and non-default rotary positions.
        for length, fully_masked in ((5, False), (9, False), (5, True), (graph.capacity - 1, False)):
            x = torch.randn(1, length, talker.config.hidden_size, device="cuda", dtype=torch.bfloat16)
            mask = torch.ones(1, length, device="cuda", dtype=torch.long)
            mask[:, :2] = 0
            if fully_masked:
                mask.zero_()
            cp = torch.arange(length, device="cuda")
            pos = cp.view(1, 1, -1).expand(3, 1, -1) + 7
            common = dict(inputs_embeds=x, attention_mask=mask, cache_position=cp,
                          position_ids=pos, use_cache=True, output_hidden_states=True)
            ref = original(**common)
            got = talker.model(**common)
            retained = None
            max_error = 0.0
            for step in range(length, min(length + 3, graph.capacity)):
                common.update(inputs_embeds=torch.randn_like(x[:, :1]),
                    attention_mask=torch.cat((mask, torch.ones(1, step + 1 - length, device="cuda", dtype=torch.long)), dim=1),
                    cache_position=torch.tensor([step], device="cuda"),
                    position_ids=torch.full((3, 1, 1), step + 7, device="cuda", dtype=torch.long))
                if fully_masked:
                    common["attention_mask"].zero_()
                ref = original(**common, past_key_values=ref.past_key_values)
                got = talker.model(**common, past_key_values=got.past_key_values)
                for rh, gh in zip(ref.hidden_states, got.hidden_states):
                    torch.testing.assert_close(rh, gh, atol=0.05, rtol=0.05)
                    max_error = max(max_error, float((rh-gh).abs().max()))
                if retained is not None:
                    assert torch.equal(retained[0], retained[1]), "Replay overwrote an earlier output"
                retained = (got.last_hidden_state, got.last_hidden_state.clone())
            print(json.dumps(dict(mask_position_test_length=length, fully_masked=fully_masked,
                                  max_hidden_error=max_error)), flush=True)
        assert graph.cache.length == graph.capacity
        common.update(cache_position=torch.tensor([graph.capacity], device="cuda"))
        try:
            talker.model(**common, past_key_values=got.past_key_values)
            raise AssertionError("Cache overflow was accepted")
        except ValueError as e:
            assert "capacity exceeded" in str(e)
        assert graph.cache.length == graph.capacity
        print("last valid cache slot and overflow rejection: PASS", flush=True)
        other = torch.cuda.Stream()
        with torch.cuda.stream(other):
            for operation in (lambda: talker.model(**common), graph.reset, graph.uninstall):
                try:
                    operation()
                    raise AssertionError("Cross-stream operation was accepted")
                except ValueError as e:
                    assert "owning CUDA stream" in str(e)
        assert talker.model._talker_graph is graph
        print("cross-stream forward/reset/uninstall rejection: PASS", flush=True)
        common["cache_position"] = torch.tensor([1], device="cuda")
        graph.reset()
        try:
            talker.model(**common, past_key_values=got.past_key_values)
            raise AssertionError("Reset cache was accepted")
        except ValueError as e:
            assert "active serial cache" in str(e)
        print("reset rejection: PASS", flush=True)
    graph.uninstall()
    hook.remove()


if __name__ == "__main__":
    main()
