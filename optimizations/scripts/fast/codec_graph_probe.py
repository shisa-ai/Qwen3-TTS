"""GPU time of graph-replayed codec decode vs eager, by frame count; right-padding check."""
import sys, time, torch
import _paths  # noqa: F401  (sys.path)
from qwen_tts import Qwen3TTSModel
model = Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice', device_map='cuda:0',
                                      dtype=torch.bfloat16, attn_implementation='sdpa')
dec = model.model.speech_tokenizer.model.decoder
print('decoder dtype', next(dec.parameters()).dtype)
torch.manual_seed(0)
with torch.inference_mode():
    for n in (48, 96, 144, 192, 256, 320):
        inp = torch.randint(0, 2048, (1, 16, n), device='cuda')
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                dec(inp)
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = dec(inp)
        torch.cuda.synchronize()
        a = torch.cuda.Event(True); b = torch.cuda.Event(True); a.record()
        for _ in range(20):
            g.replay()
        b.record(); torch.cuda.synchronize(); gt = a.elapsed_time(b) / 20
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(5):
            dec(inp)
        torch.cuda.synchronize(); et = (time.perf_counter() - t) / 5 * 1e3
        ref = dec(inp)
        g.replay(); torch.cuda.synchronize()
        print(f'{n:4d} frames: graph {gt:6.2f} ms  eager {et:6.1f} ms  graph-vs-eager maxdiff {(out - ref).abs().max():.2e}')
        del g
