"""Breakdown of the engine: prefill, frame replay, predictor share, codec; kernel table of one frame."""
import json, sys, time
import _paths  # noqa: F401  (sys.path)
import torch
from qwen_tts import Qwen3TTSModel
from bench_common import TEXTS

size = sys.argv[1]; modname = sys.argv[2] if len(sys.argv) > 2 else 'talker_engine'
ekw = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
mod = __import__(modname)
model = Qwen3TTSModel.from_pretrained(f'Qwen/Qwen3-TTS-12Hz-{size}-CustomVoice', device_map='cuda:0',
                                      dtype=torch.bfloat16, attn_implementation='sdpa')
talker = model.model.talker
h = mod.install_talker_engine(talker, **ekw); eng = h.engine
T = {}
inner_run = eng.run
def run(kw, **k):
    torch.cuda.synchronize(); t = time.perf_counter(); r = inner_run(kw, **k); torch.cuda.synchronize()
    T['talker_s'] = time.perf_counter() - t; T['frames'] = r[1]; return r
eng.run = run
dec = model.model.speech_tokenizer if hasattr(model.model, 'speech_tokenizer') else None
with torch.inference_mode():
    for i in range(3):
        torch.manual_seed(i); torch.cuda.synchronize(); t = time.perf_counter()
        wavs, sr = model.generate_custom_voice(text=TEXTS['long'], speaker='Ryan', language='English')
        torch.cuda.synchronize(); el = time.perf_counter() - t
        print(f"total {el*1e3:.1f} ms audio {len(wavs[0])/sr:.2f}s talker {T['talker_s']*1e3:.1f} ms frames {T['frames']} "
              f"=> {T['talker_s']*1e3/T['frames']:.3f} ms/frame; outside talker (prep+codec) {(el-T['talker_s'])*1e3:.1f} ms")
    g = next(iter(eng.graphs.values()))
    def ev(fn, n=100):
        torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
        for _ in range(n): fn()
        e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / n
    eng.state[1] = 300  # mid-length context
    print('frame replay ms', round(ev(g.replay), 3))
    pu = eng.uni[eng.vocab:].view(1, 15, eng.pvocab)
    pg = torch.cuda.CUDAGraph()
    with torch.cuda.graph(pg):
        eng.pred(eng.tok, eng.first, eng.past, 1 / 0.9, 50, pu)
    print('predictor-only replay ms', round(ev(pg.replay), 3))
    tg = torch.cuda.CUDAGraph()
    with torch.cuda.graph(tg):
        eng._transformer()
    print('talker-transformer-only replay ms', round(ev(tg.replay), 3))
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            g.replay()
        torch.cuda.synchronize()
    ka = prof.key_averages()
    rows = sorted([(k.key, k.count / 10, k.device_time_total / 10) for k in ka if k.device_time_total > 0], key=lambda r: -r[2])
    tot = sum(r[2] for r in rows); cnt = sum(r[1] for r in rows)
    print(f'kernels/frame {cnt:.0f}, summed kernel time/frame {tot/1e3:.3f} ms')
    for name, c, t in rows[:25]:
        print(f'{t/1e3:8.3f} ms  {c:6.0f}x  {name[:110]}')
