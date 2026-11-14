"""Tune the PDL GEMM chain configs (talker + predictor) and the attention kernel on this GPU.

Stage 1: each GEMM shape alone, PDL chain of 24 launches over 24 distinct weights (no L2 reuse).
Stage 2: coordinate descent over the stage-1 top-k per name, scored by the real captured graphs.
Writes scripts/fast/tuned/tuned-<size>.json (loaded automatically by install_fast_tts) and caches
stage-1 timings per GPU type in tuned/gemm_stage1.json.
"""
import itertools, json, sys, time
import _paths  # noqa: F401  (sys.path)
import torch, triton
from qwen_tts import Qwen3TTSModel
import talker_engine as te

size = sys.argv[1]
TOP = 4
model = Qwen3TTSModel.from_pretrained(f'Qwen/Qwen3-TTS-12Hz-{size}-CustomVoice', device_map='cuda:0',
                                      dtype=torch.bfloat16, attn_implementation='sdpa')
eng = te.TalkerEngine(model.model.talker)
dev = eng.device


def fits(cfg, n, k, name):
    if cfg[0] == 's':
        _, bn, bk, sk = cfg[:4]
        ok = n % bn == 0 and k % (bk * sk) == 0 and (k // sk) % 64 == 0 and (k // sk) >= bk
    else:
        bn, bk, nch = cfg[:3]
        ok = n % bn == 0 and k % (bk * nch) == 0 and nch in (1, 2, 4)
        ok = ok and bn * bk * nch <= 16384 * cfg[4]
    if name == 'gu':
        ok = ok and (cfg[1] if cfg[0] == 's' else cfg[0]) == te._PDL_GU_BN
    return ok


def candidates(name, n, k):
    out = []
    for bn, bk, nch, w in itertools.product([32, 64, 128], [128, 256], [1, 2, 4], [4, 8]):
        c = (bn, bk, nch, 16, w, 1, 1)
        if fits(c, n, k, name):
            out.append(c)
    for bn, bk, sk, w, st in itertools.product([64, 128], [64, 128], [1, 2, 4, 8], [4], [3, 4]):
        c = ('s', bn, bk, sk, 16, w, st)
        if fits(c, n, k, name):
            out.append(c)
    return out


def timed_graph(fn, reps=3, n=20):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    best = 1e9
    for _ in range(reps):
        torch.cuda.synchronize(); a = torch.cuda.Event(True); b = torch.cuda.Event(True); a.record()
        for _ in range(n):
            g.replay()
        b.record(); torch.cuda.synchronize(); best = min(best, a.elapsed_time(b) / n)
    del g
    return best


CACHE = str(_paths.TUNED / 'gemm_stage1.json')


def stage1(shapes):
    """shapes: name -> (N, K, epi). Returns name -> ranked [(us, cfg)]. Cached per (name, N, K)."""
    import os
    res = {}
    for name, (N, K, epi) in shapes.items():
        key = f'{torch.cuda.get_device_name(dev)}:{name}:{N}x{K}'  # per GPU type
        cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
        if key in cache:
            res[name] = [(u, tuple(c)) for u, c in cache[key]]
            print(name, (N, K), 'cached', res[name][:TOP], flush=True)
            continue
        Ws = [torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(24)]
        a = torch.randn(1, K, device=dev, dtype=torch.bfloat16)
        out = torch.zeros(1, N // 2 if epi == 1 else N, device=dev, dtype=torch.bfloat16)
        if epi == 2:
            out = torch.zeros(1, eng.hidden if N == eng.hidden else N, device=dev, dtype=torch.bfloat16)
        ss = torch.ones(max(N, K) // 16 + 1, 1, device=dev, dtype=torch.float32)
        ss2 = torch.zeros_like(ss)
        ranked = []
        for c in candidates(name, N, K):
            def fn():
                for W in Ws:
                    if epi == 2:
                        eng._gemm(c, a, W, out, None, 0, ss2, 2)
                    else:
                        eng._gemm(c, a, W, out, ss, 1, None, epi, 1e-6)
            try:
                us = timed_graph(fn) / len(Ws) * 1e3
            except Exception as e:  # noqa: BLE001  (resource limits for a config)
                continue
            ranked.append((us, c))
        ranked.sort()
        res[name] = ranked
        cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
        cache[key] = ranked[:12]
        json.dump(cache, open(CACHE, 'w'))
        print(name, (N, K), 'best', [(round(u, 2), c) for u, c in ranked[:TOP]], flush=True)
        del Ws
    return res


L0 = eng.layers[0]
talker_shapes = {'qkv': (*L0['qkv'].shape, 0), 'o': (*L0['o'].shape, 2), 'gu': (*L0['gu'].shape, 1),
                 'down': (*L0['down'].shape, 2), 'lm': (*eng.head.shape, 0)}
P = eng.pred
pw = P.pdl_w[0]
pred_shapes = {'qkv': (*pw['qkv'].shape, 0), 'o': (*pw['o'].shape, 2), 'gu': (*pw['gu'].shape, 1),
               'down': (*pw['down'].shape, 2), 'lm': (*P.pdl_lm[0].shape, 0)}
print('talker shapes', talker_shapes, 'pred shapes', pred_shapes)
# the GEMM tuner calls eng._gemm with eng.hidden as HID; only numerics depend on it
t1 = stage1(talker_shapes)
p1 = stage1(pred_shapes)

# Stage 2: real graphs. Need a valid state: run one short request to populate everything.
kw_store = {}
talker = model.model.talker
orig = talker.generate
def spy(*a, **kw):
    kw_store['kw'] = kw; return orig(*a, **kw)
talker.generate = spy
with torch.inference_mode():
    model.generate_custom_voice(text='Hello there, this is a tuning run.', speaker='Ryan', language='English', max_new_tokens=8)
talker.generate = orig
kw = dict(kw_store['kw']); kw['max_new_tokens'] = 4
with torch.inference_mode():
    eng.run(kw)
eng.state[1] = 300
pu = eng.uni[eng.vocab:].view(1, 15, eng.pvocab)


def talker_time():
    return timed_graph(eng._transformer, reps=5, n=30)


def pred_time():
    return timed_graph(lambda: eng.pred(eng.tok, eng.first, eng.past, 1 / 0.9, 50, pu), reps=5, n=30)


with torch.inference_mode():
    # attention sweep first (talker graph objective)
    best_t = talker_time(); print('talker start ms', round(best_t, 4))
    for S, bk, w in itertools.product([2, 4, 8, 16, 32], [16, 32, 64], [2, 4, 8]):
        old = (eng.S, eng.abk, eng.awarps)
        eng.S, eng.abk, eng.awarps = S, bk, w
        try:
            t = talker_time()
        except Exception:
            t = 1e9
        if t < best_t:
            best_t = t; print('  attn', (S, bk, w), round(t, 4), flush=True)
        else:
            eng.S, eng.abk, eng.awarps = old
    for _ in range(2):
        for name, ranked in t1.items():
            for us, c in ranked[:TOP]:
                old = eng.cfgs[name]
                eng.cfgs[name] = c
                t = talker_time()
                if t < best_t - 0.002:
                    best_t = t; print('  talker', name, c, round(t, 4), flush=True)
                else:
                    eng.cfgs[name] = old
    print('talker final ms', round(best_t, 4), eng.cfgs, (eng.S, eng.abk, eng.awarps))
    best_p = pred_time(); print('pred start ms', round(best_p, 4))
    for _ in range(2):
        for name, ranked in p1.items():
            for us, c in ranked[:TOP]:
                cur = eng.pred_configs(); old = cur[name]
                cur[name] = c; eng.set_pred_configs(cur)
                t = pred_time()
                if t < best_p - 0.002:
                    best_p = t; print('  pred', name, c, round(t, 4), flush=True)
                else:
                    cur[name] = old; eng.set_pred_configs(cur)
    print('pred final ms', round(best_p, 4), eng.pred_configs())
out = dict(configs=eng.cfgs, attn_splits=eng.S, attn_bk=eng.abk, attn_warps=eng.awarps,
           pred_configs=eng.pred_configs(), talker_ms=best_t, pred_ms=best_p)
tag = te._gpu_tag()
json.dump(out, open(_paths.TUNED / f'tuned-{size}-{tag}.json', 'w'), indent=1, default=list)
json.dump(out, open(_paths.TUNED / f'tuned-{size}.json', 'w'), indent=1, default=list)
print('WROTE', f'tuned-{size}-{tag}.json', out)
