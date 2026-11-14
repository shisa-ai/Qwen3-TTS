import sys, time, torch
import _paths  # noqa: F401  (sys.path)
from qwen_tts import Qwen3TTSModel
from bench_common import TEXTS
import talker_engine as te
size = sys.argv[1]
model = Qwen3TTSModel.from_pretrained(f'Qwen/Qwen3-TTS-12Hz-{size}-CustomVoice', device_map='cuda:0',
                                      dtype=torch.bfloat16, attn_implementation='sdpa')
t = model.model.talker
eng = te.TalkerEngine(t, **te.load_tuned(size))
cap = {}; o = t.generate
def spy(*a, **kw):
    cap['kw'] = kw; raise SystemExit
t.generate = spy
rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()
with torch.inference_mode():
    for tid in TEXTS:
        try:
            model.generate_custom_voice(text=TEXTS[tid], speaker='Ryan', language='English')
        except SystemExit:
            pass
        kw = cap['kw']; x = kw['inputs_embeds']; S = x.shape[1]
        out = t.forward(inputs_embeds=x, attention_mask=kw['attention_mask'], use_cache=True, return_dict=True,
                        trailing_text_hidden=kw['trailing_text_hidden'], tts_pad_embed=kw['tts_pad_embed'])
        ref_logits, ref_past = out.logits[0, -1].float(), out.past_hidden[0, -1].float()
        ref_k = torch.stack([l.keys[0] for l in out.past_key_values.layers])  # [L, KVH, S, HD]
        ref_v = torch.stack([l.values[0] for l in out.past_key_values.layers])
        eng._prefill(x); torch.cuda.synchronize()
        k = eng.kc[:, :, :S]; v = eng.vc[:, :, :S]
        print(f'{tid:8s} S={S:3d}: logits rel {rel(eng.logits[0], ref_logits):.4f} argmax same {int(eng.logits[0].argmax()) == int(ref_logits.argmax())} '
              f'top5 overlap {len(set(eng.logits[0].topk(5).indices.tolist()) & set(ref_logits.topk(5).indices.tolist()))}/5; '
              f'past rel {rel(eng.past[0], ref_past):.4f}; K rel {rel(k, ref_k):.4f} V rel {rel(v, ref_v):.4f}')
        def tm(fn, n=10):
            torch.cuda.synchronize(); s0 = time.perf_counter()
            for _ in range(n): fn()
            torch.cuda.synchronize(); return (time.perf_counter() - s0) / n * 1e3
        hf = tm(lambda: t.forward(inputs_embeds=x, attention_mask=kw['attention_mask'], use_cache=True, return_dict=True,
                                  trailing_text_hidden=kw['trailing_text_hidden'], tts_pad_embed=kw['tts_pad_embed']))
        own = tm(lambda: eng._prefill(x))
        print(f'          HF prefill {hf:.2f} ms   own prefill {own:.2f} ms')
