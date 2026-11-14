"""Talker numerics vs an FP32 reference, teacher-forced on the engine's own codes.

Compares (a) HF BF16 parallel forward and (b) the engine's incremental hidden states against the
same model in FP32: relative hidden error and KL(fp32 || x) of the codec-head distribution after
temperature 0.9 (the distribution actually sampled).
"""
import copy, sys, torch
import _paths  # noqa: F401
from qwen_tts import Qwen3TTSModel
from bench_common import TEXTS
import talker_engine as te

size = sys.argv[1]
variant = sys.argv[2] if len(sys.argv) > 2 else 'default'
model = Qwen3TTSModel.from_pretrained(f'Qwen/Qwen3-TTS-12Hz-{size}-CustomVoice', device_map='cuda:0',
                                      dtype=torch.bfloat16, attn_implementation='sdpa')
talker = model.model.talker
cap = {}
orig = talker.generate
def spy(*a, **kw):
    cap['kw'] = kw; return orig(*a, **kw)
talker.generate = spy
with torch.inference_mode():
    model.generate_custom_voice(text=TEXTS['medium'], speaker='Ryan', language='English', max_new_tokens=4)
talker.generate = orig
kw = cap['kw']
h = te.install_talker_engine(talker, **te.load_tuned(size)); eng = h.engine
t32 = Qwen3TTSModel.from_pretrained(f'Qwen/Qwen3-TTS-12Hz-{size}-CustomVoice', device_map='cuda:0',
                                    dtype=torch.float32, attn_implementation='sdpa').model.talker
with torch.inference_mode():
    torch.manual_seed(0)
    kw2 = dict(kw); kw2['max_new_tokens'] = 120
    out = h.generate(**kw2)
    codes = torch.cat([x[-1] for x in out.hidden_states if x[-1] is not None], 0)
    nf = len(codes)
    x = kw['inputs_embeds']; S0 = x.shape[1]
    emb0 = talker.get_input_embeddings(); embs = talker.code_predictor.get_input_embeddings()
    rows = []
    for i in range(nf):
        c = codes[i:i + 1]
        e = torch.cat([emb0(c[:, :1])] + [embs[j](c[:, j + 1:j + 2]) for j in range(15)], 1).sum(1, keepdim=True)
        rows.append(e + kw['tts_pad_embed'])
    full = torch.cat([x] + rows, 1)
    mask = torch.ones(full.shape[:2], device=x.device, dtype=torch.long)
    hb = talker.model(inputs_embeds=full, attention_mask=mask, use_cache=False).last_hidden_state[0, S0 - 1:S0 + nf]
    h32 = t32.model(inputs_embeds=full.float(), attention_mask=mask, use_cache=False).last_hidden_state[0, S0 - 1:S0 + nf]
    he = eng.hist[:nf + 1]
    def stats(hx, name):
        rel = ((hx.float() - h32).norm(dim=-1) / h32.norm(dim=-1))
        lp32 = torch.log_softmax(t32.codec_head(h32) / 0.9, -1)
        lpx = torch.log_softmax(talker.codec_head(hx.bfloat16()).float() / 0.9, -1)
        kl = (lp32.exp() * (lp32 - lpx)).sum(-1)
        print(f'{name:22s} hidden rel-err vs fp32: mean {rel.mean():.4f} p95 {rel.quantile(.95):.4f} | '
              f'KL(fp32||x) mean {kl.mean():.2e} max {kl.max():.2e}')
    print(f'{size} variant={variant}, {nf} frames')
    stats(hb, 'HF bf16 (parallel)')
    stats(he, 'engine (incremental)')
