"""Correctness + speed check of TalkerEngine against the unmodified HF talker."""
import argparse, sys, time
import _paths  # noqa: F401  (sys.path)
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
from bench_common import TEXTS
from talker_engine import install_talker_engine

ap = argparse.ArgumentParser()
ap.add_argument('size'); ap.add_argument('--mode', default='pdl'); ap.add_argument('--splits', type=int, default=8)
ap.add_argument('--out', default=None)
args = ap.parse_args()
model = Qwen3TTSModel.from_pretrained(f'Qwen/Qwen3-TTS-12Hz-{args.size}-CustomVoice', device_map='cuda:0',
                                      dtype=torch.bfloat16, attn_implementation='sdpa')
talker = model.model.talker
captured = {}
orig = talker.generate
def spy(*a, **kw):
    captured['kw'] = kw
    r = orig(*a, **kw); captured['out'] = r; return r
talker.generate = spy

def codes_of(out):
    return torch.cat([h[-1] for h in out.hidden_states if h[-1] is not None], 0)

with torch.inference_mode():
    torch.manual_seed(0)
    model.generate_custom_voice(text=TEXTS['medium'], speaker='Ryan', language='English',
                                do_sample=False, subtalker_dosample=False)
    kw_greedy = dict(captured['kw']); ref_codes = codes_of(captured['out'])
    talker.generate = orig
    h = install_talker_engine(talker, predictor_mode=args.mode, attn_splits=args.splits)
    eng = h.engine
    t = time.perf_counter(); out = h.generate(**kw_greedy); torch.cuda.synchronize()
    print('first engine call (incl capture) s', round(time.perf_counter() - t, 2))
    out = h.generate(**kw_greedy)
    e_codes = codes_of(out)
    n = min(len(ref_codes), len(e_codes))
    eq = (ref_codes[:n] == e_codes[:n]).all(1)
    div = int((~eq).nonzero()[0]) if (~eq).any() else n
    print(f'greedy frames ref {len(ref_codes)} engine {len(e_codes)}; identical prefix {div} frames; '
          f'cb0 match {(ref_codes[:n,0]==e_codes[:n,0]).float().mean():.3f} all-cb match {(ref_codes[:n]==e_codes[:n]).float().mean():.3f}')
    # Teacher-forced: HF hidden states over the engine's own codes.
    kw = kw_greedy; x = kw['inputs_embeds']; S0 = x.shape[1]; nf = len(e_codes)
    emb0 = talker.get_input_embeddings(); embs = talker.code_predictor.get_input_embeddings()
    rows = []
    tr = kw['trailing_text_hidden']
    for i in range(nf):
        c = e_codes[i:i+1]
        e = torch.cat([emb0(c[:, :1])] + [embs[j](c[:, j+1:j+2]) for j in range(15)], 1).sum(1, keepdim=True)
        e = e + (tr[:, i:i+1] if i < tr.shape[1] else kw['tts_pad_embed'])
        rows.append(e)
    full = torch.cat([x] + rows, 1)
    mask = torch.ones(full.shape[:2], device=x.device, dtype=kw['attention_mask'].dtype)
    ho = talker(inputs_embeds=full, attention_mask=mask, use_cache=False, return_dict=True,
                trailing_text_hidden=tr, tts_pad_embed=kw['tts_pad_embed'])
    ref_h = ho.past_hidden if False else None
    hf_last = talker.model(inputs_embeds=full, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state[0]
    hf_h = hf_last[S0 - 1:S0 - 1 + nf + 1].float()
    en_h = eng.hist[:nf + 1].float()
    rel = ((hf_h - en_h).norm(dim=-1) / hf_h.norm(dim=-1))
    cos = torch.nn.functional.cosine_similarity(hf_h, en_h, dim=-1)
    lg_hf = talker.codec_head(hf_h.bfloat16()).float(); lg_en = talker.codec_head(en_h.bfloat16()).float()
    print(f'teacher-forced hidden rel-err mean {rel.mean():.4f} max {rel.max():.4f}; cos min {cos.min():.5f}; '
          f'logit argmax agree {(lg_hf.argmax(-1)==lg_en.argmax(-1)).float().mean():.3f}; '
          f'logit max abs diff {(lg_hf-lg_en).abs().max():.3f}')
    # Sampled end-to-end
    for seed in range(4):
        torch.manual_seed(seed); torch.cuda.synchronize(); t = time.perf_counter()
        wavs, sr = model.generate_custom_voice(text=TEXTS['medium'], speaker='Ryan', language='English')
        torch.cuda.synchronize(); el = time.perf_counter() - t
        dur = len(wavs[0]) / sr
        print(f'sampled seed {seed}: {el*1e3:.1f} ms, audio {dur:.2f}s, rtf {el/dur:.4f}')
        if args.out and seed == 0:
            sf.write(args.out, wavs[0], sr)
    # per-frame replay time
    g = next(iter(eng.graphs.values()))
    torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    st = eng.state.clone()
    s.record()
    for _ in range(50):
        g.replay()
    e.record(); torch.cuda.synchronize()
    print('frame graph replay ms', s.elapsed_time(e) / 50, 'calls', eng.calls, 'fallbacks', eng.fallback_calls)
