"""Offline suite: 2 voices x 4 texts x N seeds, full generate_custom_voice (incl. codec).

--stack base   : previous accepted stack (fused predictor + talker fusion + graph + frame loop)
--stack engine : whole-frame graph TalkerEngine
--stack hf     : unmodified HF reference
Writes results/fast/<size>-<label>.json, WAVs under samples/fast/<size>-<label>/, and an ASR manifest.
"""
import argparse, json, statistics, sys, time
from pathlib import Path
import _paths  # noqa: F401  (sys.path)
import torch, soundfile as sf
from qwen_tts import Qwen3TTSModel
from bench_common import TEXTS

ROOT = _paths.ROOT
ap = argparse.ArgumentParser()
ap.add_argument('--size', required=True); ap.add_argument('--stack', required=True, choices=['base', 'engine', 'fast', 'hf'])
ap.add_argument('--label', required=True); ap.add_argument('--seeds', type=int, default=3)
ap.add_argument('--engine-kw', default='{}'); ap.add_argument('--engine-module', default='talker_engine')
ap.add_argument('--texts', default='short,medium,long,numbers'); ap.add_argument('--voices', default='Ryan,Aiden')
args = ap.parse_args()
model = Qwen3TTSModel.from_pretrained(f'Qwen/Qwen3-TTS-12Hz-{args.size}-CustomVoice', device_map='cuda:0',
                                      dtype=torch.bfloat16, attn_implementation='sdpa')
talker = model.model.talker
if args.stack == 'base':
    from qwen_fused_norm import install_fused_predictor_norm
    from qwen_fused_predictor import install_fused_predictor
    from qwen_talker_fusion import install_talker_fusion
    from qwen_talker_graph import install_talker_graph
    from qwen_fast_decode import install_fast_decode
    from qwen_frame_loop import install_frame_loop
    install_fused_predictor_norm(talker.code_predictor)
    install_fused_predictor(talker.code_predictor, mode='nonpdl' if args.size == '0.6B' else 'pdl')
    install_talker_fusion(talker); tg = install_talker_graph(talker); install_fast_decode(talker); install_frame_loop(talker)
    reset = tg.reset
elif args.stack == 'fast':
    mod = __import__(args.engine_module)
    h = mod.install_fast_tts(model, **json.loads(args.engine_kw))
    reset = lambda: None
elif args.stack == 'engine':
    mod = __import__(args.engine_module)
    h = mod.install_talker_engine(talker, **json.loads(args.engine_kw))
    reset = lambda: None
else:
    reset = lambda: None
# capture the talker frame counts
frames = {}
inner = talker.generate
def counted(*a, **kw):
    out = inner(*a, **kw)
    frames['n'] = sum(1 for x in out.hidden_states if x[-1] is not None)
    frames['eos'] = int(out.sequences[0, -1]) == kw['eos_token_id']
    return out
talker.generate = counted
samples = ROOT / f'samples/fast/{args.size.lower()}-{args.label}'; samples.mkdir(parents=True, exist_ok=True)
rows, manifest = [], []
texts = args.texts.split(',')
with torch.inference_mode():
    for voice in args.voices.split(','):
        for tid in texts:
            reset(); torch.manual_seed(1234)
            model.generate_custom_voice(text=TEXTS[tid], speaker=voice, language='English')  # warmup
            for seed in range(args.seeds):
                reset(); torch.manual_seed(seed); torch.cuda.synchronize(); t = time.perf_counter()
                wavs, sr = model.generate_custom_voice(text=TEXTS[tid], speaker=voice, language='English')
                torch.cuda.synchronize(); el = time.perf_counter() - t
                dur = len(wavs[0]) / sr
                p = samples / f'{voice}-{tid}-s{seed}.wav'; sf.write(p, wavs[0], sr)
                rows.append(dict(voice=voice, text=tid, seed=seed, gen_s=el, audio_s=dur, rtf=el / dur,
                                 frames=frames['n'], eos=frames['eos'], ms_per_frame=el * 1e3 / max(1, frames['n'])))
                manifest.append(dict(wav=str(p), text=TEXTS[tid]))
                print(f"{voice:5s} {tid:8s} s{seed} {el*1e3:7.1f} ms audio {dur:5.2f}s rtf {el/dur:.4f} frames {frames['n']} eos {frames['eos']}", flush=True)
cond = {}
for r in rows:
    cond.setdefault((r['voice'], r['text']), []).append(r['rtf'])
mean_med = statistics.mean(statistics.median(v) for v in cond.values())
agg = dict(mean_condition_median_rtf=mean_med, total_gen_s=sum(r['gen_s'] for r in rows),
           total_audio_s=sum(r['audio_s'] for r in rows), all_eos=all(r['eos'] for r in rows))
agg['pooled_rtf'] = agg['total_gen_s'] / agg['total_audio_s']
print('SUMMARY', json.dumps(agg))
out = ROOT / f'results/fast/{args.size.lower()}-{args.label}.json'; out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(dict(args=vars(args), summary=agg, rows=rows), indent=1))
(out.with_suffix('.manifest.json')).write_text(json.dumps(manifest, indent=1))
