"""Per-request timeline: prompt build, prefill, first sample, frame loop, codec, numpy."""
import sys, time, torch
import _paths  # noqa: F401  (sys.path)
from qwen_tts import Qwen3TTSModel
from bench_common import TEXTS
import talker_engine as te
size = sys.argv[1]
model = Qwen3TTSModel.from_pretrained(f'Qwen/Qwen3-TTS-12Hz-{size}-CustomVoice', device_map='cuda:0',
                                      dtype=torch.bfloat16, attn_implementation='sdpa')
h = te.install_fast_tts(model); eng = h.engine
T = []
def mark(name):
    torch.cuda.synchronize(); T.append((name, time.perf_counter()))
talker = model.model.talker
# wrap pieces
orig_run = eng.run
def run(kw, **k):
    mark('talker.generate entered (prompt built)')
    return orig_run(kw, **k)
eng.run = run
orig_fwd = talker.forward
def fwd(*a, **k):
    r = orig_fwd(*a, **k); mark('HF prefill done'); return r
talker.forward = fwd
orig_graph = eng._graph
def graph(*a):
    r = orig_graph(*a); return r
orig_sample = eng._sample
def sample(key, adv):
    r = orig_sample(key, adv)
    if adv == 0:
        mark('state setup + first sample')
    return r
eng._sample = sample
st = model.model.speech_tokenizer
orig_dec = st.decode
def dec(e):
    mark('frame loop done'); r = orig_dec(e); mark('codec decode + to numpy'); return r
st.decode = dec
with torch.inference_mode():
    for tid in ['short', 'short', 'short', 'long']:
        T.clear(); torch.manual_seed(0)
        mark('start')
        wavs, sr = model.generate_custom_voice(text=TEXTS[tid], speaker='Ryan', language='English')
        mark('end')
        print(f'--- {tid}: total {1e3*(T[-1][1]-T[0][1]):.1f} ms, audio {len(wavs[0])/sr:.2f}s, frames {len(wavs[0])//1920}')
        for (n0, t0), (n1, t1) in zip(T, T[1:]):
            print(f'   {1e3*(t1-t0):8.2f} ms  {n1}')
