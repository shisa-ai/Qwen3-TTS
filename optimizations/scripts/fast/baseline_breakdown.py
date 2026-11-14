"""Per-component timing of the current accepted stack (combined + frame loop)."""
import sys, time, torch
import _paths  # noqa: F401  (sys.path)
from qwen_tts import Qwen3TTSModel
from bench_common import TEXTS
from qwen_fused_norm import install_fused_predictor_norm
from qwen_fused_predictor import install_fused_predictor
from qwen_talker_fusion import install_talker_fusion
from qwen_talker_graph import install_talker_graph
from qwen_fast_decode import install_fast_decode
from qwen_frame_loop import install_frame_loop
size = sys.argv[1]
model = Qwen3TTSModel.from_pretrained(f'Qwen/Qwen3-TTS-12Hz-{size}-CustomVoice', device_map='cuda:0', dtype=torch.bfloat16, attn_implementation='sdpa')
talker = model.model.talker; pred = talker.code_predictor
install_fused_predictor_norm(pred)
pg = install_fused_predictor(pred, mode='nonpdl' if size == '0.6B' else 'pdl')
install_talker_fusion(talker); tg = install_talker_graph(talker); install_fast_decode(talker); loop = install_frame_loop(talker)
def ev(fn, n=200):
    torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / n
with torch.inference_mode():
    for seed in range(3):
        torch.manual_seed(seed)
        torch.cuda.synchronize(); t = time.perf_counter()
        wavs, sr = model.generate_custom_voice(text=TEXTS['medium'], speaker='Ryan', language='English')
        torch.cuda.synchronize(); el = time.perf_counter() - t
        print(f'request {el*1e3:.1f} ms audio {len(wavs[0])/sr:.2f}s rtf {el/(len(wavs[0])/sr):.4f}')
    print('talker graph replay ms', ev(tg.graph.replay))
    (key, g), = pg.graphs.items()
    print('predictor graph replay ms', ev(g.graph.replay))
    # time talker.generate alone vs codec decode
    import qwen_tts.core.models.modeling_qwen3_tts as M
    orig = talker.generate; T = {}
    def timed(*a, **kw):
        torch.cuda.synchronize(); t = time.perf_counter(); r = orig(*a, **kw); torch.cuda.synchronize(); T['gen'] = time.perf_counter() - t; T['frames'] = len(r.hidden_states); return r
    talker.generate = timed
    torch.manual_seed(0); torch.cuda.synchronize(); t = time.perf_counter()
    wavs, sr = model.generate_custom_voice(text=TEXTS['medium'], speaker='Ryan', language='English')
    torch.cuda.synchronize(); el = time.perf_counter() - t
    print(f"total {el*1e3:.1f} talker.generate {T['gen']*1e3:.1f} frames {T['frames']} per-frame {T['gen']*1e3/T['frames']:.2f} other(codec+prep) {(el-T['gen'])*1e3:.1f}")
