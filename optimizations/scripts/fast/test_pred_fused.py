import sys, torch
import _paths  # noqa: F401  (sys.path)
from qwen_tts import Qwen3TTSModel
from qwen_fused_predictor import _k, _mapped
from pred_fused import FusedAttnOPredictor
size = sys.argv[1]
model = Qwen3TTSModel.from_pretrained(f'Qwen/Qwen3-TTS-12Hz-{size}-CustomVoice', device_map='cuda:0',
                                      dtype=torch.bfloat16, attn_implementation='sdpa')
t = model.model.talker; cp = t.code_predictor
import json
tuned = {k: tuple(v) for k, v in json.load(open(_paths.TUNED / f'tuned-{size}.json'))['pred_configs'].items()}
def mk(cls, o=None):
    P = cls(_mapped(cp), max_batch=1, mode='pdl')
    c = dict(tuned)
    if o: c['o'] = o
    P._pdl_configs = ((1, c),)
    P.pdl_ws = torch.zeros(64 * P._rows * 8192, device='cuda', dtype=torch.float32)
    return P
A = mk(_k.FusedCodePredictor)
B = mk(FusedAttnOPredictor, (64, 256, 2, 16, 4, 1, 1))
H = t.config.hidden_size
emb = t.get_input_embeddings().weight
same = tot = 0
with torch.inference_mode():
    for trial in range(200):
        g = torch.Generator(device='cuda').manual_seed(trial)
        code = torch.randint(0, 2048, (1,), device='cuda', generator=g)
        first = emb[code].view(1, H)
        past = torch.randn(1, H, device='cuda', generator=g, dtype=torch.bfloat16)
        u = torch.rand(1, 15, 2048, device='cuda', generator=g).clamp_(1e-7, 1 - 1e-7)
        if trial % 2:
            u.fill_(0.5); k = 1
        else:
            k = 50
        ca = A(code, first, past, 1 / 0.9, k, u); cb = B(code, first, past, 1 / 0.9, k, u)
        same += int((ca == cb).sum()); tot += ca.numel()
    print(f'code agreement {same / tot:.5f} ({tot} codes)')
    def bench(P):
        code = torch.zeros(1, dtype=torch.long, device='cuda'); first = emb[code].view(1, H)
        past = torch.randn(1, H, device='cuda', dtype=torch.bfloat16); u = torch.rand(1, 15, 2048, device='cuda')
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            P(code, first, past, 1.1, 50, u)
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            P(code, first, past, 1.1, 50, u)
        best = 1e9
        for _ in range(5):
            a = torch.cuda.Event(True); b = torch.cuda.Event(True); torch.cuda.synchronize(); a.record()
            for _ in range(50):
                gr.replay()
            b.record(); torch.cuda.synchronize(); best = min(best, a.elapsed_time(b) / 50)
        return best
    print(f'vendored(tuned) {bench(A):.4f} ms')
    for o in [(64, 256, 2, 16, 4, 1, 1), (64, 128, 2, 16, 4, 1, 1), (32, 256, 1, 16, 4, 1, 1), (64, 256, 1, 16, 4, 1, 1),
              (32, 128, 2, 16, 4, 1, 1), (64, 128, 4, 16, 4, 1, 1), (128, 128, 2, 16, 8, 1, 1)]:
        try:
            print(f'fused attn+o o={o}: {bench(mk(FusedAttnOPredictor, o)):.4f} ms')
        except Exception as e:
            print('fail', o, str(e)[:100])
