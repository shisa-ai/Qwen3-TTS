import sys, torch
import _paths  # noqa: F401  (sys.path)
from qwen_tts import Qwen3TTSModel
from torch.profiler import profile, ProfilerActivity
model = Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice', device_map='cuda:0', dtype=torch.bfloat16, attn_implementation='sdpa')
dec = model.model.speech_tokenizer.model.decoder
print(dec)
inp = torch.randint(0, 2048, (1, 16, 256), device='cuda')
with torch.inference_mode():
    for _ in range(2): dec(inp)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        dec(inp); torch.cuda.synchronize()
rows = sorted([(k.key, k.count, k.device_time_total) for k in prof.key_averages() if k.device_time_total > 0], key=lambda r: -r[2])
print('total ms', sum(r[2] for r in rows) / 1e3, 'kernels', sum(r[1] for r in rows))
for n, c, t in rows[:22]: print(f'{t/1e3:7.2f} ms {c:5d}x {n[:120]}')
