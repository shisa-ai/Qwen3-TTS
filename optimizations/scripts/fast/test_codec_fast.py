import sys, time, torch
import _paths  # noqa: F401  (sys.path)
from qwen_tts import Qwen3TTSModel
from bench_common import TEXTS
from codec_fast import install_codec_fast
model = Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice', device_map='cuda:0', dtype=torch.bfloat16, attn_implementation='sdpa')
dec = model.model.speech_tokenizer.model.decoder
st = model.model.speech_tokenizer
cap = {}; orig = st.decode
def spy(e): cap['codes'] = e[0]['audio_codes']; return orig(e)
st.decode = spy
with torch.inference_mode():
    torch.manual_seed(0); model.generate_custom_voice(text=TEXTS['long'], speaker='Ryan', language='English')
    codes = cap['codes'].t().unsqueeze(0).clamp(min=0)
    def bench(label):
        for _ in range(2): dec(codes)
        torch.cuda.synchronize(); s = time.perf_counter()
        for _ in range(5): o = dec(codes)
        torch.cuda.synchronize(); print(f'{label}: {(time.perf_counter()-s)/5*1e3:.1f} ms for {codes.shape[-1]} frames'); return o.float()
    ref = bench('original')
    pad = torch.cat([codes, codes[..., -10:]], -1); alt = dec(pad)[..., :ref.shape[-1]].float()
    snr = lambda o: 10 * torch.log10(ref.pow(2).mean() / (o - ref).pow(2).mean())
    print(f'noise floor snr {snr(alt):.1f} dB')
    for kw in [dict(snake=True, dilated=False), dict(snake=False, dilated=True)]:
        pass
    import copy
    d32 = copy.deepcopy(dec).float()
    r32 = d32(codes).float()
    snr32 = lambda o: 10 * torch.log10(r32.pow(2).mean() / (o - r32).pow(2).mean())
    print(f'original bf16 vs fp32 reference: {snr32(ref):.1f} dB')
    del d32
    print(install_codec_fast(dec))
    o = bench('fast (snake+polyphase)')
    print(f'fast vs original snr {snr(o):.1f} dB, max abs {(o-ref).abs().max():.2e}; fast vs fp32 reference {snr32(o):.1f} dB')
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        dec(codes); torch.cuda.synchronize()
    rows = sorted([(k.key, k.count, k.device_time_total) for k in prof.key_averages() if k.device_time_total > 0], key=lambda r: -r[2])
    print('total ms', sum(r[2] for r in rows) / 1e3)
    for n, c, t in rows[:14]: print(f'{t/1e3:7.2f} ms {c:5d}x {n[:110]}')
