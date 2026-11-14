"""Overlapped chunked decode vs upstream decode on real generated codes: SNR per (chunk, ctx)."""
import sys, time, torch
import _paths  # noqa: F401  (sys.path)
from qwen_tts import Qwen3TTSModel
from bench_common import TEXTS
from talker_engine import install_talker_engine
from codec_overlap import OverlapCodec
model = Qwen3TTSModel.from_pretrained(f'Qwen/Qwen3-TTS-12Hz-{sys.argv[1]}-CustomVoice', device_map='cuda:0',
                                      dtype=torch.bfloat16, attn_implementation='sdpa')
h = install_talker_engine(model.model.talker); eng = h.engine
st = model.model.speech_tokenizer; dec = st.model.decoder
cap = {}
orig = st.decode
def spy(encoded):
    cap['codes'] = encoded[0]['audio_codes']; r = orig(encoded); cap['wav'] = r[0][0]; return r
st.decode = spy
with torch.inference_mode():
    for tid in ['long', 'medium']:
        torch.manual_seed(0)
        model.generate_custom_voice(text=TEXTS[tid], speaker='Ryan', language='English')
        codes = cap['codes']; n = codes.shape[0]
        ref = torch.from_numpy(cap['wav']).cuda()
        torch.cuda.synchronize(); t = time.perf_counter(); orig([{'audio_codes': codes}]); torch.cuda.synchronize()
        print(f'{tid}: {n} frames; upstream decode {1e3*(time.perf_counter()-t):.1f} ms')
        padded = torch.cat([codes, codes[-10:]], 0).t().unsqueeze(0).clamp(min=0)
        alt = dec(padded).reshape(-1)[:n * dec.total_upsample].float()
        e2 = alt - ref
        print(f'  noise floor (causal, +10 frames then trim): snr {10*torch.log10(ref.pow(2).mean()/e2.pow(2).mean().clamp_min(1e-20)):.1f} dB')
        eng.codes_out[:n].copy_(codes)
        for chunk, ctx in [(25, 25), (25, 72), (48, 48), (48, 96), (48, 150), (96, 96), (48, 300)]:
            oc = OverlapCodec(eng, dec, chunk=chunk, ctx=ctx); oc.reset()
            torch.cuda.synchronize(); t = time.perf_counter()
            for upto in range(1, n + 1):
                oc.launch(upto)
            out = oc.finish(n).float(); torch.cuda.synchronize(); el = time.perf_counter() - t
            err = out - ref
            snr = 10 * torch.log10(ref.pow(2).mean() / err.pow(2).mean().clamp_min(1e-20))
            print(f'  chunk {chunk:3d} ctx {ctx:3d}: snr {snr:6.1f} dB  max abs err {err.abs().max():.2e}  serial decode {el*1e3:6.1f} ms')
