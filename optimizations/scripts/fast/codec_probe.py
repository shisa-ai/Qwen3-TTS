"""Is the 12 Hz codec decoder causal (prefix decode == full decode prefix)? How long does decode take?"""
import sys, time, torch
import _paths  # noqa: F401  (sys.path)
from qwen_tts import Qwen3TTSModel
model = Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice', device_map='cuda:0',
                                      dtype=torch.bfloat16, attn_implementation='sdpa')
st = model.model.speech_tokenizer
dec = st.model.decoder
up = int(dec.total_upsample)
print(type(dec).__name__, next(dec.parameters()).dtype, 'upsample', up)
torch.manual_seed(0)
codes = torch.randint(0, 2048, (1, 16, 260), device='cuda')
with torch.inference_mode():
    full = dec(codes)
    for n in (50, 120, 200):
        pre = dec(codes[..., :n])
        d = (pre[..., :n * up] - full[..., :n * up]).abs()
        print(f'prefix {n}: max abs diff vs full {d.max():.3e}; last-frame region {d[..., -up:].max():.3e}; '
              f'first {n - 2} frames {d[..., :(n - 2) * up].max():.3e}; rms {full.pow(2).mean().sqrt():.3e}')
    for n in (25, 100, 250):
        c = codes[..., :n]
        for _ in range(2):
            dec(c)
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(5):
            dec(c)
        torch.cuda.synchronize()
        print(f'decode {n} frames: {(time.perf_counter() - t) / 5 * 1e3:.1f} ms')
    # chunk [s-L, e) vs full decode, measured on the new region [s, e)
    s, e = 150, 200
    base = full[..., s * up:e * up]
    for L in (0, 25, 50, 72, 100, 150):
        out = dec(codes[..., s - L:e])[..., L * up:]
        d = (out - base).abs()
        print(f'ctx {L:3d}: max {d.max():.3e} mean {d.mean():.3e} snr_db {10*torch.log10(base.pow(2).mean()/(out-base).pow(2).mean().clamp_min(1e-20)):.1f}')
    # numerical noise floor: same input, different total length (kernel shapes differ)
    alt = dec(codes[..., :240])[..., s * up:e * up]
    d = (alt - base).abs()
    print(f'noise floor (prefix 240 vs 260): max {d.max():.3e} snr_db {10*torch.log10(base.pow(2).mean()/(alt-base).pow(2).mean().clamp_min(1e-20)):.1f}')
