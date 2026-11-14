"""Generate CustomVoice audio with the whole-frame graph engine (offline, batch one).

Example:
  CUDA_VISIBLE_DEVICES=0 python optimizations/scripts/fast/run_fast.py --size 0.6B \
      --speaker ryan --text "Hello from the fast engine." --output out.wav
"""
import argparse
import time
from pathlib import Path

import _paths  # noqa: F401  (sys.path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--size', choices=['0.6B', '1.7B'], default='0.6B')
    p.add_argument('--text', required=True)
    p.add_argument('--speaker', default='ryan')
    p.add_argument('--language', default='English')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--temperature', type=float, default=0.9)
    p.add_argument('--top-k', type=int, default=50)
    p.add_argument('--subtalker-temperature', type=float, default=0.9)
    p.add_argument('--subtalker-top-k', type=int, default=50)
    p.add_argument('--greedy', action='store_true', help='argmax for both talker and predictor')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--repeat', type=int, default=1, help='generate N times and report steady-state timing')
    p.add_argument('--no-codec-fast', action='store_true', help='use the stock codec kernels')
    p.add_argument('--hf-prefill', action='store_true', help='use the HF prefill instead of the engine prefill')
    args = p.parse_args()
    if args.output.exists():
        p.error(f'{args.output} exists; refusing to overwrite')

    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel
    from talker_engine import install_fast_tts

    model = Qwen3TTSModel.from_pretrained(f'Qwen/Qwen3-TTS-12Hz-{args.size}-CustomVoice', device_map='cuda:0',
                                          dtype=torch.bfloat16, attn_implementation='sdpa')
    t0 = time.perf_counter()
    h = install_fast_tts(model, codec_fast=not args.no_codec_fast, own_prefill=not args.hf_prefill)
    print(f'install (kernel compile + graph capture of prefill/codec): {time.perf_counter() - t0:.1f}s')
    gen = dict(do_sample=not args.greedy, subtalker_dosample=not args.greedy,
               temperature=args.temperature, top_k=args.top_k,
               subtalker_temperature=args.subtalker_temperature, subtalker_top_k=args.subtalker_top_k)
    with torch.inference_mode():
        for i in range(args.repeat):
            if args.seed is not None:
                torch.manual_seed(args.seed + i)
            torch.cuda.synchronize(); s = time.perf_counter()
            wavs, sr = model.generate_custom_voice(text=args.text, speaker=args.speaker, language=args.language, **gen)
            torch.cuda.synchronize(); el = time.perf_counter() - s
            dur = len(wavs[0]) / sr
            note = ' (includes first-use frame-graph capture)' if i == 0 else ''
            print(f'run {i}: {el * 1e3:.1f} ms for {dur:.2f}s audio, RTF {el / dur:.4f}{note}')
    sf.write(args.output, wavs[0], sr)
    e = h.engine
    print(f'wrote {args.output}; engine calls {e.calls}, fallbacks {e.fallback_calls}')


if __name__ == '__main__':
    main()
