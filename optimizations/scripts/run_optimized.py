"""Run the frozen, experimental CustomVoice optimization stack from this checkout."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--size', choices=['0.6B', '1.7B'], default='0.6B')
    p.add_argument('--text', required=True)
    p.add_argument('--speaker', default='ryan')
    p.add_argument('--language', default='English')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--streaming', action='store_true', help='Write chunks incrementally to a WAV file')
    p.add_argument('--temperature', type=float, default=0.9)
    p.add_argument('--subtalker-temperature', type=float, default=0.9)
    p.add_argument('--top-p', type=float, default=1.0)
    p.add_argument('--subtalker-top-p', type=float, default=1.0)
    p.add_argument('--greedy', action='store_true')
    p.add_argument('--max-new-tokens', type=int, default=8192)
    p.add_argument('--cache-length', type=int, default=1024)
    p.add_argument('--disable-cudnn-sdpa', action='store_true')
    args = p.parse_args()
    if args.output.exists():
        p.error('output already exists; choose a new path')
    import torch
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel
    from qwen_fused_norm import install_fused_predictor_norm
    from qwen_fused_predictor import install_fused_predictor
    from qwen_talker_fusion import install_talker_fusion
    from qwen_talker_graph import install_talker_graph
    from qwen_fast_decode import install_fast_decode
    from qwen_frame_loop import install_frame_loop
    if not torch.cuda.is_available():
        p.error('CUDA is required; select a GPU with CUDA_VISIBLE_DEVICES')
    torch.cuda.set_device(0)
    torch.set_num_threads(8)
    if args.disable_cudnn_sdpa:
        torch.backends.cuda.enable_cudnn_sdp(False)
    handles = []
    with torch.inference_mode():
        model = Qwen3TTSModel.from_pretrained(
            f'Qwen/Qwen3-TTS-12Hz-{args.size}-CustomVoice',
            device_map='cuda:0', dtype=torch.bfloat16, attn_implementation='sdpa')
        talker = model.model.talker
        try:
            handles.append(install_fused_predictor_norm(talker.code_predictor))
            handles.append(install_fused_predictor(talker.code_predictor,
                mode='nonpdl' if args.size == '0.6B' else 'pdl'))
            handles.append(install_talker_fusion(talker))
            handles.append(install_talker_graph(talker, max_cache_len=args.cache_length))
            handles.append(install_fast_decode(talker))
            handles.append(install_frame_loop(talker))
            kw = dict(text=args.text, speaker=args.speaker, language=args.language,
                      temperature=args.temperature, subtalker_temperature=args.subtalker_temperature,
                      top_p=args.top_p, subtalker_top_p=args.subtalker_top_p,
                      do_sample=not args.greedy, subtalker_dosample=not args.greedy,
                      max_new_tokens=args.max_new_tokens)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if args.streaming:
                from qwen_frame_streaming import stream_custom_voice
                sr = model.model.speech_tokenizer.model.get_output_sample_rate()
                with sf.SoundFile(args.output, mode='x', samplerate=sr, channels=1, subtype='PCM_16') as audio:
                    def receive(chunk):
                        audio.write(chunk.samples.detach().float().cpu().numpy().reshape(-1))
                        audio.flush()
                    result = stream_custom_voice(model, on_audio=receive, **kw)
                    print(result)
            else:
                wavs, sr = model.generate_custom_voice(**kw)
                sf.write(args.output, wavs[0], sr)
            print(f'Wrote {args.output}')
        finally:
            for handle in reversed(handles):
                handle.uninstall()


if __name__ == '__main__':
    main()
