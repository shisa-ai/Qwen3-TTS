"""Diagnose complete-request reproducibility on physical GPU3 only.

Run with OMP_NUM_THREADS=8 and MKL_NUM_THREADS=8. Each request reseeds to zero.
--assert-repeat checks exact predictor tokens and decoded audio, not duration
alone. --no-cudnn-sdpa and --math-sdpa are explicit diagnostic backend controls;
they do not alter the installed predictor or production benchmark settings.
"""
import argparse
import os
import json
import numpy as np
assert os.environ.get('CUDA_VISIBLE_DEVICES') == 'GPU-88395248-f4ce-ee5f-0cc5-1bdd951de1e8'
assert os.environ.get('OMP_NUM_THREADS') == os.environ.get('MKL_NUM_THREADS') == '8'
import torch
from qwen_tts import Qwen3TTSModel
from bench_common import TEXTS
from qwen_predictor_graph import install_predictor_graph

@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--size', default='0.6B')
    p.add_argument('--mode', choices=['baseline', 'predictor', 'talker', 'combined'], default='predictor')
    p.add_argument('--math-sdpa', action='store_true')
    p.add_argument('--no-cudnn-sdpa', action='store_true')
    p.add_argument('--text', choices=list(TEXTS), default='medium')
    p.add_argument('--runs', type=int, default=3)
    p.add_argument('--assert-repeat', action='store_true')
    args = p.parse_args()
    if args.no_cudnn_sdpa:
        torch.backends.cuda.enable_cudnn_sdp(False)
    if args.math_sdpa:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    model = Qwen3TTSModel.from_pretrained(f'Qwen/Qwen3-TTS-12Hz-{args.size}-CustomVoice',
        local_files_only=True, device_map='cuda:0', dtype=torch.bfloat16, attn_implementation='sdpa')
    predictor = model.model.talker.code_predictor
    if args.mode in ('predictor', 'combined'):
        install_predictor_graph(predictor)
    if args.mode in ('talker', 'combined'):
        from qwen_talker_graph import install_talker_graph
        install_talker_graph(model.model.talker)
    original = predictor.generate
    records = []
    def record(*a, **kw):
        before = torch.cuda.get_rng_state().clone()
        x = kw['inputs_embeds'].clone()
        result = original(*a, **kw)
        records.append((x, result.sequences.clone(), before, torch.cuda.get_rng_state().clone()))
        return result
    predictor.generate = record
    prior = None
    for run in range(args.runs):
        records.clear()
        torch.manual_seed(0)
        wavs, sr = model.generate_custom_voice(text=TEXTS[args.text], speaker='ryan', language='English')
        torch.cuda.synchronize()
        current = [(x.cpu(), y.cpu(), b, a) for x, y, b, a in records]
        row = dict(size=args.size, mode=args.mode, text=args.text, math_sdpa=args.math_sdpa,
                   no_cudnn_sdpa=args.no_cudnn_sdpa, run=run, frames=len(current), audio_s=len(wavs[0])/sr)
        if prior is not None:
            names = ['input', 'tokens', 'rng_before', 'rng_after']
            for idx, name in enumerate(names):
                row['first_different_' + name] = next((i for i, (a,b) in enumerate(zip(prior,current)) if not torch.equal(a[idx],b[idx])), None)
            row['audio_equal'] = np.array_equal(previous_audio, wavs[0])
            first = row['first_different_input']
            if first is not None:
                row['first_input_max_abs_diff'] = (prior[first][0].float() - current[first][0].float()).abs().max().item()
            row['different_tokens_with_identical_input_and_rng'] = sum(
                torch.equal(a[0],b[0]) and torch.equal(a[2],b[2]) and not torch.equal(a[1],b[1])
                for a,b in zip(prior,current))
        print(json.dumps(row), flush=True)
        if prior is not None and args.assert_repeat:
            assert len(prior) == len(current)
            assert all(torch.equal(a[1], b[1]) for a,b in zip(prior,current)), row
            assert row['audio_equal'], row
        prior, previous_audio = current, wavs[0]

if __name__ == '__main__':
    main()
