"""GPU1 deterministic hook/direct and eager/prepared-codec integration audit."""
import argparse
import os
import json
import time
from pathlib import Path
from contextlib import nullcontext


def main():
    p=argparse.ArgumentParser();p.add_argument('--size',required=True)
    p.add_argument('--gpu',required=True);p.add_argument('--default-cudnn',action='store_true');a=p.parse_args()
    assert a.gpu=='GPU-3e620c1c-c243-7353-580e-cde9e5eb4b8a'
    assert os.environ['CUDA_VISIBLE_DEVICES']==a.gpu
    import hashlib
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in map(Path,[
        'scripts/qwen_frame_loop.py','scripts/qwen_frame_streaming.py','scripts/qwen_codec_graph.py',
        'scripts/qwen_talker_graph.py','scripts/qwen_fused_predictor.py','scripts/qwen_streaming.py'])}
    import torch
    from qwen_tts import Qwen3TTSModel
    from qwen_fused_norm import install_fused_predictor_norm
    from qwen_fused_predictor import install_fused_predictor
    from qwen_talker_fusion import install_talker_fusion
    from qwen_talker_graph import install_talker_graph
    from qwen_fast_decode import install_fast_decode
    from qwen_frame_loop import install_frame_loop
    from qwen_streaming import stream_custom_voice as hook
    from qwen_frame_streaming import stream_custom_voice as direct
    from qwen_codec_graph import CodecGraphs
    from bench_common import TEXTS
    torch.set_num_threads(8);torch.backends.cuda.enable_cudnn_sdp(a.default_cudnn)
    m=Qwen3TTSModel.from_pretrained(f'Qwen/Qwen3-TTS-12Hz-{a.size}-CustomVoice',device_map='cuda:0',dtype=torch.bfloat16,attn_implementation='sdpa')
    t=m.model.talker
    install_fused_predictor_norm(t.code_predictor);install_fused_predictor(t.code_predictor,mode='nonpdl' if a.size=='0.6B' else 'pdl')
    install_talker_fusion(t);tg=install_talker_graph(t);install_fast_decode(t);loop=install_frame_loop(t)
    decoder=m.model.speech_tokenizer.model.decoder
    result={'size':a.size,'rows':[],'checks':[],'complete':False,'context':588,'schedule':[4,25],
        'gpu':a.gpu,'cudnn_sdpa':a.default_cudnn,'source_start':sources,'seed':123}
    tag='final-default-cudnn' if a.default_cudnn else 'final-deterministic'
    out=Path(f'results/gap/frame-streaming/{a.size}-{tag}.json')
    if out.exists():raise FileExistsError(out)
    def save():out.write_text(json.dumps(result,indent=2))
    with torch.inference_mode():
        cg=CodecGraphs(decoder,max_graphs=32)
        for tid in ['short','medium']:
            kwargs=dict(text=TEXTS[tid],speaker='ryan',language='English',do_sample=True,subtalker_dosample=True)
            # Untimed offline reference explicitly verifies EOS and complete frames.
            original=t.generate;observed={}
            def capture(**kw):
                v=original(**kw);observed['value']=v;return v
            t.generate=capture
            torch.manual_seed(123);m.generate_custom_voice(**kwargs);t.generate=original
            raw=observed.pop('value');assert raw.sequences[0,-1].item()==t.config.codec_eos_token_id
            frames=[h[1].clone() for h in raw.hidden_states if h[1] is not None]
            result['checks'].append(tid+' offline EOS/no truncation')
            # Prepare every exact window shape outside requests using reference codes.
            count=len(frames)
            ends=list(range(4,count+1,25))
            if not ends or ends[-1]!=count:ends.append(count)
            for end in ends:
                start=max(0,end-(588+(4 if end==4 else 25)))
                cg.prepare(torch.stack(frames[start:end],dim=-1).contiguous())
            reference_wave=None
            for variant,fn in ([('direct',direct)] if a.default_cudnn else [('hook',hook),('direct',direct)]):
                for backend in (['graph'] if a.default_cudnn else ['eager','graph']):
                    for repeat in range(4):
                        events=[];chunks=[];codes=[]
                        def audio(c):events.append(time.perf_counter());chunks.append(c.samples)
                        extras={'on_frame':lambda f:codes.append(f.clone())} if variant=='direct' else {}
                        before=(cg.captures,cg.misses,cg.hits,len(tg.capture_seconds))
                        with cg.installed() if backend=='graph' else nullcontext():
                            torch.manual_seed(123);torch.cuda.synchronize();start=time.perf_counter()
                            r=fn(m,**kwargs,on_audio=audio,chunk_frames=25,first_chunk_frames=4,context_frames=588,**extras)
                            torch.cuda.synchronize();elapsed=time.perf_counter()-start
                        wav=torch.cat(chunks)
                        if reference_wave is None:reference_wave=wav.clone()
                        if not a.default_cudnn:
                            assert torch.equal(wav,reference_wave)
                            assert r.frames==count
                            if variant=='direct':assert len(codes)==len(frames) and all(torch.equal(x,y) for x,y in zip(codes,frames))
                        assert torch.isfinite(wav).all() and wav.numel()>0
                        assert before[0]==cg.captures and before[3]==len(tg.capture_seconds)
                        playback=None;starvation=0.
                        for at,chunk in zip(events,chunks):
                            if playback is not None:starvation+=max(0.,at-playback)
                            playback=max(at,playback or at)+chunk.numel()/r.sample_rate
                        if repeat:
                            result['rows'].append(dict(text=tid,variant=variant,codec=backend,repeat=repeat,
                                first_audio_s=events[0]-start,total_s=elapsed,rtf=elapsed/(r.samples/r.sample_rate),
                                starvation_s=starvation,frames=r.frames,samples=r.samples,
                                eos=True if not a.default_cudnn else None,truncated=False if not a.default_cudnn else None,
                                eos_evidence='matched offline EOS frames' if not a.default_cudnn else 'not exposed by streaming API',
                                frame_sha256=hashlib.sha256(torch.stack(codes).cpu().numpy().tobytes()).hexdigest() if codes else None,
                                waveform_sha256=hashlib.sha256(wav.numpy().tobytes()).hexdigest(),
                                misses=cg.misses-before[1],hits=cg.hits-before[2],captures=cg.captures-before[0]))
                            save()
            result['checks'].append(tid+(' finite audio/capture-free smoke; trajectories not forced equal' if a.default_cudnn else ' exact waveform all variants and direct frames'))
        result['source_end']={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}
        assert result['source_end']==sources
        result['complete']=True;save()


if __name__=='__main__':main()
