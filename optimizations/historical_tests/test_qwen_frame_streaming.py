"""GPU1 direct-frame interface correctness and hook-stream comparison."""
import os
import argparse
import json
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--size',required=True);args=p.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES']=='GPU-74eb9316-76d6-7587-c3de-2153fc1ea3dc'
    import torch
    from qwen_tts import Qwen3TTSModel
    from qwen_fused_norm import install_fused_predictor_norm
    from qwen_fused_predictor import install_fused_predictor
    from qwen_talker_fusion import install_talker_fusion
    from qwen_talker_graph import install_talker_graph
    from qwen_fast_decode import install_fast_decode
    from qwen_frame_loop import install_frame_loop
    from qwen_frame_streaming import stream_frames,stream_custom_voice
    from qwen_streaming import stream_custom_voice as hook_stream, StreamingCancelled
    torch.set_num_threads(8);torch.backends.cuda.enable_cudnn_sdp(False)
    model=Qwen3TTSModel.from_pretrained(f'Qwen/Qwen3-TTS-12Hz-{args.size}-CustomVoice',device_map='cuda:0',dtype=torch.bfloat16,attn_implementation='sdpa')
    t=model.model.talker
    install_fused_predictor_norm(t.code_predictor)
    install_fused_predictor(t.code_predictor,mode='nonpdl' if args.size=='0.6B' else 'pdl')
    install_talker_fusion(t);install_talker_graph(t);install_fast_decode(t)
    loop=install_frame_loop(t)
    inputs={};original=t.generate
    def capture(**kw):
        inputs.update(kw);return original(**kw)
    t.generate=capture
    text='Hello there. How can I help you today?'
    with torch.inference_mode():
        model.generate_custom_voice(text=text,speaker='ryan',language='English',do_sample=False,subtalker_dosample=False)
    t.generate=original
    out=Path(f'results/gap/frame-streaming/{args.size}-v3.json');out.parent.mkdir(parents=True,exist_ok=True)
    result={'size':args.size,'checks':[],'timings':[],'complete':False}
    def check(name,ok=True):
        assert ok,name
        result['checks'].append(name)
    with torch.inference_mode():
        for label,options in [('greedy',{}),('sampled',dict(do_sample=True,subtalker_dosample=True)),('top-p',dict(do_sample=True,top_p=.85,temperature=.7,subtalker_temperature=1.1)),('penalty',dict(repetition_penalty=1.2))]:
            kw=inputs|dict(max_new_tokens=24,min_new_tokens=24)|options
            torch.manual_seed(123);reference=loop.generate(**kw)
            torch.manual_seed(123);frames=[];value=stream_frames(loop,kw,frames.append)
            expected=[h[1] for h in reference.hidden_states if h[1] is not None]
            check(label+' frame count',len(frames)==len(expected))
            check(label+' exact frames',all(torch.equal(a,b) for a,b in zip(frames,expected)))
            check(label+' no history return',value is None)
            snapshots=[f.clone() for f in frames]
            stream_frames(loop,kw,lambda f:None)
            check(label+' owned outputs',all(torch.equal(a,b) for a,b in zip(frames,snapshots)))
        kw=inputs|dict(max_new_tokens=24,min_new_tokens=24)
        delivered=[]
        try:
            stream_frames(loop,kw,delivered.append,cancelled=lambda:True)
        except StreamingCancelled:
            check('cancel before first is empty and restores',not delivered and not loop.active)
        empty=[]
        stream_frames(loop,kw|dict(max_new_tokens=1,min_new_tokens=1),empty.append)
        check('one-token generation has empty complete-frame output',not empty and not loop.active)
        try:
            stream_frames(loop,kw,delivered.append,cancelled=lambda:len(delivered)==3)
        except StreamingCancelled:
            check('cancel cleans active state',not loop.active and len(delivered)==3)
        else:raise AssertionError('cancel')
        def fail(frame):raise RuntimeError('consumer failure')
        try:stream_frames(loop,kw,fail)
        except RuntimeError as error:check('callback error cleanup',str(error)=='consumer failure' and not loop.active)
        stream_frames(loop,kw,lambda f:None);check('recovery')
        delayed=[]
        def slow(frame):
            time.sleep(.005);delayed.append(frame)
        start=time.perf_counter();stream_frames(loop,kw,slow)
        check('synchronous backpressure',time.perf_counter()-start>=len(delayed)*.005)
        # Recursion must reject before supported dispatch, including final tail.
        for target in ['frame','audio','tail']:
            for unsupported in [False,True]:
                calls=[]
                def recurse(value):
                    calls.append(1)
                    mkw=dict(text=text,speaker='ryan',language='English',do_sample=False,subtalker_dosample=False)
                    if unsupported:mkw['subtalker_top_p']=.85
                    model.generate_custom_voice(**mkw)
                opts=dict(on_audio=recurse if target!='frame' else lambda c:None,
                    first_chunk_frames=100 if target=='tail' else 4,max_new_tokens=8)
                if target=='frame':opts['on_frame']=recurse
                try:
                    stream_custom_voice(model,text=text,speaker='ryan',do_sample=False,subtalker_dosample=False,**opts)
                except RuntimeError as error:
                    check(f'{target} recursion unsupported={unsupported} rejects without fallback', 'non-reentrant' in str(error) and len(calls)==1)
                    check(f'{target} recursion unsupported={unsupported} cache inactive and restored',not loop.tg.active and not loop.active and t.generate==original and not hasattr(t,'_streaming_active'))
                else:raise AssertionError('recursion accepted')
        owner=torch.cuda.current_stream()
        switched=torch.cuda.Stream()
        def switch_stream(chunk):torch.cuda.set_stream(switched)
        try:
            stream_custom_voice(model,text=text,speaker='ryan',on_audio=switch_stream,max_new_tokens=8,do_sample=False,subtalker_dosample=False)
        except ValueError:
            torch.cuda.set_stream(owner)
            check('stream-switch callback rejected and owner-cache reset',not loop.tg.active and t.generate==original)
        finally:torch.cuda.set_stream(owner)
        decoder=model.model.speech_tokenizer.model.decoder
        old_rope=decoder.pre_transformer.rotary_emb.rope_type
        decoder.pre_transformer.rotary_emb.rope_type='dynamic'
        try:
            stream_custom_voice(model,text=text,speaker='ryan',on_audio=lambda c:None)
        except ValueError:
            check('incompatible codec rotary rejected before entry',not hasattr(t,'_streaming_active'))
        else:raise AssertionError('rotary accepted')
        finally:decoder.pre_transformer.rotary_emb.rope_type=old_rope
        # Unsupported predictor top-p takes the existing strict hook path.
        fallback_audio=[]
        fallback_result=stream_custom_voice(model,text=text,speaker='ryan',on_audio=lambda c:fallback_audio.append(c.samples),
            subtalker_top_p=.85,do_sample=False,subtalker_dosample=False,max_new_tokens=24)
        check('unsupported sampler strict streaming fallback',fallback_result.frames==23 and len(fallback_audio)>0)
        check('fallback wrapper cleanup',not hasattr(t,'_streaming_active') and t.generate==original)
        for label,fn in [('hook',hook_stream),('direct',stream_custom_voice)]:
            for repeat in range(3):
                torch.manual_seed(123);events=[];chunks=[]
                def audio(chunk):events.append(time.perf_counter());chunks.append(chunk.samples)
                torch.cuda.synchronize();start=time.perf_counter()
                r=fn(model,text=text,speaker='ryan',language='English',on_audio=audio,
                     chunk_frames=25,first_chunk_frames=4,context_frames=588,
                     do_sample=False,subtalker_dosample=False)
                torch.cuda.synchronize();end=time.perf_counter()
                duration=r.samples/r.sample_rate
                playback=start;starvation=0.
                for arrived,chunk in zip(events,chunks):
                    if playback>start:starvation+=max(0.,arrived-playback)
                    playback=max(playback,arrived)+chunk.numel()/r.sample_rate
                result['timings'].append({'variant':label,'repeat':repeat,'warmup':repeat==0,
                    'first_audio_s':events[0]-start,'total_s':end-start,'rtf':(end-start)/duration,
                    'starvation_s':starvation,'frames':r.frames,'samples':r.samples})
                check(label+' finite audio',all(torch.isfinite(c).all().item() for c in chunks))
                if repeat==2:
                    import soundfile as sf
                    path=Path(f'samples/gap/frame-streaming/{args.size}-{label}-v3.wav')
                    path.parent.mkdir(parents=True,exist_ok=True)
                    sf.write(path,torch.cat(chunks).numpy(),r.sample_rate)
                    if label=='hook':
                        hook_wave=torch.cat(chunks).clone()
                    else:
                        check('hook/direct waveform exact',torch.equal(hook_wave,torch.cat(chunks)))
    result['complete']=True;out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
