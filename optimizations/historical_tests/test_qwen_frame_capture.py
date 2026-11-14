"""GPU2 isolated unified frame benchmark and forced-history cache check."""
import os
import argparse
import json
import time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--size',required=True)
    args=parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES']=='GPU-3e620c1c-c243-7353-580e-cde9e5eb4b8a'
    import torch
    from qwen_tts import Qwen3TTSModel
    from qwen_fused_norm import install_fused_predictor_norm
    from qwen_fused_predictor import install_fused_predictor
    from qwen_talker_fusion import install_talker_fusion
    from qwen_talker_graph import install_talker_graph
    from qwen_fast_decode import install_fast_decode
    from qwen_frame_loop import install_frame_loop
    from qwen_frame_capture import CapturedFrame, enable_capture_experiment
    torch.set_num_threads(8)
    torch.backends.cuda.enable_cudnn_sdp(False)
    model=Qwen3TTSModel.from_pretrained(f'Qwen/Qwen3-TTS-12Hz-{args.size}-CustomVoice',device_map='cuda:0',dtype=torch.bfloat16,attn_implementation='sdpa')
    t=model.model.talker
    install_fused_predictor_norm(t.code_predictor)
    install_fused_predictor(t.code_predictor,mode='nonpdl' if args.size=='0.6B' else 'pdl')
    install_talker_fusion(t)
    g=install_talker_graph(t)
    install_fast_decode(t)
    original=t.generate
    inputs={}
    def save_inputs(**kw):
        inputs.update(kw)
        return original(**kw)
    t.generate=save_inputs
    with torch.inference_mode():
        model.generate_custom_voice(text='Hello there. How can I help you today?',speaker='ryan',language='English',do_sample=False,subtalker_dosample=False)
    del t.generate
    loop=install_frame_loop(t)
    enable_capture_experiment(loop)
    loop.capture_settings=(False,1.,0)
    result={'size':args.size,'no_cudnn':True,'checks':[],'complete':False}
    out=Path(f'results/gap/frame-decode/{args.size}-capture-diagnostic-v3.json')
    @torch.inference_mode()
    def prefill():
        g.reset()
        x=inputs['inputs_embeds']
        return t(inputs_embeds=x,attention_mask=inputs['attention_mask'],cache_position=torch.arange(x.shape[1],device=x.device),use_cache=True,output_hidden_states=True,return_dict=True,trailing_text_hidden=inputs['trailing_text_hidden'],tts_pad_embed=inputs['tts_pad_embed'])
    try:
        with torch.inference_mode():
            value=prefill()
            position=inputs['inputs_embeds'].shape[1]
            token=torch.tensor([[17]],device='cuda')
            frame=CapturedFrame(loop,token,value.past_hidden,inputs['tts_pad_embed'],position)
            result['checks'].append('outer capture succeeds')
            # Compare eager raw operations and outer replay using arbitrary forced
            # tokens and distinct positions, with all layers of cache restored.
            for index, code in enumerate([17,1003,0,2047]):
                token.fill_(code)
                baseline=prefill()
                frame.token.copy_(token);frame.hidden.copy_(baseline.past_hidden)
                frame.position.fill_(position+index)
                frame.run()
                expected=frame.logits.clone()
                keys=[(layer.keys.clone(),layer.values.clone()) for layer in g.cache.layers]
                baseline=prefill()
                actual=frame.replay(token,baseline.past_hidden,inputs['tts_pad_embed'],position+index)[0]
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                for layer,(k,v) in zip(g.cache.layers,keys):
                    torch.testing.assert_close(layer.keys,k,rtol=0,atol=0)
                    torch.testing.assert_close(layer.values,v,rtol=0,atol=0)
            result['checks'].append('arbitrary tokens distinct scalar positions all-layer KV/logits exact')
            result['timings']={}
            for count in [24,72]:
                rows={}
                for label in ['eager_raw','outer_graph']:
                    times=[]
                    for repeat in range(3):
                        value=prefill();torch.cuda.synchronize();start=time.perf_counter()
                        for i in range(count):
                            if label=='outer_graph':
                                frame.replay(token,value.past_hidden,inputs['tts_pad_embed'],position+i)
                            else:
                                frame.position.fill_(position+i);frame.run()
                        torch.cuda.synchronize();times.append(time.perf_counter()-start)
                    rows[label]=times
                result['timings'][str(count)]=rows
            result['loop_comparison'] = {}
            settings = dict(inputs)
            settings.update(do_sample=False, subtalker_dosample=False)
            for count in [24,72]:
                settings.update(max_new_tokens=count,min_new_tokens=count)
                rows={}
                reference=None
                for label in ['full_stack','explicit_loop','unified_compute']:
                    loop.frame_factory = CapturedFrame if label == 'unified_compute' else None
                    snapshots=[]
                    def observe(step, logits, cache, codes, hidden):
                        import hashlib
                        def digest(value):
                            return hashlib.sha256(value.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
                        top=logits.topk(2,dim=-1).values
                        snapshots.append({'logits':digest(logits),'hidden':digest(hidden),
                            'cache':[(digest(l.keys[:,:,:cache.length]),digest(l.values[:,:,:cache.length])) for l in cache.layers],
                            'margin':float((top[0,0]-top[0,1]).item())})
                    if not hasattr(loop,'frames'):
                        loop.frames={}
                    generate = original if label == 'full_stack' else loop.generate
                    warm = generate(**settings)
                    if reference is None:
                        reference=warm
                    else:
                        result.setdefault('comparisons', []).append({'count': count, 'variant': label,
                            'sequence_equal': torch.equal(reference.sequences,warm.sequences),
                            'first_token_difference': next((i for i,(a,b) in enumerate(zip(reference.sequences[0].tolist(),warm.sequences[0].tolist())) if a!=b),None)})
                    # Separate untimed instrumented execution; compare every layer.
                    snapshots.clear()
                    if label == 'full_stack':
                        def hook(module, call_inputs, output):
                            if output.logits.shape[1] == 1:
                                observe(len(snapshots), output.logits[:, -1].float(), g.cache, output.hidden_states[1], output.past_hidden)
                        observer_handle=t.register_forward_hook(hook)
                    else:
                        loop.step_observer=observe
                    torch.manual_seed(123)
                    generate(**settings)
                    if label == 'full_stack':
                        observer_handle.remove()
                    else:
                        loop.step_observer=None
                    result.setdefault('step_digests',{}).setdefault(str(count),{})[label]=list(snapshots)
                    times=[]
                    repeats=[]
                    for repeat in range(3):
                        torch.manual_seed(123)
                        torch.cuda.synchronize();start=time.perf_counter()
                        repeated=generate(**settings)
                        torch.cuda.synchronize();times.append(time.perf_counter()-start)
                        hidden_diff=None
                        for step,(a,b) in enumerate(zip(warm.hidden_states,repeated.hidden_states)):
                            if any(not torch.equal(av,bv) for av,bv in zip(a[0],b[0])):
                                hidden_diff=step;break
                        repeats.append({'sequence':repeated.sequences[0].tolist(),
                            'first_hidden_difference_vs_warm':hidden_diff,
                            'rng':__import__('hashlib').sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest()})
                    result.setdefault('repeats',{}).setdefault(str(count),{})[label]=repeats
                    rows[label]=times
                result['loop_comparison'][str(count)]=rows
            result['complete']=True
    except Exception as error:
        result['error']=repr(error)
        raise
    finally:
        out.write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
