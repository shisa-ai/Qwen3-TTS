"""GPU2 real-input validation and matched latency measurements."""
import os
GPU = 'GPU-3e620c1c-c243-7353-580e-cde9e5eb4b8a'
assert os.environ.get('CUDA_VISIBLE_DEVICES') == GPU
assert os.environ.get('OMP_NUM_THREADS') == os.environ.get('MKL_NUM_THREADS') == '8'
import argparse
import json
from pathlib import Path
import statistics
import time
import torch
from qwen_tts import Qwen3TTSModel
from qwen_fused_predictor import install_fused_predictor
from qwen_predictor_graph import install_predictor_graph
from qwen_fused_norm import install_fused_predictor_norm
from qwen_fast_decode import install_fast_decode
from qwen_talker_graph import install_talker_graph
from bench_common import TEXTS


def timed(fn, n=20):
    vals=[]
    for _ in range(n):
        torch.cuda.synchronize(); t=time.perf_counter(); fn(); torch.cuda.synchronize()
        vals.append((time.perf_counter()-t)*1000)
    return statistics.median(vals)


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--size',required=True); ap.add_argument('--deep', action='store_true'); ap.add_argument('--diagnose',action='store_true'); ap.add_argument('--mode',choices=['pdl','nonpdl'],default='pdl'); args=ap.parse_args()
    out=Path('results/gap/fused-predictor') / (args.size+('' if args.mode=='pdl' else '-'+args.mode)+'.json')
    r={'size':args.size,'gpu':GPU,'complete':False,'torch':torch.__version__}
    def save(): out.write_text(json.dumps(r,indent=2))
    if not args.deep:
        save()
    torch.manual_seed(42)
    m=Qwen3TTSModel.from_pretrained(f'Qwen/Qwen3-TTS-12Hz-{args.size}-CustomVoice',local_files_only=True,device_map='cuda:0',dtype=torch.bfloat16,attn_implementation='sdpa')
    p=m.model.talker.code_predictor
    r['config']=p.config.to_dict(); inputs=[]; original=p.generate
    def collect(**kw):
        if len(inputs)<8: inputs.append(kw['inputs_embeds'].clone())
        return original(**kw)
    p.generate=collect
    m.generate_custom_voice(text=TEXTS['short'],speaker='ryan',language='English',do_sample=False,subtalker_dosample=False,max_new_tokens=24)
    p.generate=original
    if args.deep:
        deep_validate(p, inputs, args.size, m.model.talker, args.diagnose, args.mode)
        return
    norm=install_fused_predictor_norm(p)
    kw=dict(max_new_tokens=15,do_sample=False,output_hidden_states=False,return_dict_in_generate=True)
    pg=install_predictor_graph(p)
    refs=[p.generate(inputs_embeds=x,**kw).sequences for x in inputs]
    r['strict_ms']=timed(lambda:p.generate(inputs_embeds=inputs[0],**kw))
    pg.uninstall()
    fp=install_fused_predictor(p,mode=args.mode)
    outs=[p.generate(inputs_embeds=x,**kw).sequences for x in inputs]
    r['greedy_agreement']=torch.cat([a==b for a,b in zip(outs,refs)]).float().mean().item()
    r['fused_ms']=timed(lambda:p.generate(inputs_embeds=inputs[0],**kw))
    r['speedup']=r['strict_ms']/r['fused_ms']; r['pdl']=bool(fp.engine._pdl_configs)
    saved=outs[0].clone(); fp.engine.cache.fill_(123)
    assert torch.equal(p.generate(inputs_embeds=inputs[0],**kw).sequences,saved)
    for x in inputs[::-1]: p.generate(inputs_embeds=x,**kw)
    assert torch.equal(outs[0],saved)
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        try: p.generate(inputs_embeds=inputs[0],**kw)
        except ValueError: pass
        else: raise AssertionError('stream accepted')
    r['ownership_cache_stream']=True
    for t,k,prob in [(0.7,50,1.),(1.2,0,1.),(0.9,50,0.8)]:
        z=p.generate(inputs_embeds=inputs[0],**(kw|dict(do_sample=True,temperature=t,top_k=k,top_p=prob))).sequences
        assert z.shape==(1,15) and ((z>=0)&(z<p.config.vocab_size)).all()
    assert fp.fallback_calls==1
    r['sampling_fallback']=True
    r['captures']=[dict(setup_s=g.setup_seconds,capture_s=g.capture_seconds) for g in fp.graphs.values()]
    save()
    # Teacher-forced layer test: compare the fused layer to HF on identical rows,
    # then compare each against an FP32 copy. This avoids autoregressive drift.
    import copy
    e=fp.engine
    # The PDL execution drops cuBLAS weights; construct a non-PDL copy for layer diagnostics.
    from qwen_fused_predictor import _k, _mapped
    support=_k._pdl_supported; _k._pdl_supported=lambda dev:False
    try: diag=_k.FusedCodePredictor(_mapped(p),1)
    finally: _k._pdl_supported=support
    errs=[]
    for inp in inputs[:4]:
        x=p.small_to_mtp_projection(inp).reshape(2,-1)
        pos=torch.arange(2,device=x.device).unsqueeze(0)
        cos,sin=p.model.rotary_emb(x.unsqueeze(0),pos)
        mask=torch.zeros(1,1,2,2,device=x.device,dtype=x.dtype).masked_fill(torch.tensor([[False,True],[False,False]],device=x.device),torch.finfo(x.dtype).min)
        for i,l in enumerate(p.model.layers):
            hf=l(x.unsqueeze(0),attention_mask=mask,position_ids=pos,position_embeddings=(cos,sin),use_cache=False)[0].reshape_as(x)
            got=x.clone(); diag._layer(got,i,1,2,0)
            pdlgot=None
            if e._pdl_configs:
                c=e._pdl_config(1); L=e.layers[i]; W=e.pdl_w[i]
                px=e.pdl_x; px[:2].copy_(x)
                ss, nxt=e.pdl_ss
                torch.sum(px[:2].float().square(),dim=1,out=ss[0,:2])
                e._gemm(c['qkv'],px,1,0,W['qkv'],e.pdl_qkv,ss,1,None,2,0,L['eps1'])
                _k._cp_attention_kernel[(1,e.heads,2)](e.pdl_qkv,e.cache[i,0],e.cache[i,1],L['qn'],L['kn'],e.cos,e.sin,e.pdl_att,0,L['head_eps'],e.scale,NQ=2,H=e.heads,KVH=e.kv_heads,HD=e.head_dim,MAXP=e.max_pos,BK=e._attn_keys,PDL=True,num_warps=1,launch_pdl=True)
                nt=e._gemm(c['o'],e.pdl_att,1,0,W['o'],px,None,0,nxt,2,2)
                e._gemm(c['gu'],px,1,0,W['gu'],e.pdl_act,nxt,nt,None,2,1,L['eps2'])
                e._gemm(c['down'],e.pdl_act,1,0,W['down'],px,None,0,ss,2,2)
                pdlgot=px[:2].clone()
            lf=copy.deepcopy(l).float()
            hf32=lf(x.float().unsqueeze(0),attention_mask=mask.float(),position_ids=pos,position_embeddings=(cos.float(),sin.float()),use_cache=False)[0].reshape_as(x)
            err=lambda a,b:float((a.float()-b.float()).square().mean().sqrt())
            errs.append({'layer':i,'fused_hf_rmse':err(got,hf),'fused_fp32_rmse':err(got,hf32),'hf_fp32_rmse':err(hf,hf32),'reference_rms':float(hf32.square().mean().sqrt())})
            assert torch.isfinite(got).all()
            assert err(got,hf32) / float(hf32.square().mean().sqrt()) < 0.04
            if pdlgot is not None:
                errs[-1]['pdl_fp32_rmse']=err(pdlgot,hf32)
                assert err(pdlgot,hf32) / errs[-1]['reference_rms'] < 0.04
            del lf
            x=hf
    r['layer_errors']=errs; save()
    fp.uninstall()
    tg=install_talker_graph(m.model.talker)
    r['end_to_end']={}
    for name,installer in [('strict',install_predictor_graph),('fused',install_fused_predictor)]:
        h=installer(p) if name=='strict' else installer(p,mode=args.mode)
        fast=install_fast_decode(m.model.talker)
        def request(text):
            return m.generate_custom_voice(text=TEXTS[text],speaker='ryan',language='English',temperature=0.7,subtalker_temperature=0.9,max_new_tokens=180)
        request('short')
        rows=[]
        for text in ['short','medium']:
            for run in range(3):
                torch.manual_seed(100+run); torch.cuda.synchronize(); t=time.perf_counter()
                wavs,sr=request(text); torch.cuda.synchronize(); wall=time.perf_counter()-t
                duration=len(wavs[0])/sr
                rows.append(dict(text=text,run=run,wall_s=wall,audio_s=duration,rtf=wall/duration))
        r['end_to_end'][name]=rows; fast.uninstall(); h.uninstall(); save()
    r['complete']=True; save(); print(json.dumps({k:v for k,v in r.items() if k not in ('config','layer_errors')},indent=2),flush=True)

@torch.inference_mode()
def deep_validate(p, inputs, size, talker, diagnose=False, mode="pdl"):
    import copy
    from transformers.cache_utils import DynamicCache, StaticCache
    from qwen_fused_predictor import _k
    p32=copy.deepcopy(p).float()
    handle=install_fused_predictor(p,mode=mode); e=handle.engine
    report={'size':size,'complete':False,'rows':[],'guard_failures':[]}
    path=Path('results/gap/fused-predictor')/(size+('' if mode=='pdl' else '-'+mode)+('-diagnose-progress.json' if diagnose else '-deep.json'))
    def save(): path.write_text(json.dumps(report,indent=2))
    def error(a,b):
        return float((a.float()-b.float()).square().mean().sqrt()/b.float().square().mean().sqrt().clamp_min(1e-8))
    def fused_step(inp, token, step):
        nq=2 if step==0 else 1; first=0 if step==0 else step+1
        if not e._pdl_configs:
            if step==0:
                raw=inp.reshape(2,-1)
                x=torch.addmm(e.proj_b,raw,e.proj_wt) if e.proj_wt is not None else raw.clone()
            else: x=e.tables[step-1][token].reshape(1,-1).clone()
            layers=[]; sources=[]
            for li in range(len(e.layers)):
                sources.append(x.clone()); e._layer(x,li,1,nq,first);layers.append(x.clone())
            hf=torch.empty(1,e.hidden,device=x.device,dtype=x.dtype)
            last=x[-1:]
            _k._rms_rows_kernel[(1,)](last,last.stride(0),e.norm_w,hf,e.eps,N=e.hidden)
            e._diagnostic_inputs=sources
            return torch.mm(hf,e.lm_heads[step]),layers
        x=e.pdl_x
        if step==0:
            raw=inp.reshape(2,-1)
            if e.proj_wt is None: x[:2].copy_(raw)
            else: torch.addmm(e.proj_b,raw,e.proj_wt,out=x[:2])
        else: x[:1].copy_(e.tables[step-1][token].reshape(1,-1))
        ss,nxt=e.pdl_ss
        torch.sum(x[:nq].float().square(),dim=1,out=ss[0,:nq]); nt=1
        c=e._pdl_config(1)
        fused_layers=[]; fused_inputs=[]
        for li,(L,W) in enumerate(zip(e.layers,e.pdl_w)):
            fused_inputs.append(x[:nq].clone())
            e._gemm(c['qkv'],x,1,0,W['qkv'],e.pdl_qkv,ss,nt,None,nq,0,L['eps1'])
            _k._cp_attention_kernel[(1,e.heads,nq)](e.pdl_qkv,e.cache[li,0],e.cache[li,1],L['qn'],L['kn'],e.cos,e.sin,e.pdl_att,first,L['head_eps'],e.scale,NQ=nq,H=e.heads,KVH=e.kv_heads,HD=e.head_dim,MAXP=e.max_pos,BK=e._attn_keys,PDL=True,num_warps=1,launch_pdl=True)
            nt=e._gemm(c['o'],e.pdl_att,1,0,W['o'],x,None,0,nxt,nq,2); ss,nxt=nxt,ss
            e._gemm(c['gu'],x,1,0,W['gu'],e.pdl_act,ss,nt,None,nq,1,L['eps2'])
            nt=e._gemm(c['down'],e.pdl_act,1,0,W['down'],x,None,0,nxt,nq,2); ss,nxt=nxt,ss
            fused_layers.append(x[:nq].clone())
        e._gemm(c['lm'],x,nq,nq-1,e.pdl_lm[step],e.pdl_logits,ss,nt,None,1,0,e.eps)
        e._diagnostic_inputs=fused_inputs
        return e.pdl_logits.clone(), fused_layers
    histories=[]
    # Reuse four actual frames in a nonmonotonic sequence, without resetting
    # fused cache between frames. Only future positions are poisoned.
    for frame in [0,1,3,2,0]:
        inp=inputs[frame]; caches=[DynamicCache(config=p32.config),DynamicCache(config=p.config),StaticCache(config=p.config,max_cache_len=16)]
        token=None; prev=[]; frame_logits=[]
        for step in range(15):
            nq=2 if step==0 else 1; first=0 if step==0 else step+1; end=first+nq
            pos=torch.arange(first,end,device=inp.device)
            logits=[]; layer_outputs=[]; local_inputs=[]; prior_caches=[copy.deepcopy(c) for c in caches] if diagnose and frame==3 and step==13 else None
            for model,cache in zip([p32,p,p],caches):
                raw=inp.to(next(model.parameters()).dtype) if step==0 else model.get_input_embeddings()[step-1](token.reshape(1,1))
                x=model.small_to_mtp_projection(raw)
                keys=16 if isinstance(cache,StaticCache) else end
                mask=torch.zeros(1,1,nq,keys,device=x.device,dtype=x.dtype).masked_fill(torch.arange(keys,device=x.device)[None,:]>pos[:,None],torch.finfo(x.dtype).min)
                captured=[]; captured_inputs=[]
                prehooks=[l.register_forward_pre_hook(lambda module,args: captured_inputs.append(args[0].clone())) for l in model.model.layers]
                hooks=[l.register_forward_hook(lambda module,args,out: captured.append(out[0].clone() if isinstance(out,tuple) else out.clone())) for l in model.model.layers]
                try:
                    z=model.model(inputs_embeds=x,past_key_values=cache,use_cache=True,cache_position=pos,position_ids=pos[None],attention_mask={'full_attention':mask},return_dict=True)
                finally:
                    for hook in hooks+prehooks: hook.remove()
                local_inputs.append(captured_inputs)
                layer_outputs.append(captured)
                logits.append(model.lm_head[step](z.last_hidden_state[:,-1]))
            # Poison slots outside the history, preserving already written ones.
            before=e.cache.clone()
            e.cache[:,:,:,:,first:,:].fill_(123)
            a, fused_layers=fused_step(inp,token,step)
            cache_a=e.cache.clone()
            e.cache.copy_(before); e.cache[:,:,:,:,first:,:].fill_(-321)
            b, _=fused_step(inp,token,step)
            assert torch.equal(a,b), 'Unused-slot poison affected causal result'
            assert torch.equal(cache_a[:,:,:,:,:end],e.cache[:,:,:,:,:end])
            if first:
                assert torch.equal(before[:,:,:,:,:first],e.cache[:,:,:,:,:first]), 'Historical KV overwritten'
            kv=[]
            for li in range(len(e.layers)):
                ref=caches[0].layers[li]
                bf=caches[1].layers[li]; st=caches[2].layers[li]
                for index,name in enumerate(['keys','values']):
                    base=getattr(ref,name)[:,:,:end]
                    kv.append({'layer':li,'kind':name,'fused':error(e.cache[li,index,:,:,:end],base),
                               'bf16':error(getattr(bf,name)[:,:,:end],base),
                               'static':error(getattr(st,name)[:,:,:end],base),
                               'new_fused':error(e.cache[li,index,:,:,first:end],base[:,:,first:end]),
                               'new_bf16':error(getattr(bf,name)[:,:,first:end],base[:,:,first:end])})
            top=logits[0].float().topk(2).values
            row={'frame':frame,'step':step,'fused_error':error(a,logits[0]),'bf16_error':error(logits[1],logits[0]),'static_error':error(logits[2],logits[0]),'top_margin':float(top[0,0]-top[0,1]),'fused_max_abs':float((a.float()-logits[0]).abs().max()),'kv':kv}
            row['layers']=[{'layer':li,'fused':error(fused_layers[li],layer_outputs[0][li].reshape_as(fused_layers[li])),
                            'bf16':error(layer_outputs[1][li],layer_outputs[0][li])} for li in range(len(e.layers))]
            if diagnose and frame==3 and step==13:
                diagnose_layer(p,p32,e,inp,first,local_inputs,layer_outputs,prior_caches,before,fused_layers)
                handle.uninstall()
                return
            report['rows'].append(row);save()
            for label,value in [('layer',max(v['fused'] for v in row['layers'])),('new_kv',max(v['new_fused'] for v in kv)),('logits',row['fused_error']),('historical_kv',max(v['fused'] for v in kv))]:
                if not value<0.04:
                    report['guard_failures'].append({'frame':frame,'step':step,'kind':label,'value':value,'limit':0.04})
            assert torch.equal(logits[1],logits[2]), 'BF16 dynamic/static mismatch'
            token=logits[0].argmax(-1);frame_logits.append(a)
        if frame==0:
            if histories: assert all(torch.equal(a,b) for a,b in zip(histories[0],frame_logits)), 'Sequential-frame contamination'
            histories.append(frame_logits)
    # Alter only prefill row 1: row 0 cannot attend to this future row.
    e.cache.fill_(456)
    _, base_layers=fused_step(inputs[0],None,0)
    base_cache=e.cache.clone()
    perturbed=inputs[0].clone(); perturbed[:,1].mul_(3).add_(1)
    e.cache.fill_(-789)
    _, perturbed_layers=fused_step(perturbed,None,0)
    assert all(torch.equal(a[0],b[0]) for a,b in zip(base_layers,perturbed_layers)), 'Prefill future row leaked'
    assert torch.equal(base_cache[:,:,:,:,:1],e.cache[:,:,:,:,:1])
    report['prefill_causal_perturbation']=True
    kw=dict(inputs_embeds=inputs[0],max_new_tokens=15,do_sample=True,temperature=0.9,top_k=50,top_p=0.8,output_hidden_states=False,return_dict_in_generate=True)
    # Warm the fallback before reseeding, then compare exact outputs.
    handle.generate(**kw)
    torch.manual_seed(912); a=handle.generate(**kw).sequences
    torch.manual_seed(912); b=handle.strict.generate(**kw).sequences
    assert torch.equal(a,b)
    report['top_p_exact_strict']=True
    # Observe actual inverse temperature arriving at the execution object,
    # rather than merely checking dictionary keys or plausible output tokens.
    cls=type(e); orig=cls.__call__; observed=[]
    def spy(self,*args,**kwargs):
        observed.append(args[3]); return orig(self,*args,**kwargs)
    cls.__call__=spy
    try:
        for t in [0.5,1.5]:
            handle.generate(**(kw|{'temperature':t,'top_p':1.0}))
    finally: cls.__call__=orig
    assert 2.0 in observed and 1/1.5 in observed
    report['observed_inverse_temperatures']=observed
    extra_semantics(handle, inputs, report)
    report['measurements_complete']=True
    report['complete']=not report['guard_failures'];save()
    fast=install_fast_decode(talker)
    fast.uninstall()
    handle.uninstall()
    assert not hasattr(talker,'_fast_decode')
    report['fast_decode_uninstall_order']=True
    # Restores the exact preexisting instance method, not only an equivalent call.
    original=p.__dict__.get('generate')
    assert original is not None
    h=install_fused_predictor(p); h.uninstall()
    assert p.__dict__.get('generate') is original
    assert not hasattr(p,'_fused_predictor') and not hasattr(p,'_predictor_graph')
    # Also restore the class-method lookup when no instance override existed.
    del p.generate
    h=install_fused_predictor(p);h.uninstall()
    assert 'generate' not in p.__dict__
    assert not hasattr(p,'_fused_predictor') and not hasattr(p,'_predictor_graph')
    p.generate=original
    report['uninstall_identity_markers']=True;save()
    assert not report['guard_failures'], report['guard_failures']
    print('Deep validation passed',size,flush=True)


@torch.inference_mode()
def extra_semantics(handle, inputs, report):
    from qwen_fused_predictor import _k
    e=handle.engine; device=inputs[0].device
    # Fixed BF16 logits include cutoff ties and both zero encodings.
    logits=torch.linspace(-4,4,e.vocab,device=device).to(torch.bfloat16).reshape(1,-1)
    logits[0,-4:]=4; logits[0,0]=0.; logits[0,1]=-0.
    uniforms=torch.linspace(0.0001,0.9999,e.vocab,device=device).reshape(1,-1)
    codes=torch.zeros(1,e.groups,device=device,dtype=torch.long)
    cases=[]
    for temperature in [0.5,1.5]:
        for k in [0,1,50,e.vocab]:
            _k._cp_sample_kernel[(1,)](logits,uniforms,uniforms.stride(0),codes,1,e.groups,e._dummy,e._dummy,1/temperature,V=e.vocab,TOPK=k,HID=e.hidden,HAS_NEXT=False,num_warps=4)
            scores=logits.float()/temperature
            if k: scores=scores.masked_fill(logits < logits.topk(k).values[:,-1:],-float('inf'))
            expected=(scores-(-uniforms.log()).log()).argmax(-1)
            assert torch.equal(codes[:,1],expected), (temperature,k)
            cases.append({'temperature':temperature,'top_k':k,'token':int(expected)})
    # All-zero ties include signed zeros at the cutoff.
    logits.zero_(); logits[:,::2]=-0.
    for k in [1,e.vocab]:
        _k._cp_sample_kernel[(1,)](logits,uniforms,uniforms.stride(0),codes,1,e.groups,e._dummy,e._dummy,1.,V=e.vocab,TOPK=k,HID=e.hidden,HAS_NEXT=False,num_warps=4)
        assert int(codes[0,1])==e.vocab-1
    report['fixed_sampler']=cases
    kw=dict(inputs_embeds=inputs[0],max_new_tokens=15,do_sample=True,temperature=0.73,top_k=50,top_p=1.,output_hidden_states=False,return_dict_in_generate=True)
    torch.manual_seed(545)
    first=handle.generate(**kw).sequences.clone()  # includes capture
    state=torch.cuda.get_rng_state().clone()
    second=handle.generate(**kw).sequences.clone()
    assert not torch.equal(state,torch.cuda.get_rng_state()), 'RNG did not advance'
    torch.manual_seed(545)
    assert torch.equal(first,handle.generate(**kw).sequences), 'Capture consumed RNG'
    assert torch.equal(second,handle.generate(**kw).sequences), 'Replay reseed mismatch'
    alternate=kw|{'temperature':1.31}
    torch.manual_seed(54)
    a=handle.generate(**kw).sequences.clone();b=handle.generate(**alternate).sequences.clone()
    torch.manual_seed(54)
    assert torch.equal(a,handle.generate(**kw).sequences)
    assert torch.equal(b,handle.generate(**alternate).sequences)
    report['rng_capture_reseed_alternating']=True
    for override in [{'output_hidden_states':True}, {'inputs_embeds':inputs[0].repeat(2,1,1)}]:
        args=kw|override
        handle.generate(**args)
        torch.manual_seed(11);a=handle.generate(**args)
        torch.manual_seed(11);b=handle.strict.generate(**args)
        assert torch.equal(a.sequences,b.sequences)
        saved=a.sequences.clone()
        states=None if a.hidden_states is None else [[t.clone() for t in s] for s in a.hidden_states]
        handle.generate(**args)
        assert torch.equal(saved,a.sequences)
        if states is not None:
            for x,y,z in zip(a.hidden_states,b.hidden_states,states):
                assert all(torch.equal(i,j) and torch.equal(i,k) for i,j,k in zip(x,y,z))
    report['hidden_batch_fallback_exact_ownership']=True


@torch.inference_mode()
def diagnose_layer(p,p32,e,inp,first,local_inputs,outputs,caches,fused_cache,fused_outputs):
    import copy
    from qwen_fused_predictor import _k,_mapped
    li=4; pos=torch.tensor([[first]],device=inp.device)
    support=_k._pdl_supported;_k._pdl_supported=lambda dev:False
    try: diag=_k.FusedCodePredictor(_mapped(p),1)
    finally:_k._pdl_supported=support
    rms=lambda a,b:float((a.float()-b.float()).square().mean().sqrt()/b.float().square().mean().sqrt())
    rows=[]
    def hf(layer,x,cache):
        cache=copy.deepcopy(cache)
        rot=p32.model.rotary_emb if x.dtype==torch.float32 else p.model.rotary_emb
        cos,sin=rot(x,pos)
        mask=torch.zeros(1,1,1,first+1,device=x.device,dtype=x.dtype)
        return layer(x,attention_mask=mask,position_ids=pos,position_embeddings=(cos,sin),past_key_values=cache,cache_position=pos[0],use_cache=True)[0]
    def fused(x,cache,pdl):
        engine=e if pdl else diag
        engine.cache.copy_(cache)
        if not pdl:
            out=x.reshape(1,-1).clone();engine._layer(out,li,1,1,first);return out.reshape_as(x)
        out=e.pdl_x;out[:1].copy_(x.reshape(1,-1)); ss,nxt=e.pdl_ss
        torch.sum(out[:1].float().square(),dim=1,out=ss[0,:1]);c=e._pdl_config(1);L=e.layers[li];W=e.pdl_w[li]
        e._gemm(c['qkv'],out,1,0,W['qkv'],e.pdl_qkv,ss,1,None,1,0,L['eps1'])
        _k._cp_attention_kernel[(1,e.heads,1)](e.pdl_qkv,e.cache[li,0],e.cache[li,1],L['qn'],L['kn'],e.cos,e.sin,e.pdl_att,first,L['head_eps'],e.scale,NQ=1,H=e.heads,KVH=e.kv_heads,HD=e.head_dim,MAXP=e.max_pos,BK=e._attn_keys,PDL=True,num_warps=1,launch_pdl=True)
        nt=e._gemm(c['o'],e.pdl_att,1,0,W['o'],out,None,0,nxt,1,2)
        e._gemm(c['gu'],out,1,0,W['gu'],e.pdl_act,nxt,nt,None,1,1,L['eps2'])
        e._gemm(c['down'],e.pdl_act,1,0,W['down'],out,None,0,ss,1,2)
        return out[:1].clone().reshape_as(x)
    global_ref=outputs[0][li]
    for source in ['fp32_local','bf16_accumulated','fused_accumulated']:
        index=0 if source=='fp32_local' else 1
        x=local_inputs[index][li] if source!='fused_accumulated' else e._diagnostic_inputs[li].unsqueeze(0).clone()
        cache=copy.deepcopy(caches[index])
        if source=='fused_accumulated':
            for j,l in enumerate(cache.layers):
                l.keys=fused_cache[j,0,:,:,:first].clone();l.values=fused_cache[j,1,:,:,:first].clone()
        cf=copy.deepcopy(cache)
        cb=copy.deepcopy(cache)
        packed=torch.zeros_like(e.cache)
        for j,(lf,lb) in enumerate(zip(cf.layers,cb.layers)):
            lf.keys=lf.keys.float();lf.values=lf.values.float()
            lb.keys=lb.keys.bfloat16();lb.values=lb.values.bfloat16()
            packed[j,0,:,:,:first]=lb.keys;packed[j,1,:,:,:first]=lb.values
        ref=hf(p32.model.layers[li],x.float(),cf)
        bf=hf(p.model.layers[li],x.bfloat16(),cb)
        pd=fused(x.bfloat16(),packed,True);np=fused(x.bfloat16(),packed,False)
        rows.append(dict(source=source,input_global_error=rms(x,local_inputs[0][li]),reference_global_error=rms(ref,global_ref),bf16_local_error=rms(bf,ref),pdl_local_error=rms(pd,ref),nonpdl_local_error=rms(np,ref),pdl_nonpdl_error=rms(pd,np),pdl_global_error=rms(pd,global_ref)))
    # Isolate folding from the BF16 GEMM and normalization cast ordering.
    x=local_inputs[0][li].float(); layer=p32.model.layers[li]
    ln=layer.input_layernorm; a=layer.self_attn
    w=torch.cat([a.q_proj.weight,a.k_proj.weight,a.v_proj.weight])
    scale=torch.rsqrt(x.square().mean(-1,keepdim=True)+ln.variance_epsilon)
    exact=torch.nn.functional.linear(x*scale*ln.weight,w)
    folded32=torch.nn.functional.linear(x,w*ln.weight[None,:])*scale
    folded_bf=torch.nn.functional.linear(x.bfloat16(),(w*ln.weight[None,:]).bfloat16()).float()*scale
    hf_order=torch.nn.functional.linear(ln(x.bfloat16()).bfloat16(),w.bfloat16()).float()
    result={'frame':3,'step':13,'layer':4,'local_comparisons':rows,'qkv_projection':{'fp32_folded_error':rms(folded32,exact),'bf16_folded_error':rms(folded_bf,exact),'hf_cast_order_error':rms(hf_order,exact)},'original_fused_global_error':rms(fused_outputs[li],global_ref)}
    Path('results/gap/fused-predictor/0.6B-diagnosis.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
