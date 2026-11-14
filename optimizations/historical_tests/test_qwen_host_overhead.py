"""GPU2 regression and timing checks for cached immutable generation defaults."""
import os
GPU = 'GPU-3e620c1c-c243-7353-580e-cde9e5eb4b8a'
assert os.environ.get('CUDA_VISIBLE_DEVICES') == GPU
assert os.environ.get('OMP_NUM_THREADS') == os.environ.get('MKL_NUM_THREADS') == '8'
import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import time
import numpy as np
import torch
from transformers import GenerationConfig
from qwen_tts import Qwen3TTSModel
from bench_common import TEXTS
from qwen_fused_norm import install_fused_predictor_norm
from qwen_fused_predictor import install_fused_predictor
from qwen_talker_fusion import install_talker_fusion
from qwen_talker_graph import install_talker_graph
from qwen_fast_decode import install_fast_decode


def measure(fn, n, sync=False):
    values=[]
    for _ in range(3):
        if sync: torch.cuda.synchronize()
        start=time.perf_counter()
        for _ in range(n): fn()
        if sync: torch.cuda.synchronize()
        values.append((time.perf_counter()-start)*1000/n)
    return values


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--size',required=True,choices=['0.6B','1.7B']); args=ap.parse_args()
    out=Path('results/gap/host-overhead'); out.mkdir(parents=True,exist_ok=True)
    r={'size':args.size,'complete':False,'checks':[],'variants':{},'gpu':GPU,
       'hashes':{p:hashlib.sha256(Path('scripts',p).read_bytes()).hexdigest() for p in ['qwen_predictor_graph.py','qwen_fused_predictor.py','qwen_talker_graph.py','test_qwen_host_overhead.py']}}
    def save(): (out/(args.size+'-gpu2.json')).write_text(json.dumps(r,indent=2))
    def rejects(label,fn):
        try: fn()
        except (ValueError,IndexError,RuntimeError): r['checks'].append(label)
        else: raise AssertionError(label)
    torch.set_num_threads(8); torch.manual_seed(42)
    m=Qwen3TTSModel.from_pretrained(f'Qwen/Qwen3-TTS-12Hz-{args.size}-CustomVoice',local_files_only=True,device_map='cuda:0',dtype=torch.bfloat16,attn_implementation='sdpa')
    t=m.model.talker; p=t.code_predictor
    norm=install_fused_predictor_norm(p)
    fp=install_fused_predictor(p,mode='nonpdl' if args.size=='0.6B' else 'pdl')
    fusion=install_talker_fusion(t); tg=install_talker_graph(t); fast=install_fast_decode(t)
    captured={}; orig=p.generate
    def observe_predictor(**kw):
        if not captured: captured.update({k:v.clone() if torch.is_tensor(v) else v for k,v in kw.items()})
        return orig(**kw)
    p.generate=observe_predictor
    def request(text): return m.generate_custom_voice(text=TEXTS[text],speaker='ryan',language='English',max_new_tokens=256)
    request('short'); p.generate=orig
    kw=dict(captured,do_sample=False)
    reference=p.generate(**kw).sequences.clone()
    strict_ref=fp.strict.generate(**kw).sequences.clone()
    config=copy.deepcopy(p.generation_config)
    for target in (fp,fp.strict):
        for name,value in [('num_beams',2),('repetition_penalty',1.1),('eos_token_id',[1]),('bad_words_ids',[[1]]),('unknown_setting',{'nested':[2]})]:
            p.generation_config=copy.deepcopy(config); setattr(p.generation_config,name,value)
            rejects(type(target).__name__+' mutation '+name,lambda:target.generate(**kw))
        p.generation_config=copy.deepcopy(config)
        p.generation_config.bad_words_ids=[]
        rejects(type(target).__name__+' mutable empty list',lambda:target.generate(**kw))
        p.generation_config.bad_words_ids.append([3])
        rejects(type(target).__name__+' nested mutation',lambda:target.generate(**kw))
        p.generation_config=copy.deepcopy(config)
        for change in [dict(do_sample=True,temperature=0),dict(do_sample=True,top_k=-1),dict(do_sample=True,top_p=1.1),dict(max_new_tokens=14),dict(num_beams=2)]:
            rejects(type(target).__name__+' invalid '+str(change),lambda:target.generate(**(kw|change)))
    p.generation_config=config
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        rejects('fused wrong stream',lambda:p.generate(**kw))
        rejects('strict wrong stream',lambda:fp.strict.generate(**kw))
        rejects('talker wrong stream',tg.reset)
    # Cache positions are intentionally still validated using device values.
    x=torch.randn(1,3,t.config.hidden_size,device='cuda',dtype=torch.bfloat16)
    common=dict(inputs_embeds=x,cache_position=torch.arange(3,device='cuda'),use_cache=True,output_hidden_states=True)
    for positions in [[0,2,1],[0,1,1],[-1,0,1],[1023,1024,1025]]:
        rejects('invalid positions '+str(positions),lambda: t.model(**(common|{'cache_position':torch.tensor(positions,device='cuda')})))
    a=t.model(**common)
    one=common|dict(inputs_embeds=x[:,:1],past_key_values=a.past_key_values)
    for positions in [[4],[2],[3,4]]:
        rejects('invalid decode '+str(positions),lambda:t.model(**(one|{'cache_position':torch.tensor(positions,device='cuda')})))
    tg.reset()
    rejects('reset stale cache',lambda:t.model(**(one|{'cache_position':torch.tensor([3],device='cuda')})))
    default_method=fp.strict._default_generation_settings
    for variant in ['rebuild_defaults','cached_defaults']:
        fp.strict._default_generation_settings=(lambda:GenerationConfig().to_dict()) if variant=='rebuild_defaults' else default_method
        assert torch.equal(reference,p.generate(**kw).sequences)
        assert torch.equal(strict_ref,fp.strict.generate(**kw).sequences)
        v={}; r['variants'][variant]=v
        def validate_config():
            defaults=fp.strict._default_generation_settings()
            allowed={'inputs_embeds','max_new_tokens','do_sample','temperature','top_k','top_p','output_hidden_states','return_dict_in_generate','bos_token_id','pad_token_id','_from_model_config','transformers_version'}
            for name,value in p.generation_config.to_dict().items():
                if name not in allowed and value != defaults.get(name): raise ValueError(name)
        v['host_config_ms']=measure(validate_config,1000)
        v['fused_predictor_ms']=measure(lambda:p.generate(**kw),100,True)
        v['strict_predictor_ms']=measure(lambda:fp.strict.generate(**kw),30,True)
        sequences={}; old=t.generate
        def observe(*a,**kw):
            output=old(*a,**kw); sequences['seq']=output.sequences; return output
        t.generate=observe; v['e2e']=[]
        for text in ['short','medium']:
            torch.manual_seed(0); request(text)
            for seed in range(3):
                torch.manual_seed(seed); before=tg.replays
                torch.cuda.synchronize(); start=time.perf_counter(); wavs,sr=request(text); torch.cuda.synchronize()
                elapsed=time.perf_counter()-start; audio=len(wavs[0])/sr
                row=dict(text=text,seed=seed,seconds=elapsed,audio_s=audio,rtf=elapsed/audio,frames=tg.replays-before,
                         eos=bool(sequences['seq'][0,-1]==t.config.codec_eos_token_id),finite=bool(np.isfinite(wavs[0]).all()))
                row['truncated'] = not row['eos']
                assert row['finite'] and audio>0
                v['e2e'].append(row); save(); print(variant,row,flush=True)
        t.generate=old
    r['checks'].append('greedy strict and fused outputs exact across variants')
    r['captures']={'talker':tg.capture_seconds,'fused':[g.capture_seconds for g in fp.graphs.values()],'strict':[g.capture_seconds for g in fp.strict.graphs.values()]}
    fast.uninstall();tg.uninstall();fusion.uninstall();fp.uninstall();norm.uninstall()
    r['complete']=True;save()

if __name__=='__main__':main()
