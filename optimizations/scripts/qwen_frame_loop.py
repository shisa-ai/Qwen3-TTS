"""Experimental explicit batch-one Qwen frame loop with separate engine graphs.

Prefill remains upstream. EOS is read synchronously once per frame. All promised
hidden states are retained as real cloned transformer outputs. No streaming fast
path; unsupported requests use the original generate implementation.
"""
import hashlib
import inspect
import math
import torch
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.generation.logits_process import (
    LogitsProcessorList, RepetitionPenaltyLogitsProcessor,
    SuppressTokensLogitsProcessor, MinNewTokensLengthLogitsProcessor,
    TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper)
from qwen_fused_predictor import _Graph


def processors(kw, device):
    result = LogitsProcessorList()
    penalty = kw.get('repetition_penalty', 1.0)
    if penalty != 1:
        result.append(RepetitionPenaltyLogitsProcessor(penalty))
    minimum = kw.get('min_new_tokens', 0)
    if minimum:
        result.append(MinNewTokensLengthLogitsProcessor(0, minimum, kw['eos_token_id'], device=device))
    if kw.get('suppress_tokens') is not None:
        result.append(SuppressTokensLogitsProcessor(kw['suppress_tokens'], device=device))
    if kw.get('do_sample', False):
        if kw.get('temperature', 1.) != 1:
            result.append(TemperatureLogitsWarper(kw['temperature']))
        if kw.get('top_k', 0):
            result.append(TopKLogitsWarper(kw['top_k']))
        if kw.get('top_p', 1.) < 1:
            result.append(TopPLogitsWarper(kw['top_p']))
    return result


class FrameLoop:
    def __init__(self, talker):
        if hasattr(talker, '_frame_loop'):
            raise ValueError('Frame loop already installed')
        if talker.__class__.__name__ != 'Qwen3TTSTalkerForConditionalGeneration':
            raise ValueError('Expected Qwen3-TTS talker')
        # Pin the upstream forward family whose frame alignment is implemented here.
        import qwen_tts.core.models.modeling_qwen3_tts as upstream
        self.upstream_path = inspect.getfile(upstream)
        from pathlib import Path
        self.upstream_digest = hashlib.sha256(Path(self.upstream_path).read_bytes()).hexdigest()
        if self.upstream_digest != '25c42656bcf810f06ef6bc1839bd7083f3c8cfedac3a147c4060b4262b1c96a0':
            raise ValueError('Unsupported upstream frame semantics')
        if hasattr(talker, '_frame_decode'):
            raise ValueError('Uninstall the rejected position-only adapter first')
        self.talker = talker
        self.tg = talker.model._talker_graph
        self.pg = talker.code_predictor._fused_predictor
        self.original = talker.generate
        self.prior = talker.__dict__.get('generate')
        self.wrapper = self.generate
        self.active = False
        self.calls = self.fallback_calls = 0
        self.step_observer = None  # Test-only observer; never used by normal requests.
        from transformers import GenerationConfig
        self.defaults = GenerationConfig().to_dict()
        talker.generate = self.wrapper
        talker._frame_loop = self

    def supported(self, args, kw):
        allowed = {'inputs_embeds', 'attention_mask', 'trailing_text_hidden', 'tts_pad_embed',
            'max_new_tokens', 'min_new_tokens', 'do_sample', 'temperature', 'top_k', 'top_p',
            'subtalker_dosample', 'subtalker_temperature', 'subtalker_top_k', 'subtalker_top_p',
            'repetition_penalty', 'suppress_tokens', 'eos_token_id', 'output_hidden_states',
            'return_dict_in_generate'}
        if args or set(kw) - allowed:
            return False
        # Require explicit values emitted by the upstream custom-voice entry point.
        required = allowed - {'suppress_tokens'}
        if required - set(kw):
            return False
        changed = {k for k,v in self.talker.generation_config.to_dict().items()
                   if v != self.defaults.get(k)}
        if changed - {'bos_token_id', 'eos_token_id', 'pad_token_id', '_from_model_config', 'transformers_version'}:
            return False
        x, mask = kw['inputs_embeds'], kw['attention_mask']
        if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] < 2 or mask.shape != x.shape[:2]:
            return False
        if x.device != self.tg.device or x.dtype != next(self.talker.parameters()).dtype:
            return False
        if mask.device != x.device:
            return False
        for name, shape in [('trailing_text_hidden', None), ('tts_pad_embed', (1, 1, x.shape[-1]))]:
            value = kw[name]
            if (not isinstance(value, torch.Tensor) or value.device != x.device or value.dtype != x.dtype
                    or value.layout != torch.strided or not value.is_contiguous()
                    or value.ndim != 3 or value.shape[0] != 1 or value.shape[-1] != x.shape[-1]
                    or (shape is not None and value.shape != shape)):
                raise ValueError(f'Frame loop {name} requires contiguous model-dtype/device embeddings')
        if not bool((mask == 1).all()):
            return False  # One boundary synchronization, never one per decode position.
        if not kw['output_hidden_states'] or not kw['return_dict_in_generate']:
            return False
        if not isinstance(kw['eos_token_id'], int) or not 0 <= kw['eos_token_id'] < self.talker.config.vocab_size:
            return False
        if not isinstance(kw['max_new_tokens'], int) or kw['max_new_tokens'] < 1:
            return False
        if not isinstance(kw['min_new_tokens'], int) or not 0 <= kw['min_new_tokens'] <= kw['max_new_tokens']:
            return False
        for prefix, sample_name in [('', 'do_sample'), ('subtalker_', 'subtalker_dosample')]:
            if not isinstance(kw[sample_name], bool):
                return False
            t,k,p = [kw[prefix+n] for n in ('temperature', 'top_k', 'top_p')]
            if not isinstance(t, (int,float)) or not math.isfinite(t) or t <= 0:
                return False
            if not isinstance(k,int) or isinstance(k,bool) or k < 0 or not 0 < p <= 1:
                return False
        if not isinstance(kw['repetition_penalty'], (float,int)) or not math.isfinite(kw['repetition_penalty']) or kw['repetition_penalty'] <= 0:
            return False
        suppress = kw.get('suppress_tokens')
        if suppress is not None and (not isinstance(suppress, list) or any(not isinstance(i,int) or not 0 <= i < self.talker.config.vocab_size for i in suppress)):
            return False
        if kw['subtalker_top_p'] != 1.0:
            return False
        if any(m._forward_hooks or m._forward_pre_hooks for m in self.talker.modules()):
            return False
        return True

    @torch.inference_mode()
    def generate(self, *args, **kw):
        if self.active:
            raise RuntimeError('Frame loop is serial and non-reentrant')
        self.tg._check_owner()
        if self.talker.training or torch.is_autocast_enabled('cuda'):
            raise ValueError('Frame loop requires eval without autocast')
        # Serialize the live predictor configuration on every request, exactly as
        # the fused adapter does. Never cache mutable/replaced nested settings.
        allowed_predictor = {'inputs_embeds', 'max_new_tokens', 'do_sample', 'temperature',
            'top_k', 'top_p', 'output_hidden_states', 'return_dict_in_generate',
            'bos_token_id', 'pad_token_id', '_from_model_config', 'transformers_version'}
        for name, value in self.pg.predictor.generation_config.to_dict().items():
            if name not in allowed_predictor and value != self.pg.strict._default_generation_settings().get(name):
                raise ValueError(f'Unsupported predictor generation_config setting: {name}={value!r}')
        if self.pg.predictor.training:
            raise ValueError('Predictor must remain in eval mode')
        if not self.supported(args, kw):
            self.fallback_calls += 1
            return self.original(*args, **kw)
        self.active = True
        self.calls += 1
        try:
            return self.run(kw)
        finally:
            self.active = False

    def run(self, kw, *, on_frame=None, poll=lambda: None, retain_history=True):
        t,g,p = self.talker,self.tg,self.pg
        x = kw['inputs_embeds']
        length = x.shape[1]
        g.reset()
        output = t(inputs_embeds=x, attention_mask=kw['attention_mask'],
                   cache_position=torch.arange(length, device=x.device), use_cache=True,
                   output_hidden_states=True, return_dict=True,
                   trailing_text_hidden=kw['trailing_text_hidden'], tts_pad_embed=kw['tts_pad_embed'])
        hidden = [output.hidden_states] if retain_history else None
        logits = output.logits[:, -1].float()
        past_hidden = output.past_hidden
        if not retain_history:
            del output
        seq = torch.empty((1,0), dtype=torch.long, device=x.device)
        process = processors(kw, x.device)
        sample = kw['subtalker_dosample']
        key = (sample, float(kw['subtalker_temperature']) if sample else 1.,
               min(kw['subtalker_top_k'],p.engine.vocab) if sample else 0)
        for step in range(kw['max_new_tokens']):
            poll()
            scores = process(seq, logits.clone())
            token = (torch.multinomial(scores.softmax(-1), 1) if kw['do_sample']
                     else scores.argmax(-1, keepdim=True))
            seq = torch.cat((seq,token),dim=1)
            # HF stops after selecting EOS: the EOS frame itself is not materialized.
            if bool((token == kw['eos_token_id']).all()) or step + 1 == kw['max_new_tokens']:
                break
            if length >= g.capacity:
                raise ValueError('TalkerGraph cache capacity exceeded')
            first = t.get_input_embeddings()(token)
            predictor_input = torch.cat((past_hidden,first),dim=1)
            if key not in p.graphs:
                if len(p.graphs) >= p.max_graphs:
                    raise ValueError('Fused predictor graph variant limit reached')
                with torch.random.fork_rng(devices=[x.device.index]):
                    p.graphs[key] = _Graph(p.engine,predictor_input,*key)
            residual = p.graphs[key].replay(predictor_input)
            codes = torch.cat((token,residual),dim=-1)
            embeds = torch.cat([first] + [t.code_predictor.get_input_embeddings()[i](residual[:,i:i+1])
                for i in range(t.config.num_code_groups-1)],dim=1).sum(1,keepdim=True)
            trailing = kw['trailing_text_hidden']
            embeds = embeds + (trailing[:,step:step+1] if step < trailing.shape[1] else kw['tts_pad_embed'])
            positions = (torch.full((3,1,1),length,device=x.device,dtype=torch.long) + t.rope_deltas)
            if g.graph is None:
                # Capture the existing static engine once with checked metadata.
                model_out = t.model(inputs_embeds=embeds, attention_mask=torch.ones((1,length+1),device=x.device,dtype=kw['attention_mask'].dtype),
                    position_ids=positions, cache_position=torch.tensor([length],device=x.device),
                    past_key_values=g.cache,use_cache=True,output_hidden_states=True,return_dict=True)
                past_hidden = model_out.last_hidden_state
                states = model_out.hidden_states
            else:
                g.embeds.copy_(embeds)
                g.positions.copy_(positions)
                g.position.fill_(length)
                g.mask.fill_(float('-inf'))
                g.mask[..., :length+1].zero_()
                g.graph.replay()
                g.replays += 1
                g.cache.length = length+1
                past_hidden = g.output.last_hidden_state.clone()
                states = tuple(v.clone() for v in g.output.hidden_states)
            if retain_history:
                hidden.append((states,codes))
            if on_frame is not None:
                on_frame(codes.clone())
            logits = t.codec_head(past_hidden)[:, -1].float()
            length += 1
            if self.step_observer is not None:
                self.step_observer(step, logits, g.cache, codes, past_hidden)
        if not retain_history:
            return None
        return GenerateDecoderOnlyOutput(sequences=seq,hidden_states=tuple(hidden),past_key_values=g.cache)

    def uninstall(self):
        self.tg._check_owner()
        if self.active or self.talker.generate is not self.wrapper:
            raise RuntimeError('Cannot uninstall active/replaced frame loop')
        if self.prior is None:
            del self.talker.generate
        else:
            self.talker.generate = self.prior
        del self.talker._frame_loop


def install_frame_loop(talker):
    return FrameLoop(talker)
