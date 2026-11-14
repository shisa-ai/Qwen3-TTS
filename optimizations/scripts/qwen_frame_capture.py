"""Experimental unified predictor/embedding/talker frame CUDA graph.

A frame consumes the already selected codec token and produces the next logits.
Sampling and synchronous EOS stay in the reviewed explicit loop for this bounded
experiment. This is not a device-history/sampler capture implementation.
"""
import torch


def enable_capture_experiment(loop):
    """Install a test-only run variant without modifying the reviewed loop file."""
    import inspect
    import textwrap
    import types
    from qwen_frame_loop import FrameLoop, processors, GenerateDecoderOnlyOutput
    source = textwrap.dedent(inspect.getsource(FrameLoop.run))
    marker = '        first = t.get_input_embeddings()(token)'
    insertion = '''        if getattr(self, 'frame_factory', None) is not None:
            trailing = kw['trailing_text_hidden']
            text = trailing[:,step:step+1] if step < trailing.shape[1] else kw['tts_pad_embed']
            self.capture_settings = key
            if key not in self.frames:
                self.frames[key] = self.frame_factory(self, token, past_hidden, text, length)
            logits, past_hidden, states, codes = self.frames[key].replay(token, past_hidden, text, length)
            g.cache.length = length + 1
            hidden.append((states, codes))
            length += 1
            if self.step_observer is not None:
                self.step_observer(step, logits, g.cache, codes, past_hidden)
            continue
'''
    if source.count(marker) != 1:
        raise ValueError('Reviewed loop source changed; re-audit experiment')
    namespace = dict(torch=torch, processors=processors,
                     GenerateDecoderOnlyOutput=GenerateDecoderOnlyOutput)
    exec(source.replace(marker, insertion + marker), namespace)
    loop.run = types.MethodType(namespace['run'], loop)


class CapturedFrame:
    def __init__(self, loop, token, hidden, text, position):
        self.loop = loop
        self.token = token.clone()
        self.hidden = hidden.clone()
        self.text = text.clone()
        self.position = torch.tensor([position], device=token.device)
        self.positions = torch.empty((3, 1, 1), device=token.device, dtype=torch.long)
        self.keys = torch.arange(loop.tg.capacity, device=token.device)
        self.mask = torch.empty((1,1,1,loop.tg.capacity), device=hidden.device, dtype=hidden.dtype)
        self.uniforms = torch.empty((1,15,loop.pg.engine.vocab),device=hidden.device)
        self.sample, self.temperature, self.top_k = loop.capture_settings
        self.owner = torch.cuda.current_stream()
        with torch.random.fork_rng(devices=[token.device.index]):
            stream = torch.cuda.Stream()
            stream.wait_stream(self.owner)
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.run()
            self.owner.wait_stream(stream)
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.run()
            torch.cuda.synchronize()

    def run(self):
        t, g, p = self.loop.talker, self.loop.tg, self.loop.pg
        first = t.get_input_embeddings()(self.token)
        if self.sample:
            self.uniforms.uniform_().clamp_(1e-7, 1-1e-7)
        else:
            self.uniforms.fill_(.5)
        p.engine.cache.zero_()
        self.codes = p.engine(self.token[:,0],first[:,0],self.hidden[:,0],
                              1/self.temperature,self.top_k,self.uniforms)
        residual = self.codes[:,1:]
        embeds = torch.cat([first] + [t.code_predictor.get_input_embeddings()[i](residual[:,i:i+1])
            for i in range(15)],dim=1).sum(1,keepdim=True) + self.text
        self.positions.copy_(self.position.reshape(1,1,1) + t.rope_deltas)
        self.mask.zero_().masked_fill_(self.keys.view(1,1,1,-1) > self.position, float('-inf'))
        self.output = g.original(inputs_embeds=embeds,position_ids=self.positions,
            attention_mask=self.mask,past_key_values=g.cache,cache_position=self.position,
            use_cache=True,output_hidden_states=True,output_attentions=False,return_dict=True)
        self.logits = t.codec_head(self.output.last_hidden_state)[:, -1].float()

    def replay(self, token, hidden, text, position):
        if torch.cuda.current_stream() != self.owner:
            raise ValueError('Captured frame requires owning stream')
        self.token.copy_(token)
        self.hidden.copy_(hidden)
        self.text.copy_(text)
        self.position.fill_(position)
        self.graph.replay()
        return (self.logits.clone(), self.output.last_hidden_state.clone(),
                tuple(v.clone() for v in self.output.hidden_states), self.codes.clone())
