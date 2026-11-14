"""Opt-in compiled talker decode using TalkerGraph's serial cache safety checks."""
import time
import torch
from qwen_talker_graph import TalkerGraph


class _Replay:
    def __init__(self, owner):
        self.owner = owner

    def replay(self):
        self.owner.output = self.owner.compiled()


class CompiledTalker(TalkerGraph):
    """Fixed-capacity, batch-one decode; prefill stays eager. Not a CUDA graph.

    Inherits device/stream ownership, contiguous cache position, reset, capacity,
    dtype and retained-output checks from TalkerGraph. Install before generation.
    Default fusion can change BF16 rounding; it does not promise token equality.
    """
    def __init__(self, talker, max_cache_len=256, emulate_precision_casts=False):
        self.emulate_precision_casts = emulate_precision_casts
        super().__init__(talker, max_cache_len)

    def _capture(self):
        torch.cuda.synchronize()
        start = time.perf_counter()
        self.compiled = torch.compile(self._run, fullgraph=True,
            options={'emulate_precision_casts': self.emulate_precision_casts})
        self.output = self.compiled()
        torch.cuda.synchronize()
        self.capture_seconds.append(time.perf_counter() - start)
        self.graph = _Replay(self)


def install_compiled_talker(talker, max_cache_len=256, emulate_precision_casts=False):
    return CompiledTalker(talker, max_cache_len, emulate_precision_casts)
