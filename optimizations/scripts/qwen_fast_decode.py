"""Opt-in elimination of unused predictor hidden states in one Qwen talker.

Install after predictor/talker graphs and fused predictor norms. Only the pinned
upstream talker forward call site changes: output_hidden_states=False for its
predictor.generate call. Direct predictor calls still honor requested outputs.
Talker outputs (including the streaming codec-frame hook) are unchanged. Graph
ownership, cache checks, sampling and output clones remain the strict adapters'.
Serial inference only; uninstall this handle before other adapters. Unknown
upstream implementations fail closed rather than guessing about output usage.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
import types


# AST of the inspected upstream forward, including its can_return_tuple decorator.
_FORWARD_SHA256 = 'ab3aa131c453b71709b8d6d320eaee482cd3f86460cc801c40d33454c6d7f64b'


class FastDecode:
    def __init__(self, talker):
        if hasattr(talker, '_fast_decode'):
            raise ValueError('Fast decode is already installed')
        if talker.__class__.__name__ != 'Qwen3TTSTalkerForConditionalGeneration':
            raise ValueError('Expected Qwen3-TTS talker')
        if not hasattr(talker.code_predictor, '_predictor_graph'):
            raise ValueError('Install the strict predictor graph first')
        if talker.training:
            raise ValueError('Fast decode requires eval mode')
        self.talker = talker
        self.original = talker.forward
        self.prior_instance = talker.__dict__.get('forward')
        if self.prior_instance is not None:
            raise ValueError('Install fast decode before any talker.forward instance wrapper')
        original = inspect.unwrap(self.original)
        tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
        forward = tree.body[0]
        digest = hashlib.sha256(ast.dump(forward, include_attributes=False).encode()).hexdigest()
        if digest != _FORWARD_SHA256:
            raise ValueError('Unsupported upstream talker forward; re-audit predictor output consumers')
        calls = [n for n in ast.walk(forward) if isinstance(n, ast.Call)
                 and ast.unparse(n.func) == 'self.code_predictor.generate']
        if len(calls) != 1:
            raise ValueError('Expected exactly one internal predictor call')
        flags = [k for k in calls[0].keywords if k.arg == 'output_hidden_states']
        if len(flags) != 1 or not isinstance(flags[0].value, ast.Constant) or flags[0].value.value is not True:
            raise ValueError('Unexpected internal predictor hidden-state flag')
        flags[0].value.value = False
        # Reuse upstream globals and decorator, without modifying its module/class.
        namespace = dict(original.__globals__)
        exec(compile(ast.fix_missing_locations(tree), '<qwen_fast_decode>', 'exec'), namespace)
        self.patched = types.MethodType(namespace['forward'], talker)
        talker.forward = self.patched
        talker._fast_decode = self

    def uninstall(self):
        if self.talker.__dict__.get('_fast_decode') is not self or self.talker.forward is not self.patched:
            raise RuntimeError('Fast decode is not installed or forward was replaced')
        del self.talker.forward
        del self.talker._fast_decode


def install_fast_decode(talker):
    """Change only the verified internal predictor call; return uninstall handle."""
    return FastDecode(talker)
