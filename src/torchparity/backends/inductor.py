from __future__ import annotations

import torch

from ._base import Backend, quiet


class Inductor(Backend):
    name = "inductor"
    label = "torch.compile"
    mode = 'torch.compile(backend="inductor", fullgraph=True, dynamic=False), CPU'
    packages = ("torch",)
    dtypes = ("f32", "f64", "f16", "bf16", "i64", "i32", "i8", "u8")

    def compile(self, module, inputs):
        torch._dynamo.reset()
        compiled = torch.compile(module, backend="inductor", fullgraph=True, dynamic=False)
        with quiet(), torch.no_grad():
            first = compiled(*inputs)  # compiles; errors here are conversion failures

        def run(args):
            if args is inputs:
                return first
            with torch.no_grad():
                return compiled(*args)

        return run
