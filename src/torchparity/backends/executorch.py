from __future__ import annotations

import torch
from executorch.exir import to_edge
from executorch.runtime import Runtime

from ._base import Backend, quiet


class ExecuTorch(Backend):
    name = "executorch"
    label = "ExecuTorch"
    mode = "to_edge(torch.export.export(...)).to_executorch(), default CPU kernels"
    packages = ("torch", "executorch")
    dtypes = ("f32", "f64", "f16", "bf16", "i64", "i32", "i8", "u8")

    def __init__(self):
        self.runtime = Runtime.get()

    def compile(self, module, inputs):
        with quiet():
            program = to_edge(torch.export.export(module, inputs)).to_executorch()
            method = self.runtime.load_program(program.buffer).load_method("forward")

        def run(args):
            return tuple(method.execute(list(args)))

        return run
