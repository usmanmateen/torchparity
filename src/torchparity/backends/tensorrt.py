from __future__ import annotations

import torch
import torch_tensorrt

from ._base import Backend, quiet


class TensorRt(Backend):
    name = "tensorrt"
    label = "Torch-TensorRT"
    mode = 'torch_tensorrt.compile(ir="dynamo", require_full_compilation=True, use_explicit_typing=True), CUDA'
    packages = ("torch", "torch-tensorrt", "tensorrt")
    dtypes = ("f32", "f16", "i64", "i32")
    device = "cuda"

    def compile(self, module, inputs):
        with quiet():
            compiled = torch_tensorrt.compile(
                module.eval().cuda(), ir="dynamo", inputs=list(inputs), min_block_size=1,
                require_full_compilation=True, use_explicit_typing=True)

        def run(args):
            with torch.no_grad():
                return compiled(*args)

        return run
