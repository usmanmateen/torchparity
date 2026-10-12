from __future__ import annotations

import numpy as np
import openvino as ov
import torch

from ._base import Backend, quiet


class OpenVino(Backend):
    name = "openvino"
    label = "OpenVINO"
    mode = "ov.convert_model(example_input=...), CPU, f32 inference precision"
    packages = ("torch", "openvino")
    # The CPU plugin runs f64 in f32, so f64 is not checked.
    dtypes = ("f32", "f16", "i64", "i32", "i8", "u8")

    def __init__(self):
        self.core = ov.Core()

    def compile(self, module, inputs):
        with quiet():
            model = ov.convert_model(module, example_input=inputs)
            compiled = self.core.compile_model(model, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})

        def run(args):
            result = compiled([a.numpy() for a in args])
            return tuple(torch.from_numpy(np.asarray(result[o])) for o in compiled.outputs)

        return run
