from __future__ import annotations

import numpy as np
import onnxruntime as ort
import torch

from ._base import Backend, quiet


class Onnx(Backend):
    name = "onnx"
    label = "ONNX Runtime"
    mode = "torch.onnx.export(dynamo=True), CPUExecutionProvider"
    packages = ("torch", "onnxscript", "onnx", "onnxruntime")
    dtypes = ("f32", "f64", "f16", "i64", "i32", "i8", "u8")

    def __init__(self):
        self.options = ort.SessionOptions()
        self.options.log_severity_level = 3

    def compile(self, module, inputs):
        with quiet():
            program = torch.onnx.export(module, inputs, dynamo=True, verbose=False)
        model = program.model_proto.SerializeToString()
        session = ort.InferenceSession(model, self.options, providers=["CPUExecutionProvider"])
        names = [i.name for i in session.get_inputs()]

        def run(args):
            outs = session.run(None, {n: a.numpy() for n, a in zip(names, args, strict=False)})
            return tuple(torch.from_numpy(np.asarray(o)) for o in outs)

        return run
