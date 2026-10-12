"""Core ML via coremltools.

On macOS the converted model runs on the Core ML CPU runtime. Elsewhere there is no
Core ML runtime, so the inputs are baked into the model as constants and the values
come from the converter's constant folding (each MIL op's value inference). That
checks the converter, not the runtime.
"""

from __future__ import annotations

import sys

import coremltools as ct
import numpy as np
import torch

from ..compare import as_tuple
from ._base import Backend, NotChecked, quiet

_NP_TYPES = {torch.float32: np.float32, torch.float16: np.float16, torch.int32: np.int32,
             torch.int64: np.int32, torch.bool: np.bool_}
_PRECISION = ct.precision.FLOAT32


class _Constants(torch.nn.Module):
    """``module`` applied to buffers, plus one real input so the model has an input."""

    def __init__(self, module, inputs):
        super().__init__()
        self.inner = module
        self.count = len(inputs)
        for i, t in enumerate(inputs):
            self.register_buffer(f"c{i}", t.clone())

    def forward(self, x):
        out = as_tuple(self.inner(*[getattr(self, f"c{i}") for i in range(self.count)]))
        return (*out, x + 1)


def _tensor_types(inputs):
    return [ct.TensorType(name=f"x{i}", shape=t.shape, dtype=_NP_TYPES[t.dtype]) for i, t in enumerate(inputs)]


class CoreMl(Backend):
    name = "coreml"
    label = "Core ML"
    packages = ("torch", "coremltools")
    dtypes = ("f32", "f16", "i64", "i32")
    # Core ML has no int64; int64 tensors become int32.
    narrows = {torch.int64: torch.int32}

    def __init__(self):
        self.predict = sys.platform == "darwin"
        self.mode = ("torch.jit.trace + ct.convert(mlprogram), Core ML CPU" if self.predict else
                     "torch.jit.trace + ct.convert; values from constant folding (no Core ML runtime)")

    def compile(self, module, inputs):
        with quiet():
            traced = torch.jit.trace(module, inputs)
            if self.predict:
                return self._predictor(traced, inputs)
            ct.convert(traced, inputs=_tensor_types(inputs), convert_to="milinternal",
                       minimum_deployment_target=ct.target.iOS18, compute_precision=_PRECISION)
            exported = torch.export.export(_Constants(module, inputs), (torch.zeros(1),)).run_decompositions({})
            program = ct.convert(exported, convert_to="milinternal",
                                 minimum_deployment_target=ct.target.iOS18, compute_precision=_PRECISION)
        outputs = program.functions["main"].outputs[:-1]
        if any(var.val is None for var in outputs):
            raise NotChecked("not constant-folded")
        values = tuple(torch.from_numpy(np.array(var.val)) for var in outputs)
        return lambda args: values

    def _predictor(self, traced, inputs):
        model = ct.convert(traced, inputs=_tensor_types(inputs), convert_to="mlprogram",
                           minimum_deployment_target=ct.target.macOS15, compute_precision=_PRECISION,
                           compute_units=ct.ComputeUnit.CPU_ONLY)
        names = [o.name for o in model.get_spec().description.output]

        def run(args):
            feeds = {f"x{i}": a.numpy().astype(_NP_TYPES[a.dtype]) for i, a in enumerate(args)}
            out = model.predict(feeds)
            return tuple(torch.from_numpy(np.array(out[n])) for n in names)

        return run
