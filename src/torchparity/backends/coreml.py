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
from coremltools.converters.mil.mil import types

from ..compare import as_tuple
from ._base import Backend, NotChecked, quiet

_NP_TYPES = {torch.float32: np.float32, torch.float16: np.float16, torch.int32: np.int32,
             torch.int64: np.int32, torch.bool: np.bool_}


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


def _torch_dtype(builtin) -> torch.dtype | None:
    try:
        return torch.from_numpy(np.zeros(0, dtype=types.nptype_from_builtin(builtin))).dtype
    except Exception:
        return None


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
        # Compute in the input's precision: float16 for float16 inputs, as a float16 model would be
        # converted, and float32 otherwise, since float16 compute changes float32 results by design.
        half = any(t.dtype == torch.float16 for t in inputs)
        precision = ct.precision.FLOAT16 if half else ct.precision.FLOAT32
        with quiet():
            traced = torch.jit.trace(module, inputs)
            if self.predict:
                return self._predictor(traced, inputs, precision)
            converted = ct.convert(traced, inputs=_tensor_types(inputs), convert_to="milinternal",
                                   minimum_deployment_target=ct.target.iOS18, compute_precision=precision)
            exported = torch.export.export(_Constants(module, inputs), (torch.zeros(1),)).run_decompositions({})
            program = ct.convert(exported, convert_to="milinternal",
                                 minimum_deployment_target=ct.target.iOS18, compute_precision=precision)
        outputs = program.functions["main"].outputs[:-1]
        if any(var.val is None for var in outputs):
            raise NotChecked("not constant-folded")
        # Values come from constant folding; dtypes from the model converted with real inputs.
        dtypes = [_torch_dtype(var.dtype) for var in converted.functions["main"].outputs]
        values = tuple(torch.from_numpy(np.array(var.val)) for var in outputs)
        if len(dtypes) == len(values):
            values = tuple(v.to(d) if d is not None else v for v, d in zip(values, dtypes, strict=True))
        return lambda args: values

    def _predictor(self, traced, inputs, precision):
        model = ct.convert(traced, inputs=_tensor_types(inputs), convert_to="mlprogram",
                           minimum_deployment_target=ct.target.macOS15, compute_precision=precision,
                           compute_units=ct.ComputeUnit.CPU_ONLY)
        names = [o.name for o in model.get_spec().description.output]

        def run(args):
            feeds = {f"x{i}": a.numpy().astype(_NP_TYPES[a.dtype]) for i, a in enumerate(args)}
            out = model.predict(feeds)
            return tuple(torch.from_numpy(np.array(out[n])) for n in names)

        return run
