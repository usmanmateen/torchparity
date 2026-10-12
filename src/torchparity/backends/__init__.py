"""Backends: a converter plus the runtime that executes its output."""

from __future__ import annotations

import importlib

from ._base import Backend, NotChecked

__all__ = ["BACKENDS", "Backend", "NotChecked", "get_backend"]

# name -> "module:class", imported on first use so only the selected backend's packages are needed.
BACKENDS = {
    "onnx": "torchparity.backends.onnx:Onnx",
    "openvino": "torchparity.backends.openvino:OpenVino",
    "coreml": "torchparity.backends.coreml:CoreMl",
    "tensorrt": "torchparity.backends.tensorrt:TensorRt",
    "executorch": "torchparity.backends.executorch:ExecuTorch",
    "inductor": "torchparity.backends.inductor:Inductor",
}


def get_backend(name: str) -> Backend:
    """A backend by name, or by "module:Class" for one defined outside torchparity."""
    if name in BACKENDS:
        name = BACKENDS[name]
    elif ":" not in name:
        raise KeyError(f"unknown backend {name!r}; choose from {', '.join(BACKENDS)} or give module:Class")
    module, cls = name.split(":")
    return getattr(importlib.import_module(module), cls)()
