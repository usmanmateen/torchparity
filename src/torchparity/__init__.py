"""Check that model converters and runtimes give the same results as eager PyTorch."""

from __future__ import annotations

from collections.abc import Iterable

__version__ = "0.1.0"
__all__ = ["check", "__version__"]


def check(backend, *, ops: Iterable[str] | None = None, dtypes: Iterable[str] | None = None,
          expr: str | None = None, match: Iterable[str] | None = None, isolate: bool = False) -> list:
    """Run cases on a backend and return one ``Result`` per case.

    backend: a name such as "openvino", "module:Class", or a ``Backend`` instance.
    ops / dtypes / match / expr: which cases to run; see ``torchparity.runner.select_cases``.
    isolate: run in a worker process, so a native crash fails one case instead of the call
    (needs the backend as a name).

    Each ``Result`` has ``op``, ``dtype``, ``expr``, ``status`` ("ok", "wrong", "edge",
    "unsupported" or "skip"), ``detail`` and ``examples``.
    """
    from .backends import get_backend
    from .runner import run, run_isolated, select_cases

    instance = get_backend(backend) if isinstance(backend, str) else backend
    cases = select_cases(instance, dtypes=dtypes, ops=ops, expr=expr, match=match)
    if isolate:
        if not isinstance(backend, str):
            raise ValueError("isolate=True needs the backend as a name")
        return run_isolated(backend, cases)
    return run(instance, cases)
