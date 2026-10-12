"""Running cases on a backend and classifying the differences."""

from __future__ import annotations

import datetime
import json
import platform
import re
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from . import __version__
from .backends import Backend, NotChecked
from .cases import Case
from .compare import as_tuple, compare, edge_values, examples, ordinary_inputs, overflow, undefined

# ok: same result as eager.
# wrong: differs on ordinary values (or in dtype / shape).
# edge: differs only on edge values: NaN, infinities, -0.0, magnitudes >= 1e6 or < 1e-6,
#       integer overflow, division by zero.
# unsupported: conversion or execution fails.
# skip: eager PyTorch rejects the case, or the backend cannot be checked.
STATUSES = ("ok", "wrong", "edge", "unsupported", "skip")


@dataclass
class Result:
    op: str
    dtype: str
    expr: str
    status: str
    detail: str = ""
    examples: list[dict] = field(default_factory=list)


def _first_line(exc: BaseException, limit: int = 200) -> str:
    """First line of the innermost explicitly chained exception, which is usually the specific one."""
    while exc.__cause__ is not None:
        exc = exc.__cause__
    message = re.sub(r"\x1b\[[0-9;]*m", "", str(exc))
    lines = [line.strip() for line in message.strip().splitlines() if line.strip()]
    line = lines[0] if lines else ""
    if line.endswith(":") and len(lines) > 1:  # a location header; the last line says what failed
        line = lines[-1].lstrip("- ")
    text = f"{type(exc).__name__}: {line}".rstrip(": ")
    return text[:limit]


def _to(inputs, device):
    return tuple(t.to(device) for t in inputs)


def _execute(backend: Backend, case: Case):
    """Eager outputs, backend outputs, or a Result when one of them cannot be produced."""
    module = case.module()
    inputs = _to(case.inputs, backend.device)
    if backend.device != "cpu":
        module = module.to(backend.device)
    try:
        with torch.no_grad():
            expected = tuple(t.cpu() for t in as_tuple(module(*inputs)))
    except Exception as exc:
        return Result(case.op, case.dtype, case.expr, "skip", f"eager: {_first_line(exc)}")
    try:
        run = backend.compile(module, inputs)
    except NotChecked as exc:
        return Result(case.op, case.dtype, case.expr, "skip", str(exc))
    except Exception as exc:
        return Result(case.op, case.dtype, case.expr, "unsupported", f"convert: {_first_line(exc)}")
    try:
        actual = tuple(t.cpu() for t in as_tuple(run(inputs)))
    except Exception as exc:
        return Result(case.op, case.dtype, case.expr, "unsupported", f"run: {_first_line(exc)}")
    return expected, actual


def check(backend: Backend, case: Case) -> Result:
    out = _execute(backend, case)
    if isinstance(out, Result):
        return out
    expected, actual = out
    diff = compare(expected, actual, backend.narrows)
    if diff is None:
        return Result(case.op, case.dtype, case.expr, "ok")
    if diff.bad is None:
        return Result(case.op, case.dtype, case.expr, "wrong", diff.reason)
    e = expected[diff.output]
    a = diff.actual
    if case.elementwise and e.shape == case.inputs[0].shape:
        bad = diff.bad & ~undefined(case)
        if not bad.any():
            return Result(case.op, case.dtype, case.expr, "ok")
        edge = edge_values(e) | overflow(case, e) | ((e == 0) & (a == 0))
        for t in case.inputs:
            edge |= edge_values(t)
        ordinary_bad = bad & ~edge
        status = "wrong" if ordinary_bad.any() else "edge"
        shown = ordinary_bad if ordinary_bad.any() else bad
        return Result(case.op, case.dtype, case.expr, status, "values", examples(case, e, a, shown))
    # Not elementwise: rerun on the elements whose inputs are all ordinary.
    found = examples(case, e, a, diff.bad)
    keep = ordinary_inputs(case)
    if bool(keep.all()):
        return Result(case.op, case.dtype, case.expr, "wrong", "values", found)
    if not bool(keep.any()):
        return Result(case.op, case.dtype, case.expr, "edge", "values", found)
    sub = case.subset(keep)
    out = _execute(backend, sub)
    if isinstance(out, Result) or compare(out[0], out[1], backend.narrows) is None:
        return Result(case.op, case.dtype, case.expr, "edge", "values (ordinary inputs alone match)", found)
    sub_diff = compare(out[0], out[1], backend.narrows)
    if sub_diff.bad is not None:
        found = examples(sub, out[0][sub_diff.output], sub_diff.actual, sub_diff.bad)
    return Result(case.op, case.dtype, case.expr, "wrong", "values (also with ordinary inputs alone)", found)


def run(backend: Backend, cases: Iterable[Case],
        progress: Callable[[Case, Result], None] | None = None) -> list[Result]:
    results = []
    for case in cases:
        result = check(backend, case)
        results.append(result)
        if progress:
            progress(case, result)
    return results


def metadata(backend: Backend) -> dict:
    return {
        "backend": backend.name,
        "label": backend.label,
        "mode": backend.mode,
        "versions": backend.versions(),
        "torchparity": __version__,
        "platform": f"{platform.system()} {platform.machine()}",
        "python": platform.python_version(),
        "date": datetime.date.today().isoformat(),
    }


def save(path: str | Path, backend: Backend, results: list[Result]) -> None:
    data = {**metadata(backend), "results": [asdict(r) for r in results]}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
