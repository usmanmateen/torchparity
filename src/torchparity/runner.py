"""Running cases on a backend and classifying the differences."""

from __future__ import annotations

import datetime
import json
import multiprocessing
import platform
import re
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from . import __version__
from .backends import Backend, NotChecked, get_backend
from .cases import Case, build_cases, expression_case
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


def select_cases(backend: Backend, *, dtypes: Iterable[str] | None = None, ops: Iterable[str] | None = None,
                 expr: str | None = None, match: Iterable[str] | None = None) -> list[Case]:
    """The cases to run on ``backend``.

    dtypes: dtype names such as "f32" or "i64" (default: every dtype the backend checks, plus mixed pairs).
    ops: exact op names. match: substrings of case ids such as "div" or "/f16".
    expr: your own expression over ``x``, or over ``a`` and ``b``, instead of the built-in ops.
    """
    chosen = list(dtypes) if dtypes else list(backend.dtypes)
    unknown = sorted(set(chosen) - set(backend.dtypes))
    if unknown and not expr:
        raise ValueError(f"{backend.name} does not check {', '.join(unknown)}; it checks {', '.join(backend.dtypes)}")
    cases = expression_case(expr, chosen) if expr else build_cases(chosen, mixed=not dtypes)
    if ops:
        wanted = set(ops)
        cases = [c for c in cases if c.op in wanted]
    if match:
        cases = [c for c in cases if any(m in c.id for m in match)]
    return cases


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
    if len(text) > limit:
        # Keep whole sentences: what follows is often a signature or a list of types, whose
        # order can change from run to run.
        end = text.rfind(". ", 0, limit)
        text = text[: end + 1] if end > limit // 4 else text[: limit - 3] + "..."
    return text


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


def check_case(backend: Backend, case: Case) -> Result:
    out = _execute(backend, case)
    if isinstance(out, Result):
        return out
    expected, actual = out
    diff = compare(expected, actual, backend.narrows, case.exact)
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
    if isinstance(out, Result) or compare(out[0], out[1], backend.narrows, case.exact) is None:
        return Result(case.op, case.dtype, case.expr, "edge", "values (ordinary inputs alone match)", found)
    sub_diff = compare(out[0], out[1], backend.narrows, case.exact)
    if sub_diff.bad is not None:
        found = examples(sub, out[0][sub_diff.output], sub_diff.actual, sub_diff.bad)
    return Result(case.op, case.dtype, case.expr, "wrong", "values (also with ordinary inputs alone)", found)


def run(backend: Backend, cases: Iterable[Case],
        progress: Callable[[Case, Result], None] | None = None) -> list[Result]:
    results = []
    for case in cases:
        result = check_case(backend, case)
        results.append(result)
        if progress:
            progress(case, result)
    return results


def _worker(name: str, cases: list[Case], start: int, conn) -> None:
    backend = get_backend(name)
    for index in range(start, len(cases)):
        conn.send((index, check_case(backend, cases[index])))
    conn.close()


def run_isolated(name: str, cases: list[Case],
                 progress: Callable[[Case, Result], None] | None = None) -> list[Result]:
    """Like ``run``, in a worker process, so a crash in native code fails one case instead of the run.

    A case that crashes a worker is retried as the first case of a new worker, and only counts as a
    crash if it crashes there too: some runtimes crash on state left behind by earlier cases.
    """
    context = multiprocessing.get_context("spawn")
    results: list[Result] = []
    retry = False
    while len(results) < len(cases):
        receiver, sender = context.Pipe(duplex=False)
        worker = context.Process(target=_worker, args=(name, cases, len(results), sender), daemon=True)
        worker.start()
        sender.close()
        first = True
        while len(results) < len(cases):
            try:
                _, result = receiver.recv()
            except EOFError:
                worker.join()
                if not (retry and first):
                    retry = True  # rerun this case in a fresh worker
                    break
                case = cases[len(results)]
                code = worker.exitcode
                reason = f"signal {-code}" if code is not None and code < 0 else f"exit code {code}"
                result = Result(case.op, case.dtype, case.expr, "unsupported", f"crashed ({reason})")
            retry = first = False
            results.append(result)
            if progress:
                progress(cases[len(results) - 1], result)
        receiver.close()
        worker.join()
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
