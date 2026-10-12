"""Comparing backend outputs with eager PyTorch."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .cases import DTYPE_NAMES, Case, make_fn

# (rtol, atol) per floating-point dtype.
TOLERANCES = {
    torch.float64: (1e-10, 1e-12),
    torch.float32: (1e-5, 1e-6),
    torch.float16: (2e-3, 1e-3),
    torch.bfloat16: (1.6e-2, 1e-2),
}
# Finite values with a magnitude outside [SMALL, LARGE) count as edge values, as does -0.0.
SMALL, LARGE = 1e-6, 1e6
_DIGITS = {torch.float16: 5, torch.bfloat16: 4, torch.float32: 9, torch.float64: 17}


@dataclass
class Difference:
    """How one output differs. ``bad`` marks differing elements when the shapes match."""

    reason: str
    bad: torch.Tensor | None = None
    output: int = 0
    actual: torch.Tensor | None = None
    examples: list[dict] = field(default_factory=list)


def as_tuple(out) -> tuple[torch.Tensor, ...]:
    if isinstance(out, (tuple, list)):
        return tuple(out)
    return (out,)


def fmt(value, dtype: torch.dtype) -> str:
    if isinstance(value, bool) or dtype == torch.bool:
        return str(bool(value))
    if not dtype.is_floating_point:
        return str(int(value))
    value = float(value)
    if value != value:
        return "nan"
    if value in (float("inf"), float("-inf")):
        return "inf" if value > 0 else "-inf"
    if value == 0:
        return "-0.0" if str(value).startswith("-") else "0.0"
    text = f"{value:.{_DIGITS.get(dtype, 9)}g}"
    return text if any(c in text for c in ".en") else text + ".0"


def compare(expected, actual, narrows: dict[torch.dtype, torch.dtype] | None = None) -> Difference | None:
    """Compare all outputs; return the first difference.

    ``narrows`` maps a PyTorch dtype to the dtype a backend returns instead by design
    (for example int64 to int32 on Core ML). The eager result is cast before comparing.
    """
    expected, actual = as_tuple(expected), as_tuple(actual)
    if len(expected) != len(actual):
        return Difference(f"{len(actual)} outputs, eager has {len(expected)}")
    for i, (e, a) in enumerate(zip(expected, actual, strict=True)):
        e, a = e.detach().cpu(), a.detach().cpu()
        if narrows and e.dtype in narrows and a.dtype == narrows[e.dtype]:
            e = e.to(a.dtype)
        if e.dtype != a.dtype:
            return Difference(f"dtype {DTYPE_NAMES.get(a.dtype, a.dtype)}, eager gives "
                              f"{DTYPE_NAMES.get(e.dtype, e.dtype)}", output=i)
        if e.shape != a.shape:
            return Difference(f"shape {tuple(a.shape)}, eager gives {tuple(e.shape)}", output=i)
        if e.dtype.is_floating_point:
            rtol, atol = TOLERANCES.get(e.dtype, (1e-5, 1e-6))
            close = torch.isclose(a.double(), e.double(), rtol=rtol, atol=atol, equal_nan=True)
            # isclose treats -0.0 == 0.0 and ignores the sign of infinities only through equality;
            # check the sign of zero results so that atan2/copysign-style differences are caught.
            zeros = (e == 0) & (a == 0)
            close &= ~zeros | (torch.signbit(e) == torch.signbit(a))
        else:
            close = e == a
        if not bool(close.all()):
            return Difference("values", bad=~close.reshape(e.shape), output=i, actual=a)
    return None


def edge_values(t: torch.Tensor) -> torch.Tensor:
    """Elements that are NaN, infinite, -0.0, very large or very small."""
    if t.dtype == torch.bool:
        return torch.zeros(t.shape, dtype=torch.bool)
    v = t.detach().cpu().double()
    mag = v.abs()
    edge = mag >= LARGE
    if t.dtype.is_floating_point:
        edge |= ~torch.isfinite(v) | ((mag > 0) & (mag < SMALL)) | ((v == 0) & torch.signbit(v))
    return edge


def overflow(case: Case, expected: torch.Tensor) -> torch.Tensor:
    """Elements whose eager result changes when the integer inputs are widened to int64."""
    none = torch.zeros(expected.shape, dtype=torch.bool)
    def is_narrow(t: torch.Tensor) -> bool:
        return not t.dtype.is_floating_point and t.dtype not in (torch.int64, torch.bool)

    if not any(is_narrow(t) for t in case.inputs):
        return none
    wide_inputs = tuple(t.to(torch.int64) if is_narrow(t) else t for t in case.inputs)
    try:
        with torch.no_grad():
            wide = as_tuple(make_fn(case.expr, len(case.inputs))(*wide_inputs))[0]
    except Exception:
        return none
    if wide.shape != expected.shape:
        return none
    w, e = wide.double(), expected.double()
    return ~((w == e) | (torch.isnan(w) & torch.isnan(e)))


def undefined(case: Case) -> torch.Tensor:
    """Elements PyTorch leaves undefined: float-to-integer casts out of range or of NaN."""
    x = case.inputs[0]
    none = torch.zeros(x.shape, dtype=torch.bool)
    if case.cast_to is None or not x.dtype.is_floating_point:
        return none
    info = torch.iinfo(case.cast_to)
    v = x.double()
    t = torch.trunc(v)
    return ~torch.isfinite(v) | (t < info.min) | (t > info.max)


def ordinary_inputs(case: Case) -> torch.Tensor:
    """Elements where no input is an edge value."""
    keep = torch.ones(case.inputs[0].shape, dtype=torch.bool)
    for t in case.inputs:
        keep &= ~edge_values(t)
    return keep


def examples(case: Case, expected: torch.Tensor, actual: torch.Tensor, mask: torch.Tensor,
             limit: int = 3) -> list[dict]:
    out = []
    flat_e, flat_a = expected.reshape(-1), actual.reshape(-1)
    for idx in mask.reshape(-1).nonzero().flatten().tolist()[:limit]:
        item = {}
        if case.elementwise and expected.shape == case.inputs[0].shape:
            names = ("x",) if len(case.inputs) == 1 else ("a", "b")
            inputs = zip(names, case.inputs, strict=True)
            item["inputs"] = {n: fmt(t.reshape(-1)[idx].item(), t.dtype) for n, t in inputs}
        item["eager"] = fmt(flat_e[idx].item(), expected.dtype)
        item["backend"] = fmt(flat_a[idx].item(), actual.dtype)
        out.append(item)
    return out
