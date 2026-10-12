"""Operator cases and the values they run on.

Each op is a Python expression over ``x`` (unary) or ``a, b`` (binary). The same
string defines the function that runs and the text shown in reports.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

INF, NAN = math.inf, math.nan

# Binary ops pair FLOAT_A[i] with FLOAT_B[i]; unary ops use FLOAT_A.
FLOAT_A = [0.0, -0.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 1e-30, -3.7, 7.25, 1e10, -1e10,
           123456789.123456789, INF, -INF, NAN, 1.0, -1.0, 3.0, 4.2, 0.7, -4.2, 0.3]
FLOAT_B = [3.0, -3.0, 2.0, -2.0, 0.75, 0.75, -1.5, 1.5, 7.0, 2.0, -0.5, 3.0, 3.0,
           1000.0, 2.0, 3.0, 1.0, INF, -INF, 0.0, 1.4, 0.1, 1.4, -0.1]
INT_A = [0, 1, -1, 7, -7, 6, -6, 100, -100, 127, -128, 5, -5, 2, 3, -3]
INT_B = [3, 3, 3, 3, 3, -3, -3, 7, 7, -2, 3, -5, 5, 1, -2, 2]

DTYPE_NAMES = {
    torch.float64: "f64", torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16",
    torch.int64: "i64", torch.int32: "i32", torch.int16: "i16", torch.int8: "i8", torch.uint8: "u8",
    torch.bool: "bool",
}
DTYPES = {name: dtype for dtype, name in DTYPE_NAMES.items()}
FLOAT_DTYPES = ("f64", "f32", "f16", "bf16")
INT_DTYPES = ("i64", "i32", "i8", "u8")

# An entry is one expression for every dtype, or {"float": ..., "int": ...}.
UNARY = {
    "abs": "torch.abs(x)",
    "neg": "torch.neg(x)",
    "sign": "torch.sign(x)",
    "relu": "torch.relu(x)",
    "clamp": "torch.clamp(x, -2, 2)",
    "clamp_min": "torch.clamp(x, min=0)",
    "pow2": "torch.pow(x, 2)",
    "square": "torch.square(x)",
    "add_scalar": "x + 3",
    "mul_scalar": "x * -2",
    "div_scalar": "x / 3",
    "div_scalar_floor": 'torch.div(x, 3, rounding_mode="floor")',
    "div_scalar_trunc": 'torch.div(x, -3, rounding_mode="trunc")',
    "floor_divide_scalar": "x // 3",
    "fmod_scalar": "torch.fmod(x, 3)",
    "remainder_scalar": "torch.remainder(x, -3)",
    "rsub_scalar": "5 - x",
    "eq_scalar": "x == 3",
    "lt_scalar": "x < 0",
    "where_scalar": "torch.where(x > 0, x, 0)",
    "floor": "torch.floor(x)",
    "ceil": "torch.ceil(x)",
    "round": "torch.round(x)",
    "trunc": "torch.trunc(x)",
    "sqrt": "torch.sqrt(x)",
    "exp": "torch.exp(x)",
    "sigmoid": "torch.sigmoid(x)",
    "logical_not": "torch.logical_not(x)",
    "to_bool": "x.to(torch.bool)",
    "to_float32": "x.to(torch.float32)",
    "sum": "x.sum()",
    "cumsum": "x.cumsum(0)",
    "amax": "x.amax()",
    "argmax": "x.argmax()",
}
FLOAT_UNARY = {
    "rsqrt": "torch.rsqrt(x)",
    "reciprocal": "torch.reciprocal(x)",
    "log": "torch.log(x)",
    "log2": "torch.log2(x)",
    "log10": "torch.log10(x)",
    "log1p": "torch.log1p(x)",
    "expm1": "torch.expm1(x)",
    "sin": "torch.sin(x)",
    "cos": "torch.cos(x)",
    "tan": "torch.tan(x)",
    "asin": "torch.asin(x)",
    "acos": "torch.acos(x)",
    "atan": "torch.atan(x)",
    "sinh": "torch.sinh(x)",
    "cosh": "torch.cosh(x)",
    "tanh": "torch.tanh(x)",
    "asinh": "torch.asinh(x)",
    "acosh": "torch.acosh(x)",
    "atanh": "torch.atanh(x)",
    "erf": "torch.erf(x)",
    "pow_half": "torch.pow(x, 0.5)",
    "pow_neg1": "torch.pow(x, -1)",
    "exp2": "torch.pow(2.0, x)",
    "gelu": "F.gelu(x)",
    "gelu_tanh": 'F.gelu(x, approximate="tanh")',
    "elu": "F.elu(x)",
    "hardtanh": "F.hardtanh(x)",
    "leaky_relu": "F.leaky_relu(x)",
    "softplus": "F.softplus(x)",
    "hardsigmoid": "F.hardsigmoid(x)",
    "hardswish": "F.hardswish(x)",
    "silu": "F.silu(x)",
    "isnan": "torch.isnan(x)",
    "isinf": "torch.isinf(x)",
    "nan_to_num": "torch.nan_to_num(x, nan=0.5, posinf=9.0, neginf=-9.0)",
    "masked_fill": "x.masked_fill(x < 0, -7.0)",
    "to_int32": "x.to(torch.int32)",
    "to_int64": "x.to(torch.int64)",
    "to_uint8": "x.to(torch.uint8)",
    "mean": "x.mean()",
    "amin": "x.amin()",
    "max": "x.max()",
    "argmin": "x.argmin()",
    "softmax": "x.softmax(0)",
    "log_softmax": "x.log_softmax(0)",
    "var_mean": "torch.var_mean(x)",
    "std_mean": "torch.std_mean(x)",
}
INT_UNARY = {
    "bitwise_not": "torch.bitwise_not(x)",
    "shift_left_scalar": "torch.bitwise_left_shift(x, 1)",
    "shift_right_scalar": "torch.bitwise_right_shift(x, 1)",
    "to_int8": "x.to(torch.int8)",
    "prod": "x[:4].prod()",
    "mean_float": "x.float().mean()",
}
BINARY = {
    "add": "torch.add(a, b)",
    "sub": "torch.sub(a, b)",
    "mul": "torch.mul(a, b)",
    "div": "torch.div(a, b)",
    "div_floor": 'torch.div(a, b, rounding_mode="floor")',
    "div_trunc": 'torch.div(a, b, rounding_mode="trunc")',
    "floor_divide": "torch.floor_divide(a, b)",
    "fmod": "torch.fmod(a, b)",
    "remainder": "torch.remainder(a, b)",
    "pow": {"float": "torch.pow(a, b)", "int": "torch.pow(a, b.abs())"},
    "atan2": "torch.atan2(a, b)",
    "copysign": "torch.copysign(a, b)",
    "maximum": "torch.maximum(a, b)",
    "minimum": "torch.minimum(a, b)",
    "eq": "torch.eq(a, b)",
    "ne": "torch.ne(a, b)",
    "lt": "torch.lt(a, b)",
    "le": "torch.le(a, b)",
    "gt": "torch.gt(a, b)",
    "ge": "torch.ge(a, b)",
    "logical_and": "torch.logical_and(a, b)",
    "logical_or": "torch.logical_or(a, b)",
    "logical_xor": "torch.logical_xor(a, b)",
    "where": "torch.where(a > b, a, b)",
}
FLOAT_BINARY = {
    "clamp_tensor": "torch.clamp(a, min=b)",
    "lerp": "torch.lerp(a, b, 0.25)",
    "xlogy": "torch.xlogy(a, b)",
}
INT_BINARY = {
    "bitwise_and": "torch.bitwise_and(a, b)",
    "bitwise_or": "torch.bitwise_or(a, b)",
    "bitwise_xor": "torch.bitwise_xor(a, b)",
    "shift_left": "torch.bitwise_left_shift(a, b.abs())",
    "shift_right": "torch.bitwise_right_shift(a, b.abs())",
}
# Type promotion between different input dtypes.
MIXED = ("add", "sub", "mul", "div", "remainder", "floor_divide", "maximum", "eq")
# The right operand comes from INT_B, which has no zeros, so integer division is defined.
MIXED_PAIRS = (("f32", "i64"), ("i8", "u8"), ("bool", "i64"), ("i8", "i64"))

# Ops where an output element depends on more than the input element at the same index.
NOT_ELEMENTWISE = {"sum", "cumsum", "amax", "argmax", "mean", "amin", "max", "argmin", "softmax",
                   "log_softmax", "var_mean", "std_mean", "prod", "mean_float"}
# Float-to-integer casts. PyTorch leaves the result undefined for values out of range.
CASTS = {"to_int32": torch.int32, "to_int64": torch.int64, "to_uint8": torch.uint8}
# Ops whose result is a whole number, compared exactly instead of with a tolerance.
EXACT = {"floor", "ceil", "round", "trunc", "floor_divide", "div_floor", "div_trunc", "floor_divide_scalar",
         "div_scalar_floor", "div_scalar_trunc"}

_NAMESPACE = {"torch": torch, "F": F}


def make_fn(expr: str, arity: int) -> Callable[..., torch.Tensor]:
    params = "x" if arity == 1 else "a, b"
    return eval(f"lambda {params}: {expr}", dict(_NAMESPACE))  # noqa: S307 - expressions are ours or the user's


class Unary(torch.nn.Module):
    def __init__(self, fn: Callable[..., torch.Tensor]):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class Binary(Unary):
    def forward(self, a, b):
        return self.fn(a, b)


@dataclass(eq=False)
class Case:
    op: str
    expr: str
    dtype: str
    inputs: tuple[torch.Tensor, ...]
    elementwise: bool = True
    cast_to: torch.dtype | None = None
    exact: bool = False

    @property
    def id(self) -> str:
        return f"{self.op}/{self.dtype}"

    def module(self) -> torch.nn.Module:
        fn = make_fn(self.expr, len(self.inputs))
        return (Unary if len(self.inputs) == 1 else Binary)(fn)

    def subset(self, keep: torch.Tensor) -> Case:
        return Case(self.op, self.expr, self.dtype, tuple(t[keep] for t in self.inputs),
                    self.elementwise, self.cast_to, self.exact)


def float_tensor(values: Iterable[float], dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(list(values), dtype=torch.float64).to(dtype)


def int_tensor(values: Iterable[int], dtype: torch.dtype) -> torch.Tensor:
    if dtype == torch.uint8:
        values = [v % 256 for v in values]
    elif dtype == torch.bool:
        values = [v > 0 for v in values]
    return torch.tensor(list(values), dtype=torch.int64).to(dtype)


def _values(dtype: str, second: bool = False) -> torch.Tensor:
    if dtype in FLOAT_DTYPES:
        return float_tensor(FLOAT_B if second else FLOAT_A, DTYPES[dtype])
    return int_tensor(INT_B if second else INT_A, DTYPES[dtype])


def _expr(entry, kind: str) -> str:
    return entry if isinstance(entry, str) else entry[kind]


def op_table() -> dict[str, dict[str, str]]:
    """Expressions for every op, by kind ("float" / "int"), in report order."""
    table: dict[str, dict[str, str]] = {}
    for kinds, ops in ((("float", "int"), UNARY), (("float",), FLOAT_UNARY), (("int",), INT_UNARY),
                       (("float", "int"), BINARY), (("float",), FLOAT_BINARY), (("int",), INT_BINARY)):
        for op, entry in ops.items():
            table[op] = {kind: _expr(entry, kind) for kind in kinds}
    return table


def build_cases(dtypes: Iterable[str] = FLOAT_DTYPES + INT_DTYPES, mixed: bool = True) -> list[Case]:
    dtypes = list(dtypes)
    cases = []
    for dtype in dtypes:
        kind = "float" if dtype in FLOAT_DTYPES else "int"
        a, b = _values(dtype), _values(dtype, second=True)
        for op, exprs in op_table().items():
            if kind not in exprs:
                continue
            expr = exprs[kind]
            binary = op in BINARY or op in FLOAT_BINARY or op in INT_BINARY
            cases.append(Case(op, expr, dtype, (a, b) if binary else (a,),
                              elementwise=op not in NOT_ELEMENTWISE, cast_to=CASTS.get(op), exact=op in EXACT))
    if mixed:
        allowed = set(dtypes) | {"bool"}
        for left, right in MIXED_PAIRS:
            if left not in allowed or right not in allowed:
                continue
            a = _values(left)[:len(INT_A)] if left in FLOAT_DTYPES else int_tensor(INT_A, DTYPES[left])
            b = int_tensor(INT_B, DTYPES[right])
            for op in MIXED:
                cases.append(Case(op, _expr(BINARY[op], "float"), f"{left}+{right}", (a, b), exact=op in EXACT))
    return cases


def expression_case(expr: str, dtypes: Iterable[str]) -> list[Case]:
    """Cases for a user expression over ``x`` or ``a, b``."""
    names = set(compile(expr, "<expr>", "eval").co_names)
    binary = {"a", "b"} <= names
    if not binary and "x" not in names:
        raise ValueError("the expression must use x, or both a and b")
    cases = []
    for dtype in dtypes:
        a, b = _values(dtype), _values(dtype, second=True)
        cases.append(Case("expr", expr, dtype, (a, b) if binary else (a,), elementwise=False))
    return cases
