import math

import pytest
import torch

from torchparity.backends import Backend
from torchparity.cases import Case, build_cases, expression_case, float_tensor, int_tensor
from torchparity.runner import check


class Eager(Backend):
    name = "eager"
    dtypes = ("f64", "f32", "f16", "bf16", "i64", "i32", "i8", "u8")

    def __init__(self, change=None, fail=None):
        self.change = change
        self.fail = fail

    def compile(self, module, inputs):
        if self.fail == "convert":
            raise RuntimeError("cannot convert")

        def run(args):
            if self.fail == "run":
                raise RuntimeError("cannot run")
            with torch.no_grad():
                out = module(*args)
            return (self.change(out, args),) if self.change else out

        return run


def case(expr, values, dtype=torch.float32, second=None, elementwise=True, cast_to=None):
    make = float_tensor if dtype.is_floating_point else int_tensor
    inputs = (make(values, dtype),) if second is None else (make(values, dtype), make(second, dtype))
    return Case("op", expr, "x", inputs, elementwise, cast_to)


def test_every_case_matches_itself():
    results = [check(Eager(), c) for c in build_cases()]
    assert {r.status for r in results} <= {"ok", "skip"}
    assert sum(r.status == "ok" for r in results) > 600


def test_case_ids_are_unique():
    ids = [c.id for c in build_cases()]
    assert len(ids) == len(set(ids))


def test_wrong_on_ordinary_values():
    r = check(Eager(lambda out, args: out + 1), case("x * 2", [1.0, 2.0, math.nan]))
    assert r.status == "wrong"
    assert r.examples[0] == {"inputs": {"x": "1.0"}, "eager": "2.0", "backend": "3.0"}


def test_edge_when_only_nan_differs():
    r = check(Eager(lambda out, args: torch.nan_to_num(out)), case("x * 2", [1.0, math.nan, math.inf]))
    assert r.status == "edge"


def test_edge_when_output_overflows_the_integer_type():
    # 100 * 2 = 200 wraps to -56 in int8; a backend that saturates differs only there.
    r = check(Eager(lambda out, args: torch.clamp(args[0].to(torch.int64) * 2, -128, 127).to(torch.int8)),
              case("x * 2", [1, 100], torch.int8))
    assert r.status == "edge"


def test_edge_when_only_the_sign_of_zero_differs():
    r = check(Eager(lambda out, args: out.abs()), case("x * -1", [0.0, 0.0]))
    assert r.status == "edge"


def test_undefined_casts_are_not_compared():
    r = check(Eager(lambda out, args: torch.zeros_like(out)),
              case("x.to(torch.int32)", [math.nan, 1e10], cast_to=torch.int32))
    assert r.status == "ok"


def test_dtype_difference_is_wrong():
    r = check(Eager(lambda out, args: out.double()), case("x + 1", [1.0]))
    assert r.status == "wrong"
    assert r.detail == "dtype f64, eager gives f32"


def test_reduction_rerun_on_ordinary_inputs():
    # A sum that ignores NaN differs only because of the NaN input.
    nansum = Eager(lambda out, args: torch.nansum(args[0]))
    assert check(nansum, case("x.sum()", [1.0, math.nan], elementwise=False)).status == "edge"
    plus_one = Eager(lambda out, args: args[0].nansum() + 1)
    assert check(plus_one, case("x.sum()", [1.0, math.nan], elementwise=False)).status == "wrong"


@pytest.mark.parametrize("fail", ["convert", "run"])
def test_failures_are_unsupported(fail):
    r = check(Eager(fail=fail), case("x + 1", [1.0]))
    assert r.status == "unsupported"
    assert r.detail == f"{fail}: RuntimeError: cannot {fail}"


def test_eager_error_is_skipped():
    r = check(Eager(), case("torch.bitwise_not(x)", [1.0]))
    assert r.status == "skip"


def test_expression_case():
    cases = expression_case("torch.hypot(a, b)", ["f32"])
    assert len(cases) == 1 and len(cases[0].inputs) == 2
    assert check(Eager(), cases[0]).status == "ok"
    with pytest.raises(ValueError):
        expression_case("torch.zeros(3)", ["f32"])
