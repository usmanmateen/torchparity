import math

import torch

from torchparity.compare import compare, edge_values, fmt


def test_fmt():
    assert fmt(math.nan, torch.float32) == "nan"
    assert fmt(-math.inf, torch.float32) == "-inf"
    assert fmt(-0.0, torch.float32) == "-0.0"
    assert fmt(1 / 3, torch.float32) == "0.333333333"
    assert fmt(1 / 3, torch.float16) == "0.33333"
    assert fmt(7, torch.int8) == "7"
    assert fmt(True, torch.bool) == "True"


def test_compare_values_within_tolerance():
    e = torch.tensor([1.0, math.nan, math.inf])
    assert compare(e, e + torch.tensor([1e-7, 0.0, 0.0])) is None
    diff = compare(e, torch.tensor([1.1, math.nan, math.inf]))
    assert diff.reason == "values" and diff.bad.tolist() == [True, False, False]


def test_compare_sign_of_zero():
    diff = compare(torch.tensor([-0.0]), torch.tensor([0.0]))
    assert diff is not None and diff.bad.tolist() == [True]


def test_compare_narrowed_dtype():
    e = torch.tensor([1, 2], dtype=torch.int64)
    a = torch.tensor([1, 2], dtype=torch.int32)
    assert compare(e, a) is not None
    assert compare(e, a, {torch.int64: torch.int32}) is None


def test_compare_output_count_and_shape():
    assert compare((torch.zeros(2),), (torch.zeros(2), torch.zeros(2))).reason == "2 outputs, eager has 1"
    assert compare(torch.zeros(()), torch.zeros(1)).reason == "shape (1,), eager gives ()"


def test_edge_values():
    t = torch.tensor([0.0, -0.0, 1.0, 1e6, 1e-7, math.nan, -math.inf])
    assert edge_values(t).tolist() == [False, True, False, True, True, True, True]
    assert edge_values(torch.tensor([5, 1_000_000])).tolist() == [False, True]


def test_exact_compare_for_whole_number_results():
    e = torch.tensor([3.0], dtype=torch.float64)
    a = torch.tensor([2.9999999999999996], dtype=torch.float64)
    assert compare(e, a) is None
    assert compare(e, a, exact=True) is not None
