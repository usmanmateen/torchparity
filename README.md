# torchparity

[![CI](https://github.com/usmanmateen/torchparity/actions/workflows/ci.yml/badge.svg)](https://github.com/usmanmateen/torchparity/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)

**Does your exported model compute the same thing as your PyTorch model?**

torchparity runs about 120 PyTorch operators through **ONNX Runtime, OpenVINO, Core ML,
Torch-TensorRT, ExecuTorch and `torch.compile`** and compares every output element with eager
PyTorch. The inputs are the values where implementations tend to disagree: NaN, ±inf, -0.0, very
large and very small magnitudes, integer overflow, negative operands for division and remainder,
and mixed dtypes.

```console
$ torchparity run openvino --ops floor_divide_scalar,atan2 --dtypes f16,i32
OpenVINO: 4 cases (ov.convert_model(example_input=...), CPU, f32 inference precision)
❌ wrong       floor_divide_scalar/f16  x=3.0: eager 1.0, got 0.0
⚠️ edge        atan2/f16                a=-0.0, b=-3.0: eager -3.1406, got 3.1406
❌ wrong       floor_divide_scalar/i32  x=-1: eager -1, got 0
❌ wrong       atan2/i32                dtype i32, eager gives f32
✅ ok 0 · ❌ wrong 3 · ⚠️ edge 1 · 🚫 unsupported 0 · – skip 0
```

**[Full results for every operator and backend →](RESULTS.md)**

## Why

When you convert a model, a converter and a runtime reimplement each operator. Usually the
results match. When they don't, nothing fails: integer division truncates, `atan2` returns
integers, infinities turn into the largest finite float. You find out later from a wrong
prediction, far from the cause.

torchparity finds these differences one operator at a time, with a minimal example for each,
so you can tell whether your model is affected before you ship it.

## Results at a glance

<!-- summary:start -->
| Backend | Version | ✅ same | ❌ wrong | ⚠️ edge only | 🚫 fails |
|---|---|---|---|---|---|
| ONNX Runtime | onnxruntime 1.31.0 | 511 | 13 | 47 | 60 |
| OpenVINO | openvino 2026.4.1 | 366 | 28 | 104 | 25 |
| Core ML | coremltools 9.0 | 252 | 19 | 25 | 71 |
| Torch-TensorRT | torch-tensorrt 2.14.0+cu130 | 296 | 8 | 48 | 17 |
| ExecuTorch | executorch 1.5.1+cpu | 658 | 14 | 32 | 35 |
| torch.compile | torch 2.14.1+cpu | 716 | 3 | 20 | 0 |
<!-- summary:end -->

Counts are cases: one operator on one dtype. See [RESULTS.md](RESULTS.md) for every operator,
the examples behind each ❌ and ⚠️, and the full versions.

## Bugs found with it

<!-- found:start -->
| Project | Problem | Status |
|---|---|---|
| microsoft/onnxscript | Bitwise shifts on unsigned integers return a signed type | [![pull 3088](https://img.shields.io/github/pulls/detail/state/microsoft/onnxscript/3088)](https://github.com/microsoft/onnxscript/pull/3088) |
| microsoft/onnxscript | torch.clamp with one bound turns infinities into the largest finite value | [![pull 3089](https://img.shields.io/github/pulls/detail/state/microsoft/onnxscript/3089)](https://github.com/microsoft/onnxscript/pull/3089) |
| openvinotoolkit/openvino | torch.div(rounding_mode="trunc") overflows through int32; aten::trunc is not supported ([issue](https://github.com/openvinotoolkit/openvino/issues/38750)) | [![pull 38753](https://img.shields.io/github/pulls/detail/state/openvinotoolkit/openvino/38753)](https://github.com/openvinotoolkit/openvino/pull/38753) |
| openvinotoolkit/openvino | torch.atan2 on integer tensors returns integers instead of float32 ([issue](https://github.com/openvinotoolkit/openvino/issues/38751)) | [![pull 38752](https://img.shields.io/github/pulls/detail/state/openvinotoolkit/openvino/38752)](https://github.com/openvinotoolkit/openvino/pull/38752) |
| openvinotoolkit/openvino | x // d by a constant is one too small for exact multiples of d (float16, and some float32 divisors) ([issue](https://github.com/openvinotoolkit/openvino/issues/38832)) | [![pull 38833](https://img.shields.io/github/pulls/detail/state/openvinotoolkit/openvino/38833)](https://github.com/openvinotoolkit/openvino/pull/38833) |
| openvinotoolkit/openvino | CPU Mod and FloorMod return 0 instead of nearly the divisor for float32 multiples | [![issues 38929](https://img.shields.io/github/issues/detail/state/openvinotoolkit/openvino/38929)](https://github.com/openvinotoolkit/openvino/issues/38929) |
| pytorch/TensorRT | True division of int64 tensors or by a Python int truncates the result ([issue](https://github.com/pytorch/TensorRT/issues/4849)) | [![pull 4850](https://img.shields.io/github/pulls/detail/state/pytorch/TensorRT/4850)](https://github.com/pytorch/TensorRT/pull/4850) |
| pytorch/TensorRT | torch.fmod returns a tiny value with the wrong sign when the float quotient rounds up ([issue](https://github.com/pytorch/TensorRT/issues/4860)) | [![pull 4861](https://img.shields.io/github/pulls/detail/state/pytorch/TensorRT/4861)](https://github.com/pytorch/TensorRT/pull/4861) |
| pytorch/executorch | Portable remainder: wrong sign for integers, float64 rounded to float ([issue](https://github.com/pytorch/executorch/issues/23682)) | [![pull 23683](https://img.shields.io/github/pulls/detail/state/pytorch/executorch/23683)](https://github.com/pytorch/executorch/pull/23683) |
| pytorch/executorch | Portable floor_divide returns non-whole numbers for floats and wrong values for a zero divisor ([issue](https://github.com/pytorch/executorch/issues/23684)) | [![pull 23685](https://img.shields.io/github/pulls/detail/state/pytorch/executorch/23685)](https://github.com/pytorch/executorch/pull/23685) |
| pytorch/executorch | Optimized log_softmax does not support float64 | [![pull 23686](https://img.shields.io/github/pulls/detail/state/pytorch/executorch/23686)](https://github.com/pytorch/executorch/pull/23686) |
| pytorch/executorch | torch.var_mean and torch.std_mean fail in to_edge ([issue](https://github.com/pytorch/executorch/issues/23680)) | [![pull 23681](https://img.shields.io/github/pulls/detail/state/pytorch/executorch/23681)](https://github.com/pytorch/executorch/pull/23681) |
| apple/coremltools | float32 scalar constants on float16 inputs fail conversion in log2, log10, expm1, asinh, acosh, hardtanh and atan2 | [![pull 2894](https://img.shields.io/github/pulls/detail/state/apple/coremltools/2894)](https://github.com/apple/coremltools/pull/2894) |
<!-- found:end -->

## Install

torchparity needs Python 3.10+ and PyTorch. Install it with the extra for each backend you want
to check:

```bash
pip install "torchparity[openvino] @ git+https://github.com/usmanmateen/torchparity"
pip install "torchparity[onnx,coreml,executorch] @ git+https://github.com/usmanmateen/torchparity"
```

| Backend | Extra | What runs | Dtypes checked |
|---|---|---|---|
| `onnx` | `onnx` | `torch.onnx.export(dynamo=True)` → ONNX Runtime CPU | f32 f64 f16 i64 i32 i8 u8 |
| `openvino` | `openvino` | `ov.convert_model` → OpenVINO CPU plugin | f32 f16 i64 i32 i8 u8 |
| `coreml` | `coreml` | `ct.convert` → Core ML CPU on macOS, constant folding elsewhere | f32 f16 i64 i32 |
| `tensorrt` | `tensorrt` | `torch_tensorrt.compile(ir="dynamo")` → TensorRT, needs a CUDA GPU | f32 f16 i64 i32 |
| `executorch` | `executorch` | `to_edge(...).to_executorch()` → ExecuTorch CPU kernels | f32 f64 f16 bf16 i64 i32 i8 u8 |
| `inductor` | (none) | `torch.compile(fullgraph=True)` → Inductor CPU | f32 f64 f16 bf16 i64 i32 i8 u8 |

`torchparity backends` shows which are installed.

## Quick start

```bash
torchparity backends            # which backends can run here
torchparity run openvino        # every case on OpenVINO, about 500 cases
torchparity ops                 # the ops and their expressions
```

`run` prints one line per case that differs or fails, then a summary. Cases that match print
nothing.

## Usage

### Choose cases

```bash
torchparity run onnx --ops div,remainder            # exact op names, see `torchparity ops`
torchparity run tensorrt --dtypes f16,i64           # only these dtypes
torchparity run executorch -k floor -k remainder/i  # case ids (op/dtype) containing these
```

Case ids look like `remainder/i32`, or `add/f32+i64` for mixed-dtype promotion.

### Check your own expression

Use `x` for one input, or `a` and `b` for two. The expression runs on the same edge values:

```console
$ torchparity run onnx --expr "torch.special.erfc(x)" --dtypes f32,f64
ONNX Runtime: 2 cases (torch.onnx.export(dynamo=True), CPUExecutionProvider)
🚫 unsupported expr/f64                 convert: NotImplemented: [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for Erf(13) node with name 'n1'
✅ ok 1 · ❌ wrong 0 · ⚠️ edge 0 · 🚫 unsupported 1 · – skip 0
```

### Guard an upgrade in CI

`--fail-on` makes `run` exit with status 1 when any case has one of the given statuses. Pin the
ops your model uses and run it when you upgrade the converter or runtime:

```yaml
- run: pip install "torchparity[openvino] @ git+https://github.com/usmanmateen/torchparity"
- run: torchparity run openvino --ops div_floor,remainder,softmax,gelu --fail-on wrong
```

### Save results and build the tables

```bash
torchparity run onnx -o results/onnx.json
torchparity run openvino -o results/openvino.json
torchparity report results/*.json -o RESULTS.md --readme README.md
```

`report` writes the operator table and the details. `--known known_issues.toml` links cells to
upstream issues, and `--readme` refreshes the generated blocks in this README.

### Python API

```python
import torchparity

results = torchparity.check("openvino", ops=["floor_divide_scalar"], dtypes=["f16"])
for r in results:
    print(r.op, r.dtype, r.status, r.examples[0] if r.examples else "")
# floor_divide_scalar f16 wrong {'inputs': {'x': '3.0'}, 'eager': '1.0', 'backend': '0.0'}
```

`check` takes the same selectors as the CLI (`ops`, `dtypes`, `match`, `expr`) and returns one
`Result` per case with `op`, `dtype`, `expr`, `status`, `detail` and `examples`.

### Command reference

| Command | Does |
|---|---|
| `torchparity backends` | List backends and whether their packages are installed |
| `torchparity ops` | List ops, the dtype kinds they run on, and their expressions |
| `torchparity run BACKEND` | Run cases on one backend. Options: `--ops`, `--dtypes`, `-k`, `--expr`, `-o FILE`, `--fail-on`, `-v`, `--in-process` |
| `torchparity report FILES` | Build the Markdown report. Options: `-o FILE`, `--known FILE`, `--readme FILE` |

`BACKEND` can also be `module:Class` for a backend defined in your own code.

## Reading the results

| Status | Means |
|---|---|
| ✅ ok | Same result as eager PyTorch, same dtype and shape |
| ❌ wrong | Differs on ordinary values (finite, magnitude between 1e-6 and 1e6, no overflow), or the dtype or shape differs |
| ⚠️ edge | Differs only where an input or the expected output is NaN, ±inf, -0.0, very large or very small, or where integer arithmetic overflows |
| 🚫 unsupported | Conversion or execution fails, or the runtime crashes |
| – skip | Eager PyTorch rejects the case, or the backend cannot be checked |

A ❌ is the one to read first: it can change a real model's output. A ⚠️ matters if your model can
produce NaN, infinities or overflow. A 🚫 is loud, so at least you would notice.

**A difference is not always a bug.** PyTorch returns `0` for `sign(nan)` while most runtimes
return NaN, and some backends document narrower precision or dtypes. RESULTS.md links the
differences that have been reported upstream. Before reporting one, check the backend's
documentation and issue tracker, and report it on its own with a minimal repro.

## How it works

1. **Inputs.** Unary ops run on 24 float values or 16 integers. Binary ops pair them element by
   element with a second list, so each element tests one combination such as `-0.0 / -3.0`:

   ```
   a: 0.0  -0.0  0.5  -0.5  1.5  -1.5  2.5  -2.5  1e-30  -3.7  7.25  1e10  -1e10  123456789.12  inf  -inf  nan  1.0  -1.0  3.0  4.2  0.7  -4.2  0.3
   b: 3.0  -3.0  2.0  -2.0  0.75  0.75 -1.5  1.5  7.0    2.0   -0.5  3.0   3.0    1000.0        2.0  3.0   1.0  inf  -inf  0.0  1.4  0.1  1.4   -0.1
   integers a: 0 1 -1 7 -7 6 -6 100 -100 127 -128 5 -5 2 3 -3
   integers b: 3 3 3 3 3 -3 -3 7 7 -2 3 -5 5 1 -2 2
   ```

2. **Reference.** Eager PyTorch runs the same op on the same tensors, on the backend's device.
3. **Comparison.** Every output element is compared, with a tolerance per dtype (`rtol` 1e-5 for
   float32, 2e-3 for float16) and the sign of zero. Ops whose result is a whole number, such as
   `floor`, `round` and floor division, must match exactly. Float-to-integer casts of NaN,
   infinity or out-of-range values are skipped, because PyTorch leaves them undefined.
4. **Classification.** For elementwise ops, each differing element is checked: if any differs on
   ordinary values the case is ❌, otherwise ⚠️. Reductions such as `sum` and `softmax` mix
   elements, so they run again without the edge inputs to tell the two apart.
5. **Isolation.** Cases run in a worker process. If a runtime crashes, the case is retried in a
   fresh process and recorded as a crash only if it crashes again.

### Notes per backend

- **Core ML** has no runtime outside macOS. On Linux the inputs are folded into the model as
  constants, so the values come from coremltools' constant folding: this checks the converter,
  not the Core ML runtime. On macOS the model runs on the Core ML CPU runtime (experimental).
  float16 inputs are converted with float16 compute and the rest with float32, since float16
  compute changes float32 results by design. int64 is compared as int32, the only integer type
  Core ML has.
- **OpenVINO**: float64 is not checked, since the CPU plugin computes it in float32.
- **Torch-TensorRT**: the published results come from a local run on an RTX 5060 Ti; hosted CI
  has no GPU.

## Adding a backend

A backend converts a module for some inputs and returns a function that runs it:

```python
import myruntime
import torch
from torchparity.backends import Backend


class MyRuntime(Backend):
    name = "myruntime"
    label = "My Runtime"
    packages = ("torch", "myruntime")        # versions recorded with the results
    dtypes = ("f32", "f16", "i64", "i32")   # dtypes it supports

    def compile(self, module, inputs):
        model = myruntime.convert(module, inputs)   # raising here = conversion failure

        def run(args):
            # raising here = runtime failure
            return tuple(torch.from_numpy(o) for o in model(*[a.numpy() for a in args]))

        return run
```

Run it with `torchparity run mypackage.backends:MyRuntime`, or add it to `BACKENDS` in
`src/torchparity/backends/__init__.py` and open a pull request.

## FAQ

**How is this different from Polygraphy, NNSmith or PyTorch's own tests?**
[Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy) compares the outputs
of one model across runtimes, usually on random inputs. [NNSmith](https://github.com/ise-uiuc/nnsmith)
generates random models to fuzz compilers. PyTorch's OpInfo tests check PyTorch's own backends.
torchparity is narrower: a fixed set of operators on chosen edge values, across six backends at
once, with results published as a table you can look up before converting.

**Can it check a whole model?**
Not yet. It checks operators. For a small function, use `--expr`.

**Why does Core ML on Linux only check the converter?**
Core ML only runs on Apple platforms. Constant folding uses coremltools' own implementation of
each op, which is what runs whenever part of a model is constant.

## Contributing

Bug reports, new ops and new backends are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE)
