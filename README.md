# torchparity

Does your exported model compute the same thing as your PyTorch model?

torchparity runs about 120 PyTorch operators through **ONNX Runtime, OpenVINO, Core ML,
Torch-TensorRT, ExecuTorch and `torch.compile`**, then compares every output element with eager
PyTorch. The inputs are the values where implementations tend to disagree: NaN, ±inf, -0.0,
very large and very small magnitudes, integer overflow, negative operands for division and
remainder, and mixed dtypes.

**[Current results →](RESULTS.md)**

## Why

When you convert a model, each operator is reimplemented by the converter and the runtime.
Usually the results match. When they don't, nothing fails. Integer division truncates,
`atan2` returns integers, and infinities turn into the largest finite float. You see it later
as a wrong prediction, far from the cause.

## Bugs found with it

<!-- found:start -->
| Project | Problem | Status |
|---|---|---|
| microsoft/onnxscript | Bitwise shifts on unsigned integers return a signed type | [![pull 3088](https://img.shields.io/github/pulls/detail/state/microsoft/onnxscript/3088)](https://github.com/microsoft/onnxscript/pull/3088) |
| microsoft/onnxscript | torch.clamp with one bound turns infinities into the largest finite value | [![pull 3089](https://img.shields.io/github/pulls/detail/state/microsoft/onnxscript/3089)](https://github.com/microsoft/onnxscript/pull/3089) |
| openvinotoolkit/openvino | torch.div(rounding_mode="trunc") overflows through int32; aten::trunc is not supported ([issue](https://github.com/openvinotoolkit/openvino/issues/38750)) | [![pull 38753](https://img.shields.io/github/pulls/detail/state/openvinotoolkit/openvino/38753)](https://github.com/openvinotoolkit/openvino/pull/38753) |
| openvinotoolkit/openvino | torch.atan2 on integer tensors returns integers instead of float32 ([issue](https://github.com/openvinotoolkit/openvino/issues/38751)) | [![pull 38752](https://img.shields.io/github/pulls/detail/state/openvinotoolkit/openvino/38752)](https://github.com/openvinotoolkit/openvino/pull/38752) |
| pytorch/TensorRT | True division of int64 tensors or by a Python int truncates the result ([issue](https://github.com/pytorch/TensorRT/issues/4849)) | [![pull 4850](https://img.shields.io/github/pulls/detail/state/pytorch/TensorRT/4850)](https://github.com/pytorch/TensorRT/pull/4850) |
| pytorch/executorch | Portable remainder: wrong sign for integers, float64 rounded to float ([issue](https://github.com/pytorch/executorch/issues/23682)) | [![pull 23683](https://img.shields.io/github/pulls/detail/state/pytorch/executorch/23683)](https://github.com/pytorch/executorch/pull/23683) |
| pytorch/executorch | Portable floor_divide returns non-whole numbers for floats and wrong values for a zero divisor ([issue](https://github.com/pytorch/executorch/issues/23684)) | [![pull 23685](https://img.shields.io/github/pulls/detail/state/pytorch/executorch/23685)](https://github.com/pytorch/executorch/pull/23685) |
| pytorch/executorch | Optimized log_softmax does not support float64 | [![pull 23686](https://img.shields.io/github/pulls/detail/state/pytorch/executorch/23686)](https://github.com/pytorch/executorch/pull/23686) |
| pytorch/executorch | torch.var_mean and torch.std_mean fail in to_edge ([issue](https://github.com/pytorch/executorch/issues/23680)) | [![pull 23681](https://img.shields.io/github/pulls/detail/state/pytorch/executorch/23681)](https://github.com/pytorch/executorch/pull/23681) |
| apple/coremltools | float32 scalar constants on float16 inputs fail conversion in log2, log10, expm1, asinh, acosh, hardtanh and atan2 | [![pull 2894](https://img.shields.io/github/pulls/detail/state/apple/coremltools/2894)](https://github.com/apple/coremltools/pull/2894) |
<!-- found:end -->

## Install

```bash
pip install "torchparity @ git+https://github.com/usmanmateen/torchparity"
```

Then install the backend you want to check. `torchparity backends` shows which are available.

| Backend | Packages | What runs |
|---|---|---|
| `onnx` | `onnxscript onnx onnxruntime` | `torch.onnx.export(dynamo=True)`, ONNX Runtime CPU |
| `openvino` | `openvino` | `ov.convert_model`, CPU plugin |
| `coreml` | `coremltools` | `ct.convert`; Core ML CPU on macOS, constant folding elsewhere (see below) |
| `tensorrt` | `torch-tensorrt` | `torch_tensorrt.compile(ir="dynamo")`, needs a CUDA GPU |
| `executorch` | `executorch` | `to_edge(...).to_executorch()`, default CPU kernels |
| `inductor` | (none) | `torch.compile(fullgraph=True)`, CPU |

## Usage

```bash
torchparity run openvino                          # every case
torchparity run onnx -k div -k remainder          # only cases whose id contains "div" or "remainder"
torchparity run tensorrt --dtypes f16,i64         # only these dtypes
torchparity run coreml --expr "torch.special.erfc(x)"   # your own expression over x, or a and b
torchparity run onnx -o results/onnx.json         # save results
torchparity report results/*.json -o RESULTS.md   # build the tables
```

`--fail-on wrong` makes `run` exit with status 1 when a case gives a wrong result, so it can
guard a converter or runtime upgrade in CI.

## How it works

Unary ops run on 24 float values (16 for integers). Binary ops pair them with a second list, so
each element tests one combination, such as `-0.0 / -3.0` or `inf % 2.0`. Each case is one op on one
dtype. The expected values come from eager PyTorch on the same inputs. Floating-point outputs are
compared with a tolerance per dtype (`rtol` 1e-5 for float32, 2e-3 for float16), including the sign
of zero. Ops whose result is a whole number, such as `floor`, `round` and floor division, must match
exactly.

A case that differs is either:

- ❌ **wrong**: it differs on ordinary values (finite, magnitude between 1e-6 and 1e6, no overflow),
  or returns a different dtype or shape.
- ⚠️ **edge**: it differs only where an input or the expected output is NaN, ±inf, -0.0, very large
  or very small, or where integer arithmetic overflows. Reductions such as `sum` and `softmax` are
  run again without the edge inputs to tell the two apart.
- 🚫 **unsupported**: conversion or execution fails, which at least is loud.

Float-to-integer casts of NaN, infinity or out-of-range values are not compared, because PyTorch
leaves their result undefined.

A difference from eager PyTorch is not always a bug. PyTorch returns `0` for `sign(nan)` while most
runtimes return NaN, and some backends document a narrower precision. The results link the
differences that have been reported upstream; for the rest, check the backend's documentation and
issue tracker before reporting, and please report them one at a time with a minimal repro.

### Notes per backend

- **Core ML** models are converted with float16 compute precision for float16 inputs and float32
  for the rest, since float16 compute changes float32 results by design. Core ML has no runtime
  outside macOS. On Linux the inputs are folded into the model as
  constants, so the values come from coremltools' constant folding: this checks the converter,
  not the Core ML runtime. On macOS the model runs on the Core ML CPU runtime (experimental).
  int64 is compared as int32, the only integer type Core ML has.
- **OpenVINO**: float64 is not checked, since the CPU plugin computes it in float32.
- **Torch-TensorRT**: results come from a local run on an RTX 5060 Ti; hosted CI has no GPU.

## Adding a backend

Subclass `torchparity.backends.Backend`, implement `compile(module, inputs)` to return a function
that runs the converted module, and add it to `BACKENDS` in `torchparity/backends/__init__.py`.
Raising in `compile` counts as a conversion failure, raising in the returned function as a runtime
failure.

## License

MIT
