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
of zero.

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

- **Core ML** has no runtime outside macOS. On Linux the inputs are folded into the model as
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
