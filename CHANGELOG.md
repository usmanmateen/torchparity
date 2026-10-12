# Changelog

## 0.1.0 (unreleased)

First version.

- About 120 operators over float, integer and mixed dtypes, on edge values.
- Backends: ONNX Runtime, OpenVINO, Core ML, Torch-TensorRT, ExecuTorch and `torch.compile`.
- Each difference is classified as wrong on ordinary values, different only on edge values, or a
  conversion or runtime failure.
- Cases run in a worker process, so a runtime crash fails one case.
- `torchparity run`, `report`, `ops` and `backends` commands, and `torchparity.check()`.
- Markdown report with per-operator results, examples and links to upstream issues.
