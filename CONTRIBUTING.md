# Contributing

Thanks for helping. Issues and pull requests are both welcome.

## Set up

```bash
git clone https://github.com/usmanmateen/torchparity
cd torchparity
pip install --extra-index-url https://download.pytorch.org/whl/cpu -e ".[dev,onnx]"
pytest
ruff check .
```

Add the extra for any other backend you work on (`openvino`, `coreml`, `executorch`, `tensorrt`).

## Add an op

Ops live in `src/torchparity/cases.py` as Python expressions over `x` (unary) or `a` and `b`
(binary). Pick the table that matches the dtypes the op accepts:

- `UNARY` / `BINARY`: floats and integers
- `FLOAT_UNARY` / `FLOAT_BINARY`: floats only
- `INT_UNARY` / `INT_BINARY`: integers only

If the op mixes elements (a reduction or a scan), add it to `NOT_ELEMENTWISE`. If its result is
always a whole number, add it to `EXACT`. Then run it on a backend to see that it behaves:

```bash
torchparity run inductor --ops your_op
```

## Add a backend

Subclass `torchparity.backends.Backend` and implement `compile(module, inputs)`; see the
README's "Adding a backend" section. Register it in `BACKENDS` in
`src/torchparity/backends/__init__.py`, add its extra to `pyproject.toml`, and add a job to
`.github/workflows/results.yml` if it runs on a hosted runner.

## Update the results

```bash
torchparity run <backend> -o results/<backend>.json
torchparity report results/*.json -o RESULTS.md --readme README.md
```

Use released versions of the backend packages, so anyone can reproduce the numbers. The
`Results` workflow does this weekly for every CPU backend; Torch-TensorRT results come from a
local GPU run.

## Link an upstream issue

When a difference is reported upstream, add it to `known_issues.toml` with the backend, the ops,
optionally the dtypes, and the issue or pull request. Set `found_here = true` if it was found
with torchparity; it then appears in the README.

## Reporting a difference upstream

A difference from eager PyTorch is not always a bug. Before opening an issue on a backend's
tracker:

1. Reproduce it with a standalone script that does not use torchparity.
2. Check the backend's documentation and its open and closed issues.
3. Report one difference per issue, with the script, the output, the expected output and the
   versions.

Please don't open issues in bulk from the results table.
