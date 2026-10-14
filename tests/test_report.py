from torchparity.report import render


def run(backend, label, results):
    return {"backend": backend, "label": label, "mode": "m", "versions": {"torch": "2"}, "results": results}


def result(op, dtype, status, examples=(), detail=""):
    return {"op": op, "dtype": dtype, "expr": "e", "status": status, "detail": detail, "examples": list(examples)}


def test_render_matrix_and_links():
    runs = [
        run("onnx", "ONNX Runtime", [result("div", "i64", "ok"), result("div", "i32", "ok")]),
        run("tensorrt", "Torch-TensorRT", [
            result("div", "i64", "wrong", [{"inputs": {"a": "1", "b": "3"}, "eager": "0.333", "backend": "0"}]),
            result("div", "i32", "ok"),
            result("sign", "f32", "edge", [{"inputs": {"x": "nan"}, "eager": "0.0", "backend": "nan"}]),
            result("trunc", "f32", "unsupported", detail="convert: AssertionError"),
        ]),
    ]
    runs[0]["cpu"], runs[0]["platform"] = "Some CPU", "Linux x86_64"
    runs[1]["gpu"] = "Some GPU"
    known = [{"backend": "tensorrt", "ops": ["div"], "dtypes": ["i64"], "issue": "https://github.com/o/r/issues/7"}]
    text = render(runs, known)
    assert "| ONNX Runtime | torch 2 | m | Some CPU, Linux x86_64 | 2 | 0 | 0 | 0 |" in text
    assert "| Torch-TensorRT | torch 2 | m | Some GPU | 1 | 1 | 1 | 1 |" in text
    assert "| `div` | `torch.div(a, b)` | ✅ | ❌ i64 [#7](https://github.com/o/r/issues/7) |" in text
    assert "| `sign` | `torch.sign(x)` |  | ⚠️ f32 |" in text
    assert "| `trunc` | `torch.trunc(x)` |  | 🚫 f32 |" in text
    assert "a=1, b=3: eager 0.333, got 0" in text
    assert "convert: AssertionError" in text


def test_render_found():
    from torchparity.report import render_found

    known = [
        {"title": "A | B", "pr": "https://github.com/o/r/pull/3", "issue": "https://github.com/o/r/issues/2",
         "found_here": True},
        {"title": "other", "issue": "https://github.com/o/r/issues/9", "found_here": False},
    ]
    text = render_found(known)
    assert "| o/r | A \\| B ([issue](https://github.com/o/r/issues/2)) | " in text
    assert "https://img.shields.io/github/pulls/detail/state/o/r/3" in text
    assert "other" not in text


def test_update_readme_blocks(tmp_path):
    from torchparity.report import update_readme

    readme = tmp_path / "README.md"
    readme.write_text("top\n<!-- found:start -->\nold\n<!-- found:end -->\n<!-- summary:start -->\n"
                      "<!-- summary:end -->\nbottom\n", encoding="utf-8")
    runs = [run("onnx", "ONNX Runtime", [result("div", "i64", "ok"), result("div", "i32", "wrong")])]
    runs[0]["versions"] = {"torch": "2", "onnxruntime": "1.31"}
    update_readme(readme, [], runs)
    text = readme.read_text(encoding="utf-8")
    assert "old" not in text and text.startswith("top\n") and text.endswith("bottom\n")
    assert "| ONNX Runtime | onnxruntime 1.31 | 1 | 1 | 0 | 0 |" in text
