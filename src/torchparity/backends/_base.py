from __future__ import annotations

import contextlib
import importlib.metadata
import logging
import os
import sys
import tempfile
import warnings
from collections.abc import Callable, Iterator

import torch

Runner = Callable[[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]]


class NotChecked(Exception):
    """The backend cannot produce a result to compare for this case."""


class Backend:
    name = ""
    label = ""
    # Packages whose versions are recorded with the results.
    packages: tuple[str, ...] = ("torch",)
    dtypes: tuple[str, ...] = ("f32", "f16", "i64", "i32")
    device = "cpu"
    # PyTorch dtype -> the dtype this backend returns instead, by design.
    narrows: dict[torch.dtype, torch.dtype] = {}
    mode = ""

    def compile(self, module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> Runner:
        """Convert ``module`` for ``inputs`` and return a function that runs it.

        Raising here counts as a conversion failure; raising in the returned function
        counts as a runtime failure.
        """
        raise NotImplementedError

    def versions(self) -> dict[str, str]:
        found = {}
        for package in self.packages:
            try:
                found[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        return found


@contextlib.contextmanager
def quiet() -> Iterator[None]:
    """Silence Python and native (C/C++) output from converters."""
    sys.stdout.flush()
    sys.stderr.flush()
    with tempfile.TemporaryFile() as sink, open(os.devnull, "w") as null, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.disable(logging.CRITICAL)
        saved = []
        try:
            for fd in (1, 2):
                try:
                    copy = os.dup(fd)
                except OSError:
                    continue
                saved.append((fd, copy))
                os.dup2(sink.fileno(), fd)
            with contextlib.redirect_stdout(null), contextlib.redirect_stderr(null):
                yield
        finally:
            for fd, copy in saved:
                os.dup2(copy, fd)
                os.close(copy)
            logging.disable(logging.NOTSET)
