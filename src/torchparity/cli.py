from __future__ import annotations

import argparse
import importlib
import sys
from collections import Counter
from pathlib import Path

from . import __version__

ICONS = {"ok": "✅", "wrong": "❌", "edge": "⚠️", "unsupported": "🚫", "skip": "–"}


def _backends(args) -> int:
    from .backends import BACKENDS

    for name, target in BACKENDS.items():
        try:
            importlib.import_module(target.split(":")[0])
            state = "available"
        except Exception as exc:  # missing package or broken install
            state = f"not available ({type(exc).__name__}: {str(exc).splitlines()[0][:80] if str(exc) else ''})"
        print(f"{name:12s} {state}")
    return 0


def _ops(args) -> int:
    from .cases import op_table

    for op, exprs in op_table().items():
        kinds = "/".join(exprs)
        shown = " | ".join(dict.fromkeys(exprs.values()))
        print(f"{op:20s} {kinds:10s} {shown}")
    return 0


def _describe(result) -> str:
    if result.examples:
        ex = result.examples[0]
        inputs = ", ".join(f"{k}={v}" for k, v in ex.get("inputs", {}).items())
        return f"{inputs + ': ' if inputs else ''}eager {ex['eager']}, got {ex['backend']}"
    return result.detail


def _run(args) -> int:
    from .backends import get_backend
    from .runner import run, run_isolated, save, select_cases

    backend = get_backend(args.backend)
    try:
        cases = select_cases(backend, dtypes=args.dtypes.split(",") if args.dtypes else None,
                             ops=args.ops.split(",") if args.ops else None, expr=args.expr, match=args.k)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    print(f"{backend.label}: {len(cases)} case{'' if len(cases) == 1 else 's'} ({backend.mode})", flush=True)

    def progress(case, result):
        if result.status != "ok" and (args.verbose or result.status in ("wrong", "edge", "unsupported")):
            print(f"{ICONS[result.status]} {result.status:11s} {case.id:24s} {_describe(result)}", flush=True)

    if args.in_process:
        results = run(backend, cases, progress)
    else:
        results = run_isolated(backend.name, cases, progress)
    counts = Counter(r.status for r in results)
    print(" · ".join(f"{ICONS[s]} {s} {counts[s]}" for s in ("ok", "wrong", "edge", "unsupported", "skip")))
    if args.output:
        save(args.output, backend, results)
        print(f"saved {args.output}")
    if args.fail_on and any(counts[s] for s in args.fail_on):
        return 1
    return 0


def _report(args) -> int:
    from .report import load_known, load_results, render, update_readme

    known = load_known(args.known)
    runs = load_results(args.results)
    text = render(runs, known)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(text)
    if args.readme:
        update_readme(args.readme, known, runs)
        print(f"updated {args.readme}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="torchparity", description=__doc__ or
                                     "Check converters and runtimes against eager PyTorch on edge values.")
    parser.add_argument("--version", action="version", version=f"torchparity {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("backends", help="list backends and whether their packages are installed")
    p.set_defaults(func=_backends)

    p = sub.add_parser("ops", help="list the ops, the dtype kinds they run on, and their expressions")
    p.set_defaults(func=_ops)

    p = sub.add_parser("run", help="run the cases on one backend")
    p.add_argument("backend")
    p.add_argument("--ops", help="comma-separated op names, e.g. div,remainder (see `torchparity ops`)")
    p.add_argument("-k", action="append", help="only cases whose id (op/dtype) contains this; repeatable")
    p.add_argument("--dtypes", help="comma-separated dtypes, e.g. f32,i64 (default: all the backend checks)")
    p.add_argument("--expr", help="check your own expression over x, or over a and b, e.g. 'torch.special.erfc(x)'")
    p.add_argument("-o", "--output", help="save results as JSON")
    p.add_argument("--fail-on", nargs="+", choices=("wrong", "edge", "unsupported"),
                   help="exit with status 1 if any case has one of these statuses")
    p.add_argument("-v", "--verbose", action="store_true", help="also print skipped cases")
    p.add_argument("--in-process", action="store_true",
                   help="run cases in this process; by default a worker process runs them, so a crash fails one case")
    p.set_defaults(func=_run)

    p = sub.add_parser("report", help="write a Markdown report from saved results")
    p.add_argument("results", nargs="+")
    p.add_argument("--known", default="known_issues.toml", help="TOML file linking cases to issues")
    p.add_argument("-o", "--output")
    p.add_argument("--readme", help="also refresh the table of found issues in this README")
    p.set_defaults(func=_report)

    args = parser.parse_args(argv)
    return args.func(args)
