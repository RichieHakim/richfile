#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import richfile as rf


def build_workload(num_leaves: int) -> Dict[str, Any]:
    """
    Build a high-leaf nested payload to stress save/load metadata traversal.
    """
    return {
        "items": {
            f"leaf_{idx}": {
                "value": idx,
                "is_even": (idx % 2) == 0,
                "triple": [idx, idx + 1, idx + 2],
            }
            for idx in range(num_leaves)
        },
        "meta": {
            "num_leaves": num_leaves,
            "label": "sqlar-benchmark",
        },
    }


def count_filesystem_entries(path: Path) -> int:
    if path.is_file():
        return 1
    if path.is_dir():
        return 1 + sum(1 for _ in path.rglob("*"))
    return 0


def compute_storage_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return 0


def benchmark_backend(
    payload: Dict[str, Any],
    backend: str,
    temp_dir: Path,
) -> Dict[str, Any]:
    suffix = ".sqlar" if backend == "sqlar" else ".richfile"
    path_target = temp_dir / f"bench_{backend}{suffix}"

    timer_start = time.perf_counter()
    rf.RichFile(
        path=path_target,
        backend=backend,
        overwrite=True,
    ).save(payload)
    time_save_seconds = time.perf_counter() - timer_start

    timer_start = time.perf_counter()
    loaded = rf.RichFile(
        path=path_target,
        backend=backend,
    ).load()
    time_load_seconds = time.perf_counter() - timer_start

    if loaded["meta"]["num_leaves"] != payload["meta"]["num_leaves"]:
        raise AssertionError("Loaded payload mismatch during benchmark.")

    timer_start = time.perf_counter()
    lazy_value = rf.RichFile(
        path=path_target,
        backend=backend,
    )["items"]["leaf_0"]["value"].load()
    time_lazy_seconds = time.perf_counter() - timer_start
    if lazy_value != payload["items"]["leaf_0"]["value"]:
        raise AssertionError("Lazy loaded value mismatch during benchmark.")

    return {
        "backend": backend,
        "time_save_seconds": time_save_seconds,
        "time_load_seconds": time_load_seconds,
        "time_lazy_seconds": time_lazy_seconds,
        "size_bytes": compute_storage_size_bytes(path_target),
        "n_filesystem_entries": count_filesystem_entries(path_target),
    }


def summarize_runs(runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    summary = {}
    for backend in ["directory", "sqlar"]:
        group = [run for run in runs if run["backend"] == backend]
        summary[backend] = {
            "save_median_s": statistics.median(run["time_save_seconds"] for run in group),
            "load_median_s": statistics.median(run["time_load_seconds"] for run in group),
            "lazy_median_s": statistics.median(run["time_lazy_seconds"] for run in group),
            "size_median_bytes": statistics.median(run["size_bytes"] for run in group),
            "entries_median": statistics.median(run["n_filesystem_entries"] for run in group),
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark richfile directory vs SQLAR backend.")
    parser.add_argument("--num-leaves", type=int, default=1500)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--enforce-gate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_workload(num_leaves=args.num_leaves)

    runs = []
    with tempfile.TemporaryDirectory() as temp_dir:
        dir_temp = Path(temp_dir)
        for _ in range(args.repeats):
            runs.append(
                benchmark_backend(
                    payload=payload,
                    backend="directory",
                    temp_dir=dir_temp,
                )
            )
            runs.append(
                benchmark_backend(
                    payload=payload,
                    backend="sqlar",
                    temp_dir=dir_temp,
                )
            )

    summary = summarize_runs(runs=runs)
    gate_passed = (
        summary["sqlar"]["entries_median"] == 1
        and summary["sqlar"]["save_median_s"] < summary["directory"]["save_median_s"]
    )
    gate_details = {
        "sqlar_single_file": summary["sqlar"]["entries_median"] == 1,
        "sqlar_save_faster_than_directory": (
            summary["sqlar"]["save_median_s"] < summary["directory"]["save_median_s"]
        ),
    }

    output = {
        "num_leaves": args.num_leaves,
        "repeats": args.repeats,
        "summary": summary,
        "gate_passed": gate_passed,
        "gate_details": gate_details,
        "runs": runs,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2))

    print(json.dumps(output["summary"], indent=2))
    print(f"Gate passed: {gate_passed}. Details: {gate_details}")

    if args.enforce_gate and (not gate_passed):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
