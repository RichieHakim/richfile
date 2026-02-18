#!/usr/bin/env python
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

import richfile as rf


class BenchmarkCustomPayload:
    """
    Custom payload object used to benchmark bridge-based custom type I/O.

    Args:
        value (int):
            Integer payload value.
        label (str):
            Short text label.
    """

    def __init__(self, value: int, label: str):
        self.value = value
        self.label = label

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BenchmarkCustomPayload):
            return False
        return (self.value == other.value) and (self.label == other.label)

    def __repr__(self) -> str:
        return f"BenchmarkCustomPayload(value={self.value}, label={self.label})"


def load_custom_payload(path: str) -> BenchmarkCustomPayload:
    """
    Load a ``BenchmarkCustomPayload`` from a JSON file path.
    """
    payload = json.loads(Path(path).read_text())
    return BenchmarkCustomPayload(value=int(payload["value"]), label=str(payload["label"]))


def save_custom_payload(path: str, obj: BenchmarkCustomPayload) -> None:
    """
    Save a ``BenchmarkCustomPayload`` as JSON to a file path.
    """
    Path(path).write_text(
        json.dumps(
            {
                "value": int(obj.value),
                "label": str(obj.label),
            }
        )
    )


def count_filesystem_entries(path: Path) -> int:
    """
    Count file-system entries under a target path.
    """
    if path.is_file():
        return 1
    if path.is_dir():
        return 1 + sum(1 for _ in path.rglob("*"))
    return 0


def compute_storage_size_bytes(path: Path) -> int:
    """
    Compute total file size for a target path.
    """
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(path_child.stat().st_size for path_child in path.rglob("*") if path_child.is_file())
    return 0


def backend_suffix(backend: str) -> str:
    """
    Map backend name to benchmark output suffix.
    """
    suffix_map = {
        "directory": ".richfile",
        "sqlar": ".sqlar",
        "zip": ".zip",
        "tar": ".tar",
    }
    if backend not in suffix_map:
        raise ValueError(f"Unsupported backend for benchmark: {backend}")
    return suffix_map[backend]


def load_by_lazy_path(richfile_obj: "rf.RichFile", lazy_path: Sequence[Any]) -> Any:
    """
    Resolve a lazy path (list of keys/indices) and load the final object.
    """
    node = richfile_obj
    for key in lazy_path:
        node = node[key]
    return node.load()


def resolve_python_path(payload: Any, lazy_path: Sequence[Any]) -> Any:
    """
    Resolve a key/index path directly on in-memory Python payload objects.
    """
    node = payload
    for key in lazy_path:
        node = node[key]
    return node


def apply_type_registrations(
    richfile_obj: "rf.RichFile",
    type_registrations: Sequence[Dict[str, Any]],
) -> None:
    """
    Register scenario-specific custom types on a ``RichFile`` instance.
    """
    for registration in type_registrations:
        richfile_obj.register_type(**registration)


def benchmark_backend(
    scenario: Dict[str, Any],
    backend: str,
    temp_dir: Path,
) -> Dict[str, Any]:
    """
    Run save/load/lazy benchmark for one scenario and backend.
    """
    path_target = temp_dir / f"bench_{scenario['name']}_{backend}{backend_suffix(backend=backend)}"
    type_registrations = scenario.get("type_registrations", [])

    richfile_saver = rf.RichFile(
        path=path_target,
        backend=backend,
        overwrite=True,
    )
    apply_type_registrations(
        richfile_obj=richfile_saver,
        type_registrations=type_registrations,
    )
    timer_start = time.perf_counter()
    richfile_saver.save(scenario["payload"])
    time_save_seconds = time.perf_counter() - timer_start

    richfile_loader = rf.RichFile(
        path=path_target,
        backend=backend,
    )
    apply_type_registrations(
        richfile_obj=richfile_loader,
        type_registrations=type_registrations,
    )
    timer_start = time.perf_counter()
    loaded = richfile_loader.load()
    time_load_seconds = time.perf_counter() - timer_start

    meta_num_items_expected = scenario["payload"]["meta"]["num_items"]
    if loaded["meta"]["num_items"] != meta_num_items_expected:
        raise AssertionError(
            f"Loaded payload mismatch for scenario='{scenario['name']}', backend='{backend}'."
        )

    lazy_times: List[float] = []
    for lazy_path in scenario["lazy_paths"]:
        richfile_lazy = rf.RichFile(
            path=path_target,
            backend=backend,
        )
        apply_type_registrations(
            richfile_obj=richfile_lazy,
            type_registrations=type_registrations,
        )
        timer_start = time.perf_counter()
        value_lazy = load_by_lazy_path(richfile_obj=richfile_lazy, lazy_path=lazy_path)
        time_lazy_seconds = time.perf_counter() - timer_start
        value_expected = resolve_python_path(
            payload=scenario["payload"],
            lazy_path=lazy_path,
        )
        if value_lazy != value_expected:
            raise AssertionError(
                "Lazy loaded value mismatch during benchmark.\n"
                f"Scenario: {scenario['name']}\n"
                f"Backend: {backend}\n"
                f"Lazy path: {lazy_path}\n"
                f"Expected: {value_expected}\n"
                f"Found: {value_lazy}"
            )
        lazy_times.append(time_lazy_seconds)

    return {
        "scenario": scenario["name"],
        "backend": backend,
        "time_save_seconds": time_save_seconds,
        "time_load_seconds": time_load_seconds,
        "time_lazy_seconds_median": statistics.median(lazy_times),
        "time_lazy_seconds_all": lazy_times,
        "size_bytes": compute_storage_size_bytes(path_target),
        "n_filesystem_entries": count_filesystem_entries(path_target),
    }


def summarize_runs(runs: Sequence[Dict[str, Any]], backends: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """
    Summarize median metrics per backend for one scenario.
    """
    summary: Dict[str, Dict[str, float]] = {}
    for backend in backends:
        group = [run for run in runs if run["backend"] == backend]
        summary[backend] = {
            "save_median_s": statistics.median(run["time_save_seconds"] for run in group),
            "load_median_s": statistics.median(run["time_load_seconds"] for run in group),
            "lazy_median_s": statistics.median(run["time_lazy_seconds_median"] for run in group),
            "size_median_bytes": statistics.median(run["size_bytes"] for run in group),
            "entries_median": statistics.median(run["n_filesystem_entries"] for run in group),
        }
    return summary


def gate_summary(summary: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """
    Evaluate benchmark acceptance gates relative to directory backend.
    """
    if "directory" not in summary:
        return {
            "gate_passed": True,
            "gate_details": {},
        }

    gate_details = {
        "zip_single_file": summary["zip"]["entries_median"] == 1 if "zip" in summary else None,
        "zip_save_faster_than_directory": (
            summary["zip"]["save_median_s"] < summary["directory"]["save_median_s"]
            if "zip" in summary
            else None
        ),
        "tar_single_file": summary["tar"]["entries_median"] == 1 if "tar" in summary else None,
        "tar_save_faster_than_directory": (
            summary["tar"]["save_median_s"] < summary["directory"]["save_median_s"]
            if "tar" in summary
            else None
        ),
        "sqlar_single_file": summary["sqlar"]["entries_median"] == 1 if "sqlar" in summary else None,
        "sqlar_save_faster_than_directory": (
            summary["sqlar"]["save_median_s"] < summary["directory"]["save_median_s"]
            if "sqlar" in summary
            else None
        ),
    }
    gate_values = [value for value in gate_details.values() if value is not None]
    return {
        "gate_passed": all(gate_values),
        "gate_details": gate_details,
    }


def parse_backends(backends_raw: str) -> List[str]:
    """
    Parse comma-separated backend names.
    """
    backends = [part.strip() for part in backends_raw.split(",") if part.strip() != ""]
    if len(backends) == 0:
        raise ValueError("No backends specified.")
    for backend in backends:
        backend_suffix(backend=backend)
    return backends


def build_scenario_high_leaf(num_leaves: int) -> Dict[str, Any]:
    """
    High-leaf nested tree with many small dict/list leaves.
    """
    payload = {
        "items": {
            f"leaf_{idx}": {
                "value": idx,
                "is_even": (idx % 2) == 0,
                "triple": [idx, idx + 1, idx + 2],
            }
            for idx in range(num_leaves)
        },
        "meta": {
            "num_items": num_leaves,
            "label": "high_leaf_nested",
        },
    }
    index_mid = max(0, num_leaves // 2)
    return {
        "name": "high_leaf_nested",
        "description": "Many dictionary leaves under one large nested subtree.",
        "payload": payload,
        "lazy_paths": [
            ["items", "leaf_0", "value"],
            ["items", f"leaf_{index_mid}", "triple", 2],
        ],
        "type_registrations": [],
    }


def build_scenario_deep_chain(depth: int) -> Dict[str, Any]:
    """
    Deep nested dict chain to stress metadata recursion depth.
    """
    node: Any = {
        "terminal": [11, 22, 33, 44],
    }
    for _ in range(depth):
        node = [node]
    payload = {
        "chain": node,
        "depth": depth,
        "meta": {
            "num_items": depth,
            "label": "deep_chain",
        },
    }
    return {
        "name": "deep_chain",
        "description": "Long nested list chain ending at a small terminal dict.",
        "payload": payload,
        "lazy_paths": [
            ["depth"],
            ["chain"] + ([0] * depth) + ["terminal", 3],
        ],
        "type_registrations": [],
    }


def build_scenario_wide_scalars(num_keys: int) -> Dict[str, Any]:
    """
    Flat/wide scalar-heavy dict to stress sibling cardinality.
    """
    payload = {
        "scalars": {
            f"k{idx:05d}": (
                idx
                if (idx % 4 == 0)
                else (
                    float(idx) + 0.25
                    if (idx % 4 == 1)
                    else (f"value_{idx}" if (idx % 4 == 2) else None)
                )
            )
            for idx in range(num_keys)
        },
        "meta": {
            "num_items": num_keys,
            "label": "wide_scalars",
        },
    }
    return {
        "name": "wide_scalars",
        "description": "One very wide scalar dict with limited nesting.",
        "payload": payload,
        "lazy_paths": [
            ["scalars", "k00000"],
            ["scalars", f"k{max(0, num_keys - 1):05d}"],
        ],
        "type_registrations": [],
    }


def build_scenario_mixed_containers(num_rows: int) -> Dict[str, Any]:
    """
    Mixed list/tuple/set/frozenset/dict workload.
    """
    payload = {
        "rows": [
            {
                "id": idx,
                "coords": (idx, idx + 1, idx + 2),
                "active": (idx % 3) == 0,
                "tags": [f"t{idx % 8}", f"t{(idx + 1) % 8}"],
                "id_set": set([idx, idx + 1]),
                "frozen_window": frozenset([idx, idx + 10, idx + 20]),
            }
            for idx in range(num_rows)
        ],
        "group_lookup": {
            f"group_{idx}": [idx, idx + 100, idx + 200]
            for idx in range(max(1, num_rows // 8))
        },
        "meta": {
            "num_items": num_rows,
            "label": "mixed_containers",
        },
    }
    index_mid = max(0, num_rows // 2)
    return {
        "name": "mixed_containers",
        "description": "Container-rich rows with tuple/set/frozenset values.",
        "payload": payload,
        "lazy_paths": [
            ["rows", 0, "coords", 1],
            ["rows", index_mid, "tags", 1],
        ],
        "type_registrations": [],
    }


def build_scenario_large_strings(num_chunks: int, chunk_size: int) -> Dict[str, Any]:
    """
    Fewer leaves with larger string payloads to stress byte throughput.
    """
    blobs = [("x" * chunk_size) + f":{idx}" for idx in range(num_chunks)]
    payload = {
        "blobs": blobs,
        "meta": {
            "num_items": num_chunks,
            "label": "large_strings",
        },
    }
    return {
        "name": "large_strings",
        "description": "Large string leaves with lower metadata complexity.",
        "payload": payload,
        "lazy_paths": [
            ["blobs", 0],
            ["blobs", max(0, num_chunks - 1)],
        ],
        "type_registrations": [],
    }


def build_scenario_custom_bridge(num_items: int) -> Dict[str, Any]:
    """
    Custom object workload to exercise save/load bridge behavior.
    """
    payload = {
        "custom_items": [
            BenchmarkCustomPayload(value=idx, label=f"custom_{idx}")
            for idx in range(num_items)
        ],
        "meta": {
            "num_items": num_items,
            "label": "custom_bridge",
        },
    }
    type_registration = {
        "type_name": "benchmark_custom_payload",
        "function_load": load_custom_payload,
        "function_save": save_custom_payload,
        "object_class": BenchmarkCustomPayload,
        "library": "python",
        "suffix": "benchmark_custom_payload",
        "versions_supported": [">=3", "<4"],
    }
    return {
        "name": "custom_bridge",
        "description": "Custom class leaves through path-based callback bridge.",
        "payload": payload,
        "lazy_paths": [
            ["custom_items", 0],
            ["custom_items", max(0, num_items - 1)],
        ],
        "type_registrations": [type_registration],
    }


def build_scenarios(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """
    Build scenario matrix based on ``--scenario-set``.
    """
    params_positive = {
        "num_leaves": args.num_leaves,
        "deep_depth": args.deep_depth,
        "wide_keys": args.wide_keys,
        "mixed_rows": args.mixed_rows,
        "blob_chunks": args.blob_chunks,
        "blob_size": args.blob_size,
        "custom_items": args.custom_items,
        "repeats": args.repeats,
    }
    for name_param, value_param in params_positive.items():
        if int(value_param) <= 0:
            raise ValueError(f"Argument `{name_param}` must be > 0. Found: {value_param}")

    scenario_high_leaf = build_scenario_high_leaf(num_leaves=args.num_leaves)
    scenario_deep_chain = build_scenario_deep_chain(depth=args.deep_depth)
    scenario_mixed = build_scenario_mixed_containers(num_rows=args.mixed_rows)
    scenario_large_strings = build_scenario_large_strings(
        num_chunks=args.blob_chunks,
        chunk_size=args.blob_size,
    )
    scenario_wide = build_scenario_wide_scalars(num_keys=args.wide_keys)
    scenario_custom = build_scenario_custom_bridge(num_items=args.custom_items)

    scenario_set_map = {
        "legacy": [scenario_high_leaf],
        "standard": [
            scenario_high_leaf,
            scenario_deep_chain,
            scenario_mixed,
            scenario_large_strings,
        ],
        "all": [
            scenario_high_leaf,
            scenario_deep_chain,
            scenario_mixed,
            scenario_large_strings,
            scenario_wide,
            scenario_custom,
        ],
    }
    if args.scenario_set not in scenario_set_map:
        raise ValueError(f"Unknown scenario_set: {args.scenario_set}")
    return scenario_set_map[args.scenario_set]


def parse_args() -> argparse.Namespace:
    """
    Parse CLI args for multi-scenario archive benchmarks.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark richfile archive backends across multiple workload scenarios."
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--backends", type=str, default="directory,sqlar,zip,tar")
    parser.add_argument(
        "--scenario-set",
        type=str,
        default="standard",
        choices=["legacy", "standard", "all"],
    )
    parser.add_argument("--num-leaves", type=int, default=1500)
    parser.add_argument("--deep-depth", type=int, default=100)
    parser.add_argument("--wide-keys", type=int, default=5000)
    parser.add_argument("--mixed-rows", type=int, default=1200)
    parser.add_argument("--blob-chunks", type=int, default=220)
    parser.add_argument("--blob-size", type=int, default=4096)
    parser.add_argument("--custom-items", type=int, default=500)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--enforce-gate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backends = parse_backends(backends_raw=args.backends)
    scenarios = build_scenarios(args=args)

    scenarios_output: List[Dict[str, Any]] = []
    runs_all: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as temp_dir:
        dir_temp = Path(temp_dir)
        for scenario in scenarios:
            runs_scenario: List[Dict[str, Any]] = []
            for _ in range(args.repeats):
                for backend in backends:
                    run = benchmark_backend(
                        scenario=scenario,
                        backend=backend,
                        temp_dir=dir_temp,
                    )
                    runs_scenario.append(run)
                    runs_all.append(run)

            summary_scenario = summarize_runs(
                runs=runs_scenario,
                backends=backends,
            )
            gate = gate_summary(summary=summary_scenario)
            scenarios_output.append(
                {
                    "name": scenario["name"],
                    "description": scenario["description"],
                    "summary": summary_scenario,
                    "gate_passed": gate["gate_passed"],
                    "gate_details": gate["gate_details"],
                    "runs": runs_scenario,
                }
            )

    summary_overall = summarize_runs(
        runs=runs_all,
        backends=backends,
    )
    gate_overall = gate_summary(summary=summary_overall)

    output = {
        "scenario_set": args.scenario_set,
        "repeats": args.repeats,
        "backends": backends,
        "scenarios": scenarios_output,
        "summary_overall": summary_overall,
        "gate_passed_all": all(scenario["gate_passed"] for scenario in scenarios_output),
        "gate_passed_overall": gate_overall["gate_passed"],
        "gate_details_overall": gate_overall["gate_details"],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2))

    print(json.dumps(output["summary_overall"], indent=2))
    print(
        "Scenario gate results: "
        + ", ".join(
            f"{scenario['name']}={scenario['gate_passed']}"
            for scenario in scenarios_output
        )
    )
    print(
        f"Overall gate passed: {output['gate_passed_overall']}. "
        f"Details: {output['gate_details_overall']}"
    )

    if args.enforce_gate and (not output["gate_passed_overall"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
