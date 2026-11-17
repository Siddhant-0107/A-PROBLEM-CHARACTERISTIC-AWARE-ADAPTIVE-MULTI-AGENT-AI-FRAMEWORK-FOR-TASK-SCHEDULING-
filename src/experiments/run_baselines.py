"""Reproducible benchmark runner for resource-constrained rule-based baselines.

The runner compares FCFS, SPT and Critical Path on the same JSON instances.
Scheduler runtime is timed separately from validation and metric calculation.
HEFT is intentionally excluded because it uses a different problem model.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Callable

from src.baselines.rule_based import (
    schedule_critical_path,
    schedule_fcfs,
    schedule_spt,
)
from src.evaluation.schedule_evaluator import evaluate_schedule

ALGORITHMS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "FCFS": schedule_fcfs,
    "SPT": schedule_spt,
    "CRITICAL_PATH": schedule_critical_path,
}
METRIC_NAMES = (
    "makespan",
    "total_resource_cost",
    "total_tardiness",
    "resource_utilization",
)
CHARACTERISTIC_NAMES = (
    "task_count",
    "resource_count",
    "deadline_tightness",
    "resource_scarcity_proxy",
    "cost_variation_cv",
    "dependency_density",
    "capacity_factor",
)


def load_instances(dataset_dir: Path) -> list[dict[str, Any]]:
    """Load and validate all instance JSON files from dataset_dir/instances."""
    instance_dir = dataset_dir / "instances"
    if not instance_dir.is_dir():
        raise FileNotFoundError(
            f"Expected an instances directory at {instance_dir}. "
            "Generate a dataset first with src/generators/scheduling_generator.py."
        )
    paths = sorted(instance_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"No .json scheduling instances found in {instance_dir}.")
    instances = []
    seen_ids = set()
    for path in paths:
        try:
            instance = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read instance file {path}: {exc}") from exc
        if not isinstance(instance, dict) or not isinstance(instance.get("tasks"), list):
            raise ValueError(f"{path} is not a valid scheduling instance object.")
        instance_id = instance.get("instance_id", path.stem)
        if instance_id in seen_ids:
            raise ValueError(f"Duplicate instance_id: {instance_id}")
        instance["_benchmark_source_file"] = path.name
        instance["_benchmark_instance_id"] = instance_id
        seen_ids.add(instance_id)
        instances.append(instance)
    return instances


def _safe_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "stddev": None}
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _summarise(records: list[dict[str, Any]], group_by: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        key = tuple(record.get(field) for field in group_by)
        groups.setdefault(key, []).append(record)

    summaries = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        successful = [row for row in group if row["status"] == "ok"]
        summary: dict[str, Any] = {
            field: value for field, value in zip(group_by, key)
        }
        summary.update({
            "instances": len(group),
            "successful_runs": len(successful),
            "failed_runs": len(group) - len(successful),
            "success_rate": len(successful) / len(group) if group else 0.0,
        })
        for metric in (*METRIC_NAMES, "scheduler_runtime_ms"):
            values = [float(row[metric]) for row in successful if row.get(metric) is not None]
            stats = _safe_stats(values)
            for stat_name, value in stats.items():
                summary[f"{metric}_{stat_name}"] = value
        summaries.append(summary)
    return summaries


def run_benchmark(instances: list[dict[str, Any]]) -> dict[str, Any]:
    """Run every baseline on every supplied instance and aggregate the results."""
    if not instances:
        raise ValueError("At least one instance is required.")
    records: list[dict[str, Any]] = []

    for instance in instances:
        instance_id = instance.get("_benchmark_instance_id", instance.get("instance_id", "unknown"))
        source_file = instance.get("_benchmark_source_file")
        profile = instance.get("profile", "unspecified")
        characteristics = instance.get("characteristics", {})
        for algorithm_name, scheduler in ALGORITHMS.items():
            record: dict[str, Any] = {
                "instance_id": instance_id,
                "source_file": source_file,
                "profile": profile,
                "algorithm": algorithm_name,
                "status": "ok",
                "error": None,
                "scheduler_runtime_ms": None,
                **{
                    name: characteristics.get(name)
                    for name in CHARACTERISTIC_NAMES
                },
            }
            try:
                started = time.perf_counter_ns()
                schedule = scheduler(instance)
                elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
                # Evaluation is deliberately outside the timed scheduler section.
                metrics = evaluate_schedule(instance, schedule)
                record["scheduler_runtime_ms"] = elapsed_ms
                record.update({name: metrics[name] for name in METRIC_NAMES})
                record["schedule"] = schedule
            except Exception as exc:  # retain failed runs in the experiment record
                record["status"] = "error"
                record["error"] = f"{type(exc).__name__}: {exc}"
                record["schedule"] = None
                for name in METRIC_NAMES:
                    record[name] = None
            records.append(record)

    by_profile = _summarise(records, ("profile", "algorithm"))
    overall = _summarise(records, ("algorithm",))
    return {
        "experiment": {
            "name": "rule_based_baseline_benchmark",
            "problem_type": "non_preemptive_resource_constrained_project_scheduling",
            "algorithms": list(ALGORITHMS),
            "instance_count": len(instances),
            "run_count": len(records),
            "python_version": platform.python_version(),
            "runtime_measurement": (
                "Scheduler call only; schedule validation and objective evaluation are excluded."
            ),
            "metric_directions": {
                "makespan": "minimize",
                "total_resource_cost": "minimize",
                "total_tardiness": "minimize",
                "resource_utilization": "maximize",
                "scheduler_runtime_ms": "minimize",
            },
        },
        "records": records,
        "summary_by_profile": by_profile,
        "summary_overall": overall,
    }


def write_results(result: dict[str, Any], output_dir: Path) -> None:
    """Write detailed JSON plus flat per-run and aggregate CSV reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )

    detail_fields = [
        "instance_id", "source_file", "profile", "algorithm", "status", "error",
        *CHARACTERISTIC_NAMES, *METRIC_NAMES, "scheduler_runtime_ms",
    ]
    with (output_dir / "per_instance.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=detail_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["records"])

    summary_rows = result["summary_by_profile"]
    summary_fields = [
        "profile", "algorithm", "instances", "successful_runs", "failed_runs",
        "success_rate",
    ]
    for metric in (*METRIC_NAMES, "scheduler_runtime_ms"):
        summary_fields.extend(
            f"{metric}_{stat}" for stat in ("mean", "median", "stddev")
        )
    with (output_dir / "summary_by_profile.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    with (output_dir / "summary_overall.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["algorithm", *summary_fields[2:]]
        )
        writer.writeheader()
        writer.writerows(result["summary_overall"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark FCFS, SPT and Critical Path on one scheduling dataset."
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("scheduling_dataset"),
        help="Dataset directory containing instances/*.json.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("baseline_results"),
        help="Directory for JSON and CSV result files.",
    )
    args = parser.parse_args()
    instances = load_instances(args.dataset)
    result = run_benchmark(instances)
    write_results(result, args.output)

    failed = sum(row["status"] != "ok" for row in result["records"])
    print(f"Instances: {len(instances)}")
    print(f"Scheduler runs: {len(result['records'])}")
    print(f"Failed runs: {failed}")
    print(f"Results: {args.output.resolve()}")
    print(f"Overall summary: {args.output / 'summary_overall.csv'}")


if __name__ == "__main__":
    main()
