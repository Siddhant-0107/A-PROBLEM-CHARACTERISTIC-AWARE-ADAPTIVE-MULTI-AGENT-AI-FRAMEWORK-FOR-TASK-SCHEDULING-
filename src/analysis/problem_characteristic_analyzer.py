"""Analyze how scheduling baseline performance varies with problem characteristics.

This is a descriptive analysis stage, not an adaptive policy. It uses only
per-instance benchmark records and keeps the three scheduling objectives
separate. No causal or statistical-significance claims are made.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

DEFAULT_CHARACTERISTICS = (
    "task_count",
    "resource_count",
    "deadline_tightness",
    "resource_scarcity_proxy",
    "cost_variation_cv",
    "dependency_density",
    "capacity_factor",
)
METRIC_DIRECTIONS = {
    "makespan": "minimize",
    "total_resource_cost": "minimize",
    "total_tardiness": "minimize",
    "resource_utilization": "maximize",
    "scheduler_runtime_ms": "minimize",
}
METRIC_NAMES = tuple(METRIC_DIRECTIONS)
THREE_BIN_LABELS = ("low", "medium", "high")


def load_benchmark(path: Path) -> dict[str, Any]:
    """Load the results.json produced by src.experiments.run_baselines."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read benchmark JSON at {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("records"), list):
        raise ValueError("Benchmark JSON must contain a records list.")
    if not data["records"]:
        raise ValueError("Benchmark results contain no per-run records.")
    return data


def _numeric(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _stddev(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _bin_labels(bin_count: int) -> tuple[str, ...]:
    if bin_count == 3:
        return THREE_BIN_LABELS
    return tuple(f"bin_{index + 1}" for index in range(bin_count))


def _characteristic_bins(
    records: list[dict[str, Any]], characteristic: str, bin_count: int
) -> tuple[dict[str, str], dict[str, tuple[float, float]]]:
    """Assign each unique instance a deterministic equal-frequency rank bin.

    Equal characteristic values always stay in the same bin. Binning is based
    on unique instance values, not repeated algorithm rows.
    """
    values_by_instance: dict[str, float] = {}
    for row in records:
        instance_id = row.get("instance_id")
        value = row.get(characteristic)
        if instance_id is not None and _numeric(value):
            numeric_value = float(value)
            previous = values_by_instance.get(str(instance_id))
            if previous is not None and previous != numeric_value:
                raise ValueError(
                    f"Instance {instance_id} has inconsistent {characteristic} values."
                )
            values_by_instance[str(instance_id)] = numeric_value

    unique_values = sorted(set(values_by_instance.values()))
    if not unique_values:
        return {}, {}
    labels = _bin_labels(bin_count)
    value_to_bin: dict[float, str] = {}
    ranges: dict[str, tuple[float, float]] = {}
    grouped: dict[str, list[float]] = {label: [] for label in labels}
    for rank, value in enumerate(unique_values):
        bucket = min(bin_count - 1, rank * bin_count // len(unique_values))
        label = labels[bucket]
        value_to_bin[value] = label
        grouped[label].append(value)

    instance_bins = {
        instance_id: value_to_bin[value]
        for instance_id, value in values_by_instance.items()
    }
    for label, members in grouped.items():
        if members:
            ranges[label] = (min(members), max(members))
    return instance_bins, ranges


def _average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        average_rank = ((cursor + 1) + end) / 2.0
        for position in range(cursor, end):
            ranks[indexed[position][0]] = average_rank
        cursor = end
    return ranks


def _spearman(values_x: list[float], values_y: list[float]) -> float | None:
    """Spearman rank correlation with average ranks for ties."""
    if len(values_x) != len(values_y) or len(values_x) < 3:
        return None
    rank_x = _average_ranks(values_x)
    rank_y = _average_ranks(values_y)
    mean_x = statistics.mean(rank_x)
    mean_y = statistics.mean(rank_y)
    centered_x = [value - mean_x for value in rank_x]
    centered_y = [value - mean_y for value in rank_y]
    sum_x = sum(value * value for value in centered_x)
    sum_y = sum(value * value for value in centered_y)
    if sum_x == 0 or sum_y == 0:
        return None
    covariance = sum(x * y for x, y in zip(centered_x, centered_y))
    return covariance / math.sqrt(sum_x * sum_y)


def analyze_benchmark(
    benchmark: dict[str, Any],
    characteristics: tuple[str, ...] = DEFAULT_CHARACTERISTICS,
    bin_count: int = 3,
) -> dict[str, Any]:
    """Create per-characteristic performance tables, winners and associations."""
    if bin_count < 2:
        raise ValueError("bin_count must be at least 2.")
    records = benchmark.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Benchmark must contain non-empty records.")
    algorithms = benchmark.get("experiment", {}).get("algorithms")
    if not algorithms:
        algorithms = sorted({
            row.get("algorithm") for row in records if row.get("algorithm")
        })
    if not algorithms:
        raise ValueError("No algorithm names found in benchmark records.")

    performance_rows: list[dict[str, Any]] = []
    winner_rows: list[dict[str, Any]] = []
    association_rows: list[dict[str, Any]] = []
    bin_metadata: dict[str, Any] = {}

    for characteristic in characteristics:
        instance_bins, ranges = _characteristic_bins(records, characteristic, bin_count)
        if not instance_bins:
            continue
        bin_metadata[characteristic] = {
            label: {"minimum": bounds[0], "maximum": bounds[1]}
            for label, bounds in ranges.items()
        }
        for label, bounds in ranges.items():
            bin_records = [
                row for row in records
                if str(row.get("instance_id")) in instance_bins
                and instance_bins[str(row.get("instance_id"))] == label
            ]
            algorithm_means: dict[str, dict[str, float | None]] = {}
            for algorithm in algorithms:
                rows = [row for row in bin_records if row.get("algorithm") == algorithm]
                successful = [row for row in rows if row.get("status") == "ok"]
                unique_instances = {str(row.get("instance_id")) for row in rows}
                result_row: dict[str, Any] = {
                    "characteristic": characteristic,
                    "bin": label,
                    "bin_minimum": bounds[0],
                    "bin_maximum": bounds[1],
                    "algorithm": algorithm,
                    "instances": len(unique_instances),
                    "successful_runs": len(successful),
                    "failed_runs": len(rows) - len(successful),
                    "success_rate": len(successful) / len(rows) if rows else 0.0,
                }
                means: dict[str, float | None] = {}
                for metric in METRIC_NAMES:
                    values = [
                        float(row[metric]) for row in successful
                        if _numeric(row.get(metric))
                    ]
                    means[metric] = _mean(values)
                    result_row[f"{metric}_mean"] = _mean(values)
                    result_row[f"{metric}_median"] = _median(values)
                    result_row[f"{metric}_stddev"] = _stddev(values)
                algorithm_means[algorithm] = means
                performance_rows.append(result_row)

            for metric, direction in METRIC_DIRECTIONS.items():
                available = {
                    algorithm: means.get(metric)
                    for algorithm, means in algorithm_means.items()
                    if means.get(metric) is not None
                }
                if not available:
                    continue
                best_value = (
                    min(available.values()) if direction == "minimize"
                    else max(available.values())
                )
                winners = sorted(
                    algorithm for algorithm, value in available.items()
                    if math.isclose(float(value), float(best_value), rel_tol=1e-9, abs_tol=1e-9)
                )
                winner_rows.append({
                    "characteristic": characteristic,
                    "bin": label,
                    "metric": metric,
                    "direction": direction,
                    "winner": ";".join(winners),
                    "best_mean": best_value,
                    "algorithms_compared": len(available),
                    "interpretation": "descriptive winner by bin mean; not a significance test",
                })

        # Pair each instance's characteristic with each algorithm's metric once.
        for algorithm in algorithms:
            algo_rows = [
                row for row in records
                if row.get("algorithm") == algorithm and row.get("status") == "ok"
                and _numeric(row.get(characteristic))
            ]
            for metric in METRIC_NAMES:
                paired = [
                    (float(row[characteristic]), float(row[metric]))
                    for row in algo_rows if _numeric(row.get(metric))
                ]
                correlation = _spearman(
                    [pair[0] for pair in paired],
                    [pair[1] for pair in paired],
                )
                association_rows.append({
                    "characteristic": characteristic,
                    "algorithm": algorithm,
                    "metric": metric,
                    "spearman_rho": correlation,
                    "paired_instances": len(paired),
                    "interpretation": (
                        "descriptive monotonic association; not causal"
                        if correlation is not None
                        else "not calculable (fewer than 3 pairs or constant values)"
                    ),
                })

    return {
        "analysis": {
            "name": "problem_characteristic_performance_analysis",
            "source_experiment": benchmark.get("experiment", {}).get("name"),
            "instance_count": benchmark.get("experiment", {}).get(
                "instance_count",
                len({str(row.get("instance_id")) for row in records}),
            ),
            "algorithms": list(algorithms),
            "characteristics_analyzed": list(bin_metadata),
            "bin_count": bin_count,
            "binning_method": (
                "Equal-frequency rank bins over unique instance characteristic values; "
                "identical values remain in the same bin."
            ),
            "correlation_method": (
                "Spearman rank correlation with average ranks for ties, computed per "
                "algorithm and metric over successful paired instances."
            ),
            "limitations": [
                "Results are descriptive and do not establish causality.",
                "Per-bin winners are selected by mean metric and are not significance tests.",
                "Correlations are exploratory associations, not evidence of causal effects.",
                "Characteristics absent from benchmark records are skipped.",
            ],
        },
        "bin_ranges": bin_metadata,
        "performance_by_characteristic": performance_rows,
        "best_algorithm_by_characteristic": winner_rows,
        "characteristic_metric_associations": association_rows,
    }


def write_analysis(result: dict[str, Any], output_dir: Path) -> None:
    """Write machine-readable JSON and flat CSV analysis tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "characteristic_analysis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    tables = (
        ("performance_by_characteristic.csv", "performance_by_characteristic"),
        ("best_algorithm_by_characteristic.csv", "best_algorithm_by_characteristic"),
        ("characteristic_metric_associations.csv", "characteristic_metric_associations"),
    )
    for filename, key in tables:
        rows = result[key]
        path = output_dir / filename
        if not rows:
            path.write_text("", encoding="utf-8")
            continue
        fields = list(dict.fromkeys(field for row in rows for field in row))
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze baseline performance against scheduling problem characteristics."
    )
    parser.add_argument(
        "--input", type=Path, default=Path("baseline_results/results.json"),
        help="results.json produced by the baseline benchmark runner.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("characteristic_analysis"),
        help="Directory for characteristic analysis reports.",
    )
    parser.add_argument(
        "--bins", type=int, default=3,
        help="Number of equal-frequency characteristic bins (default: 3).",
    )
    args = parser.parse_args()
    benchmark = load_benchmark(args.input)
    result = analyze_benchmark(benchmark, bin_count=args.bins)
    write_analysis(result, args.output)
    print(f"Characteristics analyzed: {len(result['analysis']['characteristics_analyzed'])}")
    print(f"Algorithms: {', '.join(result['analysis']['algorithms'])}")
    print(f"Performance rows: {len(result['performance_by_characteristic'])}")
    print(f"Winner rows: {len(result['best_algorithm_by_characteristic'])}")
    print(f"Reports: {args.output.resolve()}")


if __name__ == "__main__":
    main()
