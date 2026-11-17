"""Tests for the problem characteristic analyzer."""
import json
import tempfile
import unittest
from pathlib import Path

from src.analysis.problem_characteristic_analyzer import (
    analyze_benchmark,
    load_benchmark,
    write_analysis,
)


def benchmark_fixture():
    records = []
    for index, characteristic in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], start=1):
        instance_id = f"SP-{index:04d}"
        for algorithm, offset in (("FCFS", 2.0), ("SPT", 3.0), ("CRITICAL_PATH", 1.0)):
            records.append({
                "instance_id": instance_id,
                "algorithm": algorithm,
                "status": "ok",
                "deadline_tightness": characteristic,
                "makespan": index * 10.0 + offset,
                "total_resource_cost": 100.0,
                "total_tardiness": index * 5.0 + offset,
                "resource_utilization": 0.5 + index * 0.01,
                "scheduler_runtime_ms": 1.0 + offset * 0.01,
            })
    return {
        "experiment": {
            "name": "fixture",
            "instance_count": 6,
            "algorithms": ["FCFS", "SPT", "CRITICAL_PATH"],
        },
        "records": records,
    }


class TestProblemCharacteristicAnalyzer(unittest.TestCase):
    def test_creates_per_characteristic_algorithm_rows(self):
        result = analyze_benchmark(
            benchmark_fixture(), characteristics=("deadline_tightness",)
        )
        self.assertEqual(len(result["performance_by_characteristic"]), 9)
        self.assertEqual(len(result["best_algorithm_by_characteristic"]), 15)
        self.assertEqual(len(result["characteristic_metric_associations"]), 15)

    def test_identifies_best_algorithm_for_minimize_metric(self):
        result = analyze_benchmark(
            benchmark_fixture(), characteristics=("deadline_tightness",)
        )
        winners = [
            row for row in result["best_algorithm_by_characteristic"]
            if row["bin"] == "low" and row["metric"] == "makespan"
        ]
        self.assertEqual(winners[0]["winner"], "CRITICAL_PATH")

    def test_bins_are_equal_frequency_and_deterministic(self):
        result = analyze_benchmark(
            benchmark_fixture(), characteristics=("deadline_tightness",)
        )
        self.assertEqual(
            set(result["bin_ranges"]["deadline_tightness"]),
            {"low", "medium", "high"},
        )
        self.assertEqual(
            result["bin_ranges"]["deadline_tightness"]["low"],
            {"minimum": 0.1, "maximum": 0.2},
        )

    def test_missing_characteristic_is_skipped(self):
        result = analyze_benchmark(
            benchmark_fixture(), characteristics=("not_a_feature",)
        )
        self.assertEqual(result["performance_by_characteristic"], [])
        self.assertEqual(result["analysis"]["characteristics_analyzed"], [])

    def test_failed_runs_are_counted_but_not_in_metric_means(self):
        benchmark = benchmark_fixture()
        row = dict(benchmark["records"][0])
        row.update({"status": "error", "makespan": None})
        benchmark["records"].append(row)
        result = analyze_benchmark(
            benchmark, characteristics=("deadline_tightness",)
        )
        low_fcfs = next(
            row for row in result["performance_by_characteristic"]
            if row["bin"] == "low" and row["algorithm"] == "FCFS"
        )
        self.assertEqual(low_fcfs["failed_runs"], 1)
        self.assertEqual(low_fcfs["successful_runs"], 2)

    def test_rejects_inconsistent_instance_characteristics(self):
        benchmark = benchmark_fixture()
        benchmark["records"][1]["deadline_tightness"] = 0.99
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            analyze_benchmark(
                benchmark, characteristics=("deadline_tightness",)
            )

    def test_rejects_invalid_bin_count(self):
        with self.assertRaisesRegex(ValueError, "at least 2"):
            analyze_benchmark(benchmark_fixture(), bin_count=1)

    def test_load_and_write_reports(self):
        result = analyze_benchmark(
            benchmark_fixture(), characteristics=("deadline_tightness",)
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "results.json"
            source.write_text(json.dumps(benchmark_fixture()), encoding="utf-8")
            loaded = load_benchmark(source)
            self.assertEqual(len(loaded["records"]), 18)
            output = root / "analysis"
            write_analysis(result, output)
            expected = {
                "characteristic_analysis.json",
                "performance_by_characteristic.csv",
                "best_algorithm_by_characteristic.csv",
                "characteristic_metric_associations.csv",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)


if __name__ == "__main__":
    unittest.main()
