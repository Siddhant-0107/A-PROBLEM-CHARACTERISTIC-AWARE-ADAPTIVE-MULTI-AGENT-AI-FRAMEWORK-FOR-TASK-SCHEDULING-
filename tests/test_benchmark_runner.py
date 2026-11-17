"""Tests for the rule-based baseline benchmarking engine."""
import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.run_baselines import (
    load_instances,
    run_benchmark,
    write_results,
)


def sample_instance():
    return {
        "instance_id": "SP-TEST-0001",
        "profile": "easy",
        "tasks": [
            {"task_id": "T1", "processing_time": 2, "deadline": 5,
             "predecessors": [], "resource_requirements": {"R1": 1}},
            {"task_id": "T2", "processing_time": 3, "deadline": 8,
             "predecessors": ["T1"], "resource_requirements": {"R1": 1}},
            {"task_id": "T3", "processing_time": 1, "deadline": 6,
             "predecessors": [], "resource_requirements": {"R1": 1}},
        ],
        "resources": [{"resource_id": "R1", "capacity": 2, "cost_per_unit_time": 1.5}],
        "characteristics": {
            "task_count": 3, "resource_count": 1, "deadline_tightness": 0.2,
            "resource_scarcity_proxy": 0.5, "cost_variation_cv": 0.0,
            "dependency_density": 1 / 3, "capacity_factor": 1.8,
        },
    }


class TestBenchmarkRunner(unittest.TestCase):
    def test_runs_all_three_algorithms_on_each_instance(self):
        result = run_benchmark([sample_instance(), sample_instance()])
        self.assertEqual(result["experiment"]["instance_count"], 2)
        self.assertEqual(result["experiment"]["run_count"], 6)
        self.assertEqual(
            {row["algorithm"] for row in result["records"]},
            {"FCFS", "SPT", "CRITICAL_PATH"},
        )
        self.assertTrue(all(row["status"] == "ok" for row in result["records"]))

    def test_metrics_are_present_and_feasible(self):
        result = run_benchmark([sample_instance()])
        for row in result["records"]:
            self.assertEqual(row["status"], "ok")
            for metric in (
                "makespan", "total_resource_cost", "total_tardiness",
                "resource_utilization", "scheduler_runtime_ms",
            ):
                self.assertIsNotNone(row[metric])
                self.assertGreaterEqual(row[metric], 0)

    def test_profile_and_overall_summaries_are_created(self):
        result = run_benchmark([sample_instance()])
        self.assertEqual(len(result["summary_by_profile"]), 3)
        self.assertEqual(len(result["summary_overall"]), 3)
        for row in result["summary_overall"]:
            self.assertEqual(row["successful_runs"], 1)
            self.assertEqual(row["failed_runs"], 0)

    def test_failed_run_is_recorded_without_dropping_experiment(self):
        instance = sample_instance()
        instance["tasks"][0]["resource_requirements"] = {"UNKNOWN": 1}
        result = run_benchmark([instance])
        self.assertEqual(len(result["records"]), 3)
        self.assertTrue(all(row["status"] == "error" for row in result["records"]))
        self.assertTrue(all(row["error"] for row in result["records"]))

    def test_load_instances_reads_sorted_json_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            instance_dir = root / "instances"
            instance_dir.mkdir()
            second = sample_instance()
            second["instance_id"] = "SP-0002"
            first = sample_instance()
            first["instance_id"] = "SP-0001"
            (instance_dir / "b.json").write_text(json.dumps(second), encoding="utf-8")
            (instance_dir / "a.json").write_text(json.dumps(first), encoding="utf-8")
            loaded = load_instances(root)
            self.assertEqual(
                [item["instance_id"] for item in loaded],
                ["SP-0001", "SP-0002"],
            )
            self.assertEqual(loaded[0]["_benchmark_source_file"], "a.json")

    def test_load_instances_rejects_empty_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / "instances").mkdir()
            with self.assertRaisesRegex(ValueError, "No .json"):
                load_instances(Path(temp))

    def test_write_results_creates_json_and_csv_reports(self):
        result = run_benchmark([sample_instance()])
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "results"
            write_results(result, output)
            expected = {
                "results.json", "per_instance.csv",
                "summary_by_profile.csv", "summary_overall.csv",
            }
            self.assertEqual({p.name for p in output.iterdir()}, expected)
            saved = json.loads((output / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["experiment"]["run_count"], 3)


if __name__ == "__main__":
    unittest.main()
