"""Unit and edge-case tests for the synthetic scheduling instance generator.

Run from the repository root:
    python -m unittest discover -s tests -v
"""

import csv
import json
import random
import tempfile
import unittest
from pathlib import Path

from src.generators.scheduling_generator import (
    generate_instance,
    save_dataset,
    validate_instance,
)


def tiny_instance(tasks, resources=None):
    """Build a compact hand-crafted instance for validator edge cases."""
    return {
        "tasks": tasks,
        "resources": resources or [
            {"resource_id": "R1", "capacity": 2, "cost_per_unit_time": 1.0}
        ],
    }


def task(task_id, predecessors=None, duration=2, demands=None, deadline=10):
    return {
        "task_id": task_id,
        "processing_time": duration,
        "deadline": deadline,
        "predecessors": predecessors or [],
        "resource_requirements": demands or {"R1": 1},
    }


class TestGeneratedInstances(unittest.TestCase):
    def test_generated_instance_has_unique_task_and_resource_ids(self):
        instance = generate_instance(1, random.Random(10), "medium", 25, 4)
        task_ids = [t["task_id"] for t in instance["tasks"]]
        resource_ids = [r["resource_id"] for r in instance["resources"]]
        self.assertEqual(len(task_ids), len(set(task_ids)))
        self.assertEqual(len(resource_ids), len(set(resource_ids)))

    def test_seeded_generation_is_reproducible(self):
        first = generate_instance(7, random.Random(1234), "hard", 40, 5)
        second = generate_instance(7, random.Random(1234), "hard", 40, 5)
        self.assertEqual(first, second)

    def test_different_seeds_change_generated_instance(self):
        first = generate_instance(1, random.Random(1), "medium", 20, 4)
        second = generate_instance(1, random.Random(2), "medium", 20, 4)
        self.assertNotEqual(first, second)

    def test_all_profiles_generate_valid_instances(self):
        for profile in ("easy", "medium", "hard", "mixed"):
            with self.subTest(profile=profile):
                instance = generate_instance(1, random.Random(99), profile, 12, 3)
                validate_instance(instance)

    def test_processing_times_and_deadlines_are_valid(self):
        instance = generate_instance(1, random.Random(5), "easy", 15, 3)
        for generated_task in instance["tasks"]:
            self.assertGreater(generated_task["processing_time"], 0)
            self.assertGreaterEqual(generated_task["deadline"], 0)

    def test_resource_demands_are_positive_and_within_capacity(self):
        instance = generate_instance(1, random.Random(6), "hard", 35, 5)
        capacities = {r["resource_id"]: r["capacity"] for r in instance["resources"]}
        for generated_task in instance["tasks"]:
            for resource_id, demand in generated_task["resource_requirements"].items():
                self.assertIn(resource_id, capacities)
                self.assertGreater(demand, 0)
                self.assertLessEqual(demand, capacities[resource_id])

    def test_generated_precedence_edges_point_to_earlier_tasks(self):
        instance = generate_instance(1, random.Random(7), "hard", 40, 5)
        for generated_task in instance["tasks"]:
            current_index = int(generated_task["task_id"][1:])
            for predecessor in generated_task["predecessors"]:
                self.assertLess(int(predecessor[1:]), current_index)

    def test_characteristic_counts_match_instance(self):
        instance = generate_instance(1, random.Random(8), "medium", 17, 4)
        self.assertEqual(instance["characteristics"]["task_count"], len(instance["tasks"]))
        self.assertEqual(instance["characteristics"]["resource_count"], len(instance["resources"]))

    def test_dependency_density_is_calculated_from_edges(self):
        instance = generate_instance(1, random.Random(9), "hard", 20, 4)
        n = len(instance["tasks"])
        edges = sum(len(t["predecessors"]) for t in instance["tasks"])
        expected = edges / (n * (n - 1) / 2)
        self.assertAlmostEqual(instance["characteristics"]["dependency_density"], round(expected, 4))

    def test_cost_variation_is_non_negative(self):
        instance = generate_instance(1, random.Random(10), "medium", 15, 4)
        self.assertGreaterEqual(instance["characteristics"]["cost_variation_cv"], 0)

    def test_single_task_single_resource_edge_case(self):
        instance = generate_instance(1, random.Random(11), "easy", 1, 1)
        validate_instance(instance)
        self.assertEqual(len(instance["tasks"]), 1)
        self.assertEqual(len(instance["resources"]), 1)
        self.assertEqual(instance["characteristics"]["dependency_density"], 0.0)

    def test_no_dependency_edge_case_is_valid(self):
        instance = tiny_instance([
            task("T1"),
            task("T2"),
            task("T3"),
        ])
        validate_instance(instance)

    def test_maximum_dag_density_edge_case_is_valid(self):
        # Every possible forward edge exists: 3 edges out of 3 possible.
        instance = tiny_instance([
            task("T1"),
            task("T2", ["T1"]),
            task("T3", ["T1", "T2"]),
        ])
        validate_instance(instance)


class TestValidationFailures(unittest.TestCase):
    def test_unknown_predecessor_is_rejected(self):
        instance = tiny_instance([task("T1", ["T99"])])
        with self.assertRaisesRegex(ValueError, "Unknown predecessor"):
            validate_instance(instance)

    def test_self_dependency_is_rejected(self):
        instance = tiny_instance([task("T1", ["T1"])])
        with self.assertRaises(ValueError):
            validate_instance(instance)

    def test_backward_edge_or_cycle_is_rejected(self):
        instance = tiny_instance([
            task("T1"),
            task("T2", ["T1"]),
            task("T3", ["T2"]),
            task("T4", ["T3", "T1"]),
        ])
        # A backward edge is represented by T4 depending on an earlier task,
        # which is valid. Explicitly malformed order: T2 depends on T3.
        instance["tasks"][1]["predecessors"] = ["T3"]
        with self.assertRaisesRegex(ValueError, "Dependency order"):
            validate_instance(instance)

    def test_zero_processing_time_is_rejected(self):
        instance = tiny_instance([task("T1", duration=0)])
        with self.assertRaisesRegex(ValueError, "processing time"):
            validate_instance(instance)

    def test_unknown_resource_is_rejected(self):
        instance = tiny_instance([task("T1", demands={"R9": 1})])
        with self.assertRaisesRegex(ValueError, "Unknown resource"):
            validate_instance(instance)

    def test_zero_resource_demand_is_rejected(self):
        instance = tiny_instance([task("T1", demands={"R1": 0})])
        with self.assertRaisesRegex(ValueError, "Invalid demand"):
            validate_instance(instance)

    def test_demand_above_capacity_is_rejected(self):
        instance = tiny_instance([task("T1", demands={"R1": 3})])
        with self.assertRaisesRegex(ValueError, "Invalid demand"):
            validate_instance(instance)

    def test_duplicate_task_ids_are_rejected(self):
        instance = tiny_instance([task("T1"), task("T1")])
        with self.assertRaisesRegex(ValueError, "Duplicate task IDs"):
            validate_instance(instance)


class TestDatasetOutput(unittest.TestCase):
    def test_dataset_writes_json_csv_and_manifest(self):
        instances = [
            generate_instance(i, random.Random(i), "easy", 5, 2)
            for i in range(1, 4)
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "dataset"
            save_dataset(instances, output)

            json_files = sorted((output / "instances").glob("*.json"))
            self.assertEqual(len(json_files), 3)
            with json_files[0].open(encoding="utf-8") as handle:
                loaded = json.load(handle)
            validate_instance(loaded)

            with (output / "summary.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertIn("dependency_density", rows[0])

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["number_of_instances"], 3)


if __name__ == "__main__":
    unittest.main()
