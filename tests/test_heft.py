"""Unit tests for the HEFT baseline and heterogeneous DAG generator."""
import random
import unittest

from src.baselines.heft import calculate_upward_ranks, schedule_heft
from src.generators.heft_instance_generator import (
    generate_heft_instance, validate_heft_instance,
)


def small_instance():
    return {
        "processors": [{"processor_id": "P1"}, {"processor_id": "P2"}],
        "tasks": [
            {"task_id": "T1", "predecessors": [], "execution_times": {"P1": 4, "P2": 8}},
            {"task_id": "T2", "predecessors": ["T1"], "execution_times": {"P1": 8, "P2": 3}},
            {"task_id": "T3", "predecessors": ["T1"], "execution_times": {"P1": 6, "P2": 4}},
            {"task_id": "T4", "predecessors": ["T2", "T3"], "execution_times": {"P1": 3, "P2": 5}},
        ],
        "communication_costs": {
            "T1->T2": 2, "T1->T3": 2, "T2->T4": 1, "T3->T4": 1,
        },
    }


class TestHEFTRanks(unittest.TestCase):
    def test_upward_rank_values(self):
        ranks = calculate_upward_ranks(small_instance())
        self.assertAlmostEqual(ranks["T4"], 4.0)
        self.assertAlmostEqual(ranks["T2"], 10.5)
        self.assertAlmostEqual(ranks["T3"], 10.0)
        self.assertAlmostEqual(ranks["T1"], 18.5)

    def test_priority_order_starts_with_upstream_critical_task(self):
        self.assertEqual(schedule_heft(small_instance())["task_order"][0], "T1")


class TestHEFTScheduling(unittest.TestCase):
    def test_every_task_is_assigned_once(self):
        instance = small_instance()
        result = schedule_heft(instance)
        ids = [x["task_id"] for x in result["assignments"]]
        self.assertEqual(set(ids), {t["task_id"] for t in instance["tasks"]})
        self.assertEqual(len(ids), len(set(ids)))

    def test_precedence_and_communication_constraints(self):
        instance = small_instance()
        result = schedule_heft(instance)
        assigned = {x["task_id"]: x for x in result["assignments"]}
        for task in instance["tasks"]:
            current = assigned[task["task_id"]]
            for predecessor in task["predecessors"]:
                parent = assigned[predecessor]
                transfer = (0 if parent["processor_id"] == current["processor_id"]
                            else instance["communication_costs"][f"{predecessor}->{task['task_id']}"])
                self.assertGreaterEqual(current["start_time"] + 1e-9,
                                        parent["finish_time"] + transfer)

    def test_processor_reservations_do_not_overlap(self):
        result = schedule_heft(small_instance())
        grouped = {}
        for item in result["assignments"]:
            grouped.setdefault(item["processor_id"], []).append(item)
        for items in grouped.values():
            items.sort(key=lambda x: x["start_time"])
            for left, right in zip(items, items[1:]):
                self.assertLessEqual(left["finish_time"], right["start_time"] + 1e-9)

    def test_reported_makespan(self):
        result = schedule_heft(small_instance())
        self.assertEqual(result["makespan"], max(x["finish_time"] for x in result["assignments"]))

    def test_empty_task_list(self):
        instance = {"processors": [{"processor_id": "P1"}], "tasks": [], "communication_costs": {}}
        self.assertEqual(schedule_heft(instance)["makespan"], 0.0)

    def test_cycle_is_rejected(self):
        instance = small_instance()
        instance["tasks"][0]["predecessors"] = ["T4"]
        with self.assertRaisesRegex(ValueError, "cycle"):
            schedule_heft(instance)

    def test_missing_processor_time_is_rejected(self):
        instance = small_instance()
        instance["tasks"][0]["execution_times"].pop("P2")
        with self.assertRaisesRegex(ValueError, "every processor"):
            schedule_heft(instance)


class TestHEFTGenerator(unittest.TestCase):
    def test_seeded_generation_is_reproducible(self):
        a = generate_heft_instance(1, random.Random(42), 10, 3, 0.3)
        b = generate_heft_instance(1, random.Random(42), 10, 3, 0.3)
        self.assertEqual(a, b)
        validate_heft_instance(a)

    def test_every_edge_has_communication_cost(self):
        instance = generate_heft_instance(2, random.Random(7), 12, 4, 0.4)
        expected = {f"{p}->{t['task_id']}" for t in instance["tasks"] for p in t["predecessors"]}
        self.assertEqual(set(instance["communication_costs"]), expected)

    def test_invalid_generator_arguments(self):
        with self.assertRaises(ValueError):
            generate_heft_instance(1, random.Random(1), 0, 2)
        with self.assertRaises(ValueError):
            generate_heft_instance(1, random.Random(1), 2, 0)
        with self.assertRaises(ValueError):
            generate_heft_instance(1, random.Random(1), 2, 2, 1.5)

    def test_unknown_predecessor_is_rejected(self):
        instance = small_instance()
        instance["tasks"][1]["predecessors"] = ["T99"]
        with self.assertRaisesRegex(ValueError, "unknown predecessor"):
            validate_heft_instance(instance)

    def test_missing_edge_cost_is_rejected(self):
        instance = small_instance()
        instance["communication_costs"].pop("T1->T2")
        with self.assertRaisesRegex(ValueError, "every dependency"):
            validate_heft_instance(instance)


if __name__ == "__main__":
    unittest.main()
