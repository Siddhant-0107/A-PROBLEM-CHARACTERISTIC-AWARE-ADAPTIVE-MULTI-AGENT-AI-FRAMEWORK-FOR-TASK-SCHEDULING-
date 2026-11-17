"""Tests for the deterministic schedule evaluator. Run: py -m unittest discover -s tests -v"""
import unittest
from src.evaluation.schedule_evaluator import (
    calculate_makespan, calculate_resource_cost, calculate_resource_utilization,
    calculate_total_tardiness, check_schedule, evaluate_schedule,
)


def make_instance():
    return {
        "tasks": [
            {"task_id": "T1", "processing_time": 3, "deadline": 2, "predecessors": [], "resource_requirements": {"R1": 1}},
            {"task_id": "T2", "processing_time": 2, "deadline": 10, "predecessors": ["T1"], "resource_requirements": {"R1": 1, "R2": 1}},
            {"task_id": "T3", "processing_time": 2, "deadline": 10, "predecessors": [], "resource_requirements": {"R1": 1}},
        ],
        "resources": [
            {"resource_id": "R1", "capacity": 1, "cost_per_unit_time": 2.0},
            {"resource_id": "R2", "capacity": 2, "cost_per_unit_time": 1.0},
        ],
    }


class TestHardConstraints(unittest.TestCase):
    def setUp(self):
        self.instance = make_instance()

    def test_valid_sequential_schedule(self):
        self.assertTrue(check_schedule(self.instance, {"assignments": [
            {"task_id": "T1", "start_time": 0}, {"task_id": "T2", "start_time": 3},
            {"task_id": "T3", "start_time": 5},
        ]})["feasible"])

    def test_missing_task(self):
        report = check_schedule(self.instance, {"assignments": [
            {"task_id": "T1", "start_time": 0}, {"task_id": "T2", "start_time": 3}]})
        self.assertTrue(any("missing" in x for x in report["violations"]))

    def test_duplicate_assignment(self):
        report = check_schedule(self.instance, {"assignments": [
            {"task_id": "T1", "start_time": 0}, {"task_id": "T1", "start_time": 3},
            {"task_id": "T2", "start_time": 3}, {"task_id": "T3", "start_time": 5}]})
        self.assertTrue(any("more than once" in x for x in report["violations"]))

    def test_unknown_task(self):
        report = check_schedule(self.instance, {"assignments": [
            {"task_id": "T1", "start_time": 0}, {"task_id": "T2", "start_time": 3},
            {"task_id": "T3", "start_time": 5}, {"task_id": "T99", "start_time": 0}]})
        self.assertTrue(any("unknown task" in x for x in report["violations"]))

    def test_negative_start_time(self):
        report = check_schedule(self.instance, {"assignments": [
            {"task_id": "T1", "start_time": -1}, {"task_id": "T2", "start_time": 3},
            {"task_id": "T3", "start_time": 5}]})
        self.assertFalse(report["feasible"])

    def test_non_numeric_start_time(self):
        report = check_schedule(self.instance, {"assignments": [
            {"task_id": "T1", "start_time": "soon"}, {"task_id": "T2", "start_time": 3},
            {"task_id": "T3", "start_time": 5}]})
        self.assertFalse(report["feasible"])

    def test_precedence_violation(self):
        report = check_schedule(self.instance, {"assignments": [
            {"task_id": "T1", "start_time": 0}, {"task_id": "T2", "start_time": 2},
            {"task_id": "T3", "start_time": 5}]})
        self.assertTrue(any("Precedence violation" in x for x in report["violations"]))

    def test_resource_overload(self):
        report = check_schedule(self.instance, {"assignments": [
            {"task_id": "T1", "start_time": 0}, {"task_id": "T2", "start_time": 3},
            {"task_id": "T3", "start_time": 0}]})
        self.assertTrue(any("Resource overload" in x for x in report["violations"]))

    def test_back_to_back_tasks_do_not_overlap(self):
        report = check_schedule(self.instance, {"assignments": [
            {"task_id": "T1", "start_time": 0}, {"task_id": "T2", "start_time": 3},
            {"task_id": "T3", "start_time": 5}]})
        self.assertTrue(report["feasible"])

    def test_soft_deadline_miss_is_not_hard_violation(self):
        report = check_schedule(self.instance, {"assignments": [
            {"task_id": "T1", "start_time": 0}, {"task_id": "T2", "start_time": 3},
            {"task_id": "T3", "start_time": 5}]})
        self.assertTrue(report["feasible"])

    def test_invalid_schedule_container(self):
        self.assertFalse(check_schedule(self.instance, {})["feasible"])


class TestObjectiveFunctions(unittest.TestCase):
    def setUp(self):
        self.instance = make_instance()
        self.schedule = {"assignments": [
            {"task_id": "T1", "start_time": 0}, {"task_id": "T2", "start_time": 3},
            {"task_id": "T3", "start_time": 5}]}
        self.timings = check_schedule(self.instance, self.schedule)["timings"]

    def test_makespan(self):
        self.assertEqual(calculate_makespan(self.timings), 7.0)

    def test_total_tardiness(self):
        self.assertEqual(calculate_total_tardiness(self.instance, self.timings), 1.0)

    def test_resource_cost(self):
        self.assertEqual(calculate_resource_cost(self.instance, self.timings), 16.0)

    def test_resource_utilization(self):
        self.assertAlmostEqual(calculate_resource_utilization(self.instance, self.timings), 7 / 21)

    def test_evaluate_returns_all_objectives(self):
        result = evaluate_schedule(self.instance, self.schedule)
        self.assertTrue(result["feasible"])
        for key in ("makespan", "total_resource_cost", "total_tardiness", "resource_utilization"):
            self.assertIn(key, result)

    def test_evaluate_rejects_infeasible_schedule(self):
        invalid = {"assignments": [
            {"task_id": "T1", "start_time": 0}, {"task_id": "T2", "start_time": 1},
            {"task_id": "T3", "start_time": 0}]}
        with self.assertRaises(ValueError):
            evaluate_schedule(self.instance, invalid)

    def test_empty_timings_have_zero_makespan(self):
        self.assertEqual(calculate_makespan({}), 0.0)


if __name__ == "__main__":
    unittest.main()
