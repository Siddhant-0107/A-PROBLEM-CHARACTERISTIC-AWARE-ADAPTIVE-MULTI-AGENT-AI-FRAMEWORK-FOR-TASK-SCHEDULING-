"""Tests for FCFS, SPT and critical-path resource-constrained baselines."""
import unittest

from src.baselines.rule_based import (
    schedule_critical_path, schedule_fcfs, schedule_spt,
)
from src.evaluation.schedule_evaluator import check_schedule, evaluate_schedule


def instance():
    return {
        "tasks": [
            {"task_id": "T1", "processing_time": 8, "deadline": 30,
             "predecessors": [], "resource_requirements": {"R1": 1}},
            {"task_id": "T2", "processing_time": 2, "deadline": 30,
             "predecessors": [], "resource_requirements": {"R1": 1}},
            {"task_id": "T3", "processing_time": 3, "deadline": 30,
             "predecessors": ["T1"], "resource_requirements": {"R1": 1}},
            {"task_id": "T4", "processing_time": 1, "deadline": 30,
             "predecessors": ["T2"], "resource_requirements": {"R1": 1}},
        ],
        "resources": [{"resource_id": "R1", "capacity": 1, "cost_per_unit_time": 1.0}],
    }


class TestBaselinePolicies(unittest.TestCase):
    def test_fcfs_uses_input_order_when_tasks_are_ready(self):
        result = schedule_fcfs(instance())
        self.assertEqual(result["priority_order"][0], "T1")
        self.assertEqual(result["policy"], "FCFS")

    def test_spt_selects_shortest_ready_task(self):
        result = schedule_spt(instance())
        self.assertEqual(result["priority_order"][0], "T2")
        self.assertEqual(result["policy"], "SPT")

    def test_critical_path_prioritizes_longest_remaining_chain(self):
        result = schedule_critical_path(instance())
        self.assertEqual(result["priority_order"][0], "T1")
        self.assertEqual(result["policy"], "CRITICAL_PATH")

    def test_all_baselines_produce_feasible_schedules(self):
        problem = instance()
        for scheduler in (schedule_fcfs, schedule_spt, schedule_critical_path):
            with self.subTest(scheduler=scheduler.__name__):
                result = scheduler(problem)
                self.assertTrue(check_schedule(problem, result)["feasible"])
                metrics = evaluate_schedule(problem, result)
                self.assertTrue(metrics["feasible"])
                self.assertGreater(metrics["makespan"], 0)

    def test_precedence_is_respected(self):
        problem = instance()
        for scheduler in (schedule_fcfs, schedule_spt, schedule_critical_path):
            result = scheduler(problem)
            timings = check_schedule(problem, result)["timings"]
            for task in problem["tasks"]:
                for predecessor in task["predecessors"]:
                    self.assertGreaterEqual(
                        timings[task["task_id"]]["start_time"],
                        timings[predecessor]["finish_time"],
                    )

    def test_shared_resource_capacity_is_respected(self):
        problem = instance()
        for scheduler in (schedule_fcfs, schedule_spt, schedule_critical_path):
            self.assertTrue(check_schedule(problem, scheduler(problem))["feasible"])

    def test_empty_instance(self):
        problem = {"tasks": [], "resources": []}
        for scheduler in (schedule_fcfs, schedule_spt, schedule_critical_path):
            result = scheduler(problem)
            self.assertEqual(result["assignments"], [])
            self.assertEqual(result["priority_order"], [])

    def test_unknown_predecessor_is_rejected(self):
        problem = instance()
        problem["tasks"][0]["predecessors"] = ["T99"]
        with self.assertRaisesRegex(ValueError, "unknown predecessor"):
            schedule_fcfs(problem)

    def test_cycle_is_rejected(self):
        problem = instance()
        problem["tasks"][0]["predecessors"] = ["T3"]
        with self.assertRaisesRegex(ValueError, "cycle|ready"):
            schedule_critical_path(problem)

    def test_task_demand_above_capacity_is_rejected(self):
        problem = instance()
        problem["tasks"][0]["resource_requirements"]["R1"] = 2
        with self.assertRaisesRegex(ValueError, "invalid demand"):
            schedule_spt(problem)

    def test_unknown_resource_is_rejected(self):
        problem = instance()
        problem["tasks"][0]["resource_requirements"] = {"R9": 1}
        with self.assertRaisesRegex(ValueError, "unknown resource"):
            schedule_fcfs(problem)


if __name__ == "__main__":
    unittest.main()
