"""Feasible non-preemptive baselines for renewable-resource scheduling.

Policies:
- FCFS: stable input-order proxy (the current instance schema has no arrival time).
- SPT: shortest processing time among currently precedence-ready tasks.
- Critical path: longest remaining downstream processing-time path first.

All policies use the same earliest-feasible-start resource allocator and the
project's deterministic schedule evaluator.
"""
from __future__ import annotations

from typing import Any, Callable

from src.evaluation.schedule_evaluator import check_schedule


def _validate_instance(instance: dict[str, Any]) -> None:
    tasks = instance.get("tasks")
    resources = instance.get("resources")
    if not isinstance(tasks, list) or not isinstance(resources, list):
        raise ValueError("Instance must contain tasks and resources lists.")
    ids = [task.get("task_id") for task in tasks]
    if len(ids) != len(set(ids)) or any(not task_id for task_id in ids):
        raise ValueError("Task IDs must be unique and non-empty.")
    task_ids = set(ids)
    resource_map = {r.get("resource_id"): r for r in resources}
    if len(resource_map) != len(resources) or None in resource_map:
        raise ValueError("Resource IDs must be unique and non-empty.")
    for task in tasks:
        duration = task.get("processing_time")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration <= 0:
            raise ValueError(f"{task['task_id']}: processing_time must be positive.")
        for predecessor in task.get("predecessors", []):
            if predecessor not in task_ids:
                raise ValueError(f"{task['task_id']}: unknown predecessor {predecessor}.")
        for rid, demand in task.get("resource_requirements", {}).items():
            if rid not in resource_map:
                raise ValueError(f"{task['task_id']}: unknown resource {rid}.")
            capacity = resource_map[rid].get("capacity")
            if (isinstance(demand, bool) or not isinstance(demand, (int, float))
                    or demand <= 0 or demand > capacity):
                raise ValueError(f"{task['task_id']}: invalid demand for {rid}.")


def _critical_path_lengths(instance: dict[str, Any]) -> dict[str, float]:
    """Longest processing-time path from each task through its descendants."""
    tasks = {t["task_id"]: t for t in instance["tasks"]}
    successors = {task_id: [] for task_id in tasks}
    indegree = {task_id: 0 for task_id in tasks}
    for task_id, task in tasks.items():
        for predecessor in task.get("predecessors", []):
            successors[predecessor].append(task_id)
            indegree[task_id] += 1
    queue = [task_id for task_id in tasks if indegree[task_id] == 0]
    order = []
    while queue:
        current = queue.pop(0)
        order.append(current)
        for child in successors[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(order) != len(tasks):
        raise ValueError("Task dependency graph contains a cycle.")
    lengths = {}
    for task_id in reversed(order):
        downstream = max((lengths[child] for child in successors[task_id]), default=0.0)
        lengths[task_id] = float(tasks[task_id]["processing_time"]) + downstream
    return lengths


def _earliest_resource_slot(task: dict[str, Any], earliest: float,
                            scheduled: dict[str, dict[str, Any]],
                            task_map: dict[str, dict[str, Any]],
                            resource_map: dict[str, dict[str, Any]]) -> float:
    """Find earliest start satisfying every renewable capacity for full duration."""
    duration = float(task["processing_time"])
    requirements = task.get("resource_requirements", {})
    if not requirements:
        return earliest
    candidate = float(earliest)
    # Every failed iteration advances to a scheduled task finish, so this terminates.
    while True:
        finish = candidate + duration
        boundaries = {candidate}
        for existing_id, timing in scheduled.items():
            if timing["start_time"] < finish and timing["finish_time"] > candidate:
                boundaries.add(max(candidate, timing["start_time"]))
                boundaries.add(min(finish, timing["finish_time"]))
        conflict_finishes = set()
        feasible = True
        for point in sorted(boundaries):
            if point >= finish:
                continue
            for rid, demand in requirements.items():
                active = []
                used = 0.0
                for existing_id, timing in scheduled.items():
                    if timing["start_time"] <= point < timing["finish_time"]:
                        existing_demand = float(task_map[existing_id].get(
                            "resource_requirements", {}).get(rid, 0))
                        if existing_demand:
                            used += existing_demand
                            active.append((existing_id, timing["finish_time"]))
                if used + float(demand) > float(resource_map[rid]["capacity"]) + 1e-9:
                    feasible = False
                    conflict_finishes.update(
                        end for existing_id, end in active
                        if task_map[existing_id].get("resource_requirements", {}).get(rid, 0)
                    )
        if feasible:
            return candidate
        future_finishes = [end for end in conflict_finishes if end > candidate + 1e-12]
        if not future_finishes:
            raise ValueError(f"Unable to find a feasible resource slot for {task['task_id']}.")
        candidate = min(future_finishes)


def _schedule(instance: dict[str, Any], policy: str,
              priority_key: Callable[[str, dict[str, Any], int], tuple]) -> dict[str, Any]:
    _validate_instance(instance)
    task_list = instance["tasks"]
    task_map = {task["task_id"]: task for task in task_list}
    resource_map = {r["resource_id"]: r for r in instance["resources"]}
    input_order = {task["task_id"]: i for i, task in enumerate(task_list)}
    unscheduled = set(task_map)
    scheduled: dict[str, dict[str, Any]] = {}
    priority_order = []

    while unscheduled:
        ready = [
            task_id for task_id in unscheduled
            if all(p in scheduled for p in task_map[task_id].get("predecessors", []))
        ]
        if not ready:
            raise ValueError("No precedence-ready task remains; dependency graph may contain a cycle.")
        chosen = min(ready, key=lambda tid: priority_key(tid, task_map[tid], input_order[tid]))
        task = task_map[chosen]
        earliest = max(
            (scheduled[p]["finish_time"] for p in task.get("predecessors", [])),
            default=0.0,
        )
        start = _earliest_resource_slot(task, earliest, scheduled, task_map, resource_map)
        finish = start + float(task["processing_time"])
        scheduled[chosen] = {"start_time": start, "finish_time": finish}
        priority_order.append(chosen)
        unscheduled.remove(chosen)

    schedule = {
        "policy": policy,
        "priority_order": priority_order,
        "assignments": [
            {"task_id": task_id, "start_time": scheduled[task_id]["start_time"]}
            for task_id in task_map
        ],
    }
    report = check_schedule(instance, schedule)
    if not report["feasible"]:
        raise AssertionError("Baseline produced an infeasible schedule: " + " | ".join(report["violations"]))
    return schedule


def schedule_fcfs(instance: dict[str, Any]) -> dict[str, Any]:
    """Schedule precedence-ready tasks in stable original input order."""
    return _schedule(instance, "FCFS", lambda tid, task, index: (index,))


def schedule_spt(instance: dict[str, Any]) -> dict[str, Any]:
    """Schedule the shortest processing-time task among ready tasks first."""
    return _schedule(
        instance, "SPT",
        lambda tid, task, index: (float(task["processing_time"]), index),
    )


def schedule_critical_path(instance: dict[str, Any]) -> dict[str, Any]:
    """Prioritize ready tasks with the longest remaining dependency path."""
    lengths = _critical_path_lengths(instance)
    return _schedule(
        instance, "CRITICAL_PATH",
        lambda tid, task, index: (-lengths[tid], index),
    )
