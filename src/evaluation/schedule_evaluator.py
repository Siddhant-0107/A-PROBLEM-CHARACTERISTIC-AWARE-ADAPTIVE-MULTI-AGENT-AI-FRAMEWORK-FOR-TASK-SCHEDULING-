"""Deterministic schedule constraint checker and objective evaluator.

Schedule format: {"assignments": [{"task_id": "T1", "start_time": 0}, ...]}
Finish times are derived from start_time + processing_time. Intervals are
half-open [start, finish). Deadlines are soft and do not invalidate a schedule.
"""
from __future__ import annotations
import math
from collections import defaultdict
from typing import Any


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _normalise_schedule(instance, schedule):
    violations = []
    tasks = {t["task_id"]: t for t in instance.get("tasks", [])}
    assignments = schedule.get("assignments") if isinstance(schedule, dict) else None
    if not isinstance(assignments, list):
        return {}, ["Schedule must contain an assignments list."]
    timings = {}
    for index, assignment in enumerate(assignments):
        if not isinstance(assignment, dict):
            violations.append(f"Assignment at index {index} must be an object.")
            continue
        task_id = assignment.get("task_id")
        if task_id not in tasks:
            violations.append(f"Assignment references unknown task '{task_id}'.")
            continue
        if task_id in timings:
            violations.append(f"Task '{task_id}' is assigned more than once.")
            continue
        start = assignment.get("start_time")
        if not _finite_number(start):
            violations.append(f"Task '{task_id}' has an invalid start_time.")
            continue
        if start < 0:
            violations.append(f"Task '{task_id}' starts before time zero.")
            continue
        duration = tasks[task_id].get("processing_time")
        if not _finite_number(duration) or duration <= 0:
            violations.append(f"Task '{task_id}' has invalid processing_time in instance.")
            continue
        timings[task_id] = {
            "start_time": float(start),
            "finish_time": float(start) + float(duration),
        }
    for task_id in tasks:
        if task_id not in timings:
            violations.append(f"Task '{task_id}' is missing from the schedule.")
    return timings, violations


def check_schedule(instance, schedule):
    """Return feasibility, all detected hard-constraint violations and task timings."""
    timings, violations = _normalise_schedule(instance, schedule)
    task_map = {t["task_id"]: t for t in instance.get("tasks", [])}
    resource_map = {r["resource_id"]: r for r in instance.get("resources", [])}

    for task_id, timing in timings.items():
        for predecessor in task_map[task_id].get("predecessors", []):
            if predecessor in timings and timing["start_time"] < timings[predecessor]["finish_time"]:
                violations.append(
                    f"Precedence violation: '{task_id}' starts before '{predecessor}' finishes."
                )

    events = defaultdict(list)
    for task_id, timing in timings.items():
        task = task_map[task_id]
        for rid, demand in task.get("resource_requirements", {}).items():
            if rid not in resource_map:
                violations.append(f"Task '{task_id}' requires unknown resource '{rid}'.")
                continue
            capacity = resource_map[rid].get("capacity")
            if not _finite_number(demand) or demand <= 0 or not _finite_number(capacity) or capacity <= 0:
                violations.append(f"Task '{task_id}' has invalid demand or resource capacity for '{rid}'.")
                continue
            if demand > capacity:
                violations.append(f"Task '{task_id}' demand exceeds resource '{rid}' capacity.")
                continue
            # End events sort before start events at the same timestamp.
            events[rid].append((timing["finish_time"], 0, task_id, float(demand)))
            events[rid].append((timing["start_time"], 1, task_id, float(demand)))

    for rid, resource_events in events.items():
        capacity = float(resource_map[rid]["capacity"])
        active = 0.0
        for timestamp, kind, task_id, demand in sorted(resource_events):
            if kind == 0:
                active -= demand
            else:
                active += demand
                if active > capacity + 1e-9:
                    violations.append(
                        f"Resource overload: '{rid}' demand {active:g} exceeds capacity "
                        f"{capacity:g} at time {timestamp:g}."
                    )
    return {"feasible": not violations, "violations": violations, "timings": timings}


def calculate_makespan(timings):
    return max((x["finish_time"] for x in timings.values()), default=0.0)


def calculate_total_tardiness(instance, timings):
    task_map = {t["task_id"]: t for t in instance.get("tasks", [])}
    total = 0.0
    for task_id, timing in timings.items():
        deadline = task_map[task_id].get("deadline")
        if not _finite_number(deadline):
            raise ValueError(f"Task '{task_id}' has invalid deadline.")
        total += max(0.0, timing["finish_time"] - float(deadline))
    return total


def calculate_resource_cost(instance, timings):
    task_map = {t["task_id"]: t for t in instance.get("tasks", [])}
    resource_map = {r["resource_id"]: r for r in instance.get("resources", [])}
    total = 0.0
    for task_id, timing in timings.items():
        duration = timing["finish_time"] - timing["start_time"]
        for rid, demand in task_map[task_id].get("resource_requirements", {}).items():
            if rid not in resource_map:
                raise ValueError(f"Unknown resource '{rid}'.")
            rate = resource_map[rid].get("cost_per_unit_time")
            if not _finite_number(rate) or rate < 0:
                raise ValueError(f"Resource '{rid}' has invalid cost rate.")
            total += float(demand) * duration * float(rate)
    return total


def calculate_resource_utilization(instance, timings):
    makespan = calculate_makespan(timings)
    resources = instance.get("resources", [])
    if makespan <= 0 or not resources:
        return 0.0
    task_map = {t["task_id"]: t for t in instance.get("tasks", [])}
    busy = 0.0
    for task_id, timing in timings.items():
        duration = timing["finish_time"] - timing["start_time"]
        busy += sum(float(x) for x in task_map[task_id].get("resource_requirements", {}).values()) * duration
    total_capacity = sum(float(r["capacity"]) for r in resources)
    denominator = makespan * total_capacity
    return busy / denominator if denominator > 0 else 0.0


def evaluate_schedule(instance, schedule):
    """Return objective metrics for a feasible schedule; reject infeasible input."""
    report = check_schedule(instance, schedule)
    if not report["feasible"]:
        raise ValueError("Schedule is infeasible: " + " | ".join(report["violations"]))
    timings = report["timings"]
    return {
        "feasible": True,
        "makespan": calculate_makespan(timings),
        "total_resource_cost": calculate_resource_cost(instance, timings),
        "total_tardiness": calculate_total_tardiness(instance, timings),
        "resource_utilization": calculate_resource_utilization(instance, timings),
        "timings": timings,
    }
