"""Standard list-based HEFT scheduler for heterogeneous processor DAGs."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Any


def _topological_order(instance: dict[str, Any]) -> list[str]:
    tasks = {t["task_id"]: t for t in instance["tasks"]}
    indegree = {tid: 0 for tid in tasks}
    successors = {tid: [] for tid in tasks}
    for tid, task in tasks.items():
        for predecessor in task.get("predecessors", []):
            if predecessor not in tasks:
                raise ValueError(f"Unknown predecessor {predecessor} for {tid}.")
            indegree[tid] += 1
            successors[predecessor].append(tid)
    queue = [tid for tid in tasks if indegree[tid] == 0]
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
    return order


def calculate_upward_ranks(instance: dict[str, Any]) -> dict[str, float]:
    """Compute upward ranks using average execution and edge communication costs."""
    tasks = {t["task_id"]: t for t in instance["tasks"]}
    processors = [p["processor_id"] for p in instance["processors"]]
    communication = instance.get("communication_costs", {})
    order = _topological_order(instance)
    successors = {tid: [] for tid in tasks}
    for tid, task in tasks.items():
        for predecessor in task.get("predecessors", []):
            successors[predecessor].append(tid)
    ranks = {}
    for tid in reversed(order):
        average_compute = sum(float(tasks[tid]["execution_times"][pid])
                              for pid in processors) / len(processors)
        average_pair_factor = (len(processors) - 1) / len(processors)
        downstream = max(
            (float(communication[`{tid}->{child}`]) * average_pair_factor + ranks[child]
             for child in successors[tid]), default=0.0)
        ranks[tid] = average_compute + downstream
    return ranks


def _earliest_gap(ready_time: float, duration: float,
                  reservations: list[tuple[float, float]]) -> float:
    """Find the earliest non-overlapping processor interval (gap insertion)."""
    candidate = ready_time
    for start, finish in sorted(reservations):
        if candidate + duration <= start + 1e-12:
            return candidate
        if candidate < finish - 1e-12 and candidate + duration > start + 1e-12:
            candidate = finish
    return candidate


def schedule_heft(instance: dict[str, Any]) -> dict[str, Any]:
    """Return assignments, HEFT task priority order, ranks and makespan."""
    processors = [p["processor_id"] for p in instance.get("processors", [])]
    task_list = instance.get("tasks", [])
    if not processors:
        raise ValueError("HEFT requires at least one processor.")
    if not task_list:
        return {"assignments": [], "task_order": [], "upward_ranks": {}, "makespan": 0.0}
    task_map = {t["task_id"]: t for t in task_list}
    if len(task_map) != len(task_list):
        raise ValueError("Task IDs must be unique.")
    for tid, task in task_map.items():
        times = task.get("execution_times", {})
        if set(times) != set(processors):
            raise ValueError(f"{tid}: execution time required for every processor.")
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) or v <= 0 for v in times.values()):
            raise ValueError(f"{tid}: execution times must be finite and positive.")
    topological = _topological_order(instance)
    ranks = calculate_upward_ranks(instance)
    position = {tid: i for i, tid in enumerate(topological)}
    priority = sorted(topological, key=lambda tid: (-ranks[tid], position[tid]))
    reservations = {pid: [] for pid in processors}
    assigned = {}
    communication = instance.get("communication_costs", {})

    for tid in priority:
        task = task_map[tid]
        best = None
        for processor_id in processors:
            ready = 0.0
            for predecessor in task.get("predecessors", []):
                parent = assigned[predecessor]
                transfer = (0.0 if parent["processor_id"] == processor_id else
                            float(communication[f"{predecessor}->{tid}"]))
                ready = max(ready, parent["finish_time"] + transfer)
            duration = float(task["execution_times"][processor_id])
            start = _earliest_gap(ready, duration, reservations[processor_id])
            finish = start + duration
            candidate = (finish, start, processors.index(processor_id), processor_id)
            if best is None or candidate < best:
                best = candidate
        finish, start, _, processor_id = best
        assigned[tid] = {"task_id": tid, "processor_id": processor_id,
                         "start_time": start, "finish_time": finish}
        reservations[processor_id].append((start, finish))

    assignments = sorted(assigned.values(),
                         key=lambda x: (x["start_time"], x["processor_id"], x["task_id"]))
    makespan = max(x["finish_time"] for x in assignments)
    return {"assignments": assignments, "task_order": priority,
            "upward_ranks": ranks, "makespan": makespan}


def main() -> None:
    """Run HEFT on a JSON instance and write the resulting schedule."""
    parser = argparse.ArgumentParser(description="Run the HEFT scheduler.")
    parser.add_argument("instance", type=Path, help="Path to a HEFT instance JSON file.")
    parser.add_argument("--output", type=Path, default=Path("heft_schedule.json"))
    args = parser.parse_args()
    instance = json.loads(args.instance.read_text(encoding="utf-8"))
    result = schedule_heft(instance)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"HEFT makespan: {result['makespan']:.3f}")
    print(f"Schedule written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
