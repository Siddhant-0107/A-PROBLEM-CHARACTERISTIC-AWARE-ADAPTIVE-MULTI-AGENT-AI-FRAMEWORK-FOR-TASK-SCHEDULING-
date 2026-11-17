"""Generate reproducible heterogeneous-computing DAG instances for standard HEFT.

Kept separate from scheduling_generator.py because HEFT uses processor-specific
execution times and edge communication costs, not renewable resource capacities.
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from typing import Any


def generate_heft_instance(instance_id: int, rng: random.Random, task_count: int = 12,
                           processor_count: int = 3,
                           dependency_probability: float = 0.2) -> dict[str, Any]:
    if task_count < 1 or processor_count < 1:
        raise ValueError("Task and processor counts must be at least 1.")
    if not 0.0 <= dependency_probability <= 1.0:
        raise ValueError("dependency_probability must be between 0 and 1.")
    processors = [{"processor_id": f"P{i + 1}"} for i in range(processor_count)]
    processor_ids = [p["processor_id"] for p in processors]
    speed = {pid: rng.uniform(0.65, 1.65) for pid in processor_ids}
    tasks = []
    for index in range(task_count):
        work = rng.randint(4, 20)
        execution_times = {pid: round(max(1.0, work / speed[pid]), 3) for pid in processor_ids}
        predecessors = [f"T{j + 1}" for j in range(index)
                        if rng.random() < dependency_probability]
        tasks.append({"task_id": f"T{index + 1}", "predecessors": predecessors,
                      "execution_times": execution_times})
    communication_costs = {}
    for task in tasks:
        for predecessor in task["predecessors"]:
            communication_costs[f"{predecessor}->{task['task_id']}"] = round(rng.uniform(0.5, 5.0), 3)
    instance = {
        "instance_id": f"HEFT-{instance_id:04d}",
        "problem_type": "heterogeneous_processors_dag_scheduling",
        "processors": processors,
        "tasks": tasks,
        "communication_costs": communication_costs,
        "assumptions": [
            "Tasks are non-preemptive.",
            "Each task executes on exactly one processor.",
            "A processor executes at most one task at a time.",
            "Communication cost is paid when dependent tasks use different processors.",
            "Communication is a fixed edge cost; network contention is ignored.",
        ],
    }
    validate_heft_instance(instance)
    return instance


def validate_heft_instance(instance: dict[str, Any]) -> None:
    processors, tasks = instance.get("processors"), instance.get("tasks")
    if not isinstance(processors, list) or not processors:
        raise ValueError("Instance must contain at least one processor.")
    if not isinstance(tasks, list):
        raise ValueError("Instance must contain a tasks list.")
    processor_ids = [p.get("processor_id") for p in processors]
    if len(processor_ids) != len(set(processor_ids)) or any(not x for x in processor_ids):
        raise ValueError("Processor IDs must be unique and non-empty.")
    processor_set = set(processor_ids)
    task_ids = [t.get("task_id") for t in tasks]
    if len(task_ids) != len(set(task_ids)) or any(not x for x in task_ids):
        raise ValueError("Task IDs must be unique and non-empty.")
    task_set, seen, edges = set(task_ids), set(), set()
    for task in tasks:
        task_id = task["task_id"]
        predecessors = task.get("predecessors")
        if not isinstance(predecessors, list):
            raise ValueError(f"{task_id}: predecessors must be a list.")
        for predecessor in predecessors:
            if predecessor not in task_set:
                raise ValueError(f"{task_id}: unknown predecessor {predecessor}.")
            if predecessor == task_id or predecessor not in seen:
                raise ValueError(f"{task_id}: dependency order is cyclic or invalid.")
            edges.add(f"{predecessor}->{task_id}")
        times = task.get("execution_times")
        if not isinstance(times, dict) or set(times) != processor_set:
            raise ValueError(f"{task_id}: execution_times must cover every processor.")
        for value in times.values():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{task_id}: execution times must be positive numbers.")
        seen.add(task_id)
    costs = instance.get("communication_costs", {})
    if set(costs) != edges:
        raise ValueError("Communication costs must be supplied for every dependency edge.")
    for value in costs.values():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError("Communication costs must be non-negative numbers.")


def save_heft_dataset(instances: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    instance_dir = output_dir / "instances"
    instance_dir.mkdir(parents=True, exist_ok=True)
    for instance in instances:
        validate_heft_instance(instance)
        (instance_dir / f"{instance['instance_id']}.json").write_text(
            json.dumps(instance, indent=2), encoding="utf-8")
    manifest = {"number_of_instances": len(instances), "instances_directory": "instances/",
                "problem_type": "heterogeneous_processors_dag_scheduling"}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate heterogeneous DAG instances for HEFT.")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--tasks", type=int, default=20)
    parser.add_argument("--processors", type=int, default=4)
    parser.add_argument("--dependency-probability", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("heft_dataset"))
    args = parser.parse_args()
    if args.count < 1:
        raise SystemExit("--count must be at least 1.")
    rng = random.Random(args.seed)
    instances = [generate_heft_instance(i, rng, args.tasks, args.processors,
                                        args.dependency_probability)
                 for i in range(1, args.count + 1)]
    save_heft_dataset(instances, args.output)
    print(f"Generated {len(instances)} HEFT instances in {args.output.resolve()}")


if __name__ == "__main__":
    main()
