#!/usr/bin/env python3
"""
Synthetic Scheduling Problem Generator

Generates reproducible non-preemptive resource-constrained project scheduling
instances for experiments on problem-characteristic-aware coordination.

Each instance includes tasks (durations, soft deadlines, resource demands and
precedence constraints), renewable resources (capacity and cost), and measured
problem characteristics. Dependencies are generated as a DAG.

Usage:
    python src/generators/scheduling_generator.py
    python src/generators/scheduling_generator.py --count 100 --profile hard
    python src/generators/scheduling_generator.py --count 50 --profile mixed --seed 7
    python src/generators/scheduling_generator.py --tasks-min 30 --tasks-max 60

Profiles: easy, medium, hard, mixed
Only Python's standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path
from typing import Any

PROFILES = {
    "easy": {
        "deadline_tightness": (0.10, 0.30),
        "capacity_factor": (1.60, 2.20),
        "cost_variation": (0.05, 0.25),
        "dependency_density": (0.02, 0.10),
        "task_count": (8, 18),
        "resource_count": (3, 5),
        "duration": (1, 8),
    },
    "medium": {
        "deadline_tightness": (0.35, 0.60),
        "capacity_factor": (1.10, 1.60),
        "cost_variation": (0.20, 0.60),
        "dependency_density": (0.10, 0.25),
        "task_count": (15, 35),
        "resource_count": (3, 7),
        "duration": (2, 14),
    },
    "hard": {
        "deadline_tightness": (0.65, 0.90),
        "capacity_factor": (0.85, 1.20),
        "cost_variation": (0.50, 1.00),
        "dependency_density": (0.25, 0.45),
        "task_count": (30, 60),
        "resource_count": (4, 8),
        "duration": (3, 20),
    },
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def generate_instance(
    instance_id: int,
    rng: random.Random,
    profile_name: str,
    tasks_override: int | None = None,
    resources_override: int | None = None,
) -> dict[str, Any]:
    """Generate one reproducible scheduling instance."""
    if profile_name == "mixed":
        profile_name = rng.choice(["easy", "medium", "hard"])
    profile = PROFILES[profile_name]

    n_tasks = tasks_override or rng.randint(*profile["task_count"])
    n_resources = resources_override or rng.randint(*profile["resource_count"])
    tightness = rng.uniform(*profile["deadline_tightness"])
    capacity_factor = rng.uniform(*profile["capacity_factor"])
    cost_variation = rng.uniform(*profile["cost_variation"])
    dependency_density = rng.uniform(*profile["dependency_density"])

    resources = []
    for index in range(n_resources):
        base_capacity = max(2, round((n_tasks / n_resources) ** 0.5 * 2))
        capacity = max(2, round(base_capacity * capacity_factor))
        base_cost = rng.uniform(1.0, 10.0)
        cost = base_cost * (1 + rng.uniform(-cost_variation, cost_variation))
        resources.append({
            "resource_id": f"R{index + 1}",
            "capacity": capacity,
            "cost_per_unit_time": round(max(0.10, cost), 2),
        })

    min_duration, max_duration = profile["duration"]
    tasks = []
    for index in range(n_tasks):
        duration = rng.randint(min_duration, max_duration)
        count = rng.choices([1, 2], weights=[0.75, 0.25])[0]
        selected = rng.sample(range(n_resources), k=min(count, n_resources))
        requirements = {
            resources[r]["resource_id"]: rng.randint(1, resources[r]["capacity"])
            for r in selected
        }
        predecessors = [
            f"T{previous + 1}"
            for previous in range(index)
            if rng.random() < dependency_density
        ]
        tasks.append({
            "task_id": f"T{index + 1}",
            "processing_time": duration,
            "deadline": None,
            "resource_requirements": requirements,
            "predecessors": predecessors,
        })

    # Precedence-only earliest finishes; predecessor indices always decrease,
    # so the generated dependency graph is acyclic by construction.
    earliest_finish: dict[str, int] = {}
    for task in tasks:
        start = max(
            (earliest_finish[p] for p in task["predecessors"]),
            default=0,
        )
        finish = start + task["processing_time"]
        earliest_finish[task["task_id"]] = finish
        slack_fraction = (1.0 - tightness) * rng.uniform(0.25, 1.25)
        task["deadline"] = finish + max(0, round(finish * slack_fraction))

    total_demand = {r["resource_id"]: 0 for r in resources}
    for task in tasks:
        for rid, amount in task["resource_requirements"].items():
            total_demand[rid] += amount

    scarcity = statistics.mean(
        clamp(
            total_demand[r["resource_id"]] / max(1, n_tasks) / r["capacity"],
            0.0,
            1.0,
        )
        for r in resources
    )
    possible_edges = n_tasks * (n_tasks - 1) / 2
    edge_count = sum(len(t["predecessors"]) for t in tasks)
    actual_density = edge_count / possible_edges if possible_edges else 0.0
    costs = [r["cost_per_unit_time"] for r in resources]
    cost_cv = statistics.pstdev(costs) / statistics.mean(costs) if len(costs) > 1 else 0.0

    return {
        "instance_id": f"SP-{instance_id:04d}",
        "profile": profile_name,
        "problem_type": "non_preemptive_resource_constrained_project_scheduling",
        "time_unit": "abstract_time_unit",
        "deadline_semantics": "soft",
        "objective_directions": {
            "makespan": "minimize",
            "total_resource_cost": "minimize",
            "total_tardiness": "minimize",
            "resource_utilization": "maximize",
        },
        "tasks": tasks,
        "resources": resources,
        "characteristics": {
            "task_count": n_tasks,
            "resource_count": n_resources,
            "deadline_tightness": round(tightness, 4),
            "resource_scarcity_proxy": round(scarcity, 4),
            "cost_variation_cv": round(cost_cv, 4),
            "dependency_density": round(actual_density, 4),
            "capacity_factor": round(capacity_factor, 4),
        },
        "notes": [
            "Precedence constraints form a directed acyclic graph.",
            "Deadlines are soft; tardiness is permitted and can be penalized.",
            "Resource capacities are renewable and shared by concurrent tasks.",
            "The scarcity proxy is descriptive, not a feasibility certificate.",
            "The generator does not solve or optimize the scheduling instance.",
        ],
    }


def validate_instance(instance: dict[str, Any]) -> None:
    """Check IDs, positive durations, resource demands and DAG ordering."""
    resource_map = {r["resource_id"]: r for r in instance["resources"]}
    tasks = instance["tasks"]
    task_ids = {t["task_id"] for t in tasks}
    if len(task_ids) != len(tasks):
        raise ValueError("Duplicate task IDs.")

    for task in tasks:
        if task["processing_time"] <= 0:
            raise ValueError(f"{task['task_id']}: processing time must be positive.")
        for predecessor in task["predecessors"]:
            if predecessor not in task_ids:
                raise ValueError(f"Unknown predecessor: {predecessor}")
            if int(predecessor[1:]) >= int(task["task_id"][1:]):
                raise ValueError("Dependency order is invalid; expected a DAG.")
        for rid, amount in task["resource_requirements"].items():
            if rid not in resource_map:
                raise ValueError(f"Unknown resource: {rid}")
            if amount <= 0 or amount > resource_map[rid]["capacity"]:
                raise ValueError(f"Invalid demand for {rid} in {task['task_id']}.")


def save_dataset(instances: list[dict[str, Any]], output_dir: Path) -> None:
    instance_dir = output_dir / "instances"
    instance_dir.mkdir(parents=True, exist_ok=True)

    for instance in instances:
        validate_instance(instance)
        path = instance_dir / f"{instance['instance_id']}.json"
        path.write_text(json.dumps(instance, indent=2), encoding="utf-8")

    columns = [
        "instance_id", "profile", "task_count", "resource_count",
        "deadline_tightness", "resource_scarcity_proxy",
        "cost_variation_cv", "dependency_density", "capacity_factor",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for instance in instances:
            row = {
                "instance_id": instance["instance_id"],
                "profile": instance["profile"],
                **instance["characteristics"],
            }
            writer.writerow({key: row[key] for key in columns})

    manifest = {
        "number_of_instances": len(instances),
        "profiles": sorted({x["profile"] for x in instances}),
        "instances_directory": "instances/",
        "summary_file": "summary.csv",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic resource-constrained scheduling instances."
    )
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--profile", choices=["easy", "medium", "hard", "mixed"], default="mixed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("scheduling_dataset"))
    parser.add_argument("--tasks-min", type=int)
    parser.add_argument("--tasks-max", type=int)
    parser.add_argument("--resources-min", type=int)
    parser.add_argument("--resources-max", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count < 1:
        raise SystemExit("--count must be at least 1.")

    rng = random.Random(args.seed)
    instances = []
    for number in range(1, args.count + 1):
        selected_profile = (
            rng.choice(["easy", "medium", "hard"])
            if args.profile == "mixed"
            else args.profile
        )
        default_tasks = PROFILES[selected_profile]["task_count"]
        default_resources = PROFILES[selected_profile]["resource_count"]

        task_low = args.tasks_min if args.tasks_min is not None else default_tasks[0]
        task_high = args.tasks_max if args.tasks_max is not None else default_tasks[1]
        resource_low = args.resources_min if args.resources_min is not None else default_resources[0]
        resource_high = args.resources_max if args.resources_max is not None else default_resources[1]

        if task_low < 1 or task_high < task_low:
            raise SystemExit("Invalid task range.")
        if resource_low < 1 or resource_high < resource_low:
            raise SystemExit("Invalid resource range.")

        instance = generate_instance(
            number,
            rng,
            selected_profile,
            rng.randint(task_low, task_high),
            rng.randint(resource_low, resource_high),
        )
        validate_instance(instance)
        instances.append(instance)

    save_dataset(instances, args.output)
    print(f"Generated {len(instances)} instances in: {args.output.resolve()}")
    print(f"JSON files: {args.output / 'instances'}")
    print(f"CSV summary: {args.output / 'summary.csv'}")
    print(f"Seed: {args.seed}")


if __name__ == "__main__":
    main()
