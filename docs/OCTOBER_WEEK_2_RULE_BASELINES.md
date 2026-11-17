# October Week 2 — Rule-Based Scheduling Baselines

## Goal

Provide three deterministic, feasible scheduling baselines for the original
non-preemptive renewable-resource scheduling model:

1. FCFS (input-order proxy)
2. SPT (Shortest Processing Time)
3. Critical Path Priority (longest remaining downstream path)

These are separate from standard HEFT, which uses a heterogeneous-processor
DAG model and its own compatible generator.

## Shared scheduling mechanism

All three policies use the same serial schedule-generation procedure:

1. Find tasks whose predecessors have already been scheduled.
2. Choose one ready task using the policy's priority rule.
3. Compute its earliest precedence-feasible start time.
4. Find the earliest interval where all renewable resource demands fit within
   their capacities for the task's entire non-preemptive duration.
5. Reserve that interval and repeat.
6. Validate the final schedule with src/evaluation/schedule_evaluator.py.

This isolates the policy difference: the priority rule changes, while resource
feasibility and validation remain common.

## Policy definitions

### FCFS

The current instance schema does not contain task arrival times. FCFS is
therefore operationalized as stable original input-order priority among tasks
that are currently precedence-ready. This is explicitly an input-order proxy,
not a claim about real arrival-time FCFS.

### SPT

Among currently ready tasks, choose the task with the shortest processing time.
Ties are broken by original input order.

### Critical Path Priority

For every task, calculate its remaining downstream path length:

critical_length(i) = processing_time(i) + max(critical_length(successor))

The maximum is zero for exit tasks. Among ready tasks, choose the one with the
largest remaining path length. Ties are broken by original input order.

## Resource feasibility

Tasks are non-preemptive. Resource requirements remain constant for the full
execution interval. A task starts only when every required renewable resource
has enough remaining capacity throughout its complete interval. Deadlines remain
soft, consistent with the deterministic evaluator.

If a task requests more than a resource's total capacity, the instance is
rejected rather than producing an invalid schedule.

## Run

Run all three schedulers from Python:

    from src.baselines.rule_based import (
        schedule_fcfs, schedule_spt, schedule_critical_path
    )

    schedule = schedule_fcfs(instance)
    schedule = schedule_spt(instance)
    schedule = schedule_critical_path(instance)

Each result contains:
- policy
- priority_order
- assignments in the evaluator-compatible format

Evaluate metrics with:

    from src.evaluation.schedule_evaluator import evaluate_schedule
    metrics = evaluate_schedule(instance, schedule)

## Tests

Run the focused tests:

    py -m unittest discover -s tests -p test_rule_based_baselines.py -v

Run the full suite:

    py -m unittest discover -s tests -v

## Files

- src/baselines/rule_based.py
- tests/test_rule_based_baselines.py

## Methodological note

FCFS, SPT and Critical Path Priority can be compared on the original generated
resource-constrained instances. Standard HEFT must be evaluated on its separate
heterogeneous processor DAG instances. Do not combine their raw metrics in one
ranking without defining a shared problem model or a carefully separated
benchmark protocol.
