# October — Week 1: Deterministic Scheduling and Evaluation Foundation

## Goal
Implement a deterministic, model-independent schedule validator and objective
evaluator. This trusted layer will later compare rule-based, single-LLM and
multi-agent scheduling approaches.

## Deliverables
- src/evaluation/schedule_evaluator.py: hard-constraint checker and metrics.
- tests/test_schedule_evaluator.py: unit tests for constraints and objectives.

## Schedule representation
A schedule contains one assignment per task, with task_id and start_time.
Finish time is derived as start_time + processing_time. Intervals are half-open
[start, finish), so back-to-back tasks do not overlap.

## Hard constraints
1. Every task appears exactly once.
2. No unknown task IDs or malformed assignments.
3. Start times are finite, numeric and non-negative.
4. Each predecessor finishes before its successor starts.
5. Resource demands reference known resources and do not exceed capacity.
6. Concurrent demand for a renewable resource never exceeds its capacity.

Deadlines are soft in this problem formulation. Deadline misses therefore do
not invalidate a schedule; they contribute to total tardiness.

## Objective metrics
- Makespan: maximum completion time.
- Total resource cost: sum of demand × processing duration × resource unit-time cost.
- Total tardiness: sum of max(0, completion time − deadline).
- Aggregate resource utilization: total resource-unit time consumed divided by
  makespan × sum of resource capacities.

Metrics are reported separately. Do not combine them into a weighted score until
objective weights are justified, or a Pareto comparison method is specified.

## Run tests
From the repository root:
py -m unittest discover -s tests -v

## Scope
The evaluator assumes non-preemptive tasks and renewable resources with constant
resource usage during execution. It checks feasibility; it does not construct,
optimize, or prove optimality of schedules.
