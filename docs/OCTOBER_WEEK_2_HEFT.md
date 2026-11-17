# October Week 2 — HEFT Baseline and Heterogeneous Scheduling Instances

## Goal

Implement the standard list-based Heterogeneous Earliest Finish Time (HEFT)
baseline and generate benchmark instances whose problem model matches HEFT.

## Why a separate problem model?

The existing generator models non-preemptive tasks consuming renewable resources
with capacities and costs. Standard HEFT models a DAG mapped to heterogeneous
processors. Each task has a processor-specific execution time, and dependency
edges have communication costs.

Combining these assumptions in one schema would make the baseline ambiguous.
The HEFT generator and scheduler are therefore separate modules. HEFT results
must not be directly compared with the original resource-constrained instances
without first defining a common model and evaluation protocol.

## Instance model

- processors: processor identifiers.
- tasks: task identifiers, predecessor lists, and execution time on every processor.
- communication_costs: fixed communication cost for every dependency edge,
  keyed as PARENT->CHILD.

The seeded generator creates a DAG by allowing a task to depend only on tasks
generated earlier.

## HEFT implementation

### 1. Upward rank

For task i:

rank_u(i) = average_compute(i) + max(average_communication(i,j) + rank_u(j))

The maximum is zero for exit tasks. Average compute time is calculated over all processors. For upward-rank
calculation, the fixed edge transfer cost is averaged over all ordered processor
pairs: same-processor communication is zero and different-processor communication
uses the edge cost. During actual processor selection, the full edge cost is
charged only when predecessor and child are mapped to different processors.

### 2. Task priority

Tasks are sorted by descending upward rank. Stable topological order breaks ties.

### 3. Processor selection

For each task, HEFT computes the earliest time its predecessors' data is
available on each candidate processor. Communication cost is zero when a
predecessor and its dependent task are mapped to the same processor; otherwise
the edge communication cost is added.

The task is assigned to the processor producing the minimum earliest finish time.

### 4. Insertion-based scheduling

The scheduler searches for the earliest available gap on the selected processor.
Tasks are non-preemptive and a processor runs at most one task at a time.

## Assumptions and limitations

- Non-preemptive task execution.
- One processor per task.
- One task at a time per processor.
- Fixed communication cost per dependency edge.
- No network contention, bandwidth sharing, processor failures, or migration.
- No renewable resource-capacity constraints, soft deadlines, or resource-cost
  objective in this HEFT-specific benchmark model.

This is a standard HEFT-style model with insertion-based EFT processor selection.
It is not the same model as the project's original resource-constrained
scheduling instances.

## Run

Generate 30 instances with 20 tasks and 4 processors:

    py src/generators/heft_instance_generator.py --count 30 --tasks 20 --processors 4 --seed 42

Output is written to heft_dataset/.\n\nRun HEFT on one generated instance:\n\n    py -m src.baselines.heft heft_dataset/instances/HEFT-0001.json --output heft_schedule.json

Run HEFT tests:

    py -m unittest discover -s tests -p test_heft.py -v

Run the full test suite:

    py -m unittest discover -s tests -v

## Files

- src/generators/heft_instance_generator.py
- src/baselines/heft.py
- tests/test_heft.py
