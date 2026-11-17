# Week 4 — Scheduling Generator Validation

This test suite validates the synthetic scheduling problem generator and supports
the Week 4 deliverable: **Validated Scheduling Dataset v1**.

## Run tests

From the repository root:

```bash
python -m unittest discover -s tests -v
```

No third-party test dependency is required; the suite uses Python's built-in
`unittest` framework.

## Test coverage

| Area | Checks |
|---|---|
| IDs and structure | Unique task/resource IDs; task and resource counts |
| Processing and deadlines | Positive processing times; non-negative deadlines |
| Precedence constraints | Earlier-task edges, no self-dependency, unknown predecessor rejection |
| DAG edge cases | No dependencies and maximum forward-edge density |
| Resource constraints | Known resources; positive demand; demand not above capacity |
| Characteristics | Dependency density, cost variation, and count fields |
| Reproducibility | Same seed produces identical instance; different seeds vary |
| Difficulty profiles | Easy, medium, hard, and mixed instances validate |
| Output files | JSON instances, CSV summary, and manifest are written correctly |

## Important scope note

The current validator checks the generator's topological ID ordering, which
guarantees acyclicity for instances produced by this generator. It is not yet a
general-purpose graph-cycle detector for arbitrary task-ID schemes. Likewise,
the current generator assigns each task one or two randomly selected resource
types; it does not model resource/task compatibility matrices or prove global
schedule feasibility. Those are documented limitations, not claims made by
these tests.

The current resource-scarcity feature is a descriptive proxy based on average
aggregate demand relative to capacity. It is not a measured peak-load ratio or
a feasibility certificate. Deadline tightness is the sampled generation
parameter; it is not recalculated from a solved schedule.
