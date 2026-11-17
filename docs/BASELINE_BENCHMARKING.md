# Baseline Benchmarking Engine

## Purpose

The benchmark runner evaluates FCFS, SPT and Critical Path using exactly the
same resource-constrained scheduling instances and deterministic evaluator.
HEFT is excluded from this runner because it uses a different heterogeneous
processor DAG problem model.

## 1. Generate a reproducible dataset

From the repository root:

```powershell
py src/generators/scheduling_generator.py --count 100 --profile mixed --seed 42 --output scheduling_dataset
```

The generator writes JSON instances under `scheduling_dataset/instances/`.
Keep the generated dataset fixed when comparing algorithms. Record its seed
and generator settings with experiment results.

## 2. Run the baseline benchmark

```powershell
py -m src.experiments.run_baselines --dataset scheduling_dataset --output baseline_results
```

The runner executes all three algorithms on every loaded instance. It records
scheduler execution time separately; schedule validation and objective
calculation are excluded from that timing.

## 3. Output files

- `results.json`: complete experiment metadata, per-run schedules, metrics,
  errors and aggregate summaries.
- `per_instance.csv`: one row per instance/algorithm pair, including problem
  characteristics and objective values.
- `summary_by_profile.csv`: mean, median and sample standard deviation by
  profile and algorithm.
- `summary_overall.csv`: corresponding aggregate statistics by algorithm.

For every metric, the summary reports mean, median and standard deviation.
Failed runs remain visible in the report and count toward the failure rate;
they are not silently discarded. Metric statistics are computed only from
successful runs. A successful run means the scheduler returned a schedule
accepted by the deterministic evaluator.

## 4. Metrics and directions

| Metric | Direction |
|---|---|
| Makespan | Minimize |
| Total resource cost | Minimize |
| Total tardiness | Minimize |
| Resource utilization | Maximize |
| Scheduler runtime (ms) | Minimize |

The current evaluator's resource utilization is aggregate resource-time
utilization over makespan. Interpret it alongside makespan and resource
capacities rather than as a standalone quality score.

## 5. Reproducibility and fair comparison

1. Generate one dataset with a fixed seed.
2. Run all algorithms against the same JSON instance files.
3. Do not regenerate or edit instances between algorithm runs.
4. Keep generator parameters, seed, repository commit and Python version with
   the experiment archive.
5. Repeat runtime experiments if making strong claims about small timing
   differences; runtime is environment-dependent.
6. Do not pool HEFT numbers into this report. HEFT needs a separate report
   for its heterogeneous processor DAG dataset and metrics.

## 6. Tests

```powershell
py -m unittest discover -s tests -p test_benchmark_runner.py -v
py -m unittest discover -s tests -v
```

## Files

- `src/experiments/run_baselines.py`
- `tests/test_benchmark_runner.py`
