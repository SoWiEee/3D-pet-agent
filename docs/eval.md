# Evaluation Report — Phase 10

> This document is the **canonical evaluation result** for the bundled
> demo dataset. It is produced by the same harness that runs in CI, so
> the numbers below match what `pytest` enforces in
> `tests/test_evaluation.py::test_runner_on_bundled_dataset_meets_threshold`.

## How to reproduce

```bash
.venv/bin/python main.py --mode eval \
  --dataset samples/eval_dataset.jsonl \
  --out runs
```

Output lands under `runs/eval_<timestamp>/`:

| File | Purpose |
|---|---|
| `report.md` | Human-readable summary (same shape as this doc) |
| `records.csv` | Per-trial fields for spreadsheet analysis |
| `records.jsonl` | Raw `EvaluationRecord` per line (`src/evaluation/schema.py`) |

The harness is in-process and deterministic — no perception models are
loaded; the dataset is pre-grounded so the run is fast and stable.

## Dataset coverage (spec §13.1)

| Trial | Scenario | Status |
|---|---|---|
| `t01_navigate` | Object navigation — "go to the cup" | ✅ |
| `t02_relation_hide` | Spatial relation — "hide behind the keyboard" | ✅ |
| `t03_avoidance` | Avoidance — "go to the box but avoid the mouse" | ✅ |
| `t04_exploration` | Exploration — "explore the desk" | ✅ |
| `t05_clarification` | Ambiguity — "go to the cup" with two cups | ✅ |
| `t06_stale_memory` | Persistent memory — `tracking_status=stale` | ✅ |
| `t07_no_match` | Negative — "go to the apple" with no apple | ✅ |
| `t08_look_at` | Look-only — no path required | ✅ |

All six required spec §13.1 demo scenarios are exercised; trials 7 and 8
add a negative case and a no-path case for completeness.

## Latest run (2026-06-02)

### Aggregate Metrics

| Metric | Value |
|---|---|
| Trials | 8 |
| Grounding success rate | 75.0% |
| Path success rate | 62.5% |
| **Task success rate** | **100.0%** |
| Mean latency | 8.3 ms |
| p95 latency | 34.0 ms |
| Mean cross-track error | 0.056 m |
| Mean heading error | 0.767 rad |
| Total collisions | 0 |

Notes:

- **Grounding success rate** is 75% because clarification / no-match
  trials (t05, t07) intentionally do not produce a single grounded
  target — they are still considered task-successful because the
  predicted outcome matches the expected outcome.
- **Path success rate** is 62.5% for the same reason — clarification,
  no-match, and look-at trials never invoke the planner.
- **Task success rate** is the headline number — it accounts for the
  expected-vs-predicted outcome match across all scenario types and is
  what the CI gate enforces.

### Per-Trial Results

| Trial | Command | Expected | Predicted | Task | Latency (ms) |
|---|---|---|---|---|---|
| t01_navigate | go to the cup | navigate/cup_001 | navigate/cup_001 | OK | 7.5 |
| t02_relation_hide | hide behind the keyboard | hide/kbd_001 | hide/kbd_001 | OK | 1.3 |
| t03_avoidance | go to the box but avoid the mouse | navigate/box_001 | navigate/box_001 | OK | 6.3 |
| t04_exploration | explore the desk | explore/— | explore/— | OK | 48.3 |
| t05_clarification | go to the cup | clarification/— | clarification/— | OK | 0.2 |
| t06_stale_memory | go to the cup | navigate/cup_001 | navigate/cup_001 | OK | 2.5 |
| t07_no_match | go to the apple | no_match/— | no_match/— | OK | 0.1 |
| t08_look_at | look at the monitor | look_at/monitor_001 | look_at/monitor_001 | OK | 0.1 |

## CI gate

`run_eval` exits with a non-zero status when task success rate drops
below 50%, so this dataset is wired into CI as a smoke gate. The
in-process test `test_runner_on_bundled_dataset_meets_threshold`
enforces the same threshold inside `pytest`.

## Spec §13.2 metric mapping

| Spec metric | Source field |
|---|---|
| Grounding accuracy | `EvaluationRecord.grounding_success` |
| Path success rate | `EvaluationRecord.path_success` |
| Collision count | `EvaluationRecord.collision_count` (sampled at waypoints) |
| Task success rate | `EvaluationRecord.task_success` |
| Cross-track error | `controller_metrics.max_cross_track_error_m` |
| Latency | `EvaluationRecord.latency_ms` (command → final action) |

Metrics specific to runtime perception (detection recall, mask quality,
depth stability, FPS) are not measured here because the harness skips
the perception models for determinism. Those numbers are produced
during live demo runs and would land in a separate `runs/perception_*`
report.
