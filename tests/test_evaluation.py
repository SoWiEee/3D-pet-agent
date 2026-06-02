"""Phase 10 — evaluation harness + metrics + report tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation import (
    DatasetEntry,
    DatasetScene,
    EvaluationRecord,
    EvaluationRunner,
    summarize,
    write_report,
)
from src.evaluation.metrics import _percentile
from src.evaluation.runner import load_dataset
from src.evaluation.schema import ControllerMetrics, DatasetSceneObject


def test_dataset_entry_roundtrip() -> None:
    entry = DatasetEntry(
        trial_id="t01",
        scene=DatasetScene(
            scene_id="s01",
            objects=[
                DatasetSceneObject(
                    object_id="cup_001",
                    class_label="cup",
                    center_3d_world=(0.5, 0.0, 0.5),
                )
            ],
        ),
        command="go to the cup",
        expected_outcome="navigate",
        expected_target="cup_001",
    )
    rebuilt = DatasetEntry(**entry.model_dump())
    assert rebuilt == entry


def test_evaluation_record_defaults() -> None:
    rec = EvaluationRecord(
        trial_id="t01",
        scene_id="s01",
        command="go to the cup",
        expected_outcome="navigate",
    )
    d = rec.to_dict()
    assert "controller_metrics" in d
    assert d["task_success"] is False
    assert d["collision_count"] == 0


def _record(*, success: bool, latency: float, xte: float = 0.0) -> EvaluationRecord:
    return EvaluationRecord(
        trial_id="x",
        scene_id="s",
        command="c",
        expected_outcome="navigate",
        task_success=success,
        grounding_success=success,
        path_success=success,
        latency_ms=latency,
        controller_metrics=ControllerMetrics(max_cross_track_error_m=xte),
    )


def test_summary_empty_input() -> None:
    s = summarize([])
    assert s.trials == 0
    assert s.task_success_rate == 0.0


def test_summary_rates_and_latency() -> None:
    records = [
        _record(success=True, latency=100.0),
        _record(success=True, latency=200.0),
        _record(success=False, latency=300.0),
    ]
    s = summarize(records)
    assert s.trials == 3
    assert s.task_success_rate == pytest.approx(2 / 3)
    assert s.path_success_rate == pytest.approx(2 / 3)
    assert s.mean_latency_ms == pytest.approx(200.0)


def test_percentile_helper() -> None:
    assert _percentile([1, 2, 3, 4, 5], 50.0) == pytest.approx(3.0)
    assert _percentile([1, 2, 3, 4, 5], 95.0) == pytest.approx(4.8)
    assert _percentile([], 95.0) == 0.0
    assert _percentile([42.0], 95.0) == 42.0


def _navigate_entry() -> DatasetEntry:
    return DatasetEntry(
        trial_id="t01",
        scene=DatasetScene(
            scene_id="s01",
            objects=[
                DatasetSceneObject(
                    object_id="cup_001",
                    class_label="cup",
                    center_3d_world=(0.5, 0.0, 0.6),
                )
            ],
        ),
        command="go to the cup",
        expected_outcome="navigate",
        expected_target="cup_001",
    )


def test_runner_navigate_success() -> None:
    runner = EvaluationRunner()
    record = runner.run_entry(_navigate_entry())
    assert record.grounding_success is True
    assert record.predicted_target == "cup_001"
    assert record.path_success is True
    assert record.task_success is True
    assert record.latency_ms > 0.0


def test_runner_no_match_marked_success_when_expected() -> None:
    runner = EvaluationRunner()
    entry = DatasetEntry(
        trial_id="t02",
        scene=DatasetScene(
            scene_id="s02",
            objects=[
                DatasetSceneObject(
                    object_id="cup_001",
                    class_label="cup",
                    center_3d_world=(0.5, 0.0, 0.5),
                )
            ],
        ),
        command="go to the apple",
        expected_outcome="no_match",
    )
    record = runner.run_entry(entry)
    assert record.predicted_outcome in ("no_match", "empty_map")
    assert record.task_success is True


def test_runner_ambiguous_command_triggers_clarification() -> None:
    runner = EvaluationRunner()
    entry = DatasetEntry(
        trial_id="t03",
        scene=DatasetScene(
            scene_id="s03",
            objects=[
                DatasetSceneObject(
                    object_id="cup_red", class_label="cup", center_3d_world=(0.4, 0.0, 0.5)
                ),
                DatasetSceneObject(
                    object_id="cup_blue", class_label="cup", center_3d_world=(-0.4, 0.0, 0.5)
                ),
            ],
        ),
        command="go to the cup",
        expected_outcome="clarification",
    )
    record = runner.run_entry(entry)
    assert record.predicted_outcome == "clarification"
    assert record.task_success is True


def test_runner_handles_stale_fixture_without_crash() -> None:
    runner = EvaluationRunner()
    entry = DatasetEntry(
        trial_id="t04",
        scene=DatasetScene(
            scene_id="s04",
            objects=[
                DatasetSceneObject(
                    object_id="cup_001",
                    class_label="cup",
                    center_3d_world=(0.5, 0.0, 0.6),
                    confidence=0.3,
                    tracking_status="stale",
                )
            ],
        ),
        command="go to the cup",
        expected_outcome="navigate",
        expected_target="cup_001",
    )
    record = runner.run_entry(entry)
    assert record.scene_id == "s04"


def test_run_dataset_returns_one_record_per_entry() -> None:
    runner = EvaluationRunner()
    records = runner.run_dataset([_navigate_entry(), _navigate_entry()])
    assert len(records) == 2


def test_run_dataset_isolates_failing_trial(monkeypatch) -> None:
    runner = EvaluationRunner()

    def boom(self, entry):
        raise RuntimeError("simulated")

    monkeypatch.setattr(EvaluationRunner, "run_entry", boom)
    records = runner.run_dataset([_navigate_entry()])
    assert records[0].task_success is False
    assert records[0].predicted_outcome.startswith("error:")


def test_write_report_produces_three_artifacts(tmp_path: Path) -> None:
    records = [_record(success=True, latency=120.0, xte=0.04)]
    artifacts = write_report(records, tmp_path / "out")
    assert artifacts["jsonl"].exists()
    assert artifacts["csv"].exists()
    assert artifacts["markdown"].exists()
    md = artifacts["markdown"].read_text(encoding="utf-8")
    assert "Aggregate Metrics" in md
    assert "Per-Trial Results" in md
    lines = [
        json.loads(line) for line in artifacts["jsonl"].read_text(encoding="utf-8").splitlines()
    ]
    assert len(lines) == 1
    assert lines[0]["trial_id"] == "x"


def test_failure_section_appears_when_any_trial_fails(tmp_path: Path) -> None:
    records = [
        _record(success=True, latency=100.0),
        _record(success=False, latency=400.0),
    ]
    artifacts = write_report(records, tmp_path)
    md = artifacts["markdown"].read_text(encoding="utf-8")
    assert "Failure Gallery" in md


def test_load_bundled_sample_dataset() -> None:
    entries = load_dataset(Path("samples/eval_dataset.jsonl"))
    assert len(entries) >= 6
    ids = {e.trial_id for e in entries}
    assert "t01_navigate" in ids
    assert "t05_clarification" in ids
    assert "t06_stale_memory" in ids


def test_runner_on_bundled_dataset_meets_threshold() -> None:
    """Acceptance: ≥ 50% of bundled scenarios must task-succeed."""
    entries = load_dataset(Path("samples/eval_dataset.jsonl"))
    runner = EvaluationRunner()
    records = runner.run_dataset(entries)
    summary = summarize(records)
    assert summary.task_success_rate >= 0.5, (
        f"task success rate {summary.task_success_rate:.0%} below 50% threshold; "
        f"failures: {[r.trial_id for r in records if not r.task_success]}"
    )
