"""Phase 10 reporting — CSV + Markdown.

Both formats are written next to the raw JSONL records so a reviewer can
open the markdown for the at-a-glance numbers and the CSV for spreadsheet
analysis.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .metrics import MetricsSummary, summarize
from .schema import EvaluationRecord


def write_records_jsonl(records: list[EvaluationRecord], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), sort_keys=True) + "\n")
    return path


def write_records_csv(records: list[EvaluationRecord], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial_id",
        "scene_id",
        "command",
        "expected_outcome",
        "expected_target",
        "predicted_outcome",
        "predicted_target",
        "grounding_success",
        "path_success",
        "collision_count",
        "task_success",
        "latency_ms",
        "max_cross_track_error_m",
        "max_heading_error_rad",
        "mean_speed_mps",
        "controller_steps",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow(
                {
                    "trial_id": r.trial_id,
                    "scene_id": r.scene_id,
                    "command": r.command,
                    "expected_outcome": r.expected_outcome,
                    "expected_target": r.expected_target or "",
                    "predicted_outcome": r.predicted_outcome,
                    "predicted_target": r.predicted_target or "",
                    "grounding_success": int(r.grounding_success),
                    "path_success": int(r.path_success),
                    "collision_count": r.collision_count,
                    "task_success": int(r.task_success),
                    "latency_ms": round(r.latency_ms, 2),
                    "max_cross_track_error_m": round(
                        r.controller_metrics.max_cross_track_error_m, 4
                    ),
                    "max_heading_error_rad": round(r.controller_metrics.max_heading_error_rad, 4),
                    "mean_speed_mps": round(r.controller_metrics.mean_speed_mps, 4),
                    "controller_steps": r.controller_metrics.steps,
                }
            )
    return path


def write_markdown(
    records: list[EvaluationRecord], summary: MetricsSummary, path: Path | str
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Evaluation Report")
    lines.append("")
    lines.append("## Aggregate Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    for name, value in summary.as_table_rows():
        lines.append(f"| {name} | {value} |")
    lines.append("")

    failures = [r for r in records if not r.task_success]
    lines.append(f"## Per-Trial Results ({len(records)} total, {len(failures)} failures)")
    lines.append("")
    lines.append("| Trial | Command | Expected | Predicted | Task | Latency (ms) |")
    lines.append("|---|---|---|---|---|---|")
    for r in records:
        ok = "OK" if r.task_success else "FAIL"
        lines.append(
            f"| {r.trial_id} | {r.command[:60]} | "
            f"{r.expected_outcome}/{r.expected_target or '—'} | "
            f"{r.predicted_outcome}/{r.predicted_target or '—'} | "
            f"{ok} | {r.latency_ms:.1f} |"
        )
    lines.append("")

    if failures:
        lines.append("## Failure Gallery")
        lines.append("")
        for r in failures:
            lines.append(f"### {r.trial_id} — `{r.command}`")
            lines.append("")
            lines.append(f"- expected: `{r.expected_outcome}` target=`{r.expected_target}`")
            lines.append(f"- predicted: `{r.predicted_outcome}` target=`{r.predicted_target}`")
            lines.append(
                f"- grounding={r.grounding_success} path={r.path_success} "
                f"collisions={r.collision_count}"
            )
            if r.notes:
                lines.append(f"- notes: {r.notes}")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_report(records: list[EvaluationRecord], out_dir: Path | str) -> dict[str, Path]:
    """Write all three artifacts under ``out_dir``. Returns paths."""
    out_dir = Path(out_dir)
    summary = summarize(records)
    return {
        "jsonl": write_records_jsonl(records, out_dir / "records.jsonl"),
        "csv": write_records_csv(records, out_dir / "records.csv"),
        "markdown": write_markdown(records, summary, out_dir / "report.md"),
    }
