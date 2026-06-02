"""Aggregated metrics over a list of :class:`EvaluationRecord`.

Per-trial metrics live on the record itself; this module summarises them
across a run (mean / rate / p95 / total) so dashboards and the markdown
report can render a single table.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass

from .schema import EvaluationRecord


@dataclass(frozen=True)
class MetricsSummary:
    """Aggregate of a run. All rates in [0, 1]; latencies in ms."""

    trials: int
    grounding_success_rate: float
    path_success_rate: float
    task_success_rate: float
    mean_latency_ms: float
    p95_latency_ms: float
    mean_cross_track_error_m: float
    mean_heading_error_rad: float
    total_collisions: int

    def as_table_rows(self) -> list[tuple[str, str]]:
        return [
            ("Trials", str(self.trials)),
            ("Grounding success rate", f"{self.grounding_success_rate:.1%}"),
            ("Path success rate", f"{self.path_success_rate:.1%}"),
            ("Task success rate", f"{self.task_success_rate:.1%}"),
            ("Mean latency", f"{self.mean_latency_ms:.1f} ms"),
            ("p95 latency", f"{self.p95_latency_ms:.1f} ms"),
            ("Mean cross-track error", f"{self.mean_cross_track_error_m:.3f} m"),
            ("Mean heading error", f"{self.mean_heading_error_rad:.3f} rad"),
            ("Total collisions", str(self.total_collisions)),
        ]


def _rate(records: list[EvaluationRecord], predicate: Callable[[EvaluationRecord], bool]) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if predicate(r)) / len(records)


def _percentile(values: list[float], pct: float) -> float:
    """Inclusive percentile — no scipy needed. ``pct ∈ [0, 100]``."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def summarize(records: list[EvaluationRecord]) -> MetricsSummary:
    if not records:
        return MetricsSummary(
            trials=0,
            grounding_success_rate=0.0,
            path_success_rate=0.0,
            task_success_rate=0.0,
            mean_latency_ms=0.0,
            p95_latency_ms=0.0,
            mean_cross_track_error_m=0.0,
            mean_heading_error_rad=0.0,
            total_collisions=0,
        )
    latencies = [r.latency_ms for r in records]
    xte = [r.controller_metrics.max_cross_track_error_m for r in records]
    he = [r.controller_metrics.max_heading_error_rad for r in records]
    return MetricsSummary(
        trials=len(records),
        grounding_success_rate=_rate(records, lambda r: r.grounding_success),
        path_success_rate=_rate(records, lambda r: r.path_success),
        task_success_rate=_rate(records, lambda r: r.task_success),
        mean_latency_ms=statistics.fmean(latencies),
        p95_latency_ms=_percentile(latencies, 95.0),
        mean_cross_track_error_m=statistics.fmean(xte) if xte else 0.0,
        mean_heading_error_rad=statistics.fmean(he) if he else 0.0,
        total_collisions=sum(r.collision_count for r in records),
    )
