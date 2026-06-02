"""Phase 10 — Evaluation harness + demo packaging.

Public surface:

- :class:`EvaluationRecord`   per-trial result (spec §3.8)
- :class:`DatasetEntry`       one trial spec (scene + command + expected target)
- :class:`MetricsSummary`     aggregated metrics over a run
- :class:`EvaluationRunner`   loads dataset, runs trials in-process, returns records
- :func:`write_report`        CSV + Markdown report
"""

from __future__ import annotations

from .metrics import MetricsSummary, summarize
from .report import write_report
from .runner import EvaluationRunner
from .schema import DatasetEntry, DatasetScene, EvaluationRecord

__all__ = [
    "DatasetEntry",
    "DatasetScene",
    "EvaluationRecord",
    "EvaluationRunner",
    "MetricsSummary",
    "summarize",
    "write_report",
]
