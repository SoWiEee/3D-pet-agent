"""§14.6.3 — mixed discrete/continuous A/B on matched seeds."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")
from src.research.rl.policy import evaluate_ab_mixed, heuristic_policy  # noqa: E402


def _const_continuous(_state: np.ndarray) -> np.ndarray:
    return np.array([0.2, 0.7], dtype=np.float32)


def test_mixed_ab_scores_both_kinds() -> None:
    policies = {
        "heuristic": ("discrete", heuristic_policy),
        "const_cont": ("continuous", _const_continuous),
    }
    summary = evaluate_ab_mixed(policies, n_scenes=3, seed0=10_000)
    assert set(summary) == {"heuristic", "const_cont"}
    for metrics in summary.values():
        assert set(metrics) == {
            "mean_return",
            "mean_coverage",
            "mean_relevant_found",
            "recall",
            "mean_steps",
        }
        assert 0.0 <= metrics["mean_coverage"] <= 1.0


def test_unknown_kind_raises() -> None:
    with pytest.raises(ValueError, match="unknown policy kind"):
        evaluate_ab_mixed({"bad": ("teleport", heuristic_policy)}, n_scenes=1, seed0=0)
