"""§14.6.3 — SAC/TQC training + continuous policy wrapper (tiny budgets)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")
pytest.importorskip("sb3_contrib")
from src.research.rl.continuous_env import ContinuousExplorationEnv  # noqa: E402
from src.research.rl.sb3_policies import (  # noqa: E402
    Sb3ContinuousPolicy,
    run_continuous_episode,
    train_sac,
    train_tqc,
)


def test_train_sac_returns_runnable_policy() -> None:
    model = train_sac(total_timesteps=200, seed=0, device="cpu")
    policy = Sb3ContinuousPolicy(model)
    action = policy(np.zeros(5, dtype=np.float32))
    assert action.shape == (2,)


def test_train_tqc_returns_runnable_policy() -> None:
    model = train_tqc(total_timesteps=200, seed=0, device="cpu")
    policy = Sb3ContinuousPolicy(model)
    action = policy(np.zeros(5, dtype=np.float32))
    assert action.shape == (2,)


def test_run_continuous_episode_reports_coverage() -> None:
    model = train_sac(total_timesteps=200, seed=0, device="cpu")
    policy = Sb3ContinuousPolicy(model)
    result = run_continuous_episode(ContinuousExplorationEnv(), policy, seed=42)
    assert 0.0 <= result.coverage <= 1.0
    assert result.relevant_total >= 0
