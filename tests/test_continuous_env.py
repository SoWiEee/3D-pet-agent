"""§14.6.3 — continuous-action exploration env (Gymnasium)."""

from __future__ import annotations

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
from src.research.rl.continuous_env import ContinuousExplorationEnv  # noqa: E402


def test_spaces_are_continuous_box() -> None:
    env = ContinuousExplorationEnv()
    assert env.action_space.shape == (2,)
    assert env.observation_space.shape == (5,)
    assert float(env.action_space.low.min()) == -1.0
    assert float(env.action_space.high.max()) == 1.0


def test_reset_returns_obs_and_info() -> None:
    env = ContinuousExplorationEnv()
    obs, info = env.reset(seed=7)
    assert obs.shape == (5,)
    assert obs.dtype == np.float32
    assert isinstance(info, dict)


def test_step_contract() -> None:
    env = ContinuousExplorationEnv()
    env.reset(seed=7)
    obs, reward, terminated, truncated, info = env.step(np.array([0.3, 0.8], dtype=np.float32))
    assert obs.shape == (5,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_same_seed_spawns_same_world_as_discrete_env() -> None:
    from src.research.rl.env import ExplorationEnv

    c = ContinuousExplorationEnv()
    c.reset(seed=123)
    d = ExplorationEnv()
    d.reset(seed=123)
    assert [(o.x, o.z, o.relevant) for o in c._inner._objects] == [
        (o.x, o.z, o.relevant) for o in d._objects
    ]


def test_truncates_at_max_steps() -> None:
    env = ContinuousExplorationEnv()
    env.reset(seed=1)
    terminated = truncated = False
    for _ in range(100):
        _, _, terminated, truncated, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
        if terminated or truncated:
            break
    assert terminated or truncated
