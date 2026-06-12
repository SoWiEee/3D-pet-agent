"""RL-based exploration policy — optional sidecar (spec §14.3).

Spec §14.3 names this module ``research/rl_explorer.py``. The implementation is
split across the :mod:`src.research.rl` package (env / dqn / policy) to keep each
file focused; this module is the stable facade that re-exports the public API.

Run the full train + A/B experiment with::

    python main.py --mode rl_exploration --episodes 500 --scenes 50

See ``src/research/rl/`` for the environment, DQN agent, and evaluation harness.
"""

from __future__ import annotations

from .rl import (
    DQNAgent,
    DQNConfig,
    EnvConfig,
    ExplorationEnv,
    RLExplorationPolicy,
    Sb3ContinuousPolicy,
    coverage_uplift,
    evaluate_ab,
    format_ab_report,
    heuristic_policy,
    random_policy,
    run_continuous_episode,
    run_episode,
    train_dqn,
    train_sac,
    train_tqc,
)

__all__ = [
    "ExplorationEnv",
    "EnvConfig",
    "DQNAgent",
    "DQNConfig",
    "RLExplorationPolicy",
    "heuristic_policy",
    "random_policy",
    "train_dqn",
    "run_episode",
    "evaluate_ab",
    "coverage_uplift",
    "format_ab_report",
    "Sb3ContinuousPolicy",
    "run_continuous_episode",
    "train_sac",
    "train_tqc",
]
