"""RL active-exploration package (spec §14.3)."""

from .continuous_env import ContinuousEnvConfig, ContinuousExplorationEnv
from .dqn import DQNAgent, DQNConfig, QNetwork, ReplayBuffer
from .env import ACTION_NAMES, N_ACTIONS, STATE_DIM, EnvConfig, ExplorationEnv
from .policy import (
    RLExplorationPolicy,
    coverage_uplift,
    evaluate_ab,
    format_ab_report,
    heuristic_policy,
    random_policy,
    run_episode,
    train_dqn,
)

__all__ = [
    "ContinuousExplorationEnv",
    "ContinuousEnvConfig",
    "ExplorationEnv",
    "EnvConfig",
    "N_ACTIONS",
    "STATE_DIM",
    "ACTION_NAMES",
    "DQNAgent",
    "DQNConfig",
    "QNetwork",
    "ReplayBuffer",
    "RLExplorationPolicy",
    "heuristic_policy",
    "random_policy",
    "train_dqn",
    "run_episode",
    "evaluate_ab",
    "coverage_uplift",
    "format_ab_report",
]
