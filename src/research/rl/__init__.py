"""RL active-exploration package (spec §14.3)."""

from .continuous_env import ContinuousEnvConfig, ContinuousExplorationEnv
from .dqn import DQNAgent, DQNConfig, QNetwork, ReplayBuffer
from .env import ACTION_NAMES, N_ACTIONS, STATE_DIM, EnvConfig, ExplorationEnv
from .policy import (
    RLExplorationPolicy,
    coverage_uplift,
    evaluate_ab,
    evaluate_ab_mixed,
    format_ab_report,
    heuristic_policy,
    random_policy,
    run_episode,
    train_dqn,
)
from .sb3_policies import Sb3ContinuousPolicy, run_continuous_episode, train_sac, train_tqc

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
    "evaluate_ab_mixed",
    "coverage_uplift",
    "format_ab_report",
    "Sb3ContinuousPolicy",
    "run_continuous_episode",
    "train_sac",
    "train_tqc",
]
