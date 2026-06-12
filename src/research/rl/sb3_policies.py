"""SAC/TQC training + continuous-policy adapter (spec §14.6.3)."""

from __future__ import annotations

import os
from collections.abc import Callable

import numpy as np
from sb3_contrib import TQC
from stable_baselines3 import SAC

from .continuous_env import ContinuousExplorationEnv
from .policy import EpisodeResult

ContinuousPolicy = Callable[[np.ndarray], np.ndarray]


def _resolve_device(device: str | None) -> str:
    if device is not None:
        return device
    return "cuda" if os.environ.get("PET_AGENT_DEVICE", "").lower() == "cuda" else "auto"


def train_sac(*, total_timesteps: int, seed: int = 0, device: str | None = None) -> SAC:
    env = ContinuousExplorationEnv()
    model = SAC("MlpPolicy", env, seed=seed, device=_resolve_device(device), verbose=0)
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    return model


def train_tqc(*, total_timesteps: int, seed: int = 0, device: str | None = None) -> TQC:
    env = ContinuousExplorationEnv()
    model = TQC("MlpPolicy", env, seed=seed, device=_resolve_device(device), verbose=0)
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    return model


class Sb3ContinuousPolicy:
    """Wrap a trained SB3 model as a deterministic state->action callable."""

    def __init__(self, model) -> None:
        self._model = model

    def __call__(self, state: np.ndarray) -> np.ndarray:
        action, _ = self._model.predict(state, deterministic=True)
        return np.asarray(action, dtype=np.float32)


def run_continuous_episode(
    env: ContinuousExplorationEnv, policy: ContinuousPolicy, seed: int
) -> EpisodeResult:
    """Run one episode; return the SAME EpisodeResult shape the discrete harness
    uses so both feed the A/B comparison identically."""
    obs, _ = env.reset(seed=seed)
    total_return = 0.0
    steps = 0
    while True:
        action = policy(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_return += reward
        steps += 1
        if terminated or truncated:
            break
    return EpisodeResult(
        total_return=total_return,
        coverage=env.coverage_fraction,
        relevant_found=env.relevant_discovered,
        relevant_total=env.relevant_total,
        steps=steps,
    )
