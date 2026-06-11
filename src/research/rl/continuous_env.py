"""Continuous-action exploration env for SAC/TQC (spec §14.6.3).

Composes the discrete :class:`ExplorationEnv` (spec §14.3) and re-parameterises
its action as a continuous next-viewpoint waypoint, so SAC/TQC train against the
identical world model + reward the discrete DQN uses. Matched seeds spawn an
identical world, keeping the A/B coverage comparison fair.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .env import STATE_DIM, EnvConfig, ExplorationEnv, reward_terms


@dataclass
class ContinuousEnvConfig:
    max_reach: float = 1.5  # metres a single waypoint action can travel


class ContinuousExplorationEnv(gym.Env):
    """Gymnasium env: action = (direction, magnitude) -> world waypoint."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_cfg: EnvConfig | None = None,
        cont_cfg: ContinuousEnvConfig | None = None,
    ) -> None:
        super().__init__()
        self._inner = ExplorationEnv(env_cfg)
        self._cfg = cont_cfg or ContinuousEnvConfig()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        obs = self._inner.reset(seed=seed)
        return obs.astype(np.float32), {}

    def step(self, action: np.ndarray):
        env = self._inner
        direction = float(np.clip(action[0], -1.0, 1.0))
        magnitude = float(np.clip(action[1], -1.0, 1.0))
        theta = env._heading + direction * math.pi
        reach = ((magnitude + 1.0) / 2.0) * self._cfg.max_reach
        target = (
            env._cat[0] + reach * math.cos(theta),
            env._cat[1] + reach * math.sin(theta),
        )

        new_pos, collided = env._move_toward(env._cat, target)
        is_move = new_pos != env._cat
        if is_move:
            env._heading = math.atan2(new_pos[1] - env._cat[1], new_pos[0] - env._cat[0])
            env._cat = new_pos

        new_area = 0
        discovered_relevant = 0
        verified_stale = 0
        if not collided:
            new_area = env._observe(env._cat, env._heading, fov=env.cfg.cone_fov)
            discovered_relevant = env._discover_at(env._cat, fov=env.cfg.cone_fov)
            verified_stale = env._verify_stale_near(env._cat)

        failed = is_move and new_area < env.cfg.failed_area_cells and not discovered_relevant
        terms = reward_terms(
            env.cfg,
            new_area=new_area,
            discovered_relevant=discovered_relevant,
            verified_stale=verified_stale,
            collided=collided,
            is_move=is_move,
            failed_inspection=failed,
            prev_failed=env._prev_failed_inspect,
        )
        env._prev_failed_inspect = failed
        reward = float(sum(terms.values()))

        env._steps += 1
        terminated = bool(env.coverage.unobserved_ratio() <= env.cfg.explored_done_ratio)
        truncated = bool(env._steps >= env.cfg.max_steps)
        info = {"reward_terms": terms, "collided": collided}
        return env._state().astype(np.float32), reward, terminated, truncated, info

    @property
    def coverage_fraction(self) -> float:
        return self._inner.coverage_fraction

    @property
    def relevant_discovered(self) -> int:
        return self._inner.relevant_discovered

    @property
    def relevant_total(self) -> int:
        return self._inner.relevant_total
