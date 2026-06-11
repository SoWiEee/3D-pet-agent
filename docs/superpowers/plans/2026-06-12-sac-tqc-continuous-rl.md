# SAC / TQC Continuous-Control Exploration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add continuous-control RL exploration (Stable-Baselines3 **SAC** + sb3-contrib **TQC**) alongside the existing discrete DQN, by reformulating the exploration action space to a continuous next-viewpoint waypoint, and extend the A/B harness to compare heuristic / random / DQN / SAC / TQC on identical seeds.

**Architecture:** A new `ContinuousExplorationEnv` (Gymnasium `Env`) **composes** the existing `ExplorationEnv` (spec §14.3) — it does NOT replace it. Its `Box(2)` action `(direction, magnitude)` is mapped to a world waypoint, then driven through the *same* `_move_toward` / `_observe` / `_discover_at` / `_verify_stale_near` machinery and a DRY-extracted reward function, so the continuous policy trains against the identical world model and the discrete DQN path is untouched. SB3 SAC/TQC train on this env (CUDA). The A/B harness runs each policy in its appropriate env variant at matched seeds (same seed → identical spawned world) and compares `coverage_fraction`. Default behavior (`--algo dqn`) is unchanged.

**Tech Stack:** Python 3.12, PyTorch 2.12+cu130 (CUDA on the RTX 4070), `gymnasium`, `stable-baselines3` (SAC), `sb3-contrib` (TQC), pytest, ruff, uv. Implements spec §14.6.3.

---

## Context the implementer needs (verified against the codebase)

- `src/research/rl/env.py`:
  - `ExplorationEnv`: `reset(seed: int | None) -> np.ndarray` (5-dim `float32` state), `step(action: int) -> (state, reward, done, StepInfo)`.
  - World helpers (same-package, reuse them): `_move_toward(a, b) -> (world_pos, collided)`, `_observe(pose, heading, *, fov) -> int new_area_cells`, `_discover_at(pose, *, fov) -> int new_relevant`, `_verify_stale_near(pose) -> int`, `_state() -> np.ndarray`, `coverage.unobserved_ratio()`, `coverage_fraction` property, `relevant_discovered` / `relevant_total` properties. Fields: `cfg` (`EnvConfig`), `_cat` (tuple), `_heading` (float), `_world_diag`.
  - Constants: `STATE_DIM = 5`, `N_ACTIONS = 5`, reward magnitudes `R_DISCOVER_RELEVANT=1.0`, `R_REDUCED_AREA=0.5`, `R_VERIFY_STALE=0.3`, `R_UNNECESSARY_MOVE=-0.2`, `R_COLLISION=-1.0`, `R_REPEAT_FAILED=-0.5`. `EnvConfig` fields incl. `max_steps=30`, `explored_done_ratio=0.12`, `area_norm=60.0`, `failed_area_cells=1`, `resolution=0.1`, `width=40`, `height=40`.
- `src/research/rl/policy.py`:
  - `Policy = Callable[[np.ndarray], int]` (discrete). `heuristic_policy`, `random_policy(rng)`, `RLExplorationPolicy` (wraps a `DQNAgent`).
  - `train_dqn(episodes=..., seed=...) -> tuple[DQNAgent, list[float]]`.
  - `run_episode(env: ExplorationEnv, policy: Policy, seed: int) -> EpisodeResult` (`EpisodeResult` has coverage / discovery fields — read the dataclass before using).
  - `evaluate_ab(policies: dict[str, Policy], *, n_scenes: int, seed0: int) -> dict[str, dict[str, float]]`.
  - `coverage_uplift(summary, rl, baseline) -> float`, `format_ab_report(summary, *, n_scenes, episodes) -> str`.
- `src/research/rl/dqn.py`: `DQNAgent` (`act(state, eps)`), `DQNConfig`.
- `src/research/rl/__init__.py` and `src/research/rl_explorer.py` (facade) export the public API — update BOTH when adding public names.
- `src/cli.py`: `run_rl_exploration(args, cfg)` at line ~315 (calls `train_dqn` then `evaluate_ab`, writes `runs/rl_exploration_<ts>/`), dispatched at line ~438. CLI args `--episodes`, `--scenes`, `--seed`, `--out` exist; you will ADD `--algo`.
- `pyproject.toml`: optional-dependencies already has `dev` and `track`. numpy is pinned `>=1.26,<2.2`. `sb3`/`gymnasium`/`sb3-contrib` resolve cleanly under numpy 2.1.3 (verified via `uv pip install --dry-run`).
- Device: `PET_AGENT_DEVICE` env exists; torch CUDA is available. SB3 takes `device="cuda"`.

---

## File Structure

- `pyproject.toml` — add `rl` extra: `stable-baselines3`, `sb3-contrib`, `gymnasium`.
- `src/research/rl/env.py` — extract a pure `reward_terms(...)` module function; refactor discrete `step` to call it (behavior-preserving).
- `src/research/rl/continuous_env.py` (create) — `ContinuousExplorationEnv(gymnasium.Env)`, `ContinuousEnvConfig`.
- `src/research/rl/sb3_policies.py` (create) — `train_sac`, `train_tqc`, `Sb3ContinuousPolicy`, `run_continuous_episode`.
- `src/research/rl/policy.py` — extend `evaluate_ab` to accept per-policy env factories (or add `evaluate_ab_mixed`) so discrete and continuous policies are scored on matched seeds; extend `format_ab_report` for 5 rows.
- `src/research/rl/__init__.py` + `src/research/rl_explorer.py` — export new public names.
- `src/cli.py` — `--algo {dqn,sac,tqc}`, `PET_AGENT_RL_ALGO`; route training + A/B.
- Tests: `tests/test_continuous_env.py`, `tests/test_sb3_policies.py`, `tests/test_rl_ab_mixed.py`, additions to existing `tests/test_rl_*.py` if present.
- `docs/spec.md` §14.6.3 status.

Each backend stays behind the existing facade; `--algo dqn` (default) reproduces today's behavior exactly.

---

## Task 1: Add the `rl` optional dependency

**Files:** `pyproject.toml`

- [ ] **Step 1: Add the extra**

Under `[project.optional-dependencies]`:

```toml
rl = [
    "gymnasium>=1.0",
    "stable-baselines3>=2.7",
    "sb3-contrib>=2.7",
]
```

- [ ] **Step 2: Install**

Run: `uv pip install -e ".[rl]"`
Expected: installs `gymnasium`, `stable-baselines3`, `sb3-contrib`, `cloudpickle`, `farama-notifications`, `pandas`. numpy STAYS 2.1.3 (no downgrade). If numpy would change, STOP and report BLOCKED.

- [ ] **Step 3: Verify imports + CUDA + TQC availability**

Run: `.venv/bin/python -c "import gymnasium as gym; from stable_baselines3 import SAC; from sb3_contrib import TQC; import torch; print('gym', gym.__version__, 'cuda', torch.cuda.is_available())"`
Expected: prints versions and `cuda True`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore(rl): add stable-baselines3 + sb3-contrib + gymnasium (.[rl])"
```
(End every commit message with: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`)

---

## Task 2: Extract a DRY reward function (behavior-preserving refactor)

The continuous env must reuse the discrete env's reward shaping without duplicating it. Extract a pure module function and make the discrete `step` call it. **No behavior change** — the existing env/RL tests must stay green.

**Files:** `src/research/rl/env.py`; Test: `tests/test_rl_reward_terms.py`

- [ ] **Step 1: Write the characterization test FIRST**

Create `tests/test_rl_reward_terms.py`:

```python
"""§14.6.3 — reward_terms is the shared shaping used by discrete + continuous envs."""

from __future__ import annotations

from src.research.rl.env import (
    R_COLLISION,
    R_DISCOVER_RELEVANT,
    R_REDUCED_AREA,
    R_UNNECESSARY_MOVE,
    R_VERIFY_STALE,
    EnvConfig,
    reward_terms,
)


def test_discover_and_area_terms() -> None:
    cfg = EnvConfig()
    terms = reward_terms(
        cfg,
        new_area=cfg.area_norm,  # full area reward
        discovered_relevant=1,
        verified_stale=0,
        collided=False,
        is_move=True,
        failed_inspection=False,
        prev_failed=False,
    )
    assert terms["area"] == R_REDUCED_AREA
    assert terms["discover"] == R_DISCOVER_RELEVANT
    assert "collision" not in terms


def test_collision_term() -> None:
    cfg = EnvConfig()
    terms = reward_terms(
        cfg, new_area=0, discovered_relevant=0, verified_stale=0,
        collided=True, is_move=True, failed_inspection=False, prev_failed=False,
    )
    assert terms["collision"] == R_COLLISION


def test_unnecessary_move_and_repeat_failed() -> None:
    cfg = EnvConfig()
    terms = reward_terms(
        cfg, new_area=0, discovered_relevant=0, verified_stale=0,
        collided=False, is_move=True, failed_inspection=True, prev_failed=True,
    )
    assert terms["move_cost"] == R_UNNECESSARY_MOVE
    assert terms["repeat_failed"] == R_VERIFY_STALE * 0 + (-0.5)  # R_REPEAT_FAILED
```

- [ ] **Step 2: Run it → fails (no `reward_terms`)**

Run: `.venv/bin/pytest tests/test_rl_reward_terms.py -v`
Expected: FAIL `ImportError: cannot import name 'reward_terms'`.

- [ ] **Step 3: Extract the function, refactor `step` to use it**

In `src/research/rl/env.py`, add a module-level function (place after the reward constants):

```python
def reward_terms(
    cfg: EnvConfig,
    *,
    new_area: int,
    discovered_relevant: int,
    verified_stale: int,
    collided: bool,
    is_move: bool,
    failed_inspection: bool,
    prev_failed: bool,
) -> dict[str, float]:
    """Spec §14.3 reward shaping, shared by the discrete and continuous envs.

    Pure: depends only on the per-step quantities + config, so both action
    parameterisations score identically and the A/B comparison stays fair."""
    terms: dict[str, float] = {}
    if collided:
        terms["collision"] = R_COLLISION
    if new_area > 0:
        terms["area"] = R_REDUCED_AREA * min(1.0, new_area / cfg.area_norm)
    if discovered_relevant:
        terms["discover"] = R_DISCOVER_RELEVANT * discovered_relevant
    if verified_stale:
        terms["verify"] = R_VERIFY_STALE * verified_stale
    if is_move and not collided and new_area < cfg.failed_area_cells and not discovered_relevant:
        terms["move_cost"] = R_UNNECESSARY_MOVE
    if failed_inspection and prev_failed:
        terms["repeat_failed"] = R_REPEAT_FAILED
    return terms
```

Then refactor `ExplorationEnv.step` so its reward block calls `reward_terms(...)` with the quantities it already computes (`new_area`, `discovered_relevant`, `verified_stale`, `collided`, `is_move`, the `failed_inspection` boolean, and `self._prev_failed_inspect`). Keep the `collision` term ordering and `self._prev_failed_inspect = failed_inspection` assignment. The resulting `terms` dict and `reward = float(sum(terms.values()))` must be identical to before.

- [ ] **Step 4: Run new + existing RL tests → all green (behavior preserved)**

Run: `.venv/bin/pytest tests/test_rl_reward_terms.py -v && .venv/bin/pytest -k "rl or exploration or env" -q`
Expected: new test passes AND every pre-existing RL/env test still passes (no reward drift). If any discrete-env test changed outcome, your extraction altered behavior — fix the extraction, not the test.

- [ ] **Step 5: ruff + commit**

```bash
.venv/bin/ruff check src/research/rl/env.py tests/test_rl_reward_terms.py && .venv/bin/ruff format src/research/rl/env.py tests/test_rl_reward_terms.py
git add src/research/rl/env.py tests/test_rl_reward_terms.py
git commit -m "refactor(rl): extract shared reward_terms for discrete + continuous envs"
```

---

## Task 3: `ContinuousExplorationEnv` (Gymnasium, Box(2) waypoint)

**Files:** Create `src/research/rl/continuous_env.py`; Test: `tests/test_continuous_env.py`

Action semantics (spec §14.6.3): `Box(low=-1, high=1, shape=(2,))` = `(direction, magnitude)`. `direction ∈ [-1,1]` maps to a turn angle `Δθ = direction·π` added to the current heading; `magnitude ∈ [-1,1]` maps to a reach `r = ((magnitude + 1) / 2) · max_reach` (so a 0-magnitude action is a near in-place look). Target world point = `_cat + r·(cos(θ+Δθ), sin(θ+Δθ))`. Drive via the env's `_move_toward`, set heading from actual displacement, then `_observe` + `_discover_at` with the cone fov, and `_verify_stale_near`. Reward via the shared `reward_terms`. Observation = `env._state()`. `terminated` when `coverage.unobserved_ratio() <= explored_done_ratio`; `truncated` at `max_steps`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_continuous_env.py`:

```python
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
    # The continuous env composes ExplorationEnv; matched seeds → identical world,
    # which is what makes the A/B coverage comparison fair.
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
    truncated = False
    for _ in range(100):
        _, _, terminated, truncated, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
        if terminated or truncated:
            break
    assert terminated or truncated
```

- [ ] **Step 2: Run → fails (no module)**

Run: `.venv/bin/pytest tests/test_continuous_env.py -v`
Expected: FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Implement the env**

Create `src/research/rl/continuous_env.py`:

```python
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
    """Gymnasium env: action = (direction, magnitude) → world waypoint."""

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
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32
        )

    def reset(self, *, seed: int | None = None, options=None):  # noqa: ANN001
        super().reset(seed=seed)
        obs = self._inner.reset(seed=seed)
        return obs.astype(np.float32), {}

    def step(self, action: np.ndarray):  # noqa: ANN001
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
        terminated = env.coverage.unobserved_ratio() <= env.cfg.explored_done_ratio
        truncated = env._steps >= env.cfg.max_steps
        info = {"reward_terms": terms, "collided": collided}
        return env._state().astype(np.float32), reward, terminated, truncated, info

    # Introspection mirrors ExplorationEnv for the A/B harness.
    @property
    def coverage_fraction(self) -> float:
        return self._inner.coverage_fraction

    @property
    def relevant_discovered(self) -> int:
        return self._inner.relevant_discovered

    @property
    def relevant_total(self) -> int:
        return self._inner.relevant_total
```

- [ ] **Step 4: Run → all pass**

Run: `.venv/bin/pytest tests/test_continuous_env.py -v`
Expected: 5 PASS. If `gymnasium`'s `Env.reset` signature complains, match the installed gymnasium's keyword-only `seed`/`options` (already done above).

- [ ] **Step 5: Export + ruff + commit**

Add `ContinuousExplorationEnv`, `ContinuousEnvConfig` to `src/research/rl/__init__.py` `__all__` and imports.

```bash
.venv/bin/ruff check src/research/rl/continuous_env.py tests/test_continuous_env.py && .venv/bin/ruff format src/research/rl/continuous_env.py tests/test_continuous_env.py
git add src/research/rl/continuous_env.py src/research/rl/__init__.py tests/test_continuous_env.py
git commit -m "feat(rl): ContinuousExplorationEnv — Box(2) waypoint over the §14.3 world"
```

---

## Task 4: SAC + TQC training and policy wrapper

**Files:** Create `src/research/rl/sb3_policies.py`; Test: `tests/test_sb3_policies.py`

- [ ] **Step 1: Write the failing tests (tiny budgets so they run in CI)**

Create `tests/test_sb3_policies.py`:

```python
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
    assert 0.0 <= result.coverage_fraction <= 1.0
```

- [ ] **Step 2: Run → fails (no module)**

Run: `.venv/bin/pytest tests/test_sb3_policies.py -v`
Expected: FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `src/research/rl/sb3_policies.py`. Read `policy.py::EpisodeResult` first and reuse it (do not redefine an episode-result type — import the existing one). Sketch:

```python
"""SAC/TQC training + continuous-policy adapter (spec §14.6.3)."""

from __future__ import annotations

import os
from collections.abc import Callable

import numpy as np
from stable_baselines3 import SAC
from sb3_contrib import TQC

from .continuous_env import ContinuousExplorationEnv
from .policy import EpisodeResult  # reuse the discrete harness's result type

ContinuousPolicy = Callable[[np.ndarray], np.ndarray]


def _device(device: str | None) -> str:
    if device is not None:
        return device
    return "cuda" if os.environ.get("PET_AGENT_DEVICE", "").lower() == "cuda" else "auto"


def train_sac(*, total_timesteps: int, seed: int = 0, device: str | None = None) -> SAC:
    env = ContinuousExplorationEnv()
    model = SAC("MlpPolicy", env, seed=seed, device=_device(device), verbose=0)
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    return model


def train_tqc(*, total_timesteps: int, seed: int = 0, device: str | None = None) -> TQC:
    env = ContinuousExplorationEnv()
    model = TQC("MlpPolicy", env, seed=seed, device=_device(device), verbose=0)
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    return model


class Sb3ContinuousPolicy:
    """Wrap a trained SB3 model as a deterministic state→action callable."""

    def __init__(self, model) -> None:  # noqa: ANN001
        self._model = model

    def __call__(self, state: np.ndarray) -> np.ndarray:
        action, _ = self._model.predict(state, deterministic=True)
        return np.asarray(action, dtype=np.float32)


def run_continuous_episode(
    env: ContinuousExplorationEnv, policy: ContinuousPolicy, seed: int
) -> EpisodeResult:
    """Run one episode and return the SAME EpisodeResult shape the discrete
    harness uses, so both feed evaluate_ab_mixed identically."""
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    while True:
        action = policy(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break
    # Construct EpisodeResult with the fields the dataclass declares — read
    # policy.py::EpisodeResult and fill coverage_fraction / relevant recall /
    # steps / total_reward to match its constructor exactly.
    return EpisodeResult(
        coverage_fraction=env.coverage_fraction,
        relevant_discovered=env.relevant_discovered,
        relevant_total=env.relevant_total,
        steps=steps,
        total_reward=total_reward,
    )
```

> NOTE: the `EpisodeResult(...)` constructor call MUST match the real dataclass field names in `policy.py`. Read it (Task context lists it) and adjust the kwargs before running. If field names differ, use the real ones — do not invent.

- [ ] **Step 4: Run → pass**

Run: `.venv/bin/pytest tests/test_sb3_policies.py -v`
Expected: 3 PASS (slow-ish; 200 timesteps on CPU is a few seconds each). If SB3 warns about the env not being wrapped in a Monitor/VecEnv, that is fine for `.learn` on a single `gym.Env` (SB3 auto-wraps). If `predict` returns shape `(2,)` already, the `np.asarray` is a no-op.

- [ ] **Step 5: Export + ruff + commit**

Add public names to `src/research/rl/__init__.py` and `src/research/rl_explorer.py`.

```bash
.venv/bin/ruff check src/research/rl/sb3_policies.py tests/test_sb3_policies.py && .venv/bin/ruff format src/research/rl/sb3_policies.py tests/test_sb3_policies.py
git add src/research/rl/sb3_policies.py src/research/rl/__init__.py src/research/rl_explorer.py tests/test_sb3_policies.py
git commit -m "feat(rl): SAC + TQC training + continuous policy wrapper"
```

---

## Task 5: 5-way A/B harness (matched seeds, mixed action spaces)

**Files:** `src/research/rl/policy.py`; Test: `tests/test_rl_ab_mixed.py`

The existing `evaluate_ab` runs discrete `Policy` callables on `ExplorationEnv`. Add `evaluate_ab_mixed` that accepts, per policy name, a `(kind, callable)` where `kind ∈ {"discrete","continuous"}`, runs each in its matching env at the SAME seed sequence, and returns the same `dict[name -> metrics]` shape so `format_ab_report` works unchanged.

- [ ] **Step 1: Write the failing test**

Create `tests/test_rl_ab_mixed.py`:

```python
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
        assert "coverage_mean" in metrics  # match the real metric key in evaluate_ab
```

> Before writing the impl, read `evaluate_ab` to learn the exact metric keys it emits (e.g. `coverage_mean`) and mirror them so `format_ab_report` consumes both identically. Fix the assertion's key name to the real one.

- [ ] **Step 2: Run → fails**

Run: `.venv/bin/pytest tests/test_rl_ab_mixed.py -v`
Expected: FAIL `ImportError: cannot import name 'evaluate_ab_mixed'`.

- [ ] **Step 3: Implement `evaluate_ab_mixed`**

In `policy.py`, add (reuse `run_episode` for discrete and `run_continuous_episode` for continuous; aggregate with the SAME summarisation `evaluate_ab` already uses — factor the per-policy summary into a shared helper if `evaluate_ab` has it inline):

```python
def evaluate_ab_mixed(
    policies: dict[str, tuple[str, object]],
    *,
    n_scenes: int,
    seed0: int,
) -> dict[str, dict[str, float]]:
    """Score discrete and continuous policies on identical seeds.

    ``policies[name] = (kind, callable)`` with ``kind in {"discrete","continuous"}``.
    Discrete callables run on ExplorationEnv; continuous on ContinuousExplorationEnv;
    both at seeds ``seed0 + i``. Returns the same metric dict as ``evaluate_ab``."""
    from .continuous_env import ContinuousExplorationEnv
    from .sb3_policies import run_continuous_episode

    summary: dict[str, dict[str, float]] = {}
    for name, (kind, fn) in policies.items():
        results = []
        for i in range(n_scenes):
            seed = seed0 + i
            if kind == "continuous":
                results.append(run_continuous_episode(ContinuousExplorationEnv(), fn, seed))
            else:
                results.append(run_episode(ExplorationEnv(), fn, seed))
        summary[name] = _summarise_results(results)  # reuse evaluate_ab's aggregation
    return summary
```

> If `evaluate_ab` aggregates inline, extract that aggregation into a `_summarise_results(results) -> dict[str, float]` helper and have BOTH `evaluate_ab` and `evaluate_ab_mixed` call it (DRY, and keeps the discrete path's numbers identical). Run the existing A/B tests to confirm no drift.

- [ ] **Step 4: Run new + existing A/B tests**

Run: `.venv/bin/pytest tests/test_rl_ab_mixed.py -v && .venv/bin/pytest -k "ab or rl" -q`
Expected: new test passes; existing A/B tests unchanged.

- [ ] **Step 5: Export + ruff + commit**

```bash
git add src/research/rl/policy.py src/research/rl/__init__.py src/research/rl_explorer.py tests/test_rl_ab_mixed.py
git commit -m "feat(rl): evaluate_ab_mixed — 5-way matched-seed A/B (discrete + continuous)"
```

---

## Task 6: CLI `--algo {dqn,sac,tqc}`

**Files:** `src/cli.py`; Test: extend `tests/` (a small CLI-routing test) or `tests/test_sb3_policies.py`

- [ ] **Step 1: Write the failing test**

Add `tests/test_rl_cli_algo.py`:

```python
"""§14.6.3 — rl_exploration --algo routing."""

from __future__ import annotations

import pytest

pytest.importorskip("stable_baselines3")


def test_algo_default_is_dqn() -> None:
    from src.cli import build_parser  # or however the parser is exposed

    args = build_parser().parse_args(["--mode", "rl_exploration"])
    assert getattr(args, "algo", "dqn") == "dqn"


def test_algo_accepts_sac_and_tqc() -> None:
    from src.cli import build_parser

    for algo in ("sac", "tqc"):
        args = build_parser().parse_args(["--mode", "rl_exploration", "--algo", algo])
        assert args.algo == algo
```

> Read `src/cli.py` to find how the parser is constructed/exposed (it may not be `build_parser`). Adapt the import to the real factory; if the parser is built inline in `main()`, refactor the `argparse` construction into a `build_parser()` function FIRST (small, test-driven) so it is unit-testable, keeping `main()` behavior identical.

- [ ] **Step 2: Run → fails**

Run: `.venv/bin/pytest tests/test_rl_cli_algo.py -v`
Expected: FAIL (no `--algo` / no `build_parser`).

- [ ] **Step 3: Add `--algo` + route training**

- Add the argument near the other rl args (`src/cli.py` ~line 90): `--algo`, `choices=["dqn","sac","tqc"]`, `default=os.environ.get("PET_AGENT_RL_ALGO", "dqn")`.
- In `run_rl_exploration` (~line 315): branch on `args.algo`:
  - `dqn` → existing path unchanged (`train_dqn` + `evaluate_ab`).
  - `sac`/`tqc` → `train_sac`/`train_tqc(total_timesteps=args.episodes*..., seed=args.seed, device=...)`, wrap in `Sb3ContinuousPolicy`, build a 5-way policy dict `{heuristic, random, dqn(optional), <algo>}` and call `evaluate_ab_mixed`, then `format_ab_report` and write the same `runs/rl_exploration_<ts>/` artifacts.
  - Keep the non-zero-exit acceptance check (RL must beat random or report inconclusive) consistent with the existing dqn path.
- Map `--episodes` to SB3 `total_timesteps` sensibly (e.g. `total_timesteps = max(2000, args.episodes * 200)`); document the mapping in a comment.

- [ ] **Step 4: Run CLI test + a tiny smoke**

Run: `.venv/bin/pytest tests/test_rl_cli_algo.py -v`
Expected: PASS.
Smoke (fast): `.venv/bin/python main.py --mode rl_exploration --algo sac --episodes 5 --scenes 2 --seed 0 --out runs` → exits 0, writes a report. (Tiny budget; just proves the wiring runs end-to-end. Do NOT commit anything under `runs/` — it is gitignored.)

- [ ] **Step 5: ruff + commit**

```bash
git add src/cli.py tests/test_rl_cli_algo.py
git commit -m "feat(rl): --mode rl_exploration --algo {dqn,sac,tqc} routing"
```

---

## Task 7: Acceptance experiment + docs

**Files:** `docs/spec.md` §14.6.3; optionally `docs/eval.md`

- [ ] **Step 1: Run the real A/B (longer budget, on CUDA)**

Run: `PET_AGENT_DEVICE=cuda .venv/bin/python main.py --mode rl_exploration --algo sac --episodes 300 --scenes 50 --seed 0 --out runs`
Then the same with `--algo tqc`. Record the coverage means for heuristic / random / DQN / SAC / TQC from the generated `report.md`.

- [ ] **Step 2: Judge honestly**

Acceptance (spec §14.6.3): SAC ≥ DQN coverage over the suite, OR the result is honestly reported as inconclusive (uplift < 10%). Capture the real numbers — do not massage them.

- [ ] **Step 3: Update spec §14.6.3 status**

Append an honest **Status — implemented** bullet to §14.6.3 with: the modules (`continuous_env.py`, `sb3_policies.py`, `evaluate_ab_mixed`), the `--algo` selector, install (`uv pip install -e ".[rl]"`), and the measured 5-row coverage table (SAC/TQC vs DQN/heuristic/random), stating clearly whether SAC/TQC beat DQN or the result was inconclusive.

- [ ] **Step 4: Final verification**

Run: `.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/pytest -q`
Expected: ruff clean; full suite green (baseline 452 + the new tests).

- [ ] **Step 5: Commit**

```bash
git add docs/spec.md
git commit -m "docs(rl): §14.6.3 SAC/TQC continuous exploration implemented + A/B results"
```

---

## Self-Review Notes

- **Spec coverage (§14.6.3):** continuous action reformulation (Task 3) ✓; SB3 SAC primary + sb3-contrib TQC (Task 4) ✓; 5-way matched-seed A/B vs heuristic/random/DQN (Task 5) ✓; `--algo` + `PET_AGENT_RL_ALGO` (Task 6) ✓; acceptance = SAC ≥ DQN or honest inconclusive (Task 7) ✓; CUDA via `device` (Tasks 4, 7) ✓; default `dqn` unchanged ✓.
- **Fairness:** the continuous env composes `ExplorationEnv` and shares `reward_terms`, so matched seeds spawn identical worlds and reward parity holds — `test_same_seed_spawns_same_world_as_discrete_env` pins this.
- **No-regression seam:** Task 2 is a behavior-preserving extraction guarded by re-running all discrete RL/env tests; the discrete DQN path and its results are untouched.
- **Type/naming consistency:** `ContinuousExplorationEnv`, `ContinuousEnvConfig`, `train_sac`, `train_tqc`, `Sb3ContinuousPolicy`, `run_continuous_episode`, `evaluate_ab_mixed` are used identically across tasks and exported from both `rl/__init__.py` and `rl_explorer.py`.
- **Execution-time risks called out:** `EpisodeResult` constructor field names (read `policy.py` — Task 4), `evaluate_ab` metric key names (read it — Task 5), the CLI parser factory shape (may need a `build_parser()` extraction — Task 6). Each task flags the read-first step rather than inventing the API.
- **Hardware:** SAC/TQC nets are tiny MLPs (<0.5 GB VRAM); the env sim is the bottleneck. Test budgets use CPU + 200 timesteps to stay CI-fast; the real A/B (Task 7) uses CUDA.
