"""Policies, training loop, and A/B evaluation for the exploration MDP (§14.3).

The RL acceptance criterion is comparative: beat the §12 heuristic on coverage
by ≥10% over 50 trials, *or* honestly report the result as inconclusive. So this
module ships three policies that act on the same 5-dim state — the trained DQN,
a heuristic baseline that mirrors the §12 myopic greedy over the RL action set,
and a random lower bound — plus ``evaluate_ab`` to compare them on identical
procedurally-generated scenes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .dqn import DQNAgent, DQNConfig
from .env import (
    INSPECT_FRONTIER,
    LOOK_AROUND,
    MOVE_TO_KNOWN,
    RETURN_TO_USER,
    EnvConfig,
    ExplorationEnv,
)

Policy = Callable[[np.ndarray], int]


def heuristic_policy(state: np.ndarray) -> int:
    """Myopic baseline over the RL action set, mirroring the §12 heuristic's
    priorities: finish exploring → verify stale → reveal area."""
    known, unknown, target_vis, frontier_dist, uncertainty = state  # noqa: F841
    if unknown < 0.15:  # mostly explored — head home
        return RETURN_TO_USER
    if uncertainty > 0.25:  # discovered-but-stale objects to re-verify
        return MOVE_TO_KNOWN
    if frontier_dist > 0.5:  # frontier is far — a cheap in-place sweep first
        return LOOK_AROUND
    return INSPECT_FRONTIER


def random_policy(rng: np.random.Generator) -> Policy:
    from .env import N_ACTIONS

    def _pick(_state: np.ndarray) -> int:
        return int(rng.integers(0, N_ACTIONS))

    return _pick


class RLExplorationPolicy:
    """Greedy wrapper around a trained DQN agent."""

    def __init__(self, agent: DQNAgent) -> None:
        self.agent = agent

    def __call__(self, state: np.ndarray) -> int:
        return self.agent.act(state, eps=0.0)

    @classmethod
    def load(cls, path: str) -> RLExplorationPolicy:
        return cls(DQNAgent.load(path))


# ── training ─────────────────────────────────────────────────────────────────
def train_dqn(
    *,
    episodes: int = 400,
    env_cfg: EnvConfig | None = None,
    dqn_cfg: DQNConfig | None = None,
    seed: int = 0,
) -> tuple[DQNAgent, list[float]]:
    """Train a DQN on procedurally-generated episodes. Returns the agent and the
    per-episode return history."""
    env = ExplorationEnv(env_cfg)
    env.reset(seed=seed)  # anchor the scene RNG; later resets advance it
    agent = DQNAgent(dqn_cfg or DQNConfig(seed=seed))
    history: list[float] = []
    global_step = 0
    for _ in range(episodes):
        state = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            eps = agent.epsilon(global_step)
            action = agent.act(state, eps)
            nxt, reward, done, _ = env.step(action)
            agent.push(state, action, reward, nxt, done)
            agent.learn()
            state = nxt
            ep_return += reward
            global_step += 1
        history.append(ep_return)
    return agent, history


# ── evaluation ─────────────────────────────────────────────────────────────────
@dataclass
class EpisodeResult:
    total_return: float
    coverage: float
    relevant_found: int
    relevant_total: int
    steps: int


def run_episode(env: ExplorationEnv, policy: Policy, seed: int) -> EpisodeResult:
    state = env.reset(seed=seed)
    done = False
    total = 0.0
    steps = 0
    while not done:
        state, reward, done, _ = env.step(policy(state))
        total += reward
        steps += 1
    return EpisodeResult(
        total_return=total,
        coverage=env.coverage_fraction,
        relevant_found=env.relevant_discovered,
        relevant_total=env.relevant_total,
        steps=steps,
    )


def _summarise_results(results: list[EpisodeResult]) -> dict[str, float]:
    n = len(results)
    return {
        "mean_return": sum(r.total_return for r in results) / n,
        "mean_coverage": sum(r.coverage for r in results) / n,
        "mean_relevant_found": sum(r.relevant_found for r in results) / n,
        "recall": _safe_recall(results),
        "mean_steps": sum(r.steps for r in results) / n,
    }


def evaluate_ab(
    policies: dict[str, Policy],
    *,
    n_scenes: int = 50,
    seed0: int = 10_000,
    env_cfg: EnvConfig | None = None,
) -> dict[str, dict[str, float]]:
    """Run every policy through the same ``n_scenes`` procedurally-generated
    scenes (identical seeds) and return aggregate metrics per policy."""
    seeds = [seed0 + i for i in range(n_scenes)]
    raw: dict[str, list[EpisodeResult]] = {name: [] for name in policies}
    for name, policy in policies.items():
        env = ExplorationEnv(env_cfg)
        for s in seeds:
            raw[name].append(run_episode(env, policy, s))

    return {name: _summarise_results(results) for name, results in raw.items()}


def _safe_recall(results: list[EpisodeResult]) -> float:
    found = sum(r.relevant_found for r in results)
    total = sum(r.relevant_total for r in results)
    return found / total if total > 0 else 0.0


def evaluate_ab_mixed(
    policies: dict[str, tuple[str, object]],
    *,
    n_scenes: int = 50,
    seed0: int = 10_000,
    env_cfg: EnvConfig | None = None,
) -> dict[str, dict[str, float]]:
    """Score discrete and continuous policies on identical seeds.

    ``policies[name] = (kind, callable)`` with ``kind in {"discrete", "continuous"}``.
    Discrete callables run on ExplorationEnv via run_episode; continuous on
    ContinuousExplorationEnv via run_continuous_episode; both at seeds seed0+i, so
    the spawned world is identical and the coverage comparison is fair. Returns the
    same metric dict as evaluate_ab (so format_ab_report / coverage_uplift work).
    """
    from .continuous_env import ContinuousExplorationEnv
    from .sb3_policies import run_continuous_episode

    seeds = [seed0 + i for i in range(n_scenes)]
    summary: dict[str, dict[str, float]] = {}
    for name, (kind, fn) in policies.items():
        results: list[EpisodeResult] = []
        for s in seeds:
            if kind == "continuous":
                results.append(run_continuous_episode(ContinuousExplorationEnv(env_cfg), fn, s))
            elif kind == "discrete":
                results.append(run_episode(ExplorationEnv(env_cfg), fn, s))
            else:
                raise ValueError(f"unknown policy kind {kind!r} for {name!r}")
        summary[name] = _summarise_results(results)
    return summary


def coverage_uplift(summary: dict[str, dict[str, float]], rl: str, baseline: str) -> float:
    """Relative coverage improvement of ``rl`` over ``baseline`` (spec's ≥10%)."""
    base = summary[baseline]["mean_coverage"]
    if base <= 0:
        return 0.0
    return (summary[rl]["mean_coverage"] - base) / base


def format_ab_report(summary: dict[str, dict[str, float]], *, n_scenes: int, episodes: int) -> str:
    """Render the A/B comparison as Markdown (spec §14.3 acceptance)."""
    lines = [
        "# RL Exploration — A/B Report (spec §14.3)",
        "",
        f"- Training episodes: **{episodes}**",
        f"- Evaluation scenes (identical seeds across policies): **{n_scenes}**",
        "",
        "| policy | mean coverage | recall (relevant found) | mean return | mean steps |",
        "|---|---|---|---|---|",
    ]
    for name, m in summary.items():
        lines.append(
            f"| {name} | {m['mean_coverage']:.3f} | {m['recall']:.2f} "
            f"| {m['mean_return']:+.2f} | {m['mean_steps']:.1f} |"
        )
    lines.append("")
    if "rl" in summary and "heuristic" in summary:
        uplift = coverage_uplift(summary, "rl", "heuristic")
        verdict = (
            f"RL beats the heuristic baseline on coverage by **{uplift * 100:+.1f}%** "
            "(spec acceptance: ≥10%)."
            if uplift >= 0.10
            else f"RL coverage uplift over heuristic is **{uplift * 100:+.1f}%** — "
            "below the +10% bar; reported honestly as inconclusive on coverage (spec §14.3)."
        )
        lines += ["## Verdict", "", verdict, ""]
    return "\n".join(lines)
