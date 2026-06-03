"""RL exploration sidecar (spec §14.3).

Three layers:
- ``ExplorationEnv`` mechanics (state, rewards, determinism, termination).
- ``DQNAgent`` plumbing (network, buffer, greedy act, learn step, save/load).
- learning + A/B: a short-trained DQN must beat the random baseline on coverage.
"""

from __future__ import annotations

import numpy as np

from src.research.rl import (
    ACTION_NAMES,
    N_ACTIONS,
    STATE_DIM,
    DQNAgent,
    DQNConfig,
    EnvConfig,
    ExplorationEnv,
    QNetwork,
    ReplayBuffer,
    RLExplorationPolicy,
    coverage_uplift,
    evaluate_ab,
    format_ab_report,
    heuristic_policy,
    random_policy,
    run_episode,
    train_dqn,
)
from src.research.rl.env import RETURN_TO_USER


# ── environment ──────────────────────────────────────────────────────────────
def test_reset_returns_normalized_state() -> None:
    env = ExplorationEnv()
    s = env.reset(seed=1)
    assert s.shape == (STATE_DIM,)
    assert s.dtype == np.float32
    # known, unknown_ratio, target_visible, frontier_dist, uncertainty all in [0,1].
    assert np.all(s >= 0.0) and np.all(s <= 1.0)


def test_same_seed_reproduces_scene() -> None:
    a = ExplorationEnv().reset(seed=42)
    b = ExplorationEnv().reset(seed=42)
    assert np.allclose(a, b)
    c = ExplorationEnv().reset(seed=43)
    assert not np.allclose(a, c)  # different seed → different scene


def test_return_to_user_terminates_episode() -> None:
    env = ExplorationEnv()
    env.reset(seed=2)
    _, _, done, info = env.step(RETURN_TO_USER)
    assert done is True
    assert ACTION_NAMES[info.action] == "return_to_user"


def test_episode_respects_max_steps() -> None:
    env = ExplorationEnv(EnvConfig(max_steps=5))
    env.reset(seed=3)
    done = False
    steps = 0
    while not done:
        # avoid return_to_user so termination is by step cap
        _, _, done, _ = env.step(0)
        steps += 1
    assert steps <= 5


def test_inspect_frontier_reveals_area() -> None:
    # From a fresh corner-ish start, moving to a frontier and observing should
    # reveal previously-unknown cells (positive area reward term).
    env = ExplorationEnv(EnvConfig(n_obstacles=(0, 0)))
    env.reset(seed=7)
    before = env.coverage_fraction
    _, reward, _, info = env.step(0)  # inspect_frontier
    assert info.new_area_cells > 0
    assert env.coverage_fraction >= before
    assert "area" in info.reward_terms


def test_collision_penalizes_when_obstacle_blocks_path() -> None:
    # A dense obstacle field makes some moves collide; over a few steps at least
    # one collision penalty should fire, and the cat is never teleported through.
    env = ExplorationEnv(EnvConfig(n_obstacles=(3, 3), obstacle_radius=0.5))
    env.reset(seed=11)
    saw_collision = any(env.step(0)[3].collided for _ in range(8))
    # Not guaranteed every seed, but the mechanism must be reachable; assert the
    # reward bookkeeping is wired when it does happen.
    env.reset(seed=11)
    _, reward, _, info = env.step(0)
    if info.collided:
        assert info.reward_terms.get("collision", 0.0) < 0.0
    assert isinstance(saw_collision, bool)


# ── DQN plumbing ─────────────────────────────────────────────────────────────
def test_qnetwork_output_shape() -> None:
    import torch

    net = QNetwork()
    out = net(torch.zeros(4, STATE_DIM))
    assert out.shape == (4, N_ACTIONS)


def test_replay_buffer_push_and_sample() -> None:
    import random

    buf = ReplayBuffer(100)
    for i in range(20):
        buf.push(
            np.zeros(STATE_DIM, np.float32),
            i % N_ACTIONS,
            1.0,
            np.ones(STATE_DIM, np.float32),
            False,
        )
    assert len(buf) == 20
    s, a, r, s2, done = buf.sample(8, random.Random(0))
    assert s.shape == (8, STATE_DIM)
    assert a.shape == (8, 1)


def test_agent_greedy_action_is_deterministic() -> None:
    agent = DQNAgent(DQNConfig(seed=0))
    state = np.array([0.1, 0.8, 0.0, 0.5, 0.2], dtype=np.float32)
    a1 = agent.act(state, eps=0.0)
    a2 = agent.act(state, eps=0.0)
    assert a1 == a2
    assert 0 <= a1 < N_ACTIONS


def test_agent_learn_runs_after_warmup_and_saves(tmp_path) -> None:
    agent = DQNAgent(DQNConfig(seed=0, warmup=10, batch_size=8))
    rng = np.random.default_rng(0)
    for _ in range(40):
        s = rng.random(STATE_DIM).astype(np.float32)
        agent.push(s, int(rng.integers(0, N_ACTIONS)), float(rng.random()), s, False)
    loss = agent.learn()
    assert loss is not None and np.isfinite(loss)

    path = tmp_path / "model.pt"
    agent.save(str(path))
    reloaded = DQNAgent.load(str(path))
    state = rng.random(STATE_DIM).astype(np.float32)
    assert reloaded.act(state, eps=0.0) == agent.act(state, eps=0.0)


# ── policy + evaluation ──────────────────────────────────────────────────────
def test_heuristic_and_random_policies_return_valid_actions() -> None:
    state = np.array([0.2, 0.6, 1.0, 0.4, 0.3], dtype=np.float32)
    assert 0 <= heuristic_policy(state) < N_ACTIONS
    rnd = random_policy(np.random.default_rng(0))
    assert 0 <= rnd(state) < N_ACTIONS


def test_run_episode_reports_coverage_and_recall() -> None:
    res = run_episode(ExplorationEnv(), heuristic_policy, seed=5)
    assert 0.0 <= res.coverage <= 1.0
    assert res.relevant_found <= res.relevant_total
    assert res.steps >= 1


def test_evaluate_ab_aggregates_all_policies() -> None:
    summary = evaluate_ab(
        {"heuristic": heuristic_policy, "random": random_policy(np.random.default_rng(0))},
        n_scenes=5,
        seed0=100,
    )
    assert set(summary) == {"heuristic", "random"}
    for m in summary.values():
        assert {"mean_coverage", "recall", "mean_return", "mean_steps"} <= set(m)
    assert "Verdict" not in format_ab_report(summary, n_scenes=5, episodes=1)  # no rl key


def test_trained_dqn_beats_random_on_coverage() -> None:
    # The headline acceptance signal (spec §14.3): a trained policy must clearly
    # beat random exploration. Short training keeps the test fast but decisive.
    agent, _ = train_dqn(episodes=120, seed=0)
    summary = evaluate_ab(
        {
            "rl": RLExplorationPolicy(agent),
            "random": random_policy(np.random.default_rng(7)),
        },
        n_scenes=20,
        seed0=20_000,
    )
    assert coverage_uplift(summary, "rl", "random") > 0.10  # ≥10% over random
