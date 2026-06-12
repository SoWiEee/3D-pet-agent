"""§14.6.3 — rl_exploration --algo routing."""

from __future__ import annotations

from src.cli import build_parser


def test_algo_default_is_dqn(monkeypatch) -> None:
    monkeypatch.delenv("PET_AGENT_RL_ALGO", raising=False)
    args = build_parser().parse_args(["--mode", "rl_exploration"])
    assert args.algo == "dqn"


def test_algo_accepts_sac_and_tqc() -> None:
    for algo in ("sac", "tqc"):
        args = build_parser().parse_args(["--mode", "rl_exploration", "--algo", algo])
        assert args.algo == algo


def test_algo_env_default(monkeypatch) -> None:
    monkeypatch.setenv("PET_AGENT_RL_ALGO", "sac")
    args = build_parser().parse_args(["--mode", "rl_exploration"])
    assert args.algo == "sac"
