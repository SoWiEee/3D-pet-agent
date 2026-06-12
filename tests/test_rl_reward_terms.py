"""§14.6.3 — reward_terms is the shared shaping used by discrete + continuous envs."""

from __future__ import annotations

import pytest

# env.py itself is dependency-light, but importing it runs src.research.rl/__init__,
# which eagerly pulls the continuous (gymnasium) + SB3 policy modules.
pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")
pytest.importorskip("sb3_contrib")

from src.research.rl.env import (  # noqa: E402
    R_COLLISION,
    R_DISCOVER_RELEVANT,
    R_REDUCED_AREA,
    R_REPEAT_FAILED,
    R_UNNECESSARY_MOVE,
    EnvConfig,
    reward_terms,
)


def test_discover_and_area_terms() -> None:
    cfg = EnvConfig()
    terms = reward_terms(
        cfg,
        new_area=int(cfg.area_norm),
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
        cfg,
        new_area=0,
        discovered_relevant=0,
        verified_stale=0,
        collided=True,
        is_move=True,
        failed_inspection=False,
        prev_failed=False,
    )
    assert terms["collision"] == R_COLLISION


def test_unnecessary_move_and_repeat_failed() -> None:
    cfg = EnvConfig()
    terms = reward_terms(
        cfg,
        new_area=0,
        discovered_relevant=0,
        verified_stale=0,
        collided=False,
        is_move=True,
        failed_inspection=True,
        prev_failed=True,
    )
    assert terms["move_cost"] == R_UNNECESSARY_MOVE
    assert terms["repeat_failed"] == R_REPEAT_FAILED
