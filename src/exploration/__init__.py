"""Phase 9 — Active exploration.

Public surface:

- :class:`CoverageGrid`        observed/unobserved cells aligned with the nav grid
- :class:`ExplorationGoal`     typed goal for the planner to chase
- :class:`ExplorationPlanner`  picks the next viewpoint by spec §12 heuristic
"""

from __future__ import annotations

from .coverage_grid import CoverageGrid, CoverageGridConfig
from .exploration_planner import (
    ExplorationCandidate,
    ExplorationGoal,
    ExplorationPlanner,
    ExplorationPlannerConfig,
)

__all__ = [
    "CoverageGrid",
    "CoverageGridConfig",
    "ExplorationCandidate",
    "ExplorationGoal",
    "ExplorationPlanner",
    "ExplorationPlannerConfig",
]
