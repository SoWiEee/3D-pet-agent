"""§14.6.2 — anchored pose-graph optimiser (numpy SE3 in/out)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pypose")
from src.research.graph_slam import PoseGraph  # noqa: E402


def _xform(dx: float) -> np.ndarray:
    t = np.eye(4)
    t[0, 3] = dx
    return t


def test_anchored_loop_corrects_drift() -> None:
    g = PoseGraph()
    g.add_node(np.eye(4))
    g.add_node(_xform(1.0))
    g.add_node(_xform(2.0))
    g.add_node(_xform(3.5))  # drift
    g.add_odometry_edge(0, 1, _xform(1.0))
    g.add_odometry_edge(1, 2, _xform(1.0))
    g.add_odometry_edge(2, 3, _xform(1.0))
    g.add_loop_edge(3, 0, _xform(-3.0))
    g.optimize(iters=20)
    assert np.allclose(g.pose(0)[:3, 3], [0, 0, 0], atol=1e-3)
    assert abs(g.pose(3)[0, 3] - 3.0) < 0.05


def test_pose_count() -> None:
    g = PoseGraph()
    g.add_node(np.eye(4))
    g.add_node(_xform(1.0))
    assert g.n_nodes == 2
