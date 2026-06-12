"""PyPose pose-graph optimiser — spec §14.6.2 SLAM back-end.

This module implements an anchored Levenberg–Marquardt pose-graph optimiser
built on PyPose.  Vertices represent ``world ← node_i`` SE(3) transforms;
edges carry relative measurements ``T_{j|i}`` (i.e. the transform that takes
a point in frame *i* to frame *j*).  The public API accepts and returns plain
4×4 NumPy homogeneous matrices; PyPose / PyTorch tensors are internal.

Gauge freedom is fixed by an anchor residual: the log-map of node 0 is
appended to the residual vector (scaled by ``anchor_weight``), which keeps
node 0 near the identity without hard-clamping any parameter.

Converter names used (verified against pypose 0.9.5):
  - ``pp.mat2SE3(tensor_4x4)`` — 4×4 homogeneous → SE3 LieTensor
  - ``se3.matrix()``           — SE3 LieTensor → 4×4 tensor (round-trip exact)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pypose as pp
import torch

if TYPE_CHECKING:
    pass

__all__ = ["PoseGraph"]

_ANCHOR_WEIGHT: float = 100.0


# ---------------------------------------------------------------------------
# Internal torch.nn.Module — forward returns the stacked residual vector
# ---------------------------------------------------------------------------


class _PoseGraphModule(torch.nn.Module):
    """LM-optimisable pose-graph with a soft anchor on node 0."""

    def __init__(self, nodes: pp.LieTensor, anchor_weight: float) -> None:
        super().__init__()
        self.nodes = pp.Parameter(nodes)
        self._anchor_weight = anchor_weight

    def forward(
        self,
        edges: torch.Tensor,  # (E, 2) long
        meas: pp.LieTensor,  # SE3 (E,)
    ) -> torch.Tensor:
        ti = self.nodes[edges[:, 0]]
        tj = self.nodes[edges[:, 1]]
        pred = ti.Inv() @ tj
        edge_res = (meas.Inv() @ pred).Log().tensor().view(-1)
        anchor = self.nodes[0].Log().tensor().view(-1) * self._anchor_weight
        return torch.cat([edge_res, anchor])


# ---------------------------------------------------------------------------
# Helpers: numpy 4×4  ↔  pp.SE3
# ---------------------------------------------------------------------------


def _np_to_se3(mat: np.ndarray) -> pp.LieTensor:
    """Convert a (4, 4) numpy homogeneous matrix to a scalar SE3 LieTensor."""
    t = torch.tensor(mat, dtype=torch.float64)
    return pp.mat2SE3(t)


def _se3_to_np(se3: pp.LieTensor) -> np.ndarray:
    """Convert a scalar SE3 LieTensor to a (4, 4) numpy matrix."""
    return se3.matrix().detach().numpy()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class _Edge:
    i: int
    j: int
    meas: np.ndarray  # 4×4


class PoseGraph:
    """Anchored pose-graph optimiser (LM back-end via PyPose).

    Usage::

        g = PoseGraph()
        g.add_node(np.eye(4))          # node 0 — world origin (anchor)
        g.add_node(odometry_pose)      # node 1 …
        g.add_odometry_edge(0, 1, T_1_from_0)
        g.add_loop_edge(3, 0, T_0_from_3)
        g.optimize(iters=20)
        corrected = g.pose(1)          # 4×4 numpy
    """

    def __init__(self, anchor_weight: float = _ANCHOR_WEIGHT) -> None:
        self._nodes: list[np.ndarray] = []
        self._edges: list[_Edge] = []
        self._anchor_weight = anchor_weight
        # Optimised poses (populated after optimize())
        self._optimised: list[np.ndarray] | None = None

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_node(self, pose: np.ndarray) -> int:
        """Append a node with an initial ``world ← node`` pose (4×4).

        Returns the new node index.
        """
        self._nodes.append(np.asarray(pose, dtype=float))
        self._optimised = None  # invalidate cached solution
        return len(self._nodes) - 1

    def add_odometry_edge(self, i: int, j: int, T_j_from_i: np.ndarray) -> None:
        """Add an odometry (or any relative) constraint ``T_{j|i}`` (4×4)."""
        self._edges.append(_Edge(i, j, np.asarray(T_j_from_i, dtype=float)))

    def add_loop_edge(self, i: int, j: int, T_j_from_i: np.ndarray) -> None:
        """Add a loop-closure constraint.  Alias for ``add_odometry_edge``."""
        self.add_odometry_edge(i, j, T_j_from_i)

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    def optimize(self, *, iters: int = 20) -> None:
        """Run LM optimisation in-place.

        No-ops when there are no nodes or no edges (nothing to optimise).
        """
        if not self._nodes or not self._edges:
            return

        # Build initial node tensor — shape (N, 7) SE3
        node_tensors = [_np_to_se3(m).tensor() for m in self._nodes]
        nodes_se3 = pp.SE3(torch.stack(node_tensors))

        module = _PoseGraphModule(nodes_se3, self._anchor_weight)

        edges = torch.tensor([[e.i, e.j] for e in self._edges], dtype=torch.long)
        meas_tensors = [_np_to_se3(e.meas).tensor() for e in self._edges]
        meas = pp.SE3(torch.stack(meas_tensors))

        opt = pp.optim.LM(module)
        for _ in range(iters):
            opt.step((edges, meas))

        # Cache optimised poses as numpy 4×4
        self._optimised = [_se3_to_np(module.nodes[k]) for k in range(len(self._nodes))]

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def n_nodes(self) -> int:
        """Number of nodes currently in the graph."""
        return len(self._nodes)

    def pose(self, i: int) -> np.ndarray:
        """Return the optimised ``world ← node_i`` pose as a (4, 4) numpy array.

        If ``optimize()`` has not been called yet, returns the initial pose.
        """
        if self._optimised is not None:
            return self._optimised[i].copy()
        return self._nodes[i].copy()
