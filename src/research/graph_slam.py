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

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pypose as pp
import torch
from scipy.spatial.transform import Rotation

from ..spatial.frame_packet import CameraIntrinsics, CameraPoseWorld

if TYPE_CHECKING:
    pass

__all__ = ["GraphSlamConfig", "GraphSlamPoseSource", "OrbBowLoopDetector", "PoseGraph"]

log = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# ORB appearance-based loop detector
# ---------------------------------------------------------------------------


class OrbBowLoopDetector:
    """Appearance-based loop detection over keyframe ORB descriptors.

    A lightweight stand-in for a full DBoW vocabulary: ratio-test BF matching of
    the current keyframe against stored keyframe descriptors; the best past
    keyframe (older than ``min_gap``) with >= ``min_matches`` good matches is a
    loop. Returns the matched keyframe id, else None.
    """

    def __init__(
        self,
        *,
        n_features: int = 1000,
        min_gap: int = 10,
        min_matches: int = 30,
        ratio: float = 0.75,
    ) -> None:
        self._orb = cv2.ORB_create(nfeatures=n_features)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._min_gap = min_gap
        self._min_matches = min_matches
        self._ratio = ratio
        self._kfs: list[tuple[int, np.ndarray | None]] = []

    def add_keyframe(self, kf_id: int, gray: np.ndarray) -> int | None:
        """Process a new keyframe image and return the best loop-closure match id.

        Parameters
        ----------
        kf_id:
            Monotonically increasing keyframe index.
        gray:
            Grayscale (H, W) or BGR (H, W, 3) uint8 image.

        Returns
        -------
        int | None
            The keyframe id of the best loop-closure candidate, or ``None``
            when no past frame exceeds the match threshold.
        """
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        _, des = self._orb.detectAndCompute(gray, None)
        best_id, best_n = None, self._min_matches - 1
        if des is not None:
            for past_id, past_des in self._kfs:
                if kf_id - past_id < self._min_gap or past_des is None:
                    continue
                n = self._count_matches(past_des, des)
                if n > best_n:
                    best_id, best_n = past_id, n
        self._kfs.append((kf_id, des))
        return best_id

    def _count_matches(self, d1: np.ndarray, d2: np.ndarray) -> int:
        """Count ratio-test-passing BF matches between two descriptor arrays."""
        if len(d1) < 2 or len(d2) < 2:
            return 0
        good = 0
        for pair in self._bf.knnMatch(d1, d2, k=2):
            if len(pair) == 2 and pair[0].distance < self._ratio * pair[1].distance:
                good += 1
        return good


# ---------------------------------------------------------------------------
# Graph-SLAM pose source
# ---------------------------------------------------------------------------

# Same axis-flip as slam_adapter._CV_TO_GRAPHICS: OpenCV cam (X right, Y down,
# Z forward) → graphics world (X right, Y up, Z back).
_CV_TO_GRAPHICS = np.diag([1.0, -1.0, -1.0])


def _se3(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Build a 4×4 homogeneous matrix from R (3×3) and t (3,)."""
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = rotation
    t[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return t


def _se3_inv(t: np.ndarray) -> np.ndarray:
    """Invert a 4×4 rigid-body transform without a general matrix inverse."""
    r = t[:3, :3]
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = r.T
    inv[:3, 3] = -r.T @ t[:3, 3]
    return inv


@dataclass
class GraphSlamConfig:
    """Tuning knobs for :class:`GraphSlamPoseSource`.

    Attributes
    ----------
    keyframe_stride:
        Add a new graph node every ``keyframe_stride`` successfully tracked
        frames (default 1 — every frame).
    min_inliers:
        VO inlier threshold below which tracking is declared lost (same
        semantic as ``SLAMPoseSource.min_inliers``).
    optimize_every:
        Run ``PoseGraph.optimize()`` every this many new keyframes.
        ``1`` (default) optimises after each new node.
    loop_min_matches:
        Minimum feature matches to accept a loop-closure hypothesis.
    loop_min_gap:
        Minimum keyframe-index gap before a loop hypothesis is considered,
        preventing false positives against very recent keyframes.
    """

    keyframe_stride: int = 1
    min_inliers: int = 12
    optimize_every: int = 1
    loop_min_matches: int = 30
    loop_min_gap: int = 10


class GraphSlamPoseSource:
    """ORB-VO front-end + pose-graph back-end with loop closure.

    Mirrors :class:`~src.research.slam_adapter.SLAMPoseSource` exactly on the
    non-loop path (identical ``_t_wc`` chaining), and additionally:

    * Maintains a :class:`PoseGraph` — each successfully tracked frame
      becomes a node; consecutive nodes are joined by an odometry edge.
    * Runs :class:`OrbBowLoopDetector` on every new keyframe.  When a loop
      is detected, a loop edge is added and the graph is optimised; the
      current ``_t_wc`` is then refreshed from the latest optimised node so
      subsequent VO chaining continues from the corrected pose.

    Source string is ``"slam"`` (same as :class:`SLAMPoseSource`) so the
    object lifter, tests, and downstream consumers that check ``.source``
    need no change.

    Parameters
    ----------
    intrinsics:
        Camera intrinsic parameters (used to build the K matrix for VO).
    vo:
        Optional :class:`~src.research.slam_adapter.VisualOdometry` backend.
        Defaults to a fresh :class:`OrbVisualOdometry`.
    scale:
        Monocular scale factor applied to VO translation (default 1.0).
    config:
        Algorithm hyper-parameters; see :class:`GraphSlamConfig`.
    """

    from src.research.slam_adapter import OrbVisualOdometry as _DefaultVO
    from src.research.slam_adapter import VisualOdometry as _VOProtocol

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        *,
        vo: object | None = None,
        scale: float = 1.0,
        config: GraphSlamConfig | None = None,
    ) -> None:
        # Lazily import to avoid circular-import issues at module load.
        from src.research.slam_adapter import OrbVisualOdometry, VisualOdometry  # noqa: F401

        self._k = np.array(
            [
                [intrinsics.fx, 0.0, intrinsics.cx],
                [0.0, intrinsics.fy, intrinsics.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self._vo = vo if vo is not None else OrbVisualOdometry()
        self._scale = float(scale)
        self._cfg = config if config is not None else GraphSlamConfig()

        # pose-graph back-end
        self._graph = PoseGraph()
        self._loop_detector = OrbBowLoopDetector(
            min_gap=self._cfg.loop_min_gap,
            min_matches=self._cfg.loop_min_matches,
        )

        # incremental keyframe counter (VO stride gate)
        self._frames_since_kf: int = 0
        # gray images stored per graph-node index for loop-closure re-estimation
        self._kf_grays: list[np.ndarray] = []

        self.reset()

    # ------------------------------------------------------------------
    # PoseSource interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset to initial state — identical behaviour to SLAMPoseSource."""
        self._t_wc = np.eye(4, dtype=np.float64)  # world←camera, OpenCV convention
        self._prev_gray: np.ndarray | None = None
        self._available = False
        self._last_frame_id = -1
        self._frames_since_kf = 0
        # Re-create graph and loop detector so reset is a clean slate.
        self._graph = PoseGraph()
        self._loop_detector = OrbBowLoopDetector(
            min_gap=self._cfg.loop_min_gap,
            min_matches=self._cfg.loop_min_matches,
        )
        self._kf_grays = []

    def track(
        self,
        frame_id: int,
        image: np.ndarray,
        depth: np.ndarray | None = None,
        timestamp: float | None = None,  # noqa: ARG002 — interface symmetry
    ) -> CameraPoseWorld:
        """Ingest one frame, advance the pose, and optionally close a loop."""
        gray = _to_gray(image)
        self._last_frame_id = frame_id

        # ── First frame: anchor world frame at camera origin ──────────────
        if self._prev_gray is None:
            self._prev_gray = gray
            self._available = True
            # Add the origin node (world←cam0 = I)
            node_idx = self._graph.add_node(self._t_wc.copy())
            self._kf_grays.append(gray.copy())
            self._loop_detector.add_keyframe(node_idx, gray)
            self._frames_since_kf = 0
            return self._current_pose()

        # ── Subsequent frames: run VO ─────────────────────────────────────
        rel = self._vo.estimate(self._prev_gray, gray, self._k, prev_depth=depth)
        if rel is None or rel.n_inliers < self._cfg.min_inliers:
            self._available = False
            log.debug("GraphSLAM tracking lost at frame %d", frame_id)
            return self._current_pose()

        # motion = T_{curr←prev} (VO convention, same as SLAMPoseSource)
        motion = _se3(rel.rotation, rel.translation * self._scale)
        # Compose: T_{world←curr} = T_{world←prev} @ T_{prev←curr}
        self._t_wc = self._t_wc @ _se3_inv(motion)
        self._prev_gray = gray
        self._available = True
        self._frames_since_kf += 1

        # ── Keyframe + graph update (every keyframe_stride good frames) ───
        if self._frames_since_kf >= self._cfg.keyframe_stride:
            self._frames_since_kf = 0
            self._add_keyframe_to_graph(gray)

        return self._current_pose()

    def get(
        self,
        frame_id: int | None = None,  # noqa: ARG002 — streaming, not indexed
        timestamp: float | None = None,  # noqa: ARG002
    ) -> CameraPoseWorld:
        return self._current_pose()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_keyframe_to_graph(self, gray: np.ndarray) -> None:
        """Add current pose as a new graph node, odometry edge, and check loops."""
        prev_node_idx = self._graph.n_nodes - 1
        curr_node_idx = self._graph.add_node(self._t_wc.copy())

        # Odometry edge measurement: T_{curr|prev} = inv(world_from_prev) @ world_from_curr
        world_from_prev = self._graph.pose(prev_node_idx)
        world_from_curr = self._graph.pose(curr_node_idx)
        meas = _se3_inv(world_from_prev) @ world_from_curr
        self._graph.add_odometry_edge(prev_node_idx, curr_node_idx, meas)

        # Store gray for potential loop-closure re-estimation
        self._kf_grays.append(gray.copy())

        # Loop detection
        loop_match = self._loop_detector.add_keyframe(curr_node_idx, gray)
        if loop_match is not None:
            log.debug("GraphSLAM loop detected: node %d → node %d", loop_match, curr_node_idx)
            self._close_loop(loop_match, curr_node_idx)

        # Periodic graph optimisation
        if curr_node_idx % self._cfg.optimize_every == 0:
            self._graph.optimize()
            # Refresh _t_wc from the optimised latest node so subsequent VO
            # chaining continues from the corrected pose.
            self._t_wc = self._graph.pose(curr_node_idx).copy()

    def _close_loop(self, matched_node: int, curr_node: int) -> None:
        """Add a loop-closure edge between two graph nodes.

        The measurement is computed from current node estimates
        (``inv(world_from_matched) @ world_from_curr``) — internally consistent
        regardless of accumulated drift.  If the gray images for the matched
        node are available, VO is attempted for a tighter measurement; on
        failure the graph-estimate-based measurement is used as fallback.
        """
        world_from_matched = self._graph.pose(matched_node)
        world_from_curr = self._graph.pose(curr_node)
        meas = _se3_inv(world_from_matched) @ world_from_curr

        # Optionally tighten with VO between stored keyframe grays
        if matched_node < len(self._kf_grays) and curr_node < len(self._kf_grays):
            matched_gray = self._kf_grays[matched_node]
            curr_gray = self._kf_grays[curr_node]
            loop_rel = self._vo.estimate(matched_gray, curr_gray, self._k)
            if loop_rel is not None and loop_rel.n_inliers >= self._cfg.min_inliers:
                loop_motion = _se3(loop_rel.rotation, loop_rel.translation * self._scale)
                # loop_motion = T_{curr←matched} — same convention as odometry
                meas = _se3_inv(loop_motion)

        self._graph.add_loop_edge(matched_node, curr_node, meas)

    def _current_pose(self) -> CameraPoseWorld:
        """Convert internal ``_t_wc`` (OpenCV convention) to a CameraPoseWorld."""
        r_wc = self._t_wc[:3, :3]
        t_wc = self._t_wc[:3, 3]
        r_pose = _CV_TO_GRAPHICS @ r_wc @ _CV_TO_GRAPHICS
        pos = _CV_TO_GRAPHICS @ t_wc
        quat = Rotation.from_matrix(r_pose).as_quat()  # (x, y, z, w)
        return CameraPoseWorld(
            available=self._available,
            source="slam",
            position=(float(pos[0]), float(pos[1]), float(pos[2])),
            quaternion=(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
        )


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Convert BGR or grayscale image to grayscale."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
