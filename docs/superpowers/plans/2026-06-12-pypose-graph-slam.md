# PyPose Graph-SLAM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a real graph-SLAM back-end — a **PyPose** (`pp.optim.LM`) pose-graph optimiser with ORB bag-of-words loop closure on the existing ORB-VO front-end — behind the existing `PoseSource` protocol, so the live pipeline gets a drift-corrected, loop-closing pose source. The raw frame-to-frame ORB-VO (`SLAMPoseSource`) stays the default fallback.

**Architecture:** A new `research/graph_slam.py` adds three units: (1) `PoseGraph` — a torch `nn.Module` of `pp.Parameter(pp.SE3)` keyframe vertices with odometry + loop-closure edges, optimised by `pp.optim.LM` over tangent-space relative-pose residuals (first node anchored to fix gauge); (2) `OrbBowLoopDetector` — ORB-descriptor appearance matching that flags revisited keyframes; (3) `GraphSlamPoseSource` — a streaming `PoseSource` (`track`/`get`) that uses `OrbVisualOdometry` for keyframe-to-keyframe odometry, inserts vertices/edges, fires loop closure + LM optimisation, and returns the optimised latest pose in the graphics-world frame. Selected via `PET_AGENT_POSE_SOURCE=graph_slam`; everything else is unchanged.

**Tech Stack:** Python 3.12, PyTorch 2.12+cu130 (CUDA), `pypose` 0.9.5 (`pp.SE3`, `pp.optim.LM`, `pp.optim.kernel`), OpenCV ORB, numpy. Implements spec §14.6.2.

---

## Context the implementer needs (verified against the codebase + a working PyPose probe)

- `src/research/slam_adapter.py` — reuse, do not break:
  - `RelativePose(rotation: np.ndarray(3,3), translation: np.ndarray(3,), n_inliers: int)` — `T_{curr←prev}` (OpenCV convention).
  - `OrbVisualOdometry().estimate(prev_gray, curr_gray, k_matrix, prev_depth=None) -> RelativePose | None`.
  - `SLAMPoseSource` — the existing raw-VO pose source. Its conventions are the template: `_t_wc` is `world←camera` SE(3) (OpenCV camera axes); `_CV_TO_GRAPHICS = np.diag([1,-1,-1])` converts a camera-frame vector/rotation to graphics-world; `_current_pose()` builds a `CameraPoseWorld`. Helpers `_se3(R,t)`, `_se3_inv(T)`, `_to_gray(img)`.
- `src/spatial/pose_source.py`:
  - `PoseSource` Protocol: `get(self, frame_id: int, timestamp: float | None = None) -> CameraPoseWorld`. Streaming sources additionally expose `track(frame_id, image, depth=None, timestamp=None) -> CameraPoseWorld` and `reset()`.
  - `CameraPoseWorld(available: bool, source: str, position: tuple[float,float,float], quaternion: tuple[float,float,float,float])` (quat is x,y,z,w).
  - `CameraIntrinsics(fx, fy, cx, cy)` (and `from_fov`).
- `src/runtime/perception_loop.py::_make_pose_source(intrinsics)` (line ~168) branches on `self.cfg.settings.pose_source`; `"slam"` → `SLAMPoseSource(intrinsics)`; else `FixedPoseSource()`.
- `src/config.py` (line ~200): `pose_source: Literal["fixed", "sim", "slam"] = "fixed"` under `Settings`. Env override `PET_AGENT_POSE_SOURCE`.
- **PyPose pose-graph API — VERIFIED WORKING** (this exact pattern optimised a 4-node loop to residual 0):
  ```python
  import torch, pypose as pp

  class PoseGraph(torch.nn.Module):
      def __init__(self, nodes):           # nodes: pp.SE3 (N,7)
          super().__init__()
          self.nodes = pp.Parameter(nodes)
      def forward(self, edges, meas):      # edges (E,2) long; meas pp.SE3 (E,) = T_j_from_i
          Ti = self.nodes[edges[:, 0]]
          Tj = self.nodes[edges[:, 1]]
          pred = Ti.Inv() @ Tj             # relative pose i->j
          return (meas.Inv() @ pred).Log().tensor().view(-1)   # tangent residual

  opt = pp.optim.LM(graph)
  for _ in range(15):
      loss = opt.step((edges, meas))       # returns scalar loss
  ```
  To fix the gauge (otherwise the solution floats), **anchor node 0**: add a unary edge pulling `nodes[0]` to identity, or freeze it by adding a strong prior edge `(0,0)` with `meas = identity` — simplest is a prior residual `nodes[0].Log()` appended to `forward`. Use `pp.identity_SE3(N)` for init, `pp.SE3(tensor)` to wrap a (…,7) xyz+quat tensor, `.matrix()` to get a 4×4, `pp.mat2SE3(T)` / `pp.from_matrix(T, ltype=pp.SE3_type)` to convert a 4×4 back (probe the exact converter name during Task 2 — `pp.mat2SE3` is the likely one; verify).

---

## File Structure

- `pyproject.toml` — add `slam` extra: `pypose`.
- `src/research/graph_slam.py` (create) — `PoseGraph`, `OrbBowLoopDetector`, `GraphSlamPoseSource`, `GraphSlamConfig`.
- `src/config.py` — extend the `pose_source` Literal with `"graph_slam"`.
- `src/runtime/perception_loop.py` — add the `graph_slam` branch in `_make_pose_source`.
- Tests: `tests/test_pypose_pgo.py`, `tests/test_graph_slam_loop_detector.py`, `tests/test_graph_slam_pose_source.py`, `tests/test_pose_source_selection.py`.
- `docs/spec.md` §14.6.2 status.

Keep `graph_slam.py` cohesive but under ~350 lines; if the three units crowd it, that's acceptable as one SLAM module, but flag if it passes ~400.

---

## Task 1: Add `pypose` as an optional dependency + verify the PGO primitive

**Files:** `pyproject.toml`; Test: `tests/test_pypose_pgo.py`

- [ ] **Step 1: Add the extra**

In `pyproject.toml` `[project.optional-dependencies]`:
```toml
slam = ["pypose>=0.9"]
```

- [ ] **Step 2: Install + confirm numpy intact**

Run: `uv pip install -e ".[slam]"`
Then: `.venv/bin/python -c "import numpy, pypose, torch; print(numpy.__version__, pypose.__version__, torch.cuda.is_available())"`
Expected: `2.1.3 0.9.5 True` (numpy MUST stay 2.1.3 — if it changes, STOP/BLOCKED). pypose is already installed in this venv, so this mainly records the dep.

- [ ] **Step 3: Pin the PGO primitive with a test (TDD anchor for Task 2's optimiser)**

Create `tests/test_pypose_pgo.py`:
```python
"""§14.6.2 — PyPose pose-graph optimisation primitive (the SLAM back-end core)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pp = pytest.importorskip("pypose")


def test_lm_closes_a_loop() -> None:
    # 4 nodes on a line; odometry says each +1 in x; a loop edge says 3->0 is -3.
    # LM must drive the tangent residuals to ~0 (consistent loop).
    class PoseGraph(torch.nn.Module):
        def __init__(self, nodes):
            super().__init__()
            self.nodes = pp.Parameter(nodes)

        def forward(self, edges, meas):
            ti = self.nodes[edges[:, 0]]
            tj = self.nodes[edges[:, 1]]
            pred = ti.Inv() @ tj
            return (meas.Inv() @ pred).Log().tensor().view(-1)

    graph = PoseGraph(pp.identity_SE3(4))
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])

    def x(dx):
        return pp.SE3(torch.tensor([dx, 0, 0, 0, 0, 0, 1.0])).tensor()

    meas = pp.SE3(torch.stack([x(1.0), x(1.0), x(1.0), x(-3.0)]))
    opt = pp.optim.LM(graph)
    loss = None
    for _ in range(15):
        loss = opt.step((edges, meas))
    assert float(loss) < 1e-6
    xs = graph.nodes.tensor()[:, 0].tolist()
    # consecutive spacing ~1.0 (loop satisfied)
    assert abs((xs[1] - xs[0]) - 1.0) < 1e-2
    assert abs((xs[3] - xs[0]) - 3.0) < 1e-2
```

- [ ] **Step 4: Run → pass**

Run: `.venv/bin/pytest tests/test_pypose_pgo.py -v`
Expected: PASS (verified working). If `pp.optim.LM` step signature differs in 0.9.5, adapt to the installed API but keep the loss<1e-6 assertion.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/test_pypose_pgo.py
git commit -m "chore(slam): add pypose optional dependency + PGO primitive test"
```
(End every commit message with: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`)

---

## Task 2: `PoseGraph` — anchored PyPose LM pose-graph

**Files:** Create `src/research/graph_slam.py`; Test: `tests/test_graph_slam_pose_graph.py`

A reusable pose graph: add nodes (4×4 `world←node` SE(3) numpy), odometry/loop edges (relative 4×4 `T_j_from_i`), `optimize()` runs anchored LM, `pose(i)` returns the optimised 4×4.

- [ ] **Step 1: Write the failing test**

Create `tests/test_graph_slam_pose_graph.py`:
```python
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
    # 4 nodes; drifted initial estimates (node 3 lands at x=3.5 instead of 3).
    g.add_node(np.eye(4))
    g.add_node(_xform(1.0))
    g.add_node(_xform(2.0))
    g.add_node(_xform(3.5))  # drift
    g.add_odometry_edge(0, 1, _xform(1.0))
    g.add_odometry_edge(1, 2, _xform(1.0))
    g.add_odometry_edge(2, 3, _xform(1.0))
    g.add_loop_edge(3, 0, _xform(-3.0))  # loop says back to origin is -3
    g.optimize(iters=20)
    # Node 0 stays anchored at the origin; node 3 corrected toward x=3.
    assert np.allclose(g.pose(0)[:3, 3], [0, 0, 0], atol=1e-3)
    assert abs(g.pose(3)[0, 3] - 3.0) < 0.05


def test_pose_count() -> None:
    g = PoseGraph()
    g.add_node(np.eye(4))
    g.add_node(_xform(1.0))
    assert g.n_nodes == 2
```

- [ ] **Step 2: Run → fails (no module)**

Run: `.venv/bin/pytest tests/test_graph_slam_pose_graph.py -v`
Expected: FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Implement `PoseGraph`**

Create `src/research/graph_slam.py` with (this is the core — get it right; reuse the verified PGO pattern, anchor node 0 with a prior residual):
```python
"""PyPose graph-SLAM back-end (spec §14.6.2).

A pose graph over keyframe poses optimised by PyPose Levenberg-Marquardt:
vertices are `world<-node` SE(3) poses, edges are relative-pose constraints
(odometry + loop closures). Node 0 is anchored (prior to identity) to fix the
gauge. All public I/O is plain 4x4 numpy SE(3); torch/pypose stays internal.
"""

from __future__ import annotations

import numpy as np
import torch
import pypose as pp


def _mat_to_se3(mats: np.ndarray) -> pp.SE3:
    """(...,4,4) world<-X numpy -> pp.SE3. Probe pp.mat2SE3 in 0.9.5; if absent,
    build from translation + quaternion via scipy Rotation."""
    t = torch.as_tensor(np.asarray(mats), dtype=torch.float64)
    return pp.mat2SE3(t)  # if this name is wrong in 0.9.5, adapt (see NOTE below)


class _PoseGraphModule(torch.nn.Module):
    def __init__(self, nodes: pp.SE3) -> None:
        super().__init__()
        self.nodes = pp.Parameter(nodes)

    def forward(self, edges: torch.Tensor, meas: pp.SE3) -> torch.Tensor:
        ti = self.nodes[edges[:, 0]]
        tj = self.nodes[edges[:, 1]]
        pred = ti.Inv() @ tj
        edge_res = (meas.Inv() @ pred).Log().tensor().view(-1)
        anchor = (self.nodes[0]).Log().tensor().view(-1) * 100.0  # strong prior
        return torch.cat([edge_res, anchor])


class PoseGraph:
    def __init__(self) -> None:
        self._nodes: list[np.ndarray] = []
        self._edges: list[tuple[int, int]] = []
        self._meas: list[np.ndarray] = []

    @property
    def n_nodes(self) -> int:
        return len(self._nodes)

    def add_node(self, world_from_node: np.ndarray) -> int:
        self._nodes.append(np.asarray(world_from_node, dtype=np.float64).copy())
        return len(self._nodes) - 1

    def add_odometry_edge(self, i: int, j: int, j_from_i: np.ndarray) -> None:
        self._edges.append((i, j))
        self._meas.append(np.asarray(j_from_i, dtype=np.float64).copy())

    add_loop_edge = add_odometry_edge  # same constraint shape; kept distinct in API

    def optimize(self, *, iters: int = 20) -> None:
        if len(self._edges) == 0 or self.n_nodes == 0:
            return
        nodes = _mat_to_se3(np.stack(self._nodes))
        module = _PoseGraphModule(nodes)
        edges = torch.tensor(self._edges, dtype=torch.long)
        meas = _mat_to_se3(np.stack(self._meas))
        opt = pp.optim.LM(module)
        for _ in range(iters):
            opt.step((edges, meas))
        out = module.nodes.detach().matrix().numpy()  # (N,4,4)
        self._nodes = [out[i] for i in range(out.shape[0])]

    def pose(self, i: int) -> np.ndarray:
        return self._nodes[i]
```
**NOTE for the implementer:** probe the exact pypose 0.9.5 converters first:
`/.venv/bin/python -c "import pypose as pp; print([n for n in dir(pp) if 'mat' in n.lower() or 'SE3' in n])"`. If `pp.mat2SE3` isn't present, convert via translation+quaternion: `pp.SE3(torch.cat([t_xyz, quat_xyzw], -1))` using `scipy.spatial.transform.Rotation.from_matrix(R).as_quat()` (already a project dep). `.matrix()` on a `pp.SE3` returns the 4×4; confirm and adapt `.matrix()` vs `.matrix4x4()` naming. Keep the test green — these are the only API-name risks.

- [ ] **Step 4: Run → pass**

Run: `.venv/bin/pytest tests/test_graph_slam_pose_graph.py -v`
Expected: 2 PASS. The loop must pull node 3 from x=3.5 toward x=3.0.

- [ ] **Step 5: ruff + commit**

```bash
.venv/bin/ruff check src/research/graph_slam.py tests/test_graph_slam_pose_graph.py && .venv/bin/ruff format src/research/graph_slam.py tests/test_graph_slam_pose_graph.py
git add src/research/graph_slam.py tests/test_graph_slam_pose_graph.py
git commit -m "feat(slam): PyPose anchored pose-graph optimiser (LM)"
```

---

## Task 3: `OrbBowLoopDetector` — appearance-based revisit detection

**Files:** `src/research/graph_slam.py`; Test: `tests/test_graph_slam_loop_detector.py`

Detect when the current keyframe revisits an earlier one by ORB-descriptor similarity. Keep it simple and deterministic: store each keyframe's ORB descriptors; a candidate loop = a past keyframe (older than a `min_gap`) whose ratio-test match count exceeds a threshold.

- [ ] **Step 1: Write the failing test**

Create `tests/test_graph_slam_loop_detector.py`:
```python
"""§14.6.2 — ORB appearance loop detector."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")
from src.research.graph_slam import OrbBowLoopDetector  # noqa: E402


def _textured(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(120, 160), dtype=np.uint8)


def test_revisited_frame_is_detected() -> None:
    det = OrbBowLoopDetector(min_gap=2, min_matches=15)
    a = _textured(1)
    for kf, img in enumerate([a, _textured(2), _textured(3), a]):
        loop = det.add_keyframe(kf, img)
        if kf < 3:
            assert loop is None
    # The 4th keyframe is image `a` again → loops back to keyframe 0.
    assert det.add_keyframe(4, a) == 0 or loop == 0  # detector returns the matched kf id


def test_distinct_frames_no_loop() -> None:
    det = OrbBowLoopDetector(min_gap=1, min_matches=20)
    for kf in range(5):
        assert det.add_keyframe(kf, _textured(100 + kf)) is None
```
> NOTE: tune the test's `min_matches`/seed so a re-shown identical image reliably matches and distinct random images don't. Random-noise images are ORB-poor; if ORB finds too few features on pure noise, switch `_textured` to a structured pattern (e.g. tiled rectangles via cv2.rectangle) so matches are stable. Adjust thresholds to make the behavioural contract (same→loop, different→none) hold honestly; do not assert on a flaky margin.

- [ ] **Step 2: Run → fails**

Run: `.venv/bin/pytest tests/test_graph_slam_loop_detector.py -v`
Expected: FAIL `cannot import name 'OrbBowLoopDetector'`.

- [ ] **Step 3: Implement the detector (append to `graph_slam.py`)**

```python
import cv2


class OrbBowLoopDetector:
    """Appearance-based loop detection over keyframe ORB descriptors.

    A lightweight stand-in for a full DBoW vocabulary: ratio-test BF matching of
    the current keyframe against stored keyframe descriptors; the best past
    keyframe (older than ``min_gap``) with >= ``min_matches`` inliers is a loop.
    """

    def __init__(self, *, n_features: int = 1000, min_gap: int = 10,
                 min_matches: int = 30, ratio: float = 0.75) -> None:
        self._orb = cv2.ORB_create(nfeatures=n_features)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._min_gap = min_gap
        self._min_matches = min_matches
        self._ratio = ratio
        self._kfs: list[tuple[int, np.ndarray]] = []  # (kf_id, descriptors)

    def add_keyframe(self, kf_id: int, gray: np.ndarray) -> int | None:
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
        if len(d1) < 2 or len(d2) < 2:
            return 0
        good = 0
        for m_n in self._bf.knnMatch(d1, d2, k=2):
            if len(m_n) == 2 and m_n[0].distance < self._ratio * m_n[1].distance:
                good += 1
        return good
```

- [ ] **Step 4: Run → pass** (tune thresholds/fixtures until honest)

Run: `.venv/bin/pytest tests/test_graph_slam_loop_detector.py -v`
Expected: 2 PASS.

- [ ] **Step 5: ruff + commit**

```bash
git add src/research/graph_slam.py tests/test_graph_slam_loop_detector.py
git commit -m "feat(slam): ORB appearance loop detector"
```

---

## Task 4: `GraphSlamPoseSource` — streaming PoseSource with loop closure

**Files:** `src/research/graph_slam.py`; Test: `tests/test_graph_slam_pose_source.py`

Ties it together: a `PoseSource` that runs ORB-VO between keyframes, adds graph nodes/odometry edges, detects loops (adds loop edges + re-optimises), and returns the optimised latest pose in graphics-world — mirroring `SLAMPoseSource`'s conventions (`_t_wc`, `_CV_TO_GRAPHICS`, `CameraPoseWorld`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_graph_slam_pose_source.py`:
```python
"""§14.6.2 — GraphSlamPoseSource: PoseSource conformance + loop-closing pose."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pypose")
from src.research.graph_slam import GraphSlamPoseSource  # noqa: E402
from src.spatial.pose_source import CameraIntrinsics, PoseSource


def test_conforms_to_pose_source_protocol() -> None:
    src = GraphSlamPoseSource(CameraIntrinsics(fx=500, fy=500, cx=320, cy=240))
    assert isinstance(src, PoseSource)  # has get(...)
    assert hasattr(src, "track") and hasattr(src, "reset")


def test_first_frame_anchors_world_origin() -> None:
    src = GraphSlamPoseSource(CameraIntrinsics(fx=500, fy=500, cx=320, cy=240))
    img = np.random.default_rng(0).integers(0, 255, (240, 320), dtype=np.uint8)
    pose = src.track(0, img)
    assert pose.available
    assert np.allclose(pose.position, (0.0, 0.0, 0.0), atol=1e-6)


def test_tracking_lost_marks_unavailable_but_holds_pose() -> None:
    # A blank second frame yields no VO → tracking lost → available False, pose held.
    src = GraphSlamPoseSource(CameraIntrinsics(fx=500, fy=500, cx=320, cy=240))
    rng = np.random.default_rng(1)
    src.track(0, rng.integers(0, 255, (240, 320), dtype=np.uint8))
    pose = src.track(1, np.zeros((240, 320), dtype=np.uint8))
    assert pose.available is False
```
> The drift-reduction acceptance over a real synthetic loop is Task 6 (it needs a rendered trajectory). Here, pin the PoseSource contract: protocol conformance, world-anchored first frame, and the tracking-lost behaviour (mirroring `SLAMPoseSource`).

- [ ] **Step 2: Run → fails**

Run: `.venv/bin/pytest tests/test_graph_slam_pose_source.py -v`
Expected: FAIL `cannot import name 'GraphSlamPoseSource'`.

- [ ] **Step 3: Implement `GraphSlamPoseSource` (append to `graph_slam.py`)**

Read `SLAMPoseSource` in `slam_adapter.py` and mirror its conventions exactly (`_k` intrinsics matrix, `_t_wc` world←camera, `_CV_TO_GRAPHICS`, `_current_pose` → `CameraPoseWorld`, the first-frame anchor, the tracking-lost `available=False` hold). Differences: keep a `PoseGraph` + `OrbBowLoopDetector`; insert a keyframe node every frame (or every `keyframe_stride` frames); add an odometry edge from the VO motion; on a detected loop, estimate the relative pose between the current and matched keyframe via the VO front-end and `add_loop_edge`, then `PoseGraph.optimize()` and refresh `_t_wc` from the optimised latest node. Import `RelativePose`, `OrbVisualOdometry`, `_se3`, `_se3_inv`, `_CV_TO_GRAPHICS`, and `CameraPoseWorld`/`CameraIntrinsics` from their modules. Expose `track`, `get`, `reset`. `GraphSlamConfig(keyframe_stride=1, min_inliers=12, optimize_every=1, loop_min_matches=30, loop_min_gap=10)`.

Provide `CameraPoseWorld` construction identical to `SLAMPoseSource._current_pose` (apply `_CV_TO_GRAPHICS`, build quaternion via `scipy....Rotation.from_matrix`).

- [ ] **Step 4: Run → pass**

Run: `.venv/bin/pytest tests/test_graph_slam_pose_source.py -v`
Expected: 3 PASS.

- [ ] **Step 5: ruff + commit**

```bash
git add src/research/graph_slam.py tests/test_graph_slam_pose_source.py
git commit -m "feat(slam): GraphSlamPoseSource — ORB-VO + loop-closing pose graph"
```

---

## Task 5: Wire `graph_slam` into config + perception loop

**Files:** `src/config.py`, `src/runtime/perception_loop.py`; Test: `tests/test_pose_source_selection.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pose_source_selection.py`:
```python
"""§14.6.2 — pose_source = graph_slam selects GraphSlamPoseSource."""

from __future__ import annotations

import pytest

pytest.importorskip("pypose")


def test_config_accepts_graph_slam() -> None:
    from src.config import Settings

    s = Settings(pose_source="graph_slam")
    assert s.pose_source == "graph_slam"


def test_perception_loop_builds_graph_slam_source() -> None:
    from src.research.graph_slam import GraphSlamPoseSource
    from src.runtime.perception_loop import PerceptionLoop  # adapt import to real class
    from src.spatial.pose_source import CameraIntrinsics

    # Build a loop with pose_source=graph_slam and assert _make_pose_source returns
    # a GraphSlamPoseSource. Construct PerceptionLoop the way the codebase does
    # (read perception_loop.py for the real constructor; if _make_pose_source is
    # accessible without a full loop, call it directly with a CameraIntrinsics).
    intr = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240)
    # ... build minimal loop or call the factory; assert isinstance(src, GraphSlamPoseSource)
```
> Read `perception_loop.py` for the real `PerceptionLoop` constructor + how `_make_pose_source` is reachable. If constructing a full loop is heavy, refactor `_make_pose_source` into a module-level `make_pose_source(cfg_value, intrinsics)` helper (small, behavior-preserving) and test that directly — update `_make_pose_source` to call it. Make the test concrete and runnable before finishing.

- [ ] **Step 2: Run → fails**

Run: `.venv/bin/pytest tests/test_pose_source_selection.py -v`
Expected: FAIL (config rejects `"graph_slam"`).

- [ ] **Step 3: Implement**

- `src/config.py`: extend the Literal to `pose_source: Literal["fixed", "sim", "slam", "graph_slam"] = "fixed"`; update the inline comment.
- `src/runtime/perception_loop.py::_make_pose_source`: add
  ```python
  if self.cfg.settings.pose_source == "graph_slam":
      from ..research.graph_slam import GraphSlamPoseSource
      return GraphSlamPoseSource(intrinsics)
  ```
  (lazy import so the `slam` extra stays optional — a missing pypose must not break `fixed`/`slam`). Keep the existing `"slam"` and fallback branches.

- [ ] **Step 4: Run → pass + full suite**

Run: `.venv/bin/pytest tests/test_pose_source_selection.py -v && .venv/bin/pytest -q`
Expected: new tests pass; full suite green (baseline 468 + the new SLAM tests).

- [ ] **Step 5: ruff + commit**

```bash
git add src/config.py src/runtime/perception_loop.py tests/test_pose_source_selection.py
git commit -m "feat(slam): select GraphSlamPoseSource via PET_AGENT_POSE_SOURCE=graph_slam"
```

---

## Task 6: Acceptance (synthetic loop drift) + docs

**Files:** Test `tests/test_graph_slam_acceptance.py`; `docs/spec.md` §14.6.2

- [ ] **Step 1: Write a synthetic-loop drift-reduction acceptance test**

Create `tests/test_graph_slam_acceptance.py` that builds a **pose graph directly** (no images) representing a closed loop with accumulated odometry drift, asserts the optimiser reduces end-to-start drift after adding the loop edge:
```python
"""§14.6.2 acceptance — loop closure reduces accumulated pose-graph drift."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pypose")
from src.research.graph_slam import PoseGraph


def _rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    t = np.eye(4)
    t[:2, :2] = [[c, -s], [s, c]]
    return t


def test_loop_closure_reduces_drift() -> None:
    # A square loop (4 legs, 90deg turns). Odometry has a small per-edge yaw bias,
    # so the open chain doesn't return to the origin. Adding the loop edge + LM
    # must pull the final node back toward the start.
    leg = np.eye(4); leg[0, 3] = 1.0
    bias = _rot_z(np.deg2rad(5))  # drift per turn
    g = PoseGraph()
    g.add_node(np.eye(4))
    pose = np.eye(4)
    rel = []
    for k in range(4):
        step = leg @ bias
        pose = pose @ step
        g.add_node(pose.copy())
        rel.append(step)
    for i, step in enumerate(rel):
        g.add_odometry_edge(i, i + 1, step)
    drift_before = np.linalg.norm(g.pose(4)[:3, 3] - g.pose(0)[:3, 3])
    g.add_loop_edge(4, 0, np.eye(4))  # node 4 == node 0 (closed loop)
    g.optimize(iters=40)
    drift_after = np.linalg.norm(g.pose(4)[:3, 3] - g.pose(0)[:3, 3])
    assert drift_after < drift_before * 0.5  # at least halved
    assert drift_after < 0.2
```

- [ ] **Step 2: Run → pass**

Run: `.venv/bin/pytest tests/test_graph_slam_acceptance.py -v`
Expected: PASS (loop closure halves the drift). If the anchor prior over-constrains and node 4 can't move, ensure only node 0 is anchored (not node 4) — that's the intended gauge.

- [ ] **Step 3: Spec §14.6.2 status**

Append a **Status — implemented** bullet to §14.6.2: modules (`graph_slam.py`: `PoseGraph` PyPose-LM, `OrbBowLoopDetector`, `GraphSlamPoseSource`), selector (`PET_AGENT_POSE_SOURCE=graph_slam`, `.[slam]` extra), the measured synthetic-loop drift reduction, and that the default fallback stays raw ORB-VO `SLAMPoseSource`. Be honest about scope: the loop-closure + global optimisation is implemented and unit-verified on synthetic graphs; a full live-camera 2-minute desk loop drift ≤ 5 cm is the hardware-validation follow-up.

- [ ] **Step 4: Final verification**

Run: `.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/pytest -q`
Expected: ruff clean; full suite green.

- [ ] **Step 5: Commit**

```bash
git add docs/spec.md
git commit -m "docs(slam): §14.6.2 PyPose graph-SLAM implemented + drift acceptance"
```

---

## Self-Review Notes

- **Spec coverage (§14.6.2):** PyPose LM pose-graph (Task 2) ✓; ORB loop closure (Task 3) ✓; `GraphSlamPoseSource` behind the `PoseSource` protocol (Task 4) ✓; `PET_AGENT_POSE_SOURCE=graph_slam` selection with lazy import + ORB-VO fallback (Task 5) ✓; loop-closure drift-reduction acceptance (Task 6) ✓; torch-native, numpy-2 safe, in-process (Task 1) ✓.
- **Reuse, don't fork:** the front-end (`OrbVisualOdometry`), pose conventions (`_t_wc`, `_CV_TO_GRAPHICS`), and `CameraPoseWorld` come straight from `slam_adapter.py`; the new source mirrors `SLAMPoseSource`'s contracts (first-frame anchor, tracking-lost hold).
- **No-regression seam:** default `pose_source` stays `fixed`; `graph_slam` is opt-in with a lazy `pypose` import so the core never depends on the `slam` extra.
- **Execution-time API risks flagged:** the exact pypose 0.9.5 converter names (`pp.mat2SE3` / `.matrix()`) are called out in Task 2 with a probe step and a scipy-quaternion fallback — verify before relying on them (this is the one place to avoid a g2o-style "assumed API" slip). The loop-detector thresholds/fixtures (Task 3) and the `_make_pose_source` reachability (Task 5) are flagged read-first.
- **Hardware:** the pose graph is a few hundred SE(3) params → negligible VRAM on the CUDA torch; ORB front-end + loop detection are CPU. Fits the §14.6.5 budget trivially.
