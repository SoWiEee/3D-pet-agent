"""Exploration MDP environment (spec §14.3).

A self-contained, deterministic simulator for the active-exploration task,
built on the project's own :class:`CoverageGrid` + :class:`OccupancyGrid` so the
RL policy trains against the same world model the live system uses. No external
RL framework — the env exposes a minimal ``reset`` / ``step`` surface.

State (5-dim, spec §14.3): known object count, unknown-area ratio, target-visible
flag, distance to nearest frontier, semantic uncertainty.

Actions (5, spec §14.3): inspect_frontier, move_to_known_object, look_around,
ask_user, return_to_user.

Reward (spec §14.3): +1.0 discovered relevant object, +0.5 reduced unknown area,
+0.3 verified stale object, −0.2 unnecessary movement, −1.0 collision, −0.5
repeated failed inspection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ...exploration.coverage_grid import CoverageGrid, CoverageGridConfig
from ...planning import GridConfig, OccupancyGrid
from ...planning.astar import iter_line_cells

# Action ids.
INSPECT_FRONTIER = 0
MOVE_TO_KNOWN = 1
LOOK_AROUND = 2
ASK_USER = 3
RETURN_TO_USER = 4
N_ACTIONS = 5
ACTION_NAMES = (
    "inspect_frontier",
    "move_to_known_object",
    "look_around",
    "ask_user",
    "return_to_user",
)

STATE_DIM = 5

# Reward magnitudes (spec §14.3).
R_DISCOVER_RELEVANT = 1.0
R_REDUCED_AREA = 0.5
R_VERIFY_STALE = 0.3
R_UNNECESSARY_MOVE = -0.2
R_COLLISION = -1.0
R_REPEAT_FAILED = -0.5


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
    Pure: depends only on per-step quantities + config so both action
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


@dataclass(frozen=True)
class SceneObject:
    x: float
    z: float
    relevant: bool
    stale: bool


@dataclass
class EnvConfig:
    resolution: float = 0.1
    origin_x: float = -2.0
    origin_z: float = -2.0
    width: int = 40
    height: int = 40
    n_objects: tuple[int, int] = (4, 8)  # inclusive range
    relevant_prob: float = 0.4
    stale_prob: float = 0.3
    n_obstacles: tuple[int, int] = (0, 2)
    obstacle_radius: float = 0.3
    sensor_range: float = 1.2
    cone_fov: float = 1.6  # radians for directed observations
    discover_range: float = 1.2
    max_steps: int = 30
    explored_done_ratio: float = 0.12  # episode ends when unknown ratio drops below
    # Cells revealed that map to a full +0.5 area reward. Kept small so the
    # cumulative area signal over an episode dominates the fixed penalties —
    # otherwise the agent games the reward by stopping early (see §14.3 notes).
    area_norm: float = 60.0
    failed_area_cells: int = 1  # only a *zero*-reveal action counts as wasted


@dataclass
class StepInfo:
    action: int
    collided: bool
    new_area_cells: int
    discovered_relevant: int
    verified_stale: int
    reward_terms: dict[str, float] = field(default_factory=dict)


class ExplorationEnv:
    """Procedurally-generated active-exploration episodes."""

    def __init__(self, cfg: EnvConfig | None = None) -> None:
        self.cfg = cfg or EnvConfig()
        cov_cfg = CoverageGridConfig(
            resolution=self.cfg.resolution,
            origin_x=self.cfg.origin_x,
            origin_z=self.cfg.origin_z,
            width=self.cfg.width,
            height=self.cfg.height,
        )
        self.coverage = CoverageGrid(cov_cfg)
        self._grid_cfg = GridConfig(
            resolution=self.cfg.resolution,
            origin_x=self.cfg.origin_x,
            origin_z=self.cfg.origin_z,
            width=self.cfg.width,
            height=self.cfg.height,
        )
        self._rng = np.random.default_rng()
        self._world_diag = math.hypot(
            self.cfg.width * self.cfg.resolution, self.cfg.height * self.cfg.resolution
        )
        self.reset()

    # ── episode lifecycle ────────────────────────────────────────────────────
    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.coverage.reset()
        self._objects = self._spawn_objects()
        self._obstacles = self._spawn_obstacles()
        self._occ = self._build_occupancy()
        self._discovered: set[int] = set()
        self._cat = (0.0, 0.0)
        self._heading = 0.0
        self._steps = 0
        self._target_hint = False
        self._prev_failed_inspect = False
        self._done = False
        # Reveal the immediate surroundings so the first state isn't all-unknown.
        self._observe(self._cat, self._heading, fov=2.0 * math.pi)
        self._discover_at(self._cat, fov=2.0 * math.pi)
        return self._state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, StepInfo]:
        if self._done:
            raise RuntimeError("step() called on a finished episode; call reset()")
        self._steps += 1

        target, is_move, full_sweep = self._resolve_action(action)
        collided = False
        new_area = 0
        discovered_relevant = 0
        verified_stale = 0

        if action == ASK_USER:
            # Free information action: reveal that a relevant object exists.
            self._target_hint = self._has_undiscovered_relevant()
        else:
            if is_move and target is not None:
                new_pos, collided = self._move_toward(self._cat, target)
                if new_pos != self._cat:
                    self._heading = math.atan2(new_pos[1] - self._cat[1], new_pos[0] - self._cat[0])
                    self._cat = new_pos
            fov = 2.0 * math.pi if full_sweep else self.cfg.cone_fov
            if not collided:
                new_area = self._observe(self._cat, self._heading, fov=fov)
                discovered_relevant = self._discover_at(self._cat, fov=fov)
                if action == MOVE_TO_KNOWN:
                    verified_stale = self._verify_stale_near(self._cat)

        # ── reward shaping (spec §14.3) ──
        failed_inspection = action in (INSPECT_FRONTIER, LOOK_AROUND) and (
            new_area < self.cfg.failed_area_cells
        )
        terms = reward_terms(
            self.cfg,
            new_area=new_area,
            discovered_relevant=discovered_relevant,
            verified_stale=verified_stale,
            collided=collided,
            is_move=is_move,
            failed_inspection=failed_inspection,
            prev_failed=self._prev_failed_inspect,
        )
        self._prev_failed_inspect = failed_inspection

        reward = float(sum(terms.values()))

        self._done = (
            action == RETURN_TO_USER
            or self._steps >= self.cfg.max_steps
            or self.coverage.unobserved_ratio() <= self.cfg.explored_done_ratio
        )

        info = StepInfo(
            action=action,
            collided=collided,
            new_area_cells=new_area,
            discovered_relevant=discovered_relevant,
            verified_stale=verified_stale,
            reward_terms=terms,
        )
        return self._state(), reward, self._done, info

    # ── action resolution ────────────────────────────────────────────────────
    def _resolve_action(self, action: int) -> tuple[tuple[float, float] | None, bool, bool]:
        """Return (move_target | None, is_move, full_360_sweep)."""
        if action == INSPECT_FRONTIER:
            return self._nearest_frontier_world(), True, False
        if action == MOVE_TO_KNOWN:
            return self._nearest_known_object_world(), True, False
        if action == LOOK_AROUND:
            return None, False, True
        if action == ASK_USER:
            return None, False, False
        if action == RETURN_TO_USER:
            return (0.0, 0.0), True, False
        raise ValueError(f"invalid action {action}")

    # ── world helpers ────────────────────────────────────────────────────────
    def _spawn_objects(self) -> list[SceneObject]:
        n = int(self._rng.integers(self.cfg.n_objects[0], self.cfg.n_objects[1] + 1))
        half_w = self.cfg.width * self.cfg.resolution / 2.0
        half_h = self.cfg.height * self.cfg.resolution / 2.0
        objs: list[SceneObject] = []
        for _ in range(n):
            x = float(self._rng.uniform(-half_w * 0.9, half_w * 0.9))
            z = float(self._rng.uniform(-half_h * 0.9, half_h * 0.9))
            objs.append(
                SceneObject(
                    x=x,
                    z=z,
                    relevant=bool(self._rng.random() < self.cfg.relevant_prob),
                    stale=bool(self._rng.random() < self.cfg.stale_prob),
                )
            )
        return objs

    def _spawn_obstacles(self) -> list[tuple[float, float]]:
        n = int(self._rng.integers(self.cfg.n_obstacles[0], self.cfg.n_obstacles[1] + 1))
        half_w = self.cfg.width * self.cfg.resolution / 2.0
        half_h = self.cfg.height * self.cfg.resolution / 2.0
        out: list[tuple[float, float]] = []
        for _ in range(n):
            # Keep obstacles away from the origin so the cat doesn't start stuck.
            x = float(self._rng.uniform(-half_w * 0.8, half_w * 0.8))
            z = float(self._rng.uniform(-half_h * 0.8, half_h * 0.8))
            if math.hypot(x, z) < self.cfg.obstacle_radius + 0.3:
                continue
            out.append((x, z))
        return out

    def _build_occupancy(self) -> OccupancyGrid:
        data = np.zeros((self.cfg.height, self.cfg.width), dtype=np.uint8)
        r_cells = max(1, int(self.cfg.obstacle_radius / self.cfg.resolution))
        for ox, oz in self._obstacles:
            cgx, cgz = self.coverage.world_to_cell(ox, oz)
            for gz in range(cgz - r_cells, cgz + r_cells + 1):
                for gx in range(cgx - r_cells, cgx + r_cells + 1):
                    if (
                        0 <= gx < self.cfg.width
                        and 0 <= gz < self.cfg.height
                        and math.hypot(gx - cgx, gz - cgz) <= r_cells
                    ):
                        data[gz, gx] = 1
        return OccupancyGrid(cfg=self._grid_cfg, data=data)

    def _move_toward(
        self, a: tuple[float, float], b: tuple[float, float]
    ) -> tuple[tuple[float, float], bool]:
        """Slide from ``a`` toward ``b`` along the Bresenham line, stopping at the
        last free cell before any obstacle. Returns (new_world_pos, collided).
        Sliding (vs. freezing on a blocked straight line) stops the cat getting
        trapped re-targeting an unreachable frontier."""
        ca = self._occ.world_to_cell(*a)
        cb = self._occ.world_to_cell(*b)
        last_free = ca
        collided = False
        for cell in iter_line_cells(ca, cb):
            if self._occ.is_blocked(*cell):
                collided = True
                break
            last_free = cell
        return self._occ.cell_to_world(*last_free), collided

    def _observe(self, pose: tuple[float, float], heading: float, *, fov: float) -> int:
        return self.coverage.observe_cone(pose, heading, fov, self.cfg.sensor_range)

    def _discover_at(self, pose: tuple[float, float], *, fov: float) -> int:
        """Discover objects within sensor range + fov of ``pose``. Returns the
        number of newly discovered *relevant* objects."""
        px, pz = pose
        new_relevant = 0
        for i, obj in enumerate(self._objects):
            if i in self._discovered:
                continue
            dx, dz = obj.x - px, obj.z - pz
            if math.hypot(dx, dz) > self.cfg.discover_range:
                continue
            if fov < 2.0 * math.pi:
                bearing = math.atan2(dz, dx)
                delta = math.atan2(
                    math.sin(bearing - self._heading), math.cos(bearing - self._heading)
                )
                if abs(delta) > fov * 0.5:
                    continue
            self._discovered.add(i)
            if obj.relevant:
                new_relevant += 1
        return new_relevant

    def _verify_stale_near(self, pose: tuple[float, float]) -> int:
        px, pz = pose
        verified = 0
        for i in list(self._discovered):
            obj = self._objects[i]
            if obj.stale and math.hypot(obj.x - px, obj.z - pz) <= self.cfg.discover_range:
                self._objects[i] = SceneObject(obj.x, obj.z, obj.relevant, stale=False)
                verified += 1
        return verified

    def _has_undiscovered_relevant(self) -> bool:
        return any(o.relevant for i, o in enumerate(self._objects) if i not in self._discovered)

    def _nearest_frontier_world(self) -> tuple[float, float] | None:
        # nearest_unknown takes a world-coord tuple and returns world coords.
        return self.coverage.nearest_unknown(self._cat)

    def _nearest_known_object_world(self) -> tuple[float, float] | None:
        best: tuple[float, float] | None = None
        best_d = float("inf")
        for i in self._discovered:
            obj = self._objects[i]
            d = math.hypot(obj.x - self._cat[0], obj.z - self._cat[1])
            if d < best_d:
                best_d, best = d, (obj.x, obj.z)
        return best

    # ── state ────────────────────────────────────────────────────────────────
    def _state(self) -> np.ndarray:
        n_objs = max(1, self.cfg.n_objects[1])
        known = len(self._discovered) / n_objs
        unknown_ratio = self.coverage.unobserved_ratio()
        target_visible = 1.0 if (self._target_hint or self._any_relevant_discovered()) else 0.0
        frontier = self._nearest_frontier_world()
        if frontier is None:
            frontier_dist = 0.0
        else:
            d = math.hypot(frontier[0] - self._cat[0], frontier[1] - self._cat[1])
            frontier_dist = min(1.0, d / self._world_diag)
        stale_unverified = sum(1 for i in self._discovered if self._objects[i].stale)
        uncertainty = stale_unverified / n_objs
        return np.array(
            [known, unknown_ratio, target_visible, frontier_dist, uncertainty],
            dtype=np.float32,
        )

    def _any_relevant_discovered(self) -> bool:
        return any(self._objects[i].relevant for i in self._discovered)

    # ── introspection (for eval / tests) ──────────────────────────────────────
    @property
    def coverage_fraction(self) -> float:
        return 1.0 - self.coverage.unobserved_ratio()

    @property
    def relevant_discovered(self) -> int:
        return sum(1 for i in self._discovered if self._objects[i].relevant)

    @property
    def relevant_total(self) -> int:
        return sum(1 for o in self._objects if o.relevant)
