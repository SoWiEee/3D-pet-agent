"""Evaluation runner (spec §13).

Drives each :class:`DatasetEntry` through the full backend stack in-process:

    1. Reset SemanticMap + tracker + planner state.
    2. Pour the scene fixture objects into SemanticMap.
    3. Run command parser → grounding resolver → A* planner → pure-pursuit
       follower (offline simulator).
    4. Score outcome vs the entry's expectations and emit one
       :class:`EvaluationRecord`.

No real perception models are touched — the dataset is pre-grounded, which
is what spec §13.3 asks for ("50 natural-language commands" over annotated
scenes). This keeps the harness deterministic, fast, and CI-friendly.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from ..control import PathFollower, PIDController, PurePursuitController, UnicycleState
from ..exploration import CoverageGrid, ExplorationPlanner
from ..language import parse_command
from ..planning import GroundingResolver, Planner
from ..planning.occupancy_grid import build_occupancy_grid
from ..spatial import SceneGraphBuilder, SemanticMap
from ..spatial.object_lifter import ObjectConfidence, ObjectState3D
from .schema import (
    ControllerMetrics,
    DatasetEntry,
    DatasetSceneObject,
    EvaluationRecord,
)

log = logging.getLogger("pet_agent.evaluation")


def _scene_object_to_state(obj: DatasetSceneObject, frame_id: int) -> ObjectState3D:
    """Turn a dataset fixture into a tracked :class:`ObjectState3D`."""
    return ObjectState3D(
        object_id=obj.object_id,
        class_label=obj.class_label,
        attributes=obj.attributes,
        bbox_xyxy=(0.0, 0.0, 10.0, 10.0),
        center_2d=(5.0, 5.0),
        center_3d_world=obj.center_3d_world,
        extent_3d=obj.extent_3d,
        median_depth=1.0,
        depth_uncertainty=0.05,
        confidence=ObjectConfidence(overall=obj.confidence),
        tracking_status=obj.tracking_status,
        last_seen_frame=frame_id,
        source_backend="mainline_grounding_sam",
    )


class EvaluationRunner:
    """Stateless across trials — each call to :meth:`run_entry` is hermetic."""

    def __init__(
        self,
        *,
        planner: Planner | None = None,
        follower: PathFollower | None = None,
        scene_graph_builder: SceneGraphBuilder | None = None,
        resolver: GroundingResolver | None = None,
        exploration_planner: ExplorationPlanner | None = None,
        coverage_grid: CoverageGrid | None = None,
    ) -> None:
        self.planner = planner or Planner()
        self.follower = follower or PathFollower(
            controller=PurePursuitController(),
            pid=PIDController(kp=1.2, ki=0.05, kd=0.02),
        )
        self.scene_graph_builder = scene_graph_builder or SceneGraphBuilder()
        self.resolver = resolver or GroundingResolver()
        self.exploration_planner = exploration_planner or ExplorationPlanner()
        self.coverage_grid = coverage_grid or CoverageGrid()

    # ── one trial ─────────────────────────────────────────────────────────
    def run_entry(self, entry: DatasetEntry) -> EvaluationRecord:
        t0 = time.perf_counter()
        smap = SemanticMap(map_id=entry.scene.scene_id)
        observations = [_scene_object_to_state(o, frame_id=1) for o in entry.scene.objects]
        smap.update(observations, frame_id=1)
        for fixture in entry.scene.objects:
            if fixture.tracking_status != "tracked":
                smap._objects[fixture.object_id] = smap._objects[fixture.object_id].model_copy(
                    update={"tracking_status": fixture.tracking_status}
                )

        record = EvaluationRecord(
            trial_id=entry.trial_id,
            scene_id=entry.scene.scene_id,
            command=entry.command,
            expected_outcome=entry.expected_outcome,
            expected_target=entry.expected_target,
            notes=entry.notes,
        )

        intent = parse_command(entry.command)
        if intent is None:
            record.predicted_outcome = "unparseable"
            record.task_success = False
            record.latency_ms = (time.perf_counter() - t0) * 1000.0
            return record

        if intent.intent_type == "explore":
            self._run_exploration_trial(intent, smap, record)
            record.latency_ms = (time.perf_counter() - t0) * 1000.0
            return record

        graph = self.scene_graph_builder.build(smap)
        resolution = self.resolver.resolve(intent, smap, graph)
        record.predicted_outcome = resolution.status

        if resolution.status in ("clarification", "no_match", "empty_map"):
            record.task_success = entry.expected_outcome == "clarification" and (
                resolution.status == "clarification"
            )
            if entry.expected_outcome == "no_match" and resolution.status in (
                "no_match",
                "empty_map",
            ):
                record.task_success = True
            record.latency_ms = (time.perf_counter() - t0) * 1000.0
            return record

        goal = resolution.goal
        assert goal is not None
        record.predicted_target = goal.target_object_id
        record.grounding_success = (
            entry.expected_target is None or record.predicted_target == entry.expected_target
        )

        if (
            intent.intent_type in ("stop", "look_at", "report")
            or goal.target_position_world is None
        ):
            record.predicted_outcome = intent.intent_type
            record.task_success = entry.expected_outcome in (
                "stop",
                "look_at",
                "report",
            ) and (record.grounding_success or entry.expected_target is None)
            record.latency_ms = (time.perf_counter() - t0) * 1000.0
            return record

        plan = self.planner.plan(goal, smap, start_world=(0.0, 0.0, 0.0))
        record.path_success = plan.status == "success" and bool(plan.path_world)
        if not record.path_success:
            record.predicted_outcome = f"plan_{plan.status}"
            record.latency_ms = (time.perf_counter() - t0) * 1000.0
            return record

        trace = self.follower.simulate(plan.path_world, UnicycleState(x=0.0, y=0.0, theta=0.0))
        record.controller_metrics = ControllerMetrics(
            max_cross_track_error_m=trace.summary.max_cross_track_error,
            max_heading_error_rad=trace.summary.max_heading_error,
            mean_speed_mps=trace.summary.mean_speed,
            steps=trace.summary.steps,
        )
        record.collision_count = self._count_collisions(plan.path_world, smap, goal)
        record.predicted_outcome = "hide" if intent.intent_type == "hide" else "navigate"
        record.task_success = (
            record.grounding_success
            and record.path_success
            and trace.summary.status == "success"
            and record.collision_count == 0
        )
        record.latency_ms = (time.perf_counter() - t0) * 1000.0
        return record

    # ── helpers ───────────────────────────────────────────────────────────
    def _run_exploration_trial(
        self, intent: Any, smap: SemanticMap, record: EvaluationRecord
    ) -> None:
        coverage = CoverageGrid(self.coverage_grid.cfg)
        coverage.observe_cone((0.0, 0.0), 0.0, 0.5, 1.0)
        target_class = intent.target.class_label if intent.target else None
        ex_goal = self.exploration_planner.next_goal(
            smap, coverage, (0.0, 0.0), target_class=target_class
        )
        if ex_goal is None:
            record.predicted_outcome = "fully_explored"
            record.task_success = False
            return
        record.predicted_outcome = "explore"
        record.predicted_target = ex_goal.related_object_id
        nav_goal = ex_goal.to_navigation_goal()
        plan = self.planner.plan(nav_goal, smap, start_world=(0.0, 0.0, 0.0))
        record.path_success = plan.status == "success" and bool(plan.path_world)
        if record.path_success:
            trace = self.follower.simulate(plan.path_world, UnicycleState(x=0.0, y=0.0, theta=0.0))
            record.controller_metrics = ControllerMetrics(
                max_cross_track_error_m=trace.summary.max_cross_track_error,
                max_heading_error_rad=trace.summary.max_heading_error,
                mean_speed_mps=trace.summary.mean_speed,
                steps=trace.summary.steps,
            )
        record.grounding_success = True
        record.task_success = record.path_success

    def _count_collisions(
        self, path: list[tuple[float, float, float]], smap: SemanticMap, goal: Any
    ) -> int:
        """Count waypoints that sit inside a non-target obstacle cell."""
        excluded = {goal.target_object_id} if goal.target_object_id else set()
        grid = build_occupancy_grid(smap, cfg=self.planner.cfg.grid, exclude_object_ids=excluded)
        hits = 0
        for x, _, z in path:
            cell = grid.world_to_cell(x, z)
            if grid.is_blocked(*cell):
                hits += 1
        return hits

    # ── batch ─────────────────────────────────────────────────────────────
    def run_dataset(self, entries: list[DatasetEntry]) -> list[EvaluationRecord]:
        out: list[EvaluationRecord] = []
        for i, entry in enumerate(entries, 1):
            try:
                record = self.run_entry(entry)
            except Exception as e:  # never let one bad trial kill the run
                log.exception("trial %s failed", entry.trial_id)
                record = EvaluationRecord(
                    trial_id=entry.trial_id,
                    scene_id=entry.scene.scene_id,
                    command=entry.command,
                    expected_outcome=entry.expected_outcome,
                    expected_target=entry.expected_target,
                    predicted_outcome=f"error:{type(e).__name__}",
                    task_success=False,
                    notes=str(e)[:200],
                )
            out.append(record)
            log.info(
                "[%d/%d] %s: %s → task_success=%s latency=%.0fms",
                i,
                len(entries),
                entry.trial_id,
                entry.command[:60],
                record.task_success,
                record.latency_ms,
            )
        return out


def load_dataset(path: Path | str) -> list[DatasetEntry]:
    """Parse a JSONL dataset file → list of :class:`DatasetEntry`."""
    path = Path(path)
    entries: list[DatasetEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(DatasetEntry(**json.loads(line)))
    return entries
