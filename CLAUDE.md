# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

**Phases 1–7 implemented (spec v2).** Backend (FastAPI + WebSocket) and frontend (Vue 3 + Three.js) both run. GroundingDINO + SAM perception produces `runs/snapshot_*.json` + masks; with `--lift` the pipeline also runs Depth Anything V2, lifts each mask via the pinhole model + `FixedPoseSource` into `runs/lifted_*.json`. Phase 4 added `tracking/tracker.py` (IoU + class + 3D-distance greedy association with stable `track_NNN` ids) and `spatial/semantic_map.py` (persistent fused store, EMA position fusion, Bayesian confidence update, `tracked → occluded → stale → lost` status machine, byte-identical save/load). Phase 5 added `spatial/relation_scorer.py` (smooth-ramp scoring for `left_of/right_of/in_front_of/behind/above/below/near/far_from/between/on_surface/occluding`, all in the graphics-world frame) and `spatial/scene_graph.py` (`SceneGraphBuilder.build()` walks SemanticMap pair- and triple-wise, keeps edges above `min_relation_score`, JSON-serializable per spec §3.3). `POST /perception/lifted` now feeds tracker → map → graph and broadcasts the graph alongside markers in `world_update`; `GET /scene/graph` returns it directly. The frontend renders subtle phosphor edges between marker pairs and a sortable RELATIONS card. The `PetAction` schema supports `move_follow_path` for the Phase 8 controller without further schema change.

**`docs/spec.md` is the authoritative design document** (v2 — 10 mainline phases + optional extensions). `docs/spec-ref.md` is research notes that fed v2. When asked to "add X" or "implement Y", locate the relevant phase in `docs/spec.md` and follow its acceptance criteria. Do not invent module APIs, file paths, or schemas — they are specified.

## Architecture (the big picture)

A nine-layer pipeline: **perception → semantic map → grounding → navigation → control → pet runtime**, with optional research backends.

```
Camera/video ──► Mainline perception (GroundingDINO → SAM → Depth)
                                                          │
                                              FramePacket + pose source
                                                          │
                                                          ▼
                                            Object lifter + tracker
                                                          │
                                                          ▼
                                              SemanticMap (persistent)
                                                          │
                                                          ▼
                                              Scene graph + relations
User command ──► Command Parser ──► Grounding Resolver ──► NavigationGoal
                                                          │
                                                          ▼
                                          A* planner (occupancy grid)
                                                          │
                                                          ▼
                                          Pure-pursuit controller
                                                          │
                                                          ▼
                                          PetAction (move_follow_path)
                                                          │
                                                          ▼
                                          3D cat runtime (Three.js browser)
```

Optional sidecars (each in `docs/spec.md §14`): Visual SLAM replaces fixed pose; OpenScene becomes a second perception backend; RL replaces the heuristic in active exploration; ROS 2 Nav2 bridges to a physical robot.

Critical design rules baked into the spec — **do not violate without flagging**:

- **GroundingDINO + SAM/SAM 2 is the mainline. OpenScene / SLAM / RL / ROS 2 are optional.** They must not block the demo. Don't lead with them.
- **Perception (2 Hz), tracking (10 Hz), control (30 Hz), renderer (60 Hz) run at different rates.** Don't couple them into one loop. See `configs/runtime.yaml`.
- **LLM is event-driven, not per-frame.** Command parsing only on user events; output is schema-validated `CommandIntent`, never per-frame motion.
- **Cross-module flows use the typed contracts in `docs/spec.md §3`**: `FramePacket`, `ObjectState`, `SceneGraph`, `SemanticMap`, `CommandIntent`, `NavigationGoal`, `PetAction`. Extend rather than replace.
- **Coordinate frame:** `world` is the default but is produced by `spatial/pose_source.py` with three swappable implementations (`fixed` / `sim` / `slam`). Phase 3+ code must work under all three; default is `fixed`.
- **`PetAction.move_to` and `PetAction.move_follow_path` coexist.** `move_to` for direct manual commands (sandbox, quick buttons); `move_follow_path` for the controller's path output (carries `path: [(x,y,z), ...]` + `speed`). The frontend uses chained Tweens for `move_follow_path` with smooth heading.
- **Source-backend tagging matters**: every `ObjectState` carries `source_backend ∈ {mainline_grounding_sam, openscene}` so backend comparison works.
- **Grounding must be explainable**: `NavigationGoal` carries an `explanation` string. Low confidence or small ambiguity margin → clarification dialog, not a guess.

## Repository Layout (current state — Phases 1–7 done)

Backend Python modules under `src/`:

- `camera_service/` — `image_reader`, `video_reader`, `webcam` (BGR frames).
- `perception/` — `detector` (GroundingDINO), `segmenter` (SAM), `depth` (Depth Anything V2, lazy-loaded with CPU fallback on CUDA failure), `pipeline` orchestrator with `run_frame_3d()` (Phase 3) and `run_frame_tracked()` (Phase 4; lifts → tracks → fuses into SemanticMap), `schema` (`PerceptionResult`, `ObjectCandidate2D`).
- `spatial/` — `frame_packet` (`FramePacket`, `CameraIntrinsics.from_fov()`, `CameraPoseWorld`), `pose_source` (`FixedPoseSource`, `SimPoseSource` JSONL reader), `object_lifter` (`ObjectLifter`, `ObjectState3D` with percentile-filtered median depth + pinhole back-projection + camera→graphics-world axis flip), `semantic_map` (`SemanticMap`: keyed by `track_id`, EMA position fusion, Bayes-like confidence update, status decay `tracked → occluded → stale → lost`, byte-identical save/load), `relation_scorer` (smooth-ramp scoring for 10 base pair-relations + Gaussian `near/far_from` + `between` segment projection + `on_surface` plane attachment + `occluding` bbox + depth ordering), `scene_graph` (`SceneGraphBuilder` walks SemanticMap, emits `SceneGraph` with sorted edges over `min_relation_score`).
- `tracking/` — `tracker.py` (`Tracker`: greedy IoU + class + 3D-distance association; mints stable `track_NNN`; prunes after `persistence_frames` consecutive misses). ByteTrack swap is a future drop-in via `_associate`.
- `language/` — `schema.py` (`CommandIntent`, `TargetSpec`, `RelationSpec`, `ConstraintSpec`), `command_parser.py` (rule-based parser handling 20+ canonical commands; `PET_AGENT_LLM_PARSER=on` seam reserved for an optional LLM adapter with JSON-schema fallback).
- `planning/` — `schema.py` (`NavigationGoal`, `NavigationConstraint`), `grounding_resolver.py` (scores candidates by 0.35·semantic + 0.20·attribute + 0.25·relation + 0.10·visibility + 0.10·feasibility; emits success / clarification / no_match / empty_map; every `NavigationGoal` carries an `explanation`), `occupancy_grid.py` (XZ-plane rasterisation of SemanticMap, obstacle inflation, `avoid_object` halos, `exclude_object_ids` for the target), `astar.py` (8-conn grid A* with Euclidean heuristic, structured failures `no_path/goal_unreachable/start_blocked`, no-corner-cut diagonals, LOS-pruning `smooth_path`), `planner.py` (`Planner` orchestrates grid + A* + smoothing; relocates blocked goal cells via nearest-free; returns `PlannerResult` with status + world-frame path).
- `control/` — `kinematic.py` (frozen `UnicycleState` + `kinematic_step` with v/ω clamps and θ wrap), `pid.py` (immutable PID + anti-windup), `pure_pursuit.py` (signed cross-track via segment tangent; arc-length `lookahead_point`; `v = clamp(base·cos²(he))` + slow-down radius), `path_follower.py` (offline simulator emitting dense `path_world` + `ControlSummary` for `EvaluationRecord`).
- `exploration/` — `coverage_grid.py` (uint16 observation counter, vectorised `observe_cone`, union-find `unknown_clusters`, `nearest_unknown` ring search), `exploration_planner.py` (`ExplorationGoal` ∈ {inspect_unknown, search_object, verify_stale, look_behind}; scoring per spec §12.1; `to_navigation_goal()` so the existing A* runs unchanged).
- `runtime/` — `pet_runtime.py` (authoritative PetState + action API; `PetAction` carries optional `scene_graph` and `controller_trace`), `websocket_server.py` (FastAPI; holds one process-wide `Tracker` + `SemanticMap` + `SceneGraphBuilder` + `GroundingResolver` + `Planner` + `PathFollower` + `CoverageGrid` + `ExplorationPlanner`; endpoints `/ws/pet`, `/perception/lifted`, `/semantic/{map,reset}`, `/scene/graph`, `/planning/occupancy`, `/command`, `/control/{last_trace,simulate}`, `/exploration/{observe,step,coverage,reset}`. `POST /command` parses → grounds → plans → runs the pure-pursuit simulator → emits `move_follow_path` with controller-densified waypoints + `controller_trace`. `intent_type="explore"` routes through the exploration heuristic. Planner / controller failures surface as `runtime.ask` speech rather than teleporting).
- `cli.py` — `--mode` dispatch; `snapshot --lift` runs the Phase 3 lifter; `snapshot --track [--frames N]` runs the Phase 4 tracker + SemanticMap and writes `runs/semantic_map_<image>.json`.
- `config.py` — pydantic `AppConfig` loaded from `configs/*.yaml`, `PET_AGENT_` env prefix. Sections: `models`, `thresholds`, `runtime`, `navigation`, `control`, `settings`.

**Phase 10** added `src/evaluation/` (`schema`, `metrics`, `runner`, `report`) — `EvaluationRunner` loads `samples/eval_dataset.jsonl`, runs each trial through the in-process backend (parse → ground → plan → controller), writes JSONL + CSV + Markdown artifacts under `runs/eval_<ts>/`, and the CLI exits non-zero when task success rate drops below 50%. Latest run on the bundled 8-trial dataset: **8/8 trials task-successful (100%), mean latency 8.3 ms**. Canonical results in [`docs/eval.md`](docs/eval.md).

**Optional sidecar (spec §14.1) — done:** `research/slam_adapter.py` ships the Visual SLAM pose source. `SLAMPoseSource` conforms to the `pose_source.py` `PoseSource` protocol (frames pushed via `track()`, pose read via `get()`) and publishes a `world ← camera` pose already in the graphics-world convention the lifter expects. Backbone is `OrbVisualOdometry` (OpenCV ORB features → RGB-D PnP when depth is supplied, else monocular essential matrix) — a pip-only, frame-to-frame VO stand-in for ORB-SLAM3 (no loop closure / global BA, so it drifts over long loops; a real ORB-SLAM3/DROID-SLAM binding drops in behind the `VisualOdometry` protocol). Enable with `PET_AGENT_POSE_SOURCE=slam`; the perception loop then pushes each webcam frame to the tracker before lifting. Default stays `fixed`.

Planned-but-not-yet:

- `research/` — `openscene_backend`, `rl_explorer` (optional spec §14.2 / §14.3).

Frontend (`frontend/`) is Vue 3 + Vite + TypeScript + native Three.js (no React wrappers). `PetScene.ts` already implements `followPath(path[])` chaining tweens with heading lerp, so the controller can hand off paths without any frontend refactor.

## Runtime Modes

A single CLI entrypoint (`main.py`) dispatches on `--mode`:

| Mode | Status | Inputs |
|---|---|---|
| `sandbox` | ✅ Phase 1 | `--target X Y Z` or `--script foo.jsonl` |
| `snapshot` | ✅ Phase 2 / 3 / 4 | `--image`, `--prompts`, `--out`, optional `--lift` + `--fov`, optional `--track [--frames N]` |
| `demo` | Phase 7–9 | `--camera`, `--prompts` |
| `replay` | ✅ Phase 10 (alias for eval) | `--dataset` |
| `eval` | ✅ Phase 10 | `--dataset samples/eval_dataset.jsonl --out runs` |
| `perception_debug` / `exploration` / `openscene_static` / `compare_backends` / `rl_exploration` / `ros_bridge` | scaffolded (rc=3) | see spec |

## Development Commands

```bash
source .venv/bin/activate
uv pip install -e ".[dev]"

# lint + tests (currently 242 passing)
.venv/bin/ruff check . && .venv/bin/ruff format .
.venv/bin/pytest -q
.venv/bin/pytest tests/test_pet_runtime.py::test_move_follow_path_snaps_state_to_end_and_broadcasts -v

# frontend
cd frontend && npm install && npm run dev
npx vue-tsc --noEmit          # type check

# servers
.venv/bin/uvicorn src.runtime.websocket_server:app --host 127.0.0.1 --port 8000 --reload
```

Environment variables use the `PET_AGENT_` prefix (`PET_AGENT_DEVICE=cuda`, `PET_AGENT_CAMERA_INDEX=0`), loaded via pydantic-settings.

## Conventions

- Python deps managed with `uv` (not pip/poetry).
- Lint and format with `ruff` (no black/isort).
- Config lives in `configs/*.yaml`, validated by Pydantic. v2 adds `configs/navigation.yaml` (Phase 7, done) and `configs/control.yaml` (Phase 8, done).
- Debug artifacts (masks, depth maps, scene graphs, eval runs) are written under `runs/` — that directory is gitignored. Don't commit its contents.
- VRAM target is RTX 4070 (~12 GB). Don't load detector + segmenter + depth at full resolution simultaneously without throttling.
- **GroundingDINO runs fp32** — fp16 trips a `grid_sample` dtype error inside its deformable attention. Documented in spec §17.1 and `docs/spec.md`.

## Phased Implementation Order

Spec §18 prescribes a strict phase order; later phases depend on contracts established earlier. **Do not jump ahead** (e.g. don't build A* before SemanticMap exists). When in doubt, the dependency direction is:

```
sandbox → 2D perception → depth+FramePacket → tracking+SemanticMap →
scene graph → command grounding → A* planning → pure-pursuit control →
exploration → evaluation → (optional sidecars)
```
