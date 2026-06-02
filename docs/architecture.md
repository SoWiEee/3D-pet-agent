# Architecture

> This document is the **canonical system overview**. Both `README.md`
> and `CLAUDE.md` link here instead of inlining their own copies, so any
> change to the architecture lands in exactly one place.

## Goal in one line

A virtual 3D cat that understands a real desk / room scene from a camera
feed and reacts to natural-language commands like "hide behind the
keyboard" or "go to the red cup but avoid the mouse".

## Nine-layer pipeline

```text
Camera/video ──► Mainline perception (GroundingDINO → SAM → Depth Anything V2)
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

Optional sidecars (each in `docs/spec.md §14`): Visual SLAM replaces
fixed pose; OpenScene becomes a second perception backend; RL replaces
the heuristic in active exploration; ROS 2 Nav2 bridges to a physical
robot. None of these gate the demo.

## Module map (spec §2.3 + implemented Phases 1–10)

| Layer | Module | Responsibility |
|---|---|---|
| Camera | `src/camera_service/` | Image / video / webcam readers, all returning BGR frames |
| Perception (2D) | `src/perception/` | GroundingDINO + SAM, packs `PerceptionResult` |
| Depth + lift | `src/perception/depth.py`, `src/spatial/object_lifter.py` | Depth Anything V2 + pinhole back-projection to graphics-world coords |
| Pose | `src/spatial/pose_source.py` | `FixedPoseSource` / `SimPoseSource` / future SLAM |
| Tracking | `src/tracking/tracker.py` | IoU + class + 3D-distance greedy association, stable `track_NNN` ids |
| SemanticMap | `src/spatial/semantic_map.py` | Persistent fused store; EMA position; Bayes confidence; `tracked → occluded → stale → lost` |
| Scene graph | `src/spatial/relation_scorer.py`, `src/spatial/scene_graph.py` | 11 smooth-ramp relations, pair + triple walks |
| Language | `src/language/` | 10-intent rule-based parser + LLM seam |
| Grounding | `src/planning/grounding_resolver.py` | `0.35·semantic + 0.20·attribute + 0.25·relation + 0.10·visibility + 0.10·feasibility` |
| Planning | `src/planning/occupancy_grid.py`, `astar.py`, `planner.py` | XZ grid + obstacle inflation + 8-conn A* + LOS smoothing |
| Control | `src/control/` | Frozen `UnicycleState` + pure-pursuit + anti-windup PID + offline simulator |
| Exploration | `src/exploration/` | `CoverageGrid` + 4-goal-kind heuristic (spec §12.1) |
| Runtime | `src/runtime/` | `PetRuntime` action API + FastAPI server holding one of each module above |
| Evaluation | `src/evaluation/` | Dataset → in-process pipeline → `EvaluationRecord` → CSV/JSONL/Markdown report |

## Critical design rules (must not be violated without flagging)

1. **GroundingDINO + SAM / SAM 2 is the mainline.** OpenScene / SLAM / RL / ROS 2 are optional. They must not block the demo. Don't lead with them.
2. **Different rates run in different loops:** perception 2 Hz, tracking 10 Hz, control 30 Hz, renderer 60 Hz. Don't couple them. See `configs/runtime.yaml`.
3. **LLM is event-driven, not per-frame.** Command parsing only on user events; output is schema-validated `CommandIntent`, never per-frame motion.
4. **Cross-module flows use the typed contracts in spec §3:** `FramePacket`, `ObjectState`, `SceneGraph`, `SemanticMap`, `CommandIntent`, `NavigationGoal`, `PetAction`. Extend rather than replace.
5. **Coordinate frame:** `world` is the default but is produced by `spatial/pose_source.py` with three swappable implementations (`fixed` / `sim` / `slam`). Phase 3+ code must work under all three; default is `fixed`.
6. **`PetAction.move_to` and `PetAction.move_follow_path` coexist.** `move_to` is for direct manual commands (sandbox, quick buttons); `move_follow_path` is for the controller's path output (`path: [(x,y,z), ...]` + `speed`). The frontend uses chained Tweens for `move_follow_path` with smooth heading.
7. **Source-backend tagging matters:** every `ObjectState` carries `source_backend ∈ {mainline_grounding_sam, openscene}` so backend comparison works.
8. **Grounding must be explainable:** `NavigationGoal` carries an `explanation` string. Low confidence or a small ambiguity margin → clarification dialog, not a guess.
9. **Planner / controller failures surface as `runtime.ask` speech, not teleports.** When A\* returns `no_path / goal_unreachable / start_blocked`, the user sees the structured failure instead of the cat jumping to the goal.
10. **GroundingDINO runs fp32.** fp16 trips a `grid_sample` dtype error inside its deformable attention. Documented in spec §17.1.

## Runtime modes

A single CLI entrypoint (`main.py`) dispatches on `--mode`:

| Mode | Status | Inputs |
|---|---|---|
| `sandbox` | ✅ Phase 1 | `--target X Y Z` or `--script foo.jsonl` |
| `snapshot` | ✅ Phase 2 / 3 / 4 | `--image`, `--prompts`, `--out`, optional `--lift` + `--fov`, optional `--track [--frames N]` |
| `demo` | ✅ Phase 7–9 | `--camera`, `--prompts` |
| `eval` | ✅ Phase 10 | `--dataset samples/eval_dataset.jsonl --out runs` |
| `replay` | ✅ Phase 10 (alias for eval) | same as eval |
| `perception_debug` / `openscene_static` / `compare_backends` / `rl_exploration` / `ros_bridge` | scaffolded (rc=3) | spec §14 |

## Configuration sources

| File | Purpose |
|---|---|
| `configs/models.yaml` | Detector / segmenter / depth model IDs + device |
| `configs/thresholds.yaml` | Grounding / tracking / relations / behavior thresholds |
| `configs/runtime.yaml` | Update rates + server host/port |
| `configs/navigation.yaml` | Phase 7 grid + planner + constraint halos |
| `configs/control.yaml` | Phase 8 kinematic + pure-pursuit + PID + preempt latency |
| `configs/prompts.txt` | Default open-vocabulary detector prompts |
| Env vars | `PET_AGENT_*` prefix overrides anything in `configs/` |

## Frontend

`frontend/` is Vue 3 + Vite + TypeScript + **native** Three.js (no React
wrappers). `PetScene.ts` implements `followPath(path[])` by chaining
tweens with heading lerp, so the controller's dense path lands directly
on the renderer with no further refactor.

## Per-phase implementation reference

What each `--mode` and endpoint actually produces. Bash invocations of
these flows live in [`README.md`](../README.md); the explanations below
say **what the bytes mean**.

### Phase 1 — 3D 寵物 sandbox (`--mode sandbox`)

Pure pet runtime, no perception models loaded. `--target X Y Z`
broadcasts a single `move_to`; `--script foo.jsonl` plays a sequence of
`idle / move_to / look_at / set_emotion / play_animation / ask` actions
with a 0.5 s gap between events.

### Phase 2 — Snapshot 偵測 + 分割 (`--mode snapshot`)

Runs GroundingDINO + SAM on a single image. **First run downloads
weights from Hugging Face** (GroundingDINO ≈ 700 MB, SAM ≈ 400 MB).

Outputs under `runs/`:

| File | Content |
|---|---|
| `runs/snapshot_<image>.json` | `PerceptionResult` — per-object bbox, mask path, confidence, normalised centre (spec §5.4) |
| `runs/snapshot_<image>.png` | Visual overlay — bbox + label + mask |
| `runs/frame_000000/obj_XXX_<label>.png` | Binary mask per object |

`configs/prompts.txt` is tuned for desk / room scenes (cup, keyboard,
mouse, monitor, …). Replace via `--prompts /path/to/my_prompts.txt`.

### Phase 3 — Depth + 3D lifting (`--mode snapshot --lift`)

Adds Depth Anything V2 + 2D→3D back-projection. Camera intrinsics are
estimated from `--fov` (default 60°) when no calibration is supplied.

Extra outputs:

| File | Content |
|---|---|
| `runs/lifted_<image>.json` | Per-object `ObjectState3D` — `center_3d_world`, `extent_3d`, `median_depth`, `depth_uncertainty`, `confidence` (`detector / mask_quality / depth_quality / overall`) |
| `runs/depth_<image>.png` | Inferno-colormap depth visualisation |

Caveats:

- Monocular depth is **relative**, not metric — values are only
  comparable within the same image.
- Without calibration, accuracy is "consistent-scale pairs", not metric.
- Default `pose_source: fixed` (camera at world origin). Phase optional
  §14.1 swaps in ORB-SLAM.
- The depth model falls back to CPU when CUDA is unavailable (≈ 1–2 s
  per frame instead of < 100 ms).

### Phase 4 — Tracker + persistent SemanticMap (`--mode snapshot --track`)

Chains the Phase 3 lifter through IoU + class + 3D-distance association
(stable `track_NNN` ids) into a persistent SemanticMap with EMA position
fusion, Bayes confidence update, and a `tracked → occluded → stale →
lost` status machine.

Extra output:

| File | Content |
|---|---|
| `runs/semantic_map_<image>.json` | Persistent map — latest `ObjectState3D` per track, `confidence.overall`, `tracking_status`, `last_seen_frame` |

`SemanticMap.save → load → save` is byte-identical (acceptance
criterion). Re-running with `--frames N` replays the same image N times
to demonstrate id persistence under repeated observation.

### Phase 5 + 6 — Scene graph + command grounding (server-driven)

Once the backend is running (`uvicorn …`), every SemanticMap update
recomputes the scene graph (11 smooth-ramp relations:
`left_of / right_of / in_front_of / behind / above / below / near /
far_from / between / on_surface / occluding`) and broadcasts it to the
RELATIONS panel.

`POST /command` runs the full pipeline: rule-based parser
(10 `intent_type`s) → grounding resolver
(`0.35·semantic + 0.20·attribute + 0.25·relation + 0.10·visibility +
0.10·feasibility`). Multiple candidates trigger clarification asks; low
confidence comes back with a verbatim `explanation`.

### Phase 7 + 8 — A\* + pure-pursuit (server-driven)

On successful grounding the server:

1. Rasterises the SemanticMap into an XZ-plane `OccupancyGrid`
   (with `obstacle_padding` inflation, per-target exclusion, and
   `avoid_object` halos).
2. Runs 8-connectivity A\* (Euclidean heuristic, no corner-cut,
   Bresenham LOS smoothing).
3. Feeds the smoothed path into the pure-pursuit offline simulator:
   `UnicycleState (x, y, θ)` + `v = clamp(base·cos²(he), v_min, v_max)`
   + `ω = Kp·he` + anti-windup PID speed smoothing + slow-down radius.
4. Broadcasts `move_follow_path` with a dense dynamically-feasible
   trajectory + a `controller_trace` summary
   (`steps / duration_s / max_cross_track_error / max_heading_error /
   mean_speed`).

Controller config lives in `configs/control.yaml` (kinematic limits,
lookahead, PID gains, preempt latency).

### Phase 9 — Active exploration (server-driven)

`CoverageGrid` uses a `uint16` per-cell observation counter (same XZ
frame as the navigation grid), updated by a vectorised cone sweep.
`ExplorationPlanner` scores 4 goal kinds per spec §12.1
(`0.40·new_area + 0.25·uncertainty + 0.20·search_relevance −
0.15·travel_cost`) and routes the winner back through the standard A\* +
controller pipeline. Newly discovered object ids are reported via the
pet's speech bubble.

### Phase 10 — Evaluation harness (`--mode eval` / `replay`)

Loads `samples/eval_dataset.jsonl`, runs each `DatasetEntry` through the
backend in-process (parse → ground → plan → controller), and writes
three artifacts under `runs/eval_<ts>/`:

| File | Purpose |
|---|---|
| `report.md` | Human-readable summary |
| `records.csv` | Per-trial fields for spreadsheets |
| `records.jsonl` | Raw `EvaluationRecord` (spec §3.8) per line |

The CLI exits non-zero when the task success rate drops below 50% — the
same threshold enforced inside `pytest`. See [`eval.md`](eval.md) for
the canonical numbers (currently 100% on the bundled dataset).

## Repository layout

```text
3D-pet-agent/
├── main.py                      # CLI entry — dispatches to src/cli.py
├── pyproject.toml
├── configs/
│   ├── models.yaml              # Model ids + device + thresholds
│   ├── thresholds.yaml          # Grounding / tracking / relations / behavior
│   ├── runtime.yaml             # Update rates + server host/port
│   ├── navigation.yaml          # Phase 7 grid + planner + constraints
│   ├── control.yaml             # Phase 8 kinematic + pure-pursuit + PID
│   └── prompts.txt
├── src/
│   ├── config.py                # AppConfig (pydantic-settings, PET_AGENT_ prefix)
│   ├── cli.py                   # --mode dispatch
│   ├── camera_service/          # image / video / webcam readers
│   ├── perception/              # detector, segmenter, depth, pipeline, schema
│   ├── spatial/                 # FramePacket, pose_source, object_lifter,
│   │                            # semantic_map, relation_scorer, scene_graph
│   ├── tracking/                # IoU + class + 3D-distance tracker
│   ├── language/                # CommandIntent + rule-based parser (+ LLM seam)
│   ├── planning/                # NavigationGoal, grounding_resolver,
│   │                            # occupancy_grid, astar, planner
│   ├── control/                 # UnicycleState, pid, pure_pursuit, path_follower
│   ├── exploration/             # coverage_grid, exploration_planner
│   ├── evaluation/              # schema, metrics, runner, report
│   └── runtime/
│       ├── pet_runtime.py       # PetState + action API (includes controller_trace)
│       └── websocket_server.py  # FastAPI app + all endpoints
├── frontend/                    # Vue 3 + Vite + TS + native Three.js
│   └── src/
│       ├── App.vue
│       ├── renderer/PetScene.ts # Three.js scene, followPath chained tweens
│       ├── composables/useWebSocket.ts
│       └── components/          # StatusBar, ModulePanel, Readouts, CommandBar,
│                                # WorldObjectsLayer, RelationEdgesLayer,
│                                # RegistrationMarks, PetSpeech
├── tests/                       # 242 unit + integration + server smoke tests
├── .github/workflows/ci.yml     # Backend + frontend CI
├── samples/
│   ├── desk.jpg
│   ├── pet_actions.jsonl
│   └── eval_dataset.jsonl
├── docs/
│   ├── architecture.md          # this file
│   ├── eval.md                  # Phase 10 canonical results
│   ├── spec.md                  # authoritative v2 spec
│   └── spec-ref.md              # research notes
└── runs/                        # (gitignored) perception + eval outputs
```

## Tech stack

### AI / vision
- PyTorch + CUDA
- GroundingDINO via `transformers` (`IDEA-Research/grounding-dino-tiny`)
- SAM (`facebook/sam-vit-base`); Phase 3+ can swap to SAM 2
- Depth Anything V2 (`depth-anything/Depth-Anything-V2-Small-hf`)
- OpenCV, Pillow, numpy, supervision

### Backend
- Python 3.12
- FastAPI + Uvicorn + websockets
- pydantic / pydantic-settings
- `uv` for env + deps
- `ruff` lint + format
- `pytest` + `pytest-asyncio`

### Frontend / 3D runtime
- Vue 3 + Vite + TypeScript
- Native Three.js (no React wrappers)
- `@tweenjs/tween.js` for waypoint interpolation
- `lil-gui` for debug panels (reserved)

## See also

- [`docs/spec.md`](spec.md) — authoritative v2 specification (10 phases + optional)
- [`docs/eval.md`](eval.md) — Phase 10 evaluation results against the bundled dataset
- [`docs/spec-ref.md`](spec-ref.md) — research notes that fed v2
