"""Typed configuration loaded from configs/*.yaml.

Spec §16. Settings use pydantic for validation; YAML is the canonical source of truth
so non-developers can tune thresholds without touching code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


class DetectorConfig(BaseModel):
    name: str
    hf_model_id: str
    device: str = "cuda"
    box_threshold: float = 0.30
    text_threshold: float = 0.25


class SegmenterConfig(BaseModel):
    name: str
    hf_model_id: str
    device: str = "cuda"


class DepthConfig(BaseModel):
    name: str
    hf_model_id: str
    device: str = "cuda"
    mode: str = "relative"


class OpenSceneConfig(BaseModel):
    enabled: bool = False
    scene_root: str = "data/openscene"


class ModelsConfig(BaseModel):
    detector: DetectorConfig
    segmenter: SegmenterConfig
    depth: DepthConfig
    openscene: OpenSceneConfig


class GroundingThresholds(BaseModel):
    min_final_score: float = 0.65
    ambiguity_margin: float = 0.12


class TrackingThresholds(BaseModel):
    backend: str = "simple_iou_then_bytetrack"
    min_iou: float = 0.35
    max_center_distance: float = 0.20
    persistence_frames: int = 3


class RelationThresholds(BaseModel):
    near_sigma: float = 0.50
    right_left_threshold: float = 0.08
    behind_front_threshold: float = 0.10


class BehaviorThresholds(BaseModel):
    safe_distance: float = 0.15
    default_speed: float = 0.8


class ThresholdsConfig(BaseModel):
    grounding: GroundingThresholds
    tracking: TrackingThresholds
    relations: RelationThresholds
    behavior: BehaviorThresholds


class RuntimeSection(BaseModel):
    perception_update_hz: float = 2.0
    tracking_update_hz: float = 10.0
    renderer_fps: int = 60
    save_debug_outputs: bool = True
    ask_clarification: bool = True


class ServerSection(BaseModel):
    host: str = "127.0.0.1"
    http_port: int = 8000
    ws_path: str = "/ws/pet"


class RuntimeConfig(BaseModel):
    runtime: RuntimeSection
    server: ServerSection


class NavigationGridConfig(BaseModel):
    resolution: float = 0.05
    origin_x: float = -3.0
    origin_z: float = -4.0
    width: int = 120
    height: int = 120
    obstacle_padding: float = 0.15


class NavigationPlannerConfig(BaseModel):
    connectivity: int = 8
    nearest_free_radius: float = 1.0
    smoothing: Literal["line_of_sight", "none"] = "line_of_sight"
    smoothing_subdivisions: int = 0
    default_speed: float = 0.45
    goal_tolerance: float = 0.10


class NavigationConstraintsConfig(BaseModel):
    avoid_default_min_distance: float = 0.25


class NavigationConfig(BaseModel):
    grid: NavigationGridConfig = Field(default_factory=NavigationGridConfig)
    planner: NavigationPlannerConfig = Field(default_factory=NavigationPlannerConfig)
    constraints: NavigationConstraintsConfig = Field(default_factory=NavigationConstraintsConfig)


# ── Phase 8: control ────────────────────────────────────────────────────────
class KinematicConfig(BaseModel):
    v_max: float = 0.80
    v_min: float = 0.05
    omega_max: float = 3.20
    dt: float = 0.05
    max_steps: int = 400


class PurePursuitConfig(BaseModel):
    lookahead_distance: float = 0.30
    base_speed: float = 0.45
    kp_heading: float = 2.40
    goal_tolerance: float = 0.08
    cross_track_tolerance: float = 0.20


class SpeedPIDConfig(BaseModel):
    kp: float = 1.20
    ki: float = 0.05
    kd: float = 0.02
    integral_clamp: float = 0.40


class PreemptConfig(BaseModel):
    max_latency_s: float = 0.10


class CarConfig(BaseModel):
    """Car-like (bicycle) kinematics for the robot avatar — spec §14.5.

    Distinct from the unicycle ``kinematic`` block: a finite ``wheelbase`` and
    ``max_steer`` give a non-zero minimum turning radius, so the robot drives
    Reeds-Shepp paths (and reverses to square up) instead of pivoting in place.
    """

    enabled: bool = True
    wheelbase: float = 0.44  # m — front-rear axle separation
    max_steer: float = 0.55  # rad (~31°) — front-wheel steering limit
    v_max: float = 0.80  # m/s
    speed: float = 0.45  # m/s — cruise magnitude along the path
    dt: float = 0.05  # s — densification step


class ManipulationConfig(BaseModel):
    """Arm/grasp parameters for the live pick path — spec §14.5 Stage C/E."""

    arm_base_height: float = 0.30  # m — shoulder height above the floor on arrival
    min_grasp_confidence: float = 0.20  # below this, the pick is refused (speech)


class ControlConfig(BaseModel):
    kinematic: KinematicConfig = Field(default_factory=KinematicConfig)
    pure_pursuit: PurePursuitConfig = Field(default_factory=PurePursuitConfig)
    speed_pid: SpeedPIDConfig = Field(default_factory=SpeedPIDConfig)
    preempt: PreemptConfig = Field(default_factory=PreemptConfig)
    car: CarConfig = Field(default_factory=CarConfig)
    manipulation: ManipulationConfig = Field(default_factory=ManipulationConfig)


class Settings(BaseSettings):
    """Environment overrides — see spec §16 (PET_AGENT_ prefix)."""

    model_config = SettingsConfigDict(env_prefix="PET_AGENT_", extra="ignore")

    device: str = "cuda"
    camera_index: int = 0
    weights_dir: str = "weights"
    # Pose source for the live perception loop. "slam" enables the frame-to-frame
    # ORB visual-odometry sidecar (spec §14.1); "graph_slam" adds a PyPose
    # pose-graph back-end with ORB loop closure on top of the same VO front-end
    # (spec §14.6.2); default "fixed" keeps the camera at the world origin.
    # Override with PET_AGENT_POSE_SOURCE=graph_slam.
    pose_source: Literal["fixed", "sim", "slam", "graph_slam"] = "fixed"

    # Local Ollama backend for the command parser (spec §14.6.4). Consulted only
    # when PET_AGENT_LLM_PARSER=on and PET_AGENT_LLM_BACKEND=ollama; every failure
    # path falls back to the rule parser. Override with PET_AGENT_OLLAMA_MODEL /
    # PET_AGENT_OLLAMA_HOST.
    ollama_model: str = "qwen2.5-coder:7b"
    ollama_host: str = "http://localhost:11434"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_models(config_dir: Path = CONFIG_DIR) -> ModelsConfig:
    return ModelsConfig(**_load_yaml(config_dir / "models.yaml"))


def load_thresholds(config_dir: Path = CONFIG_DIR) -> ThresholdsConfig:
    return ThresholdsConfig(**_load_yaml(config_dir / "thresholds.yaml"))


def load_runtime(config_dir: Path = CONFIG_DIR) -> RuntimeConfig:
    return RuntimeConfig(**_load_yaml(config_dir / "runtime.yaml"))


def load_navigation(config_dir: Path = CONFIG_DIR) -> NavigationConfig:
    """Phase 7: navigation config is optional — fall back to NavigationConfig
    defaults so existing test fixtures don't have to ship the file."""
    path = config_dir / "navigation.yaml"
    if not path.exists():
        return NavigationConfig()
    return NavigationConfig(**_load_yaml(path))


def load_control(config_dir: Path = CONFIG_DIR) -> ControlConfig:
    """Phase 8: pure-pursuit controller + PID config. Optional — defaults match
    ``configs/control.yaml`` so tests can run without the YAML file present."""
    path = config_dir / "control.yaml"
    if not path.exists():
        return ControlConfig()
    return ControlConfig(**_load_yaml(path))


def load_prompts(config_dir: Path = CONFIG_DIR) -> list[str]:
    path = config_dir / "prompts.txt"
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class AppConfig(BaseModel):
    """Bundled, validated configuration. One object passed around the app."""

    models: ModelsConfig
    thresholds: ThresholdsConfig
    runtime: RuntimeConfig
    navigation: NavigationConfig = Field(default_factory=NavigationConfig)
    control: ControlConfig = Field(default_factory=ControlConfig)
    settings: Settings = Field(default_factory=Settings)

    @classmethod
    def load(cls, config_dir: Path = CONFIG_DIR) -> AppConfig:
        return cls(
            models=load_models(config_dir),
            thresholds=load_thresholds(config_dir),
            runtime=load_runtime(config_dir),
            navigation=load_navigation(config_dir),
            control=load_control(config_dir),
        )
