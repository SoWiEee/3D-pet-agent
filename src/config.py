"""Typed configuration loaded from configs/*.yaml.

Spec §16. Settings use pydantic for validation; YAML is the canonical source of truth
so non-developers can tune thresholds without touching code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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


class Settings(BaseSettings):
    """Environment overrides — see spec §16 (PET_AGENT_ prefix)."""

    model_config = SettingsConfigDict(env_prefix="PET_AGENT_", extra="ignore")

    device: str = "cuda"
    camera_index: int = 0
    weights_dir: str = "weights"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_models(config_dir: Path = CONFIG_DIR) -> ModelsConfig:
    return ModelsConfig(**_load_yaml(config_dir / "models.yaml"))


def load_thresholds(config_dir: Path = CONFIG_DIR) -> ThresholdsConfig:
    return ThresholdsConfig(**_load_yaml(config_dir / "thresholds.yaml"))


def load_runtime(config_dir: Path = CONFIG_DIR) -> RuntimeConfig:
    return RuntimeConfig(**_load_yaml(config_dir / "runtime.yaml"))


def load_prompts(config_dir: Path = CONFIG_DIR) -> list[str]:
    path = config_dir / "prompts.txt"
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class AppConfig(BaseModel):
    """Bundled, validated configuration. One object passed around the app."""

    models: ModelsConfig
    thresholds: ThresholdsConfig
    runtime: RuntimeConfig
    settings: Settings = Field(default_factory=Settings)

    @classmethod
    def load(cls, config_dir: Path = CONFIG_DIR) -> AppConfig:
        return cls(
            models=load_models(config_dir),
            thresholds=load_thresholds(config_dir),
            runtime=load_runtime(config_dir),
        )
