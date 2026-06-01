"""Configuration loaders. Spec §16."""
from src.config import AppConfig, load_models, load_prompts, load_runtime, load_thresholds


def test_models_yaml_loads():
    m = load_models()
    assert m.detector.name == "groundingdino"
    assert m.detector.hf_model_id.startswith("IDEA-Research/")
    assert 0 < m.detector.box_threshold < 1
    assert m.segmenter.hf_model_id
    assert m.depth.hf_model_id


def test_thresholds_yaml_loads():
    t = load_thresholds()
    assert 0 < t.grounding.min_final_score <= 1
    assert t.tracking.min_iou > 0
    assert t.behavior.default_speed > 0


def test_runtime_yaml_loads():
    r = load_runtime()
    assert r.runtime.renderer_fps > 0
    assert r.runtime.perception_update_hz > 0
    assert r.server.http_port > 0
    assert r.server.ws_path.startswith("/")


def test_prompts_load():
    p = load_prompts()
    assert "cup" in p
    assert "keyboard" in p
    assert all(line.strip() == line for line in p)


def test_app_config_bundle():
    cfg = AppConfig.load()
    assert cfg.models.detector.hf_model_id
    assert cfg.thresholds.grounding.min_final_score > 0
    assert cfg.runtime.server.ws_path
    assert cfg.settings.weights_dir
