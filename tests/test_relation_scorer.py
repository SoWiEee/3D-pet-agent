"""Phase 5 — RelationScorer per-relation tests."""

from __future__ import annotations

from src.spatial import RelationConfig, RelationScorer
from src.spatial.object_lifter import ObjectConfidence, ObjectState3D


def _obj(
    *,
    object_id: str,
    label: str = "cup",
    center: tuple[float, float, float] = (0.0, 0.0, -2.0),
    extent: tuple[float, float, float] = (0.1, 0.1, 0.1),
    bbox: tuple[float, float, float, float] = (100, 100, 200, 200),
    median_depth: float = 2.0,
) -> ObjectState3D:
    return ObjectState3D(
        object_id=object_id,
        class_label=label,
        bbox_xyxy=bbox,
        mask_path=None,
        center_2d=((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
        coordinate_frame="world",
        center_3d_world=center,
        extent_3d=extent,
        median_depth=median_depth,
        depth_uncertainty=0.1,
        confidence=ObjectConfidence(
            detector=0.8, mask_quality=0.7, depth_quality=0.7, tracking=1.0, overall=0.75
        ),
        last_seen_frame=0,
    )


# ── axis-projected pair relations ──────────────────────────────────────────
def test_left_right_swap() -> None:
    s = RelationScorer()
    a = _obj(object_id="a", center=(-0.5, 0, -2))
    b = _obj(object_id="b", center=(0.5, 0, -2))
    assert s.left_of(a, b) > 0.9
    assert s.right_of(a, b) == 0.0
    assert s.right_of(b, a) > 0.9
    assert s.left_of(b, a) == 0.0


def test_axis_threshold_zero_at_zero_delta() -> None:
    s = RelationScorer()
    a = _obj(object_id="a", center=(0.0, 0.0, -2.0))
    b = _obj(object_id="b", center=(0.0, 0.0, -2.0))
    assert s.left_of(a, b) == 0.0
    assert s.right_of(a, b) == 0.0
    assert s.above(a, b) == 0.0
    assert s.in_front_of(a, b) == 0.0


def test_in_front_uses_camera_relative_z() -> None:
    """Graphics world: closer to camera (origin) means larger z (less neg)."""
    s = RelationScorer()
    near = _obj(object_id="near", center=(0, 0, -1.0))
    far = _obj(object_id="far", center=(0, 0, -3.0))
    assert s.in_front_of(near, far) > 0.9
    assert s.behind(far, near) > 0.9
    assert s.in_front_of(far, near) == 0.0


def test_above_below_uses_y() -> None:
    s = RelationScorer()
    up = _obj(object_id="up", center=(0, 0.5, -2))
    dn = _obj(object_id="dn", center=(0, -0.5, -2))
    assert s.above(up, dn) > 0.9
    assert s.below(dn, up) > 0.9


# ── distance ──────────────────────────────────────────────────────────────
def test_near_far_complementary() -> None:
    s = RelationScorer(RelationConfig(near_sigma=0.5))
    a = _obj(object_id="a", center=(0, 0, 0))
    b = _obj(object_id="b", center=(0.1, 0, 0))
    c = _obj(object_id="c", center=(3.0, 0, 0))
    near_ab = s.near(a, b)
    near_ac = s.near(a, c)
    assert near_ab > near_ac
    assert abs(s.near(a, b) + s.far_from(a, b) - 1.0) < 1e-9


# ── between ───────────────────────────────────────────────────────────────
def test_between_midpoint_high() -> None:
    s = RelationScorer()
    mid = _obj(object_id="m", center=(0.0, 0, -2))
    left = _obj(object_id="l", center=(-1.0, 0, -2))
    right = _obj(object_id="r", center=(1.0, 0, -2))
    score = s.between(mid, left, right)
    assert score > 0.9


def test_between_outside_segment_zero() -> None:
    s = RelationScorer()
    outside = _obj(object_id="o", center=(2.0, 0, -2))
    left = _obj(object_id="l", center=(-1.0, 0, -2))
    right = _obj(object_id="r", center=(1.0, 0, -2))
    assert s.between(outside, left, right) == 0.0


def test_between_off_axis_decays() -> None:
    s = RelationScorer()
    off = _obj(object_id="o", center=(0.0, 0, -1.0))  # 1m off the segment
    left = _obj(object_id="l", center=(-1.0, 0, -2))
    right = _obj(object_id="r", center=(1.0, 0, -2))
    assert s.between(off, left, right) == 0.0


# ── on_surface ────────────────────────────────────────────────────────────
def test_on_surface_attached() -> None:
    s = RelationScorer()
    desk = _obj(object_id="desk", center=(0, 0.0, -2), extent=(2.0, 0.05, 1.0))
    # cup's bottom (cup_y - 0.05) sits right on the desk's top (0.025).
    cup = _obj(object_id="cup", center=(0.0, 0.075, -2), extent=(0.1, 0.1, 0.1))
    assert s.on_surface(cup, desk) > 0.8


def test_on_surface_outside_xz_extent() -> None:
    s = RelationScorer()
    desk = _obj(object_id="desk", center=(0, 0, -2), extent=(0.5, 0.05, 0.5))
    floater = _obj(object_id="x", center=(2.0, 0.075, -2), extent=(0.1, 0.1, 0.1))
    assert s.on_surface(floater, desk) == 0.0


# ── occluding ─────────────────────────────────────────────────────────────
def test_occluding_requires_closer_depth_and_overlap() -> None:
    s = RelationScorer()
    near_o = _obj(
        object_id="near",
        bbox=(100, 100, 200, 200),
        median_depth=1.5,
    )
    far_o = _obj(
        object_id="far",
        bbox=(120, 120, 220, 220),
        median_depth=3.0,
    )
    assert s.occluding(near_o, far_o) > 0.3
    # Reverse: the far one doesn't occlude the near one.
    assert s.occluding(far_o, near_o) == 0.0


def test_occluding_zero_without_bbox_overlap() -> None:
    s = RelationScorer()
    a = _obj(object_id="a", bbox=(0, 0, 50, 50), median_depth=1.5)
    b = _obj(object_id="b", bbox=(500, 500, 600, 600), median_depth=3.0)
    assert s.occluding(a, b) == 0.0
