# ByteTrack Tracker Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a real ByteTrack tracker (8-state Kalman + Hungarian assignment via the `supervision` library) behind the existing `Tracker.update()` surface, selectable by config/env, with the current greedy tracker retained as the default fallback.

**Architecture:** A `make_tracker()` factory returns either the existing `Tracker` (default) or a new `ByteTrackTracker` adapter. The adapter runs one `supervision.ByteTrack` instance **per class label** (preserving the existing class-gated association), associates purely in the 2D image plane, maps each per-class integer `tracker_id` to a stable `track_NNN` slug, and rewrites `object_id` on the original `ObjectState3D` so the lifter's 3D centre rides along untouched. SemanticMap (keyed by `track_id`) is unaffected. Missing/broken `supervision` import falls back to the greedy tracker with a logged warning — the demo never gates on the heavy dependency.

**Tech Stack:** Python 3.12, `supervision` (numpy + scipy ByteTrack), pydantic, pytest, ruff, uv. Implements spec §14.6.1.

---

## File Structure

- `pyproject.toml` — add `supervision` to a new optional extra `[track]`.
- `src/tracking/protocol.py` (create) — `TrackerBackend` Protocol + `make_tracker()` factory.
- `src/tracking/bytetrack_adapter.py` (create) — `ByteTrackTracker` adapter.
- `src/tracking/__init__.py` (modify) — export `ByteTrackTracker`, `TrackerBackend`, `make_tracker`.
- `src/runtime/websocket_server.py:73` (modify) — build the tracker via `make_tracker(config)`.
- `tests/test_bytetrack_adapter.py` (create) — adapter behavioral tests.
- `tests/test_tracker_factory.py` (create) — factory selection + fallback tests.
- `docs/spec.md` §14.6.1 (modify) — mark implemented.
- `CLAUDE.md` (modify) — note the new backend + env var.

**Shared surface** every backend implements (already satisfied by `Tracker`):
- `update(detections: list[ObjectState3D], frame_id: int) -> list[ObjectState3D]`
- `reset() -> None`
- `active_tracks` property → `dict[str, Any]`

---

## Task 1: Add `supervision` as an optional dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the `track` extra**

In `pyproject.toml`, under `[project.optional-dependencies]`, add (create the table if absent):

```toml
[project.optional-dependencies]
track = ["supervision>=0.25"]
```

- [ ] **Step 2: Install it into the venv**

Run: `uv pip install -e ".[track]"`
Expected: resolves cleanly. Note: `supervision 0.28.0` is **already present** in
this venv (verified via `uv pip install --dry-run supervision` → "no changes"),
so this step mainly records the dependency in `pyproject.toml` for
reproducibility. numpy stays at 2.1.3 — `supervision` does not pin numpy<2.

- [ ] **Step 3: Verify the import and the API surface used by the adapter**

Run: `.venv/bin/python -c "import supervision as sv; t=sv.ByteTrack(); import numpy as np; d=sv.Detections(xyxy=np.array([[0,0,10,10.]]), confidence=np.array([0.9]), class_id=np.array([0])); r=t.update_with_detections(d); print('tracker_id', r.tracker_id); t.reset(); print('ok')"`
Expected: prints a `tracker_id` array (e.g. `[1]`) and `ok`. If `ByteTrack` rejects a kwarg in Step-3 of Task 3, pin the version here.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore(track): add supervision optional dependency (.[track])"
```

---

## Task 2: Tracker protocol + factory (default backend unchanged)

**Files:**
- Create: `src/tracking/protocol.py`
- Modify: `src/tracking/__init__.py`
- Test: `tests/test_tracker_factory.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_tracker_factory.py`:

```python
"""§14.6.1 — tracker backend factory selection + fallback."""

from __future__ import annotations

from src.config import TrackingThresholds
from src.tracking import Tracker, make_tracker


def test_default_backend_returns_greedy_tracker() -> None:
    cfg = TrackingThresholds()  # backend = "simple_iou_then_bytetrack"
    t = make_tracker(cfg)
    assert isinstance(t, Tracker)


def test_unknown_backend_falls_back_to_greedy() -> None:
    cfg = TrackingThresholds(backend="does_not_exist")
    t = make_tracker(cfg)
    assert isinstance(t, Tracker)


def test_env_override_selects_greedy(monkeypatch) -> None:
    monkeypatch.setenv("PET_AGENT_TRACKER", "greedy")
    cfg = TrackingThresholds(backend="supervision_bytetrack")
    t = make_tracker(cfg)
    assert isinstance(t, Tracker)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/pytest tests/test_tracker_factory.py -v`
Expected: FAIL with `ImportError: cannot import name 'make_tracker'`.

- [ ] **Step 3: Implement the protocol + factory**

Create `src/tracking/protocol.py`:

```python
"""Tracker backend protocol + factory (spec §14.6.1).

Every tracker backend exposes the same surface so the perception loop is
backend-agnostic. ``make_tracker`` selects the backend from config, honouring a
``PET_AGENT_TRACKER`` env override, and always degrades to the greedy
``Tracker`` rather than raising — a missing heavy dependency must never gate the
demo.
"""

from __future__ import annotations

import logging
import os
from typing import Protocol, runtime_checkable

from ..spatial.object_lifter import ObjectState3D
from .tracker import Tracker

log = logging.getLogger("pet_agent.tracking")

# Config/env values that select the real ByteTrack adapter.
_BYTETRACK_NAMES = {"supervision_bytetrack", "bytetrack"}


@runtime_checkable
class TrackerBackend(Protocol):
    """The surface the perception loop depends on."""

    def update(
        self, detections: list[ObjectState3D], frame_id: int
    ) -> list[ObjectState3D]: ...

    def reset(self) -> None: ...

    @property
    def active_tracks(self) -> dict: ...


def make_tracker(cfg) -> TrackerBackend:  # cfg: TrackingThresholds
    """Build the configured tracker. ``PET_AGENT_TRACKER`` env overrides
    ``cfg.backend``. Unknown names or a failed ByteTrack import → greedy."""
    choice = (os.environ.get("PET_AGENT_TRACKER") or cfg.backend or "").strip().lower()

    if choice in _BYTETRACK_NAMES:
        try:
            from .bytetrack_adapter import ByteTrackTracker

            return ByteTrackTracker(
                high_confidence=0.5,
                persistence_frames=cfg.persistence_frames,
                min_iou=cfg.min_iou,
            )
        except Exception as e:  # noqa: BLE001 — heavy dep optional
            log.warning("ByteTrack unavailable (%s); using greedy tracker", e)

    return Tracker(
        min_iou=cfg.min_iou,
        max_center_distance=cfg.max_center_distance,
        persistence_frames=cfg.persistence_frames,
    )
```

Modify `src/tracking/__init__.py`:

```python
"""Per-frame object association (spec §7 Phase 4 / §14.6.1)."""

from .protocol import TrackerBackend, make_tracker
from .tracker import TrackedObject, Tracker

__all__ = ["Tracker", "TrackedObject", "TrackerBackend", "make_tracker"]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_tracker_factory.py -v`
Expected: 3 PASS. (`test_env_override_selects_greedy` passes because the env override short-circuits before any ByteTrack import.)

- [ ] **Step 5: Commit**

```bash
git add src/tracking/protocol.py src/tracking/__init__.py tests/test_tracker_factory.py
git commit -m "feat(track): tracker backend protocol + make_tracker factory"
```

---

## Task 3: ByteTrackTracker — first frame mints stable ids

**Files:**
- Create: `src/tracking/bytetrack_adapter.py`
- Test: `tests/test_bytetrack_adapter.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_bytetrack_adapter.py`:

```python
"""§14.6.1 — supervision ByteTrack adapter behavioral tests."""

from __future__ import annotations

import pytest

from tests.factories import make_object

sv = pytest.importorskip("supervision")
from src.tracking.bytetrack_adapter import ByteTrackTracker  # noqa: E402


def _obs(object_id, label, bbox, center, frame, detector=0.9, overall=0.85):
    return make_object(
        object_id=object_id,
        class_label=label,
        bbox_xyxy=bbox,
        center_3d_world=center,
        last_seen_frame=frame,
        detector=detector,
        mask_quality=0.7,
        depth_quality=0.7,
        overall=overall,
    )


def test_first_frame_mints_track_ids() -> None:
    t = ByteTrackTracker()
    obs = [
        _obs("raw_a", "cup", (100, 100, 200, 200), (0.0, 0.0, -2.0), 0),
        _obs("raw_b", "keyboard", (300, 300, 500, 400), (1.0, 0.0, -2.0), 0),
    ]
    out = t.update(obs, frame_id=0)
    ids = sorted(o.object_id for o in out)
    assert all(i.startswith("track_") for i in ids)
    assert len(set(ids)) == 2


def test_3d_centre_rides_along_unchanged() -> None:
    t = ByteTrackTracker()
    obs = [_obs("raw_a", "cup", (100, 100, 200, 200), (0.4, 0.1, -2.0), 0)]
    out = t.update(obs, frame_id=0)
    assert len(out) == 1
    assert out[0].center_3d_world == (0.4, 0.1, -2.0)
    assert out[0].object_id.startswith("track_")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/pytest tests/test_bytetrack_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.tracking.bytetrack_adapter'`.

- [ ] **Step 3: Implement the adapter (mint + ride-along)**

Create `src/tracking/bytetrack_adapter.py`:

```python
"""Real ByteTrack via `supervision`, behind the `Tracker` surface (spec §14.6.1).

`supervision.ByteTrack` provides the textbook ByteTrack: an 8-state
constant-velocity Kalman filter per track plus Hungarian (linear-sum) assignment
over a two-stage high/low score cascade. We run **one instance per class label**
so association stays class-gated (matching the greedy tracker), associate purely
in the 2D image plane, and map each per-class integer ``tracker_id`` to a stable
``track_NNN`` slug. The lifter's 3D centre is never touched by ByteTrack — it
rides along on the original ``ObjectState3D`` whose ``object_id`` we rewrite.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..spatial.object_lifter import ObjectState3D

log = logging.getLogger("pet_agent.bytetrack")


@dataclass
class _LiteTrack:
    """Minimal active-track record for the `active_tracks` surface."""

    track_id: str
    class_label: str
    last_seen_frame: int


def _det_score(det: ObjectState3D) -> float:
    return max(det.confidence.detector, det.confidence.overall)


class ByteTrackTracker:
    """ByteTrack adapter exposing the same surface as `Tracker`."""

    def __init__(
        self,
        *,
        high_confidence: float = 0.5,
        persistence_frames: int = 3,
        min_iou: float = 0.35,
        frame_rate: int = 10,
    ) -> None:
        import supervision as sv  # lazy: factory catches ImportError → greedy

        self._sv = sv
        self._activation = high_confidence
        self._lost_buffer = max(1, persistence_frames)
        self._matching_threshold = float(min_iou)
        self._frame_rate = frame_rate
        self._per_class: dict[str, "object"] = {}  # label -> sv.ByteTrack
        self._id_map: dict[tuple[str, int], str] = {}  # (label, sv_id) -> slug
        self._tracks: dict[str, _LiteTrack] = {}
        self._next_id = 1

    # ── surface ──────────────────────────────────────────────────────────────
    def reset(self) -> None:
        for tracker in self._per_class.values():
            tracker.reset()
        self._per_class.clear()
        self._id_map.clear()
        self._tracks.clear()
        self._next_id = 1

    @property
    def active_tracks(self) -> dict[str, _LiteTrack]:
        return dict(self._tracks)

    def update(
        self, detections: list[ObjectState3D], frame_id: int
    ) -> list[ObjectState3D]:
        out: list[ObjectState3D] = []
        by_class: dict[str, list[int]] = {}
        for i, det in enumerate(detections):
            by_class.setdefault(det.class_label, []).append(i)
        for label, idxs in by_class.items():
            out.extend(self._update_class(label, idxs, detections, frame_id))
        return out

    # ── internals ────────────────────────────────────────────────────────────
    def _mint_id(self) -> str:
        slug = f"track_{self._next_id:03d}"
        self._next_id += 1
        return slug

    def _class_tracker(self, label: str):
        tracker = self._per_class.get(label)
        if tracker is None:
            tracker = self._sv.ByteTrack(
                track_activation_threshold=self._activation,
                lost_track_buffer=self._lost_buffer,
                minimum_matching_threshold=self._matching_threshold,
                frame_rate=self._frame_rate,
            )
            self._per_class[label] = tracker
        return tracker

    def _update_class(
        self,
        label: str,
        idxs: list[int],
        detections: list[ObjectState3D],
        frame_id: int,
    ) -> list[ObjectState3D]:
        tracker = self._class_tracker(label)
        xyxy = np.array([detections[i].bbox_xyxy for i in idxs], dtype=float)
        conf = np.array([_det_score(detections[i]) for i in idxs], dtype=float)
        det_block = self._sv.Detections(
            xyxy=xyxy,
            confidence=conf,
            class_id=np.zeros(len(idxs), dtype=int),
        )
        tracked = tracker.update_with_detections(det_block)

        # Map each returned (tracked) box back to its input index by coordinates
        # — we built the boxes, so an exact-rounded lookup is unambiguous.
        box_to_idx = {
            tuple(np.round(detections[i].bbox_xyxy, 3)): i for i in idxs
        }
        out: list[ObjectState3D] = []
        ids = tracked.tracker_id if tracked.tracker_id is not None else []
        for row in range(len(tracked)):
            sv_id = int(ids[row]) if len(ids) else -1
            if sv_id < 0:
                continue
            box = tuple(np.round(tracked.xyxy[row], 3))
            di = box_to_idx.get(box)
            if di is None:
                continue
            key = (label, sv_id)
            slug = self._id_map.get(key)
            if slug is None:
                slug = self._mint_id()
                self._id_map[key] = slug
            self._tracks[slug] = _LiteTrack(slug, label, frame_id)
            out.append(detections[di].model_copy(update={"object_id": slug}))
        return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_bytetrack_adapter.py -v`
Expected: 2 PASS. If `update_with_detections` returns boxes in a slightly different dtype/rounding, adjust the `np.round(..., 3)` precision in both the lookup and the readback so they match.

- [ ] **Step 5: Commit**

```bash
git add src/tracking/bytetrack_adapter.py tests/test_bytetrack_adapter.py
git commit -m "feat(track): ByteTrackTracker adapter — first-frame mint + 3D ride-along"
```

---

## Task 4: ByteTrackTracker — stable id across frames

**Files:**
- Modify: `tests/test_bytetrack_adapter.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_bytetrack_adapter.py`:

```python
def test_same_object_keeps_id_across_frames() -> None:
    t = ByteTrackTracker(min_iou=0.2)
    first = t.update(
        [_obs("raw_a", "cup", (100, 100, 200, 200), (0.0, 0.0, -2.0), 0)],
        frame_id=0,
    )
    assert len(first) == 1
    track_id = first[0].object_id
    # Small motion on the next two frames — IoU stays high, id must hold.
    for frame, bbox in ((1, (106, 104, 206, 204)), (2, (112, 108, 212, 208))):
        out = t.update(
            [_obs("raw_a", "cup", bbox, (0.0, 0.0, -2.0), frame)],
            frame_id=frame,
        )
        assert len(out) == 1
        assert out[0].object_id == track_id
```

- [ ] **Step 2: Run it to verify it fails or passes**

Run: `.venv/bin/pytest tests/test_bytetrack_adapter.py::test_same_object_keeps_id_across_frames -v`
Expected: PASS if the adapter is correct. If FAIL because ByteTrack needs `minimum_consecutive_frames` to activate a track before it emits a stable id, set `minimum_consecutive_frames=1` in `_class_tracker` (Task 3, Step 3) and re-run — that is the intended fix, not a test change.

- [ ] **Step 3: Apply the activation fix if needed**

If Step 2 failed, add `minimum_consecutive_frames=1` to the `sv.ByteTrack(...)` call in `src/tracking/bytetrack_adapter.py::_class_tracker`.

- [ ] **Step 4: Run again to verify pass**

Run: `.venv/bin/pytest tests/test_bytetrack_adapter.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tracking/bytetrack_adapter.py tests/test_bytetrack_adapter.py
git commit -m "test(track): ByteTrack holds id across frames"
```

---

## Task 5: ByteTrackTracker — class gating isolates ids

**Files:**
- Modify: `tests/test_bytetrack_adapter.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_bytetrack_adapter.py`:

```python
def test_overlapping_boxes_of_different_classes_get_distinct_ids() -> None:
    t = ByteTrackTracker(min_iou=0.2)
    # A cup and a book at the *same* pixel box — class gating must keep them
    # on separate per-class trackers, so they never share an id.
    out = t.update(
        [
            _obs("raw_cup", "cup", (100, 100, 200, 200), (0.0, 0.0, -2.0), 0),
            _obs("raw_book", "book", (100, 100, 200, 200), (0.0, 0.0, -2.0), 0),
        ],
        frame_id=0,
    )
    by_label = {o.class_label: o.object_id for o in out}
    assert set(by_label) == {"cup", "book"}
    assert by_label["cup"] != by_label["book"]
```

- [ ] **Step 2: Run it to verify it passes**

Run: `.venv/bin/pytest tests/test_bytetrack_adapter.py::test_overlapping_boxes_of_different_classes_get_distinct_ids -v`
Expected: PASS — per-class `sv.ByteTrack` instances mean the two boxes never compete for the same id even at identical coordinates. (If it fails, the per-class dict in Task 3 is wired wrong.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_bytetrack_adapter.py
git commit -m "test(track): per-class gating keeps overlapping classes distinct"
```

---

## Task 6: ByteTrackTracker — reset + low-confidence drop

**Files:**
- Modify: `tests/test_bytetrack_adapter.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_bytetrack_adapter.py`:

```python
def test_reset_clears_all_state() -> None:
    t = ByteTrackTracker()
    t.update([_obs("raw_a", "cup", (100, 100, 200, 200), (0, 0, -2), 0)], frame_id=0)
    assert len(t.active_tracks) >= 1
    t.reset()
    assert t.active_tracks == {}
    # After reset the id counter restarts from track_001.
    out = t.update([_obs("raw_a", "cup", (100, 100, 200, 200), (0, 0, -2), 0)], 0)
    assert out[0].object_id == "track_001"


def test_low_confidence_only_detection_is_not_emitted_as_new_track() -> None:
    t = ByteTrackTracker(high_confidence=0.5)
    # A single faint box below the activation threshold has no track to recover,
    # so ByteTrack drops it (the ByteTrack insight) — output is empty.
    out = t.update(
        [_obs("raw_a", "cup", (100, 100, 200, 200), (0, 0, -2), 0,
              detector=0.2, overall=0.2)],
        frame_id=0,
    )
    assert out == []
```

- [ ] **Step 2: Run them to verify pass**

Run: `.venv/bin/pytest tests/test_bytetrack_adapter.py -v`
Expected: all PASS. If `test_reset_clears_all_state` fails on the `track_001` assertion, confirm `reset()` zeroes `_next_id` and clears `_id_map` (Task 3).

- [ ] **Step 3: Commit**

```bash
git add tests/test_bytetrack_adapter.py
git commit -m "test(track): ByteTrack reset + low-confidence drop"
```

---

## Task 7: Wire the factory into the websocket server

**Files:**
- Modify: `src/runtime/websocket_server.py:62,73`
- Test: `tests/test_tracker_factory.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tracker_factory.py`:

```python
def test_bytetrack_backend_builds_adapter_when_available(monkeypatch) -> None:
    pytest = __import__("pytest")
    pytest.importorskip("supervision")
    monkeypatch.delenv("PET_AGENT_TRACKER", raising=False)
    from src.tracking.bytetrack_adapter import ByteTrackTracker

    cfg = TrackingThresholds(backend="supervision_bytetrack")
    t = make_tracker(cfg)
    assert isinstance(t, ByteTrackTracker)
```

- [ ] **Step 2: Run it to verify it passes**

Run: `.venv/bin/pytest tests/test_tracker_factory.py::test_bytetrack_backend_builds_adapter_when_available -v`
Expected: PASS (factory already implemented in Task 2; this pins the real-library path).

- [ ] **Step 3: Replace the hardcoded `Tracker()` in the server**

In `src/runtime/websocket_server.py`, change the import line 62 region and the construction at line 73.

Find:

```python
from ..tracking import Tracker
```

Replace with:

```python
from ..tracking import make_tracker
```

Find:

```python
tracker = Tracker()
```

Replace with:

```python
tracker = make_tracker(config.thresholds.tracking)
```

> If `config` is not already imported at module scope in this file, use the existing config accessor already used for `_ctrl_cfg` (grep for how `config`/`AppConfig` is loaded near the other singletons and mirror it). Do **not** introduce a second config load.

- [ ] **Step 4: Run the server-construction smoke + full suite**

Run: `.venv/bin/python -c "import src.runtime.websocket_server as s; print(type(s.tracker).__name__)"`
Expected: prints `Tracker` (default config backend is `simple_iou_then_bytetrack` → greedy).

Run: `.venv/bin/pytest -q`
Expected: all previously-passing tests still pass (242+ baseline) plus the new ones.

- [ ] **Step 5: Commit**

```bash
git add src/runtime/websocket_server.py tests/test_tracker_factory.py
git commit -m "feat(track): build server tracker via make_tracker factory"
```

---

## Task 8: Fallback when `supervision` is missing

**Files:**
- Test: `tests/test_tracker_factory.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tracker_factory.py`:

```python
def test_bytetrack_backend_falls_back_when_import_fails(monkeypatch) -> None:
    import builtins

    monkeypatch.delenv("PET_AGENT_TRACKER", raising=False)
    real_import = builtins.__import__

    def _no_supervision(name, *args, **kwargs):
        if name == "supervision" or name.startswith("supervision."):
            raise ImportError("simulated missing supervision")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_supervision)

    cfg = TrackingThresholds(backend="supervision_bytetrack")
    t = make_tracker(cfg)
    assert isinstance(t, Tracker)  # graceful fallback, no raise
```

- [ ] **Step 2: Run it to verify it passes**

Run: `.venv/bin/pytest tests/test_tracker_factory.py::test_bytetrack_backend_falls_back_when_import_fails -v`
Expected: PASS — the `except Exception` in `make_tracker` catches the simulated `ImportError` and returns `Tracker`. (The lazy `import supervision` lives inside `ByteTrackTracker.__init__`, so the patched import is hit.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_tracker_factory.py
git commit -m "test(track): greedy fallback when supervision import fails"
```

---

## Task 9: Acceptance — ID-switch comparison on an occlusion sequence

**Files:**
- Create: `tests/test_bytetrack_acceptance.py`

This task encodes the spec §14.6.1 acceptance criterion: on a seeded multi-object crossing/occlusion sequence, ByteTrack's ID switches do not exceed the greedy baseline.

- [ ] **Step 1: Write the acceptance test**

Create `tests/test_bytetrack_acceptance.py`:

```python
"""§14.6.1 acceptance — ByteTrack ID stability vs greedy on a crossing scene."""

from __future__ import annotations

import pytest

from src.tracking import Tracker
from tests.factories import make_object

sv = pytest.importorskip("supervision")
from src.tracking.bytetrack_adapter import ByteTrackTracker  # noqa: E402


def _obs(object_id, label, bbox, center, frame):
    return make_object(
        object_id=object_id,
        class_label=label,
        bbox_xyxy=bbox,
        center_3d_world=center,
        last_seen_frame=frame,
        detector=0.9,
        mask_quality=0.7,
        depth_quality=0.7,
        overall=0.85,
    )


def _two_cups_crossing(frames: int = 12):
    """Two same-class cups translating toward and past each other in x."""
    seq = []
    for f in range(frames):
        ax = 100 + f * 18
        bx = 360 - f * 18
        seq.append(
            [
                _obs("gtA", "cup", (ax, 100, ax + 60, 180), (-1.0 + f * 0.15, 0, -2), f),
                _obs("gtB", "cup", (bx, 100, bx + 60, 180), (1.0 - f * 0.15, 0, -2), f),
            ]
        )
    return seq


def _count_id_switches(tracker, seq) -> int:
    """Run a sequence; count how often a ground-truth object's assigned
    track id changes from the previous frame (lower is better)."""
    last: dict[str, str] = {}
    # We tag ground truth via the input order: index 0 = gtA, 1 = gtB.
    switches = 0
    for f, dets in enumerate(seq):
        out = tracker.update(dets, frame_id=f)
        # Map output back to the input by exact bbox (we control both).
        bbox_to_track = {tuple(o.bbox_xyxy): o.object_id for o in out}
        for gt_key, det in (("gtA", dets[0]), ("gtB", dets[1])):
            tid = bbox_to_track.get(tuple(det.bbox_xyxy))
            if tid is None:
                continue
            if gt_key in last and last[gt_key] != tid:
                switches += 1
            last[gt_key] = tid
    return switches


def test_bytetrack_id_switches_not_worse_than_greedy() -> None:
    seq = _two_cups_crossing()
    greedy = _count_id_switches(
        Tracker(min_iou=0.2, max_center_distance=0.5), seq
    )
    bt = _count_id_switches(ByteTrackTracker(min_iou=0.2), seq)
    assert bt <= greedy + 1, f"ByteTrack {bt} switches vs greedy {greedy}"
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/pytest tests/test_bytetrack_acceptance.py -v`
Expected: PASS. If ByteTrack reports more switches than expected, tune the crossing speed (`f * 18`) so boxes don't teleport past the matching threshold in one step — that models a realistic 10 Hz tracker, not a test fudge. Document the chosen parameters in the test docstring.

- [ ] **Step 3: Commit**

```bash
git add tests/test_bytetrack_acceptance.py
git commit -m "test(track): ByteTrack ID-switch acceptance vs greedy baseline"
```

---

## Task 10: Docs + final verification

**Files:**
- Modify: `docs/spec.md` (§14.6.1)
- Modify: `CLAUDE.md`

- [ ] **Step 1: Mark §14.6.1 implemented**

In `docs/spec.md`, in the `#### 14.6.1 ByteTrack` block, append a status line after the **Acceptance** bullet:

```markdown
- **Status — implemented:** `tracking/bytetrack_adapter.py` (`ByteTrackTracker`,
  per-class `supervision.ByteTrack`); selected via `make_tracker` from
  `thresholds.tracking.backend = "supervision_bytetrack"` or
  `PET_AGENT_TRACKER=bytetrack`; greedy `Tracker` stays the default fallback.
```

- [ ] **Step 2: Note the backend in CLAUDE.md**

In `CLAUDE.md`, in the `tracking/` bullet under Repository Layout, append:

```markdown
  A real ByteTrack backend (`bytetrack_adapter.py`, `supervision`) is selectable
  via `PET_AGENT_TRACKER=bytetrack` (or `thresholds.tracking.backend`); install
  it with `uv pip install -e ".[track]"`. Default stays the greedy tracker.
```

- [ ] **Step 3: Full lint + format + test**

Run: `.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/pytest -q`
Expected: ruff clean; full suite green (baseline 242 + new tests).

- [ ] **Step 4: Commit**

```bash
git add docs/spec.md CLAUDE.md
git commit -m "docs(track): mark §14.6.1 ByteTrack implemented"
```

---

## Self-Review Notes

- **Spec coverage (§14.6.1):** library `supervision` (Task 1) ✓; adapter with per-class association + 3D ride-along + `track_NNN` mapping (Tasks 3–5) ✓; `PET_AGENT_TRACKER` selection (Tasks 2, 7) ✓; default greedy fallback incl. missing-dep path (Tasks 2, 8) ✓; acceptance = ID switches ≤ greedy (Task 9) ✓; SemanticMap untouched (rewrites `object_id` only, no schema change) ✓.
- **Surface consistency:** every backend exposes `update(detections, frame_id)`, `reset()`, `active_tracks` — pinned by the `TrackerBackend` Protocol (Task 2) and exercised against both implementations.
- **Type consistency:** `ByteTrackTracker.__init__` kwargs (`high_confidence`, `persistence_frames`, `min_iou`, `frame_rate`) are used identically in the factory (Task 2) and tests (Tasks 3–9). `_LiteTrack` is the only adapter-internal type and is never leaked into a contract.
- **Known execution-time risk:** `supervision.ByteTrack` return-shape details (length/order of `update_with_detections`, `tracker_id` for unactivated boxes, the `minimum_consecutive_frames` activation lag) are version-sensitive. Tasks 3–4 call this out explicitly with the intended fix (box-coordinate readback; `minimum_consecutive_frames=1`) rather than hiding it.
