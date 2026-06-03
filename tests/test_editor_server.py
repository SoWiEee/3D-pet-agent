"""Scene-editor endpoints: manual object placement + deletion.

Exercises ``POST /editor/object`` and ``DELETE /editor/object/{id}`` through the
in-process FastAPI app (shared ``client`` fixture from conftest).
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def _place(client: TestClient, label: str, x: float, z: float) -> dict:
    r = client.post("/editor/object", json={"class_label": label, "x": x, "z": z})
    assert r.status_code == 200, r.text
    return r.json()


def test_place_object_adds_one_marker(client: TestClient) -> None:
    body = _place(client, "cup", -1.2, -2.0)

    assert body["placed"] is True
    assert body["object_id"].startswith("editor_")
    assert body["map_size"] == 1

    objs = client.get("/semantic/map").json()["objects"]
    assert [o["class_label"] for o in objs] == ["cup"]
    assert objs[0]["tracking_status"] == "tracked"


def test_default_extent_and_floor_rest(client: TestClient) -> None:
    body = _place(client, "cup", 0.0, -1.5)
    oid = body["object_id"]

    obj = next(o for o in client.get("/semantic/map").json()["objects"] if o["object_id"] == oid)
    # cup default extent, and y rests at half-height on the floor.
    assert tuple(obj["extent_3d"]) == (0.08, 0.10, 0.08)
    assert obj["center_3d_world"][1] == 0.05


def test_placing_second_object_keeps_first(client: TestClient) -> None:
    # The decay step in SemanticMap.update must not age out earlier objects
    # when a later placement only submits the new one.
    _place(client, "cup", -1.0, -2.0)
    _place(client, "chair", 1.5, -2.4)

    objs = client.get("/semantic/map").json()["objects"]
    labels = sorted(o["class_label"] for o in objs)
    assert labels == ["chair", "cup"]
    assert all(o["tracking_status"] == "tracked" for o in objs)


def test_empty_label_rejected(client: TestClient) -> None:
    r = client.post("/editor/object", json={"class_label": "  ", "x": 0.0, "z": 0.0})
    assert r.status_code == 400


def test_delete_removes_object(client: TestClient) -> None:
    oid = _place(client, "ball", -1.8, -3.5)["object_id"]

    r = client.delete(f"/editor/object/{oid}")
    assert r.status_code == 200
    assert r.json() == {"removed": True, "object_id": oid, "map_size": 0}
    assert client.get("/semantic/map").json()["objects"] == []


def test_delete_missing_returns_404(client: TestClient) -> None:
    r = client.delete("/editor/object/editor_doesnotexist")
    assert r.status_code == 404


def test_custom_extent_overrides_default(client: TestClient) -> None:
    r = client.post(
        "/editor/object",
        json={"class_label": "box", "x": 0.0, "z": -1.0, "extent": [0.5, 0.5, 0.5]},
    )
    assert r.status_code == 200
    oid = r.json()["object_id"]
    obj = next(o for o in client.get("/semantic/map").json()["objects"] if o["object_id"] == oid)
    assert tuple(obj["extent_3d"]) == (0.5, 0.5, 0.5)
    assert obj["center_3d_world"][1] == 0.25
