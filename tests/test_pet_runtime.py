"""Pet runtime action API. Spec §4.3."""

import asyncio
import json

import pytest

from src.runtime.pet_runtime import PetAction, PetRuntime


def test_move_to_updates_state_and_broadcasts():
    rt = PetRuntime()
    q = rt.subscribe()
    action = rt.move_to(0.5, 0.0, 1.2)
    assert rt.state.position.as_tuple() == (0.5, 0.0, 1.2)
    assert rt.state.animation == "walk"
    assert action.action == "move_to"
    assert action.target_position_3d == (0.5, 0.0, 1.2)
    delivered = q.get_nowait()
    assert delivered.target_position_3d == (0.5, 0.0, 1.2)


def test_look_at_does_not_overwrite_position():
    rt = PetRuntime()
    rt.move_to(0.5, 0.0, 1.2)
    rt.look_at(0.0, 0.3, 1.0)
    assert rt.state.position.as_tuple() == (0.5, 0.0, 1.2)
    assert rt.state.look_at and rt.state.look_at.as_tuple() == (0.0, 0.3, 1.0)


def test_animation_and_emotion():
    rt = PetRuntime()
    rt.play_animation("sit")
    assert rt.state.animation == "sit"
    rt.set_emotion("curious")
    assert rt.state.emotion == "curious"


def test_ask_records_speech():
    rt = PetRuntime()
    rt.ask("hello")
    assert rt.state.speech == "hello"


def test_apply_from_dict():
    rt = PetRuntime()
    rt.apply({"action": "move_to", "target_position_3d": [0.1, 0.0, 0.5]})
    rt.apply({"action": "play_animation", "animation": "sit"})
    rt.apply({"action": "set_emotion", "emotion": "happy"})
    assert rt.state.position.as_tuple() == (0.1, 0.0, 0.5)
    assert rt.state.animation == "sit"
    assert rt.state.emotion == "happy"


def test_unknown_action_raises():
    rt = PetRuntime()
    with pytest.raises(ValueError):
        rt.apply({"action": "fly_to_moon"})


def test_move_follow_path_snaps_state_to_end_and_broadcasts():
    rt = PetRuntime()
    q = rt.subscribe()
    path = [(0.0, 0.0, 0.0), (0.2, 0.0, 0.4), (0.5, 0.0, 1.0)]
    action = rt.move_follow_path(path, speed=0.45)
    assert action.action == "move_follow_path"
    assert action.path == path
    assert action.target_position_3d == path[-1]
    assert rt.state.position.as_tuple() == (0.5, 0.0, 1.0)
    assert rt.state.animation == "walk"
    assert rt.state.speed == 0.45
    delivered = q.get_nowait()
    assert delivered.action == "move_follow_path"
    assert delivered.path and len(delivered.path) == 3


def test_move_follow_path_rejects_empty_path():
    rt = PetRuntime()
    with pytest.raises(ValueError):
        rt.move_follow_path([])


def test_apply_dispatches_move_follow_path():
    rt = PetRuntime()
    rt.apply(
        {
            "action": "move_follow_path",
            "path": [[0.0, 0.0, 0.0], [0.3, 0.0, 0.6]],
            "speed": 0.3,
            "look_at_object_id": "cup_001",
        }
    )
    assert rt.state.position.as_tuple() == (0.3, 0.0, 0.6)
    assert rt.state.speed == 0.3


def test_pet_action_round_trips_json():
    a = PetAction(action="move_to", target_position_3d=(1.0, 2.0, 3.0), speed=0.7)
    blob = a.model_dump_json()
    parsed = json.loads(blob)
    assert parsed["action"] == "move_to"
    assert parsed["target_position_3d"] == [1.0, 2.0, 3.0]


def test_broadcast_drops_oldest_on_slow_consumer():
    rt = PetRuntime()
    q = rt.subscribe()
    # Fill the queue past its bound (maxsize=64).
    for i in range(80):
        rt.move_to(i * 0.01, 0.0, 0.0)
    # No exception; queue should still hold something near the latest.
    assert q.qsize() <= 64

    async def drain() -> list:
        out = []
        while not q.empty():
            out.append(await q.get())
        return out

    items = asyncio.get_event_loop().run_until_complete(drain())
    assert items[-1].target_position_3d == (0.79, 0.0, 0.0)
