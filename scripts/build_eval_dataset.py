"""Phase A3 — generate samples/eval_dataset.jsonl programmatically.

docs/review.md A3: bundled dataset had 8 trials; spec §13.3 calls for
50 NL + 10+ ambiguous + 5 no-target = 65. This script is the source of
truth for the dataset — change it and re-run::

    .venv/bin/python scripts/build_eval_dataset.py

JSONL is the artifact committed to git; the script is committed so a
reviewer can see the intent behind each trial.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATASET_PATH = Path(__file__).resolve().parent.parent / "samples" / "eval_dataset.jsonl"


# ── object factories ──────────────────────────────────────────────────────
def _o(
    object_id: str,
    class_label: str,
    *,
    pos: tuple[float, float, float],
    extent: tuple[float, float, float] = (0.1, 0.1, 0.1),
    attributes: list[str] | None = None,
    confidence: float = 0.85,
    status: str = "tracked",
) -> dict[str, Any]:
    return {
        "object_id": object_id,
        "class_label": class_label,
        "attributes": attributes or [],
        "center_3d_world": list(pos),
        "extent_3d": list(extent),
        "confidence": confidence,
        "tracking_status": status,
    }


# Re-usable canonical objects.
CUP_R = _o("cup_001", "cup", pos=(0.5, 0.0, 0.6), extent=(0.08, 0.12, 0.08), attributes=["red"])
CUP_B = _o("cup_002", "cup", pos=(-0.5, 0.0, 0.6), extent=(0.08, 0.12, 0.08), attributes=["blue"])
KBD = _o("kbd_001", "keyboard", pos=(0.0, 0.0, 0.6), extent=(0.4, 0.04, 0.15))
MOUSE = _o("mouse_001", "mouse", pos=(0.3, 0.0, 0.4), extent=(0.06, 0.03, 0.1))
MONITOR = _o("monitor_001", "monitor", pos=(0.0, 0.3, 1.0), extent=(0.5, 0.3, 0.05))
BOX_S = _o("box_001", "box", pos=(0.8, 0.0, 0.5), extent=(0.15, 0.15, 0.15))
BOX_L = _o("box_002", "box", pos=(-0.8, 0.0, 0.5), extent=(0.25, 0.25, 0.25), attributes=["large"])
CHAIR = _o("chair_001", "chair", pos=(-0.7, 0.0, 0.3), extent=(0.4, 0.5, 0.4))
LAMP = _o("lamp_001", "lamp", pos=(0.6, 0.4, 0.2), extent=(0.1, 0.3, 0.1))
BOTTLE = _o("bottle_001", "bottle", pos=(0.2, 0.0, 0.7), extent=(0.07, 0.2, 0.07))
BOOK = _o("book_001", "book", pos=(-0.3, 0.0, 0.5), extent=(0.18, 0.03, 0.22))
PLANT = _o("plant_001", "plant", pos=(0.9, 0.1, 0.9), extent=(0.2, 0.35, 0.2))
SPEAKER = _o("speaker_001", "speaker", pos=(-0.6, 0.0, 0.8), extent=(0.15, 0.25, 0.15))


def desk_full() -> list[dict[str, Any]]:
    """Six-object canonical desk scene."""
    return [CUP_R, KBD, MOUSE, MONITOR, BOX_S, CHAIR]


def desk_extra() -> list[dict[str, Any]]:
    """Larger scene with lamp / bottle / book / plant / speaker."""
    return [CUP_R, KBD, MOUSE, MONITOR, BOX_S, CHAIR, LAMP, BOTTLE, BOOK, PLANT, SPEAKER]


def two_cups() -> list[dict[str, Any]]:
    return [CUP_R, CUP_B, KBD]


def two_boxes() -> list[dict[str, Any]]:
    return [BOX_S, BOX_L, KBD]


def two_monitors() -> list[dict[str, Any]]:
    return [
        MONITOR,
        _o("monitor_002", "monitor", pos=(1.5, 0.3, 1.0), extent=(0.5, 0.3, 0.05)),
        KBD,
    ]


def two_chairs() -> list[dict[str, Any]]:
    return [
        CHAIR,
        _o("chair_002", "chair", pos=(0.7, 0.0, 0.3), extent=(0.4, 0.5, 0.4)),
    ]


def two_mice() -> list[dict[str, Any]]:
    return [
        MOUSE,
        _o("mouse_002", "mouse", pos=(-0.3, 0.0, 0.4), extent=(0.06, 0.03, 0.1)),
        KBD,
    ]


def two_keyboards() -> list[dict[str, Any]]:
    return [
        KBD,
        _o("kbd_002", "keyboard", pos=(0.0, 0.0, 1.0), extent=(0.4, 0.04, 0.15)),
    ]


def three_cups() -> list[dict[str, Any]]:
    return [
        CUP_R,
        CUP_B,
        _o("cup_003", "cup", pos=(0.0, 0.0, 0.9), extent=(0.08, 0.12, 0.08)),
    ]


def three_boxes() -> list[dict[str, Any]]:
    return [
        BOX_S,
        BOX_L,
        _o("box_003", "box", pos=(0.0, 0.0, 0.8), extent=(0.15, 0.15, 0.15)),
    ]


def stale_cup() -> list[dict[str, Any]]:
    obj = dict(CUP_R)
    obj["confidence"] = 0.3
    obj["tracking_status"] = "stale"
    return [obj]


# ── trial factory ─────────────────────────────────────────────────────────
def trial(
    trial_id: str,
    scene_id: str,
    objects: list[dict[str, Any]],
    command: str,
    expected_outcome: str,
    *,
    expected_target: str | None = None,
    notes: str = "",
    description: str = "",
) -> dict[str, Any]:
    return {
        "trial_id": trial_id,
        "scene": {"scene_id": scene_id, "description": description, "objects": objects},
        "command": command,
        "expected_outcome": expected_outcome,
        "expected_target": expected_target,
        "notes": notes,
    }


# ── dataset ───────────────────────────────────────────────────────────────
def build() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    # ── 1) The six spec §13.1 demo scenarios + 2 baseline cases ──
    rows += [
        trial(
            "t01_navigate",
            "desk_001",
            [CUP_R],
            "go to the cup",
            "navigate",
            expected_target="cup_001",
            notes="spec §13.1 #1",
        ),
        trial(
            "t02_relation_hide",
            "desk_002",
            [KBD, MONITOR],
            "hide behind the keyboard",
            "hide",
            expected_target="kbd_001",
            notes="spec §13.1 #2",
        ),
        trial(
            "t03_avoidance",
            "desk_003",
            [BOX_S, MOUSE],
            "go to the box but avoid the mouse",
            "navigate",
            expected_target="box_001",
            notes="spec §13.1 #3",
        ),
        trial(
            "t04_exploration",
            "desk_004",
            [CUP_R],
            "explore the desk",
            "explore",
            notes="spec §13.1 #4",
        ),
        trial(
            "t05_clarification",
            "desk_005",
            two_cups(),
            "go to the cup",
            "clarification",
            notes="spec §13.1 #5",
        ),
        trial(
            "t06_stale_memory",
            "desk_006",
            stale_cup(),
            "go to the cup",
            "navigate",
            expected_target="cup_001",
            notes="spec §13.1 #6 — object out of view but in map",
        ),
        trial(
            "t07_no_match",
            "desk_007",
            [CUP_R],
            "go to the apple",
            "no_match",
            notes="negative trial",
        ),
        trial(
            "t08_look_at",
            "desk_008",
            [MONITOR],
            "look at the monitor",
            "look_at",
            expected_target="monitor_001",
            notes="no path required",
        ),
    ]

    # ── 2) Navigation verb variants (15) ──
    rows += [
        trial(
            "t09_walk_to_cup",
            "desk_full_01",
            desk_full(),
            "walk to the cup",
            "navigate",
            expected_target="cup_001",
        ),
        trial(
            "t10_approach_kbd",
            "desk_full_02",
            desk_full(),
            "approach the keyboard",
            "navigate",
            expected_target="kbd_001",
        ),
        trial(
            "t11_head_monitor",
            "desk_full_03",
            desk_full(),
            "head to the monitor",
            "navigate",
            expected_target="monitor_001",
        ),
        trial(
            "t12_move_toward_box",
            "desk_full_04",
            desk_full(),
            "move toward the box",
            "navigate",
            expected_target="box_001",
        ),
        trial(
            "t13_go_chair",
            "desk_full_05",
            desk_full(),
            "go to the chair",
            "navigate",
            expected_target="chair_001",
        ),
        trial(
            "t14_walk_red_cup",
            "desk_full_06",
            desk_full(),
            "walk to the red cup",
            "navigate",
            expected_target="cup_001",
            notes="attribute filter",
        ),
        trial(
            "t15_go_lamp",
            "desk_extra_01",
            desk_extra(),
            "go to the lamp",
            "navigate",
            expected_target="lamp_001",
        ),
        trial(
            "t16_walk_bottle",
            "desk_extra_02",
            desk_extra(),
            "walk to the bottle",
            "navigate",
            expected_target="bottle_001",
        ),
        trial(
            "t17_approach_book",
            "desk_extra_03",
            desk_extra(),
            "approach the book",
            "navigate",
            expected_target="book_001",
        ),
        trial(
            "t18_go_plant",
            "desk_extra_04",
            desk_extra(),
            "head to the plant",
            "navigate",
            expected_target="plant_001",
        ),
        trial(
            "t19_walk_speaker",
            "desk_extra_05",
            desk_extra(),
            "walk to the speaker",
            "navigate",
            expected_target="speaker_001",
        ),
        trial(
            "t20_go_mouse",
            "desk_full_07",
            desk_full(),
            "go to the mouse",
            "navigate",
            expected_target="mouse_001",
        ),
        trial(
            "t21_run_box",
            "desk_full_08",
            desk_full(),
            "run to the box",
            "navigate",
            expected_target="box_001",
        ),
        trial(
            "t22_navigate_far_obj",
            "desk_far",
            [
                _o("box_far", "box", pos=(2.5, 0.0, 2.0), extent=(0.15, 0.15, 0.15)),
            ],
            "go to the box",
            "navigate",
            expected_target="box_far",
            notes="distant object — exercises planner over longer path",
        ),
        trial(
            "t23_navigate_around",
            "desk_around",
            [
                CUP_R,
                _o(
                    "obs_001",
                    "box",
                    pos=(0.25, 0.0, 0.4),
                    extent=(0.2, 0.2, 0.2),
                    attributes=["obstacle"],
                ),
            ],
            "go to the cup",
            "navigate",
            expected_target="cup_001",
            notes="obstacle between cat and target — planner must route",
        ),
    ]

    # ── 3) look_at / inspect / search (8) ──
    rows += [
        trial(
            "t24_look_cup",
            "desk_full_09",
            desk_full(),
            "look at the cup",
            "look_at",
            expected_target="cup_001",
        ),
        trial(
            "t25_watch_monitor",
            "desk_full_10",
            desk_full(),
            "watch the monitor",
            "look_at",
            expected_target="monitor_001",
        ),
        trial(
            "t26_stare_keyboard",
            "desk_full_11",
            desk_full(),
            "stare at the keyboard",
            "look_at",
            expected_target="kbd_001",
        ),
        trial(
            "t27_inspect_mouse",
            "desk_full_12",
            desk_full(),
            "inspect the mouse",
            "look_at",
            expected_target="mouse_001",
            notes="inspect maps to look_at outcome in eval (no path)",
        ),
        trial(
            "t28_examine_box",
            "desk_full_13",
            desk_full(),
            "examine the box",
            "look_at",
            expected_target="box_001",
        ),
        trial(
            "t29_check_chair",
            "desk_full_14",
            desk_full(),
            "check the chair",
            "look_at",
            expected_target="chair_001",
        ),
        trial(
            "t30_find_cup",
            "desk_full_15",
            desk_full(),
            "find the cup",
            "look_at",
            expected_target="cup_001",
            notes="search → look_at outcome",
        ),
        trial(
            "t31_search_kbd",
            "desk_full_16",
            desk_full(),
            "search for the keyboard",
            "look_at",
            expected_target="kbd_001",
        ),
    ]

    # ── 4) hide variants (5) ──
    rows += [
        trial(
            "t32_hide_monitor",
            "hide_01",
            [MONITOR, CHAIR],
            "hide behind the monitor",
            "hide",
            expected_target="monitor_001",
        ),
        trial(
            "t33_hide_box",
            "hide_02",
            [BOX_S, MOUSE],
            "hide behind the box",
            "hide",
            expected_target="box_001",
        ),
        trial(
            "t34_hide_chair",
            "hide_03",
            [CHAIR, KBD],
            "hide behind the chair",
            "hide",
            expected_target="chair_001",
        ),
        trial(
            "t35_hide_cup",
            "hide_04",
            [CUP_R, MOUSE],
            "hide behind the cup",
            "hide",
            expected_target="cup_001",
        ),
        trial(
            "t36_hide_kbd2",
            "hide_05",
            [KBD, BOX_S],
            "hide behind the keyboard",
            "hide",
            expected_target="kbd_001",
        ),
    ]

    # ── 5) avoidance / constraint (7) ──
    rows += [
        trial(
            "t37_avoid_mouse",
            "avoid_01",
            [CUP_R, MOUSE],
            "go to the cup but avoid the mouse",
            "navigate",
            expected_target="cup_001",
        ),
        trial(
            "t38_avoid_box",
            "avoid_02",
            [KBD, BOX_S],
            "go to the keyboard but avoid the box",
            "navigate",
            expected_target="kbd_001",
        ),
        trial(
            "t39_avoid_chair",
            "avoid_03",
            [BOX_S, CHAIR],
            "go to the box but avoid the chair",
            "navigate",
            expected_target="box_001",
        ),
        trial(
            "t40_keepaway_cup",
            "avoid_04",
            [MONITOR, CUP_R],
            "approach the monitor but stay away from the cup",
            "navigate",
            expected_target="monitor_001",
        ),
        trial(
            "t41_avoid_kbd",
            "avoid_05",
            [CUP_R, KBD],
            "go to the cup but avoid the keyboard",
            "navigate",
            expected_target="cup_001",
        ),
        trial(
            "t42_avoid_box2",
            "avoid_06",
            [CHAIR, BOX_S],
            "head to the chair but avoid the box",
            "navigate",
            expected_target="chair_001",
        ),
        trial(
            "t43_avoid_two",
            "avoid_07",
            [CUP_R, MOUSE, KBD],
            "approach the cup but avoid the mouse",
            "navigate",
            expected_target="cup_001",
            notes="single avoid keyword — multi-avoid not parsed yet",
        ),
    ]

    # ── 6) ambiguous → clarification (12) ──
    rows += [
        trial("t44_ambig_cup", "ambig_01", two_cups(), "go to the cup", "clarification"),
        trial("t45_ambig_box", "ambig_02", two_boxes(), "go to the box", "clarification"),
        trial(
            "t46_ambig_monitor", "ambig_03", two_monitors(), "approach the monitor", "clarification"
        ),
        trial("t47_ambig_chair", "ambig_04", two_chairs(), "head to the chair", "clarification"),
        trial("t48_ambig_mouse", "ambig_05", two_mice(), "go to the mouse", "clarification"),
        trial("t49_ambig_cup_look", "ambig_06", two_cups(), "look at the cup", "clarification"),
        trial("t50_ambig_kbd", "ambig_07", two_keyboards(), "go to the keyboard", "clarification"),
        trial("t51_ambig_cup_hide", "ambig_08", two_cups(), "hide behind the cup", "clarification"),
        trial("t52_ambig_three_cup", "ambig_09", three_cups(), "go to the cup", "clarification"),
        trial(
            "t53_ambig_three_box", "ambig_10", three_boxes(), "approach the box", "clarification"
        ),
        trial("t54_ambig_chair2", "ambig_11", two_chairs(), "go to the chair", "clarification"),
        trial("t55_ambig_cup_examine", "ambig_12", two_cups(), "examine the cup", "clarification"),
    ]

    # ── 7) no_match — target class absent (6) ──
    rows += [
        trial("t56_no_lamp", "neg_01", [CUP_R, KBD], "go to the lamp", "no_match"),
        trial("t57_no_keys", "neg_02", desk_full(), "find the keys", "no_match"),
        trial("t58_no_dog", "neg_03", desk_full(), "approach the dog", "no_match"),
        trial("t59_no_pizza", "neg_04", desk_full(), "go to the pizza", "no_match"),
        trial("t60_no_spoon", "neg_05", desk_full(), "look at the spoon", "no_match"),
        trial("t61_no_door", "neg_06", desk_full(), "hide behind the door", "no_match"),
    ]

    # ── 8) no-target commands (4 new + t04 = 5 total) ──
    rows += [
        trial("t62_stop", "ctrl_01", desk_full(), "stop", "stop"),
        trial("t63_halt", "ctrl_02", desk_full(), "halt", "stop"),
        trial("t64_explore_room", "ctrl_03", desk_full(), "explore the room", "explore"),
        trial("t65_report", "ctrl_04", desk_full(), "what do you see", "report"),
    ]

    return rows


def main() -> None:
    rows = build()
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DATASET_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    by_outcome: dict[str, int] = {}
    for r in rows:
        by_outcome[r["expected_outcome"]] = by_outcome.get(r["expected_outcome"], 0) + 1
    print(f"Wrote {len(rows)} trials to {DATASET_PATH}")
    for k in sorted(by_outcome):
        print(f"  {k:>14s}: {by_outcome[k]}")


if __name__ == "__main__":
    main()
