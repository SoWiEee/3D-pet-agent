"""CLI dispatch for main.py. Spec §3.3 runtime modes.

Phase 1–2 implement: sandbox, snapshot. Other modes are scaffolded to refuse cleanly
with NotImplementedError so we can ship the live demo in phases.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from .config import AppConfig
from .runtime.pet_runtime import PetRuntime

log = logging.getLogger("pet_agent.cli")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pet-agent", description="3D Pet Agent")
    p.add_argument(
        "--mode",
        required=True,
        choices=[
            "sandbox", "snapshot", "demo", "replay",
            "record", "eval", "openscene_static", "compare_backends",
        ],
    )
    # Common
    p.add_argument("--config-dir", type=Path, default=None, help="override configs/ dir")
    # Sandbox
    p.add_argument("--target", nargs=3, type=float, metavar=("X", "Y", "Z"))
    p.add_argument("--script", type=Path, help="JSONL of pet actions to replay")
    p.add_argument("--serve", action="store_true", help="start WebSocket server (sandbox/demo)")
    # Snapshot / replay
    p.add_argument("--image", type=Path)
    p.add_argument("--video", type=Path)
    p.add_argument("--command", type=str)
    p.add_argument("--prompts", type=Path, help="override configs/prompts.txt")
    p.add_argument("--out", type=Path, default=Path("runs"))
    p.add_argument(
        "--lift",
        action="store_true",
        help="snapshot mode: also run depth + 3D lifting (Phase 3)",
    )
    p.add_argument(
        "--fov",
        type=float,
        default=60.0,
        help="estimated camera horizontal FOV in degrees (used when no intrinsics file)",
    )
    p.add_argument(
        "--track",
        action="store_true",
        help="snapshot mode: also feed lifted objects through the Phase 4 tracker + SemanticMap "
             "(writes runs/semantic_map_<image>.json)",
    )
    p.add_argument(
        "--frames",
        type=int,
        default=1,
        help="snapshot --track: replay the same image N times to demonstrate id persistence",
    )
    # Demo
    p.add_argument("--camera", type=int, default=0)
    return p


def _load_config(args: argparse.Namespace) -> AppConfig:
    if args.config_dir:
        return AppConfig.load(args.config_dir)
    return AppConfig.load()


def _load_prompts(args: argparse.Namespace, cfg: AppConfig) -> list[str]:
    if args.prompts and args.prompts.exists():
        return [
            line.strip()
            for line in args.prompts.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    from .config import load_prompts
    cfg_dir = args.config_dir or Path(__file__).resolve().parent.parent / "configs"
    return load_prompts(cfg_dir)


# ── modes ───────────────────────────────────────────────────────────────────

def run_sandbox(args: argparse.Namespace, cfg: AppConfig) -> int:
    runtime = PetRuntime()
    if args.target:
        x, y, z = args.target
        log.info("moving pet to (%.3f, %.3f, %.3f)", x, y, z)
        runtime.move_to(x, y, z)
    elif args.script:
        for line in args.script.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            action = json.loads(line)
            log.info("script action: %s", action.get("action"))
            runtime.apply(action)
            time.sleep(0.5)
    else:
        runtime.play_animation("idle")
    log.info("final state: %s", runtime.state.model_dump_json())

    if args.serve:
        _serve(cfg)
    return 0


def run_snapshot(args: argparse.Namespace, cfg: AppConfig) -> int:
    if not args.image:
        log.error("snapshot mode requires --image")
        return 2
    if not args.image.exists():
        log.error("image not found: %s", args.image)
        return 2

    from .camera_service.image_reader import read_image
    from .perception.pipeline import PerceptionPipeline

    prompts = _load_prompts(args, cfg)
    frame = read_image(args.image)
    log.info("loaded %s (shape=%s)", args.image, frame.shape)

    pipeline = PerceptionPipeline(cfg)
    args.out.mkdir(parents=True, exist_ok=True)
    out_json = args.out / f"snapshot_{args.image.stem}.json"

    if args.lift or args.track:
        from PIL import Image

        from .spatial import CameraIntrinsics, FixedPoseSource, SemanticMap
        from .tracking import Tracker

        intrinsics = CameraIntrinsics.from_fov(
            image_size=frame.shape[:2], horizontal_fov_deg=args.fov
        )

        if args.track:
            tracker = Tracker(
                min_iou=cfg.thresholds.tracking.min_iou,
                max_center_distance=cfg.thresholds.tracking.max_center_distance,
                persistence_frames=cfg.thresholds.tracking.persistence_frames,
            )
            smap = SemanticMap(map_id=f"snapshot_{args.image.stem}")
            n_frames = max(1, args.frames)
            for fi in range(n_frames):
                result, depth, tracked = pipeline.run_frame_tracked(
                    frame,
                    prompts=prompts,
                    tracker=tracker,
                    semantic_map=smap,
                    frame_id=fi,
                    intrinsics=intrinsics,
                    pose_source=FixedPoseSource(),
                    save_masks=(fi == 0),  # masks are deterministic; write once
                )
            lifted = smap.values()
        else:
            result, depth, lifted = pipeline.run_frame_3d(
                frame,
                prompts=prompts,
                frame_id=0,
                intrinsics=intrinsics,
                pose_source=FixedPoseSource(),
            )
            smap = None

        out_json.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        lifted_json = args.out / f"lifted_{args.image.stem}.json"
        lifted_json.write_text(
            json.dumps(
                {
                    "frame_id": result.frame_id,
                    "image_size": list(result.image_size),
                    "intrinsics": intrinsics.model_dump(),
                    "objects": [o.to_dict() for o in lifted],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if smap is not None:
            map_json = args.out / f"semantic_map_{args.image.stem}.json"
            smap.save(map_json)
            log.info("wrote %s (%d objects)", map_json, len(smap.objects))
        log.info(
            "wrote %s (%d objects) + %s (%d lifted)",
            out_json, len(result.objects_2d), lifted_json, len(lifted),
        )
        if cfg.runtime.runtime.save_debug_outputs:
            viz = pipeline.visualize(frame, result)
            viz_path = args.out / f"snapshot_{args.image.stem}.png"
            Image.fromarray(viz).save(viz_path)
            depth_path = args.out / f"depth_{args.image.stem}.png"
            Image.fromarray(pipeline.colorize_depth(depth)).save(depth_path)
            log.info("wrote %s + %s", viz_path, depth_path)
        return 0

    result = pipeline.run_frame(frame, prompts=prompts, frame_id=0)
    out_json.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    log.info("wrote %s (%d objects)", out_json, len(result.objects_2d))
    if cfg.runtime.runtime.save_debug_outputs:
        viz = pipeline.visualize(frame, result)
        viz_path = args.out / f"snapshot_{args.image.stem}.png"
        from PIL import Image
        Image.fromarray(viz).save(viz_path)
        log.info("wrote %s", viz_path)
    return 0


def _serve(cfg: AppConfig) -> None:
    import uvicorn

    from .runtime.websocket_server import app
    log.info("serving on http://%s:%d", cfg.runtime.server.host, cfg.runtime.server.http_port)
    uvicorn.run(
        app,
        host=cfg.runtime.server.host,
        port=cfg.runtime.server.http_port,
        log_level="info",
    )


def run_demo(args: argparse.Namespace, cfg: AppConfig) -> int:  # noqa: ARG001
    log.info("demo mode: starting backend (frontend served via Vite separately)")
    _serve(cfg)
    return 0


def run_not_implemented(mode: str) -> int:
    log.error("mode %r is scaffolded but not implemented in Phase 1–2", mode)
    return 3


# ── entrypoint ──────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    args = build_parser().parse_args(argv)
    cfg = _load_config(args)
    log.info("mode=%s device=%s", args.mode, cfg.settings.device)

    if args.mode == "sandbox":
        return run_sandbox(args, cfg)
    if args.mode == "snapshot":
        return run_snapshot(args, cfg)
    if args.mode == "demo":
        return run_demo(args, cfg)
    return run_not_implemented(args.mode)


if __name__ == "__main__":
    sys.exit(main())
