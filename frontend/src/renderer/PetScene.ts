/**
 * Three.js scene for the 3D pet sandbox (spec §4).
 *
 * Stage aesthetic: dark obsidian floor with a softly glowing phosphor grid,
 * subtle fog, registration crosshairs at world origin. The cat itself is
 * matte ceramic-pearl — the one warm subject in a cold instrument room.
 */
import * as THREE from "three";
import { Tween, Easing, Group as TweenGroup } from "@tweenjs/tween.js";
import { Cat } from "./Cat";
import { Robot } from "./Robot";
import {
  CoverageLayer,
  ExplorationGoalMarker,
  OccupancyLayer,
  PlannedPathLayer,
  RelationEdgesLayer,
  TargetMarker,
  WorldObjectsLayer,
} from "./sceneLayers";
import type { ExplorationGoalView } from "./sceneLayers";
import type { CoveragePayload, MotionSample, OccupancyPayload } from "../composables/useWebSocket";

// The overlay-layer classes live in ./sceneLayers (code-health split).
// Re-exported so existing importers keep resolving the view type from here.
export type { ExplorationGoalView } from "./sceneLayers";

export interface PetSceneOptions {
  el: HTMLElement;
}

/** Largest index ``i`` with ``cum[i] <= d`` (binary search over arc lengths). */
function sampleIndexAt(cum: number[], d: number): number {
  let lo = 0;
  let hi = cum.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (cum[mid] <= d) lo = mid;
    else hi = mid - 1;
  }
  return Math.min(lo, cum.length - 2 < 0 ? 0 : cum.length - 2);
}

/** Interpolate between two angles along the shortest arc. */
function lerpAngle(a: number, b: number, f: number): number {
  let d = b - a;
  d = Math.atan2(Math.sin(d), Math.cos(d));
  return a + d * f;
}

export class PetScene {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  cat: Cat;
  robot: Robot;
  /** The avatar currently receiving motion (cat by default; robot in Robot Mode). */
  private avatar: Cat | Robot;
  private mode: "cat" | "robot" = "cat";
  /** A pick queued while the base is still driving; fired on arrival. */
  private pendingPick: (() => void) | null = null;
  // Per-frame motion estimate (for differential wheel speeds).
  private prevAvatarPos = new THREE.Vector3();
  private prevAvatarHeading = 0;
  // While a backend motion profile plays, the robot's controls come from the
  // profile (setControl), so the per-frame v/ω estimate is suppressed.
  private profileActive = false;
  targetMarker: TargetMarker;
  worldObjects!: WorldObjectsLayer;
  relationEdges!: RelationEdgesLayer;
  plannedPath!: PlannedPathLayer;
  coverage!: CoverageLayer;
  occupancy!: OccupancyLayer;
  explorationGoal!: ExplorationGoalMarker;
  private clock = new THREE.Clock();
  private tweens = new TweenGroup();
  private activeMotion:
    | { kind: "single" | "path"; tweens: Tween<{ x: number; y: number; z: number }>[] }
    | null = null;
  private gridHelper!: THREE.GridHelper;
  private axes!: THREE.AxesHelper;
  private origin!: THREE.Group;
  private raf = 0;
  private resizeObserver?: ResizeObserver;

  // Scene-editor placement (manual object authoring). When on, a left click
  // that doesn't drag raycasts the floor plane and reports world (x, z).
  private editorMode = false;
  private onFloorClick: ((x: number, z: number) => void) | null = null;
  private raycaster = new THREE.Raycaster();
  private groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
  private pointerMoveDist = 0;

  // Custom drag controls — left button pans, right button orbits. We track
  // the camera as an offset from `cameraTarget` so panning and orbiting
  // share the same anchor: pan moves the target along the camera's right /
  // up plane; orbit rotates around it via spherical coordinates.
  private cameraTarget = new THREE.Vector3(0, 0.4, 0);
  private cameraSpherical = new THREE.Spherical();
  private activeDrag: "pan" | "orbit" | null = null;
  private lastPointer = { x: 0, y: 0 };
  private static readonly PAN_SPEED = 0.0025;     // metres per pixel (scaled by distance)
  private static readonly ORBIT_SPEED = 0.005;    // radians per pixel
  private static readonly MIN_PHI = 0.1;          // keep above horizon to avoid gimbal flip
  private static readonly MAX_PHI = Math.PI / 2 - 0.05;
  private static readonly MIN_RADIUS = 1.2;
  private static readonly MAX_RADIUS = 20.0;

  constructor(private opts: PetSceneOptions) {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x07090a);
    this.scene.fog = new THREE.Fog(0x07090a, 6, 18);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    opts.el.appendChild(this.renderer.domElement);

    const { clientWidth: w, clientHeight: h } = opts.el;
    this.camera = new THREE.PerspectiveCamera(38, w / h, 0.1, 100);
    this.camera.position.set(2.7, 2.1, 3.4);
    this.camera.lookAt(this.cameraTarget);
    // Seed the spherical state from the initial pose so the first orbit drag
    // doesn't snap the camera.
    this.cameraSpherical.setFromVector3(
      this.camera.position.clone().sub(this.cameraTarget),
    );

    this.installDragControls(this.renderer.domElement);
    this.buildEnvironment();
    this.cat = new Cat();
    this.scene.add(this.cat.group);
    this.robot = new Robot();
    this.robot.group.visible = false;
    this.scene.add(this.robot.group);
    this.avatar = this.cat;
    this.targetMarker = new TargetMarker();
    this.scene.add(this.targetMarker.group);
    this.worldObjects = new WorldObjectsLayer();
    this.scene.add(this.worldObjects.group);
    this.relationEdges = new RelationEdgesLayer();
    this.scene.add(this.relationEdges.group);
    // Coverage sits lowest (floor decal); planned path + goal beacon ride above.
    this.coverage = new CoverageLayer();
    this.scene.add(this.coverage.group);
    this.occupancy = new OccupancyLayer();
    this.scene.add(this.occupancy.group);
    this.plannedPath = new PlannedPathLayer();
    this.scene.add(this.plannedPath.group);
    this.explorationGoal = new ExplorationGoalMarker();
    this.scene.add(this.explorationGoal.group);

    this.handleResize();
    this.resizeObserver = new ResizeObserver(() => this.handleResize());
    this.resizeObserver.observe(opts.el);

    this.animate();
  }

  private buildEnvironment() {
    // Hemisphere fill — cool sky, warmer ground bounce.
    const hemi = new THREE.HemisphereLight(0x8fc8ff, 0x4a3a22, 0.55);
    this.scene.add(hemi);

    // Key light — angled, soft shadow.
    const key = new THREE.DirectionalLight(0xfff2d8, 1.6);
    key.position.set(3.5, 6, 2.5);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.camera.left = -4;
    key.shadow.camera.right = 4;
    key.shadow.camera.top = 4;
    key.shadow.camera.bottom = -4;
    key.shadow.bias = -0.0005;
    this.scene.add(key);

    // Rim — phosphor backlight for atmosphere.
    const rim = new THREE.DirectionalLight(0x74f7d0, 0.45);
    rim.position.set(-3, 2, -3);
    this.scene.add(rim);

    // Floor — dark obsidian disk.
    const floorGeo = new THREE.CircleGeometry(7, 96);
    const floorMat = new THREE.MeshStandardMaterial({
      color: 0x0c1112,
      roughness: 0.92,
      metalness: 0.05,
    });
    const floor = new THREE.Mesh(floorGeo, floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    this.scene.add(floor);

    // Phosphor grid.
    this.gridHelper = new THREE.GridHelper(12, 24, 0x2f6757, 0x152622);
    (this.gridHelper.material as THREE.Material).transparent = true;
    (this.gridHelper.material as THREE.Material).opacity = 0.6;
    this.scene.add(this.gridHelper);

    // World axes (red/green/blue).
    this.axes = new THREE.AxesHelper(0.8);
    this.axes.position.y = 0.001;
    this.scene.add(this.axes);

    // Origin registration glyph: thin ring + crosshair.
    this.origin = new THREE.Group();
    const ringGeo = new THREE.RingGeometry(0.36, 0.38, 64);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x74f7d0,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.35,
    });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = 0.002;
    this.origin.add(ring);
    this.scene.add(this.origin);
  }

  // ── public API ────────────────────────────────────────────────────────
  moveTo(x: number, y: number, z: number, speed = 0.8) {
    this.cancelActiveMotion();
    // Manual move — no planned path or exploration goal to show.
    this.plannedPath.set([]);
    this.explorationGoal.clear();
    const start = this.avatar.group.position.clone();
    const dist = start.distanceTo(new THREE.Vector3(x, y, z));
    const duration = Math.max(300, (dist / Math.max(0.2, speed)) * 1000);
    this.targetMarker.placeAt(x, y, z);
    this.avatar.setAnimation("walk");
    // Face the destination.
    this.avatar.faceTowards(x, z);

    const tween = new Tween(start, this.tweens)
      .to({ x, y, z }, duration)
      .easing(Easing.Quadratic.InOut)
      .onUpdate((v) => this.avatar.group.position.set(v.x, v.y, v.z))
      .onComplete(() => {
        this.avatar.setAnimation("idle");
        this.activeMotion = null;
      })
      .start();
    this.activeMotion = { kind: "single", tweens: [tween] };
  }

  /**
   * Traverse a sequence of waypoints with smooth heading. Used by the
   * controller (spec §11) — each waypoint is one chained tween whose duration
   * is sized by segment length and target speed.
   */
  followPath(path: [number, number, number][], speed = 0.35, profile: MotionSample[] | null = null) {
    if (!path || path.length === 0) return;
    // The robot drives like a car. With a backend motion profile it replays the
    // controller's exact (position, heading, v, ω, steer, gear) — reversing and
    // full-lock maneuvers and all. Without one (e.g. a manual move in Robot
    // Mode) it falls back to a smooth continuous sweep.
    if (this.mode === "robot") {
      if (profile && profile.length >= 2) this.followProfile(profile, speed);
      else this.followPathContinuous(path, speed);
      return;
    }
    this.cancelActiveMotion();
    const safeSpeed = Math.max(0.1, speed);
    const cat = this.avatar;

    // Mark the final goal + draw the planned trajectory.
    const goal = path[path.length - 1];
    this.targetMarker.placeAt(goal[0], goal[1], goal[2]);
    this.plannedPath.set(path);
    cat.setAnimation("walk");

    const tweens: Tween<{ x: number; y: number; z: number }>[] = [];
    // Snap to first waypoint if the cat is far from it (handles plan replans).
    const here = cat.group.position;
    const first = new THREE.Vector3(path[0][0], path[0][1], path[0][2]);
    if (here.distanceTo(first) > 0.6) {
      cat.group.position.copy(first);
    }

    let prev = cat.group.position.clone();
    for (let i = 1; i < path.length; i++) {
      const [tx, ty, tz] = path[i];
      const next = new THREE.Vector3(tx, ty, tz);
      const dist = prev.distanceTo(next);
      const duration = Math.max(180, (dist / safeSpeed) * 1000);
      const fromHeading = prev.clone();
      const t = new Tween(fromHeading, this.tweens)
        .to({ x: tx, y: ty, z: tz }, duration)
        .easing(Easing.Quadratic.InOut)
        .onStart(() => cat.faceTowards(tx, tz))
        .onUpdate((v) => cat.group.position.set(v.x, v.y, v.z));
      if (tweens.length > 0) {
        tweens[tweens.length - 1].chain(t);
      }
      tweens.push(t);
      prev = next.clone();
    }
    // Final tween's onComplete returns the cat to idle.
    if (tweens.length > 0) {
      tweens[tweens.length - 1].onComplete(() => {
        cat.setAnimation("idle");
        this.activeMotion = null;
        // Arrived — fade the planned path and retire the goal beacon.
        this.plannedPath.set([]);
        this.explorationGoal.clear();
        // If a pick was queued while driving, run it now that we've arrived.
        if (this.pendingPick) {
          const pick = this.pendingPick;
          this.pendingPick = null;
          pick();
        }
      });
      tweens[0].start();
    }
    this.activeMotion = { kind: "path", tweens };
  }

  /**
   * Continuous, car-like traversal: fit a Catmull-Rom curve through the
   * waypoints and sweep position + heading along it with a single ease, so the
   * robot accelerates once, holds speed through corners (steering its front
   * wheels), and decelerates once at the goal — no per-waypoint stops.
   */
  private followPathContinuous(path: [number, number, number][], speed: number) {
    this.cancelActiveMotion();
    const safeSpeed = Math.max(0.1, speed);
    const avatar = this.avatar;

    const goal = path[path.length - 1];
    this.targetMarker.placeAt(goal[0], goal[1], goal[2]);
    this.plannedPath.set(path);
    avatar.setAnimation("walk");

    // Snap to the path start if we're far from it (handles replans).
    const first = new THREE.Vector3(path[0][0], path[0][1], path[0][2]);
    if (avatar.group.position.distanceTo(first) > 0.6) avatar.group.position.copy(first);

    const points = [avatar.group.position.clone(), ...path.slice(1).map((p) => new THREE.Vector3(p[0], p[1], p[2]))];
    if (points.length < 2) {
      avatar.setAnimation("idle");
      return;
    }
    const curve = new THREE.CatmullRomCurve3(points, false, "catmullrom", 0.5);
    const length = Math.max(0.001, curve.getLength());
    const duration = Math.max(400, (length / safeSpeed) * 1000);

    const cursor = { t: 0 };
    const tan = new THREE.Vector3();
    const tween = new Tween(cursor, this.tweens)
      .to({ t: 1 }, duration)
      .easing(Easing.Quadratic.InOut)
      .onUpdate((c) => {
        const tt = Math.min(1, Math.max(0, c.t));
        const p = curve.getPointAt(tt);
        avatar.group.position.copy(p);
        curve.getTangentAt(tt, tan);
        // Steer toward the curve tangent — heading stays continuous, so the
        // per-frame ω estimate (and thus the front-wheel steer) is smooth.
        avatar.faceTowards(p.x + tan.x, p.z + tan.z);
      })
      .onComplete(() => {
        avatar.setAnimation("idle");
        this.activeMotion = null;
        this.plannedPath.set([]);
        this.explorationGoal.clear();
        if (this.pendingPick) {
          const pick = this.pendingPick;
          this.pendingPick = null;
          pick();
        }
      })
      .start();
    // The cursor tween animates {t}, but activeMotion only ever calls .stop();
    // store it under the shared motion-tween type.
    this.activeMotion = {
      kind: "path",
      tweens: [tween as unknown as Tween<{ x: number; y: number; z: number }>],
    };
  }

  /**
   * §14.5 car kinematics: replay the backend's Reeds-Shepp control profile.
   * Position and chassis heading come straight from the samples (so a reverse
   * segment keeps the car facing its travel-forward heading while moving
   * backward), and each tick's real (v, ω, steer, gear) drives the wheels,
   * steering and reverse lights. Played at constant rate so wheel speed matches
   * ground speed.
   */
  private followProfile(profile: MotionSample[], speed: number) {
    this.cancelActiveMotion();
    const avatar = this.avatar;
    const pts = profile.map((p) => [p.x, 0, p.z] as [number, number, number]);
    const goal = pts[pts.length - 1];
    this.targetMarker.placeAt(goal[0], goal[1], goal[2]);
    this.plannedPath.set(pts);
    avatar.setAnimation("walk");

    // Snap to the profile start if we're far from it (handles replans).
    const first = new THREE.Vector3(profile[0].x, avatar.group.position.y, profile[0].z);
    if (avatar.group.position.distanceTo(first) > 0.6) avatar.group.position.copy(first);

    // Cumulative arc length → constant-rate playback (so v matches motion).
    const cum = [0];
    for (let i = 1; i < profile.length; i++) {
      const dx = profile[i].x - profile[i - 1].x;
      const dz = profile[i].z - profile[i - 1].z;
      cum.push(cum[i - 1] + Math.hypot(dx, dz));
    }
    const total = Math.max(1e-3, cum[cum.length - 1]);
    const duration = Math.max(400, (total / Math.max(0.1, speed)) * 1000);

    this.profileActive = true;
    const cursor = { d: 0 };
    const tween = new Tween(cursor, this.tweens)
      .to({ d: total }, duration)
      .easing(Easing.Linear.None)
      .onUpdate((c) => {
        const i = sampleIndexAt(cum, c.d);
        const j = Math.min(i + 1, profile.length - 1);
        const span = Math.max(1e-6, cum[j] - cum[i]);
        const f = Math.min(1, Math.max(0, (c.d - cum[i]) / span));
        const a = profile[i];
        const b = profile[j];
        avatar.group.position.set(a.x + (b.x - a.x) * f, avatar.group.position.y, a.z + (b.z - a.z) * f);
        // rotation.y = −theta (the body's yaw convention); lerp the short way.
        avatar.group.rotation.y = -lerpAngle(a.theta, b.theta, f);
        const v = a.v + (b.v - a.v) * f;
        const omega = a.omega + (b.omega - a.omega) * f;
        const steer = a.steer + (b.steer - a.steer) * f;
        this.robot.setControl(v, omega, steer, v < -0.02 ? -1 : 1);
      })
      .onComplete(() => {
        this.profileActive = false;
        this.robot.setControl(0, 0, 0, 1);
        avatar.setAnimation("idle");
        this.activeMotion = null;
        this.plannedPath.set([]);
        this.explorationGoal.clear();
        if (this.pendingPick) {
          const pick = this.pendingPick;
          this.pendingPick = null;
          pick();
        }
      })
      .start();
    this.activeMotion = {
      kind: "path",
      tweens: [tween as unknown as Tween<{ x: number; y: number; z: number }>],
    };
  }

  private cancelActiveMotion() {
    if (!this.activeMotion) return;
    for (const t of this.activeMotion.tweens) {
      t.stop();
    }
    this.profileActive = false;
    this.activeMotion = null;
  }

  lookAt(x: number, y: number, z: number) {
    this.avatar.lookAtPoint(x, y, z);
  }

  setEmotion(name: string) {
    this.avatar.setEmotion(name);
  }

  setAnimation(name: string) {
    this.avatar.setAnimation(name);
  }

  /** Swap the active avatar (Robot Mode toggle, spec §14.5 Stage E). The new
   *  avatar inherits the old one's pose so the swap is seamless. */
  setMode(mode: "cat" | "robot") {
    if (mode === this.mode) return;
    this.mode = mode;
    const next: Cat | Robot = mode === "robot" ? this.robot : this.cat;
    const prev = this.avatar;
    next.group.position.copy(prev.group.position);
    next.group.rotation.y = prev.group.rotation.y;
    this.cat.group.visible = mode === "cat";
    this.robot.group.visible = mode === "robot";
    if (mode === "cat") this.robot.resetArm();
    this.avatar = next;
  }

  getMode(): "cat" | "robot" {
    return this.mode;
  }

  /** Stage E: animate the robot arm picking the object at `targetWorld`. If the
   *  base is still driving, queue it to run on arrival. In cat mode it's a
   *  no-op (the cat can't manipulate) beyond a glance at the target. */
  playPick(targetWorld: [number, number, number]) {
    const run = () => {
      if (this.mode !== "robot") {
        this.avatar.lookAtPoint(targetWorld[0], targetWorld[1], targetWorld[2]);
        return;
      }
      this.robot.faceTowards(targetWorld[0], targetWorld[2]);
      this.robot.playPickSequence(this.tweens);
    };
    if (this.activeMotion?.kind === "path") {
      this.pendingPick = run;
    } else {
      run();
    }
  }

  /** Phase 3/4: render lifted + tracked object centroids as phosphor markers.
   *
   * ``tracking_status`` controls the marker's opacity envelope so stale /
   * occluded tracks visibly fade without disappearing — the SemanticMap is
   * authoritative, the renderer is a window onto it.
   */
  setWorldObjects(
    markers: {
      center: [number, number, number];
      label: string;
      depth?: number;
      tracking_status?: "tracked" | "occluded" | "stale" | "lost";
    }[],
  ) {
    this.worldObjects.update(markers);
  }

  /** Phase 5: render top scene-graph edges as faint phosphor lines between
   * marker centroids. Spatial relations come straight from the SemanticMap,
   * so the renderer just draws what the backend ranked. */
  setSceneGraph(
    edges: { from: [number, number, number]; to: [number, number, number]; score: number }[],
  ) {
    this.relationEdges.update(edges);
  }

  /** Show / hide the CoverageGrid heatmap (Phase 9). Pass null to hide. */
  setCoverage(payload: CoveragePayload | null) {
    this.coverage.set(payload);
  }

  /** Show / hide the OccupancyGrid overlay (Phase 7). Used to explain *why* a
   *  plan failed — blocked cells render red. Pass null to hide. */
  setOccupancy(payload: OccupancyPayload | null) {
    this.occupancy.set(payload);
  }

  /** Mark the exploration planner's chosen viewpoint with a colour-coded beacon. */
  setExplorationGoal(goal: ExplorationGoalView) {
    this.explorationGoal.set(goal);
  }

  clearExplorationGoal() {
    this.explorationGoal.clear();
  }

  /** Toggle scene-editor placement. While on, a non-drag left click raycasts
   *  the floor and reports world (x, z) via `onFloorClick`; camera orbit
   *  (right drag) and zoom (wheel) keep working, and left-drag still pans. */
  setEditorMode(on: boolean, onFloorClick?: (x: number, z: number) => void) {
    this.editorMode = on;
    this.onFloorClick = on ? (onFloorClick ?? null) : null;
    this.renderer.domElement.style.cursor = on ? "crosshair" : "";
  }

  /** Raycast a viewport pixel onto the y=0 floor plane → world point, or null. */
  private raycastFloor(clientX: number, clientY: number): THREE.Vector3 | null {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1,
    );
    this.raycaster.setFromCamera(ndc, this.camera);
    const hit = new THREE.Vector3();
    return this.raycaster.ray.intersectPlane(this.groundPlane, hit) ? hit : null;
  }

  // ── camera drag controls ──────────────────────────────────────────────
  /** Wire pointer / contextmenu / wheel listeners on the canvas. Left button
   *  pans the look-at target; right button orbits around it. Wheel zooms
   *  along the current camera-to-target axis. */
  private installDragControls(el: HTMLElement) {
    el.addEventListener("pointerdown", this.onPointerDown);
    el.addEventListener("pointermove", this.onPointerMove);
    el.addEventListener("pointerup", this.onPointerEnd);
    el.addEventListener("pointercancel", this.onPointerEnd);
    el.addEventListener("pointerleave", this.onPointerEnd);
    el.addEventListener("contextmenu", this.onContextMenu);
    el.addEventListener("wheel", this.onWheel, { passive: false });
  }

  private uninstallDragControls(el: HTMLElement) {
    el.removeEventListener("pointerdown", this.onPointerDown);
    el.removeEventListener("pointermove", this.onPointerMove);
    el.removeEventListener("pointerup", this.onPointerEnd);
    el.removeEventListener("pointercancel", this.onPointerEnd);
    el.removeEventListener("pointerleave", this.onPointerEnd);
    el.removeEventListener("contextmenu", this.onContextMenu);
    el.removeEventListener("wheel", this.onWheel);
  }

  private onPointerDown = (e: PointerEvent) => {
    if (e.button === 0) this.activeDrag = "pan";
    else if (e.button === 2) this.activeDrag = "orbit";
    else return;
    this.pointerMoveDist = 0;
    this.lastPointer = { x: e.clientX, y: e.clientY };
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    e.preventDefault();
  };

  private onPointerMove = (e: PointerEvent) => {
    if (!this.activeDrag) return;
    const dx = e.clientX - this.lastPointer.x;
    const dy = e.clientY - this.lastPointer.y;
    this.lastPointer = { x: e.clientX, y: e.clientY };
    this.pointerMoveDist += Math.abs(dx) + Math.abs(dy);
    if (this.activeDrag === "pan") this.pan(dx, dy);
    else this.orbit(dx, dy);
  };

  private onPointerEnd = (e: PointerEvent) => {
    if (!this.activeDrag) return;
    // A left press that barely moved is a click, not a pan — in editor mode
    // that drops an object on the floor under the cursor.
    const wasClick = this.activeDrag === "pan" && this.pointerMoveDist < 6;
    this.activeDrag = null;
    try {
      (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId);
    } catch {
      // ignore — pointer may already have been released
    }
    if (this.editorMode && wasClick && e.type === "pointerup" && this.onFloorClick) {
      const hit = this.raycastFloor(e.clientX, e.clientY);
      if (hit) this.onFloorClick(hit.x, hit.z);
    }
  };

  private onContextMenu = (e: MouseEvent) => {
    // Swallow the native menu so right-click can drive orbiting.
    e.preventDefault();
  };

  private onWheel = (e: WheelEvent) => {
    e.preventDefault();
    // Zoom by adjusting spherical radius — keeps the look-at point fixed.
    const factor = Math.exp(e.deltaY * 0.001);
    this.cameraSpherical.radius = THREE.MathUtils.clamp(
      this.cameraSpherical.radius * factor,
      PetScene.MIN_RADIUS,
      PetScene.MAX_RADIUS,
    );
    this.applyCamera();
  };

  private pan(dx: number, dy: number) {
    // Translate the camera target along the camera-relative right / up
    // basis. Pan speed scales with current zoom so the per-pixel motion
    // feels consistent regardless of distance.
    const distance = this.cameraSpherical.radius;
    const scale = PetScene.PAN_SPEED * distance;
    const right = new THREE.Vector3();
    const up = new THREE.Vector3();
    this.camera.matrixWorld.extractBasis(right, up, new THREE.Vector3());
    const offset = right.multiplyScalar(-dx * scale).add(up.multiplyScalar(dy * scale));
    this.cameraTarget.add(offset);
    this.applyCamera();
  }

  private orbit(dx: number, dy: number) {
    this.cameraSpherical.theta -= dx * PetScene.ORBIT_SPEED;
    this.cameraSpherical.phi = THREE.MathUtils.clamp(
      this.cameraSpherical.phi - dy * PetScene.ORBIT_SPEED,
      PetScene.MIN_PHI,
      PetScene.MAX_PHI,
    );
    this.applyCamera();
  }

  private applyCamera() {
    const offset = new THREE.Vector3().setFromSpherical(this.cameraSpherical);
    this.camera.position.copy(this.cameraTarget).add(offset);
    this.camera.lookAt(this.cameraTarget);
  }

  // ── loop ──────────────────────────────────────────────────────────────
  private handleResize = () => {
    const { clientWidth: w, clientHeight: h } = this.opts.el;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  };

  private animate = () => {
    this.raf = requestAnimationFrame(this.animate);
    const t = performance.now();
    this.tweens.update(t);
    const dt = this.clock.getDelta();
    // Estimate the active avatar's linear speed v and yaw rate ω from this
    // frame's motion, then feed the robot's differential wheels (spec §14.5).
    // Suppressed while a backend motion profile plays — there the controls come
    // straight from the profile (setControl), so estimating would fight it. We
    // still track prev pose so the estimate resumes cleanly afterwards.
    if (dt > 1e-4) {
      const pos = this.avatar.group.position;
      const heading = this.avatar.group.rotation.y;
      if (!this.profileActive) {
        const v = pos.distanceTo(this.prevAvatarPos) / dt;
        let dHeading = heading - this.prevAvatarHeading;
        // Wrap to (−π, π] so a ±2π flip doesn't spike ω.
        dHeading = Math.atan2(Math.sin(dHeading), Math.cos(dHeading));
        // rotation.y = −atan2(dz,dx), so world yaw rate is the negated derivative.
        const omega = -dHeading / dt;
        this.robot.setDrive(v, omega);
      }
      this.prevAvatarPos.copy(pos);
      this.prevAvatarHeading = heading;
    }
    this.cat.update(dt, t);
    this.robot.update(dt, t);
    this.targetMarker.update(dt, t);
    this.plannedPath.updateAnim(t);
    this.explorationGoal.updateAnim(t);
    this.renderer.render(this.scene, this.camera);
  };

  dispose() {
    cancelAnimationFrame(this.raf);
    this.resizeObserver?.disconnect();
    this.uninstallDragControls(this.renderer.domElement);
    this.renderer.dispose();
    this.opts.el.removeChild(this.renderer.domElement);
  }
}
