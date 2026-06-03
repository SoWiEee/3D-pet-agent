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
import { buildObjectModel, isBoxyLabel } from "./objectMeshes";
import type { CoveragePayload, OccupancyPayload } from "../composables/useWebSocket";

/** Argument for {@link PetScene.setExplorationGoal}. */
export interface ExplorationGoalView {
  position: [number, number, number];
  kind: string;
  score: number;
}

export interface PetSceneOptions {
  el: HTMLElement;
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
  followPath(path: [number, number, number][], speed = 0.35) {
    if (!path || path.length === 0) return;
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

  private cancelActiveMotion() {
    if (!this.activeMotion) return;
    for (const t of this.activeMotion.tweens) {
      t.stop();
    }
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
    if (dt > 1e-4) {
      const pos = this.avatar.group.position;
      const heading = this.avatar.group.rotation.y;
      const v = pos.distanceTo(this.prevAvatarPos) / dt;
      let dHeading = heading - this.prevAvatarHeading;
      // Wrap to (−π, π] so a ±2π flip doesn't spike ω.
      dHeading = Math.atan2(Math.sin(dHeading), Math.cos(dHeading));
      // rotation.y = −atan2(dz,dx), so world yaw rate is the negated derivative.
      const omega = -dHeading / dt;
      this.robot.setDrive(v, omega);
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

// ─────────────────────────────────────────────────────────────────────────────
// World objects layer — phosphor dot + thin vertical stalk + class label for
// each lifted object centroid. Spec §6 acceptance: "debug view shows 3D
// centroids in the Three.js scene as small dots aligned with the camera view".
// ─────────────────────────────────────────────────────────────────────────────
type TrackingStatusLite = "tracked" | "occluded" | "stale" | "lost";

// Per-status opacity envelope (dot, halo, stalk). "tracked" is the brightest;
// "lost" is barely visible — a stale track that hasn't been pruned yet.
const STATUS_FADE: Record<TrackingStatusLite, [number, number, number]> = {
  tracked: [0.95, 0.4, 0.55],
  occluded: [0.7, 0.28, 0.4],
  stale: [0.4, 0.18, 0.22],
  lost: [0.2, 0.08, 0.12],
};

class WorldObjectsLayer {
  group = new THREE.Group();
  private items: { group: THREE.Group; mats: THREE.Material[]; geos: THREE.BufferGeometry[] }[] = [];
  // Smaller centroid pin — the real-size box is the dominant visual cue now.
  private dotGeo = new THREE.SphereGeometry(0.012, 12, 10);
  private haloGeo = new THREE.RingGeometry(0.05, 0.06, 32);
  // Floor extent for missing / degenerate `extent_3d` values so a 0-sized
  // box never produces a degenerate mesh.
  private static readonly MIN_EXTENT = 0.02;

  update(
    markers: {
      center: [number, number, number];
      extent?: [number, number, number];
      label: string;
      depth?: number;
      tracking_status?: TrackingStatusLite;
    }[],
  ) {
    // Reset (dispose per-marker materials + per-marker stalk geometry; shared
    // dot/halo geos are kept).
    for (const it of this.items) {
      this.group.remove(it.group);
      for (const mat of it.mats) mat.dispose();
      for (const g of it.geos) g.dispose();
    }
    this.items = [];

    // Build new markers.
    for (const m of markers) {
      const [x, y, z] = m.center;
      const [dotOp, haloOp, stalkOp] = STATUS_FADE[m.tracking_status ?? "tracked"];

      const dotMat = new THREE.MeshBasicMaterial({
        color: 0x74f7d0,
        transparent: true,
        opacity: dotOp,
      });
      const stalkMat = new THREE.LineBasicMaterial({
        color: 0x2f6757,
        transparent: true,
        opacity: stalkOp,
      });
      const haloMat = new THREE.MeshBasicMaterial({
        color: 0x74f7d0,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: haloOp,
      });

      const item = new THREE.Group();

      // Class-appropriate model sized from extent_3d so the user sees each
      // object at its actual physical scale relative to the cat — a cup as a
      // cylinder, a ball as a sphere, a plant as pot+foliage, etc.
      const ex = Math.max(m.extent?.[0] ?? 0.08, WorldObjectsLayer.MIN_EXTENT);
      const ey = Math.max(m.extent?.[1] ?? 0.08, WorldObjectsLayer.MIN_EXTENT);
      const ez = Math.max(m.extent?.[2] ?? 0.08, WorldObjectsLayer.MIN_EXTENT);
      // Round / compound shapes carry no bright wireframe (unlike boxes), so
      // they need a more solid fill to read as objects rather than haze.
      const boxy = isBoxyLabel(m.label);
      const fillMat = new THREE.MeshStandardMaterial({
        color: 0x74f7d0,
        transparent: true,
        opacity: dotOp * (boxy ? 0.3 : 0.55),
        roughness: 0.7,
        metalness: 0.0,
        emissive: 0x1f5247,
        emissiveIntensity: 0.6,
        depthWrite: false,
      });
      const model = buildObjectModel(m.label, ex, ey, ez, fillMat);
      // center_3d_world reports the centroid; the model is centred on its local
      // origin, so positioning the node places that centroid on the centre.
      model.node.position.set(x, y, z);
      item.add(model.node);

      // Crisp wireframe edges for box-like shapes so silhouettes stay readable
      // through the fog. Round/compound shapes skip this (edges read as noise).
      let edgesMat: THREE.LineBasicMaterial | null = null;
      if (model.outline) {
        edgesMat = new THREE.LineBasicMaterial({
          color: 0x74f7d0,
          transparent: true,
          opacity: dotOp * 0.7,
        });
        const edges = new THREE.LineSegments(model.outline, edgesMat);
        edges.position.set(x, y, z);
        item.add(edges);
      }

      // Centroid pin (small) — keeps the debug "this is where backend
      // thinks the object centre is" cue.
      const dot = new THREE.Mesh(this.dotGeo, dotMat);
      dot.position.set(x, y, z);
      item.add(dot);

      // Vertical stalk down to the floor for readability when the box is
      // floating in space (e.g. a monitor above the desk plane).
      const stalkGeo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(x, y, z),
        new THREE.Vector3(x, 0.002, z),
      ]);
      item.add(new THREE.Line(stalkGeo, stalkMat));

      // Ground halo at the foot of the stalk.
      const halo = new THREE.Mesh(this.haloGeo, haloMat);
      halo.rotation.x = -Math.PI / 2;
      halo.position.set(x, 0.003, z);
      item.add(halo);

      // Labels live in the SpatialInsightsModal panel — see App.vue.

      const mats: THREE.Material[] = [fillMat, dotMat, stalkMat, haloMat];
      const geos: THREE.BufferGeometry[] = [...model.geometries, stalkGeo];
      if (edgesMat && model.outline) {
        mats.push(edgesMat);
        geos.push(model.outline);
      }

      this.group.add(item);
      this.items.push({ group: item, mats, geos });
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Relation edges layer — thin phosphor segments between two world-object
// centroids, opacity proportional to the relation's score. The backend caps
// the broadcast to its top edges so this layer stays cheap.
// ─────────────────────────────────────────────────────────────────────────────
class RelationEdgesLayer {
  group = new THREE.Group();
  private lines: { line: THREE.Line; mat: THREE.LineBasicMaterial; geo: THREE.BufferGeometry }[] =
    [];

  update(
    edges: {
      from: [number, number, number];
      to: [number, number, number];
      score: number;
    }[],
  ) {
    for (const item of this.lines) {
      this.group.remove(item.line);
      item.mat.dispose();
      item.geo.dispose();
    }
    this.lines = [];
    for (const e of edges) {
      const geo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(...e.from),
        new THREE.Vector3(...e.to),
      ]);
      const mat = new THREE.LineBasicMaterial({
        color: 0x74f7d0,
        transparent: true,
        opacity: Math.max(0.15, Math.min(0.7, e.score * 0.7)),
      });
      const line = new THREE.Line(geo, mat);
      this.group.add(line);
      this.lines.push({ line, mat, geo });
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Planned-path layer — draws the controller's dense trajectory as a glowing
// phosphor polyline with waypoint nodes, sitting just above the floor. Set with
// the path from a `move_follow_path`; cleared (empty array) on arrival.
// ─────────────────────────────────────────────────────────────────────────────
class PlannedPathLayer {
  group = new THREE.Group();
  // A glowing tube (real width — WebGL caps GL_LINES at 1px, which is too faint
  // for a headline overlay) plus endpoint pucks at the start and goal.
  private tube?: THREE.Mesh;
  private ends?: THREE.Points;
  private tubeGeo?: THREE.TubeGeometry;
  private tubeMat?: THREE.MeshBasicMaterial;
  private endGeo?: THREE.BufferGeometry;
  private endMat?: THREE.PointsMaterial;

  set(path: [number, number, number][]) {
    this.dispose();
    if (!path || path.length < 2) return;
    const pts = path.map(([x, y, z]) => new THREE.Vector3(x, Math.max(0.002, y) + 0.05, z));

    const curve = new THREE.CatmullRomCurve3(pts, false, "catmullrom", 0.5);
    const segments = Math.max(32, pts.length * 2);
    this.tubeGeo = new THREE.TubeGeometry(curve, segments, 0.024, 8, false);
    this.tubeMat = new THREE.MeshBasicMaterial({
      color: 0x74f7d0,
      transparent: true,
      opacity: 0.6,
      blending: THREE.AdditiveBlending, // reads as phosphor glow over the dark floor
      depthWrite: false,
    });
    this.tube = new THREE.Mesh(this.tubeGeo, this.tubeMat);
    this.group.add(this.tube);

    // Emphasise the two ends — where the cat starts and where it is headed.
    this.endGeo = new THREE.BufferGeometry().setFromPoints([pts[0], pts[pts.length - 1]]);
    this.endMat = new THREE.PointsMaterial({
      color: 0xaef7e6,
      size: 0.12,
      transparent: true,
      opacity: 0.9,
      sizeAttenuation: true,
      depthWrite: false,
    });
    this.ends = new THREE.Points(this.endGeo, this.endMat);
    this.group.add(this.ends);
  }

  updateAnim(tMs: number) {
    if (this.tubeMat) this.tubeMat.opacity = 0.45 + 0.2 * Math.sin((tMs / 1000) * 4);
  }

  private dispose() {
    if (this.tube) this.group.remove(this.tube);
    if (this.ends) this.group.remove(this.ends);
    this.tubeGeo?.dispose();
    this.tubeMat?.dispose();
    this.endGeo?.dispose();
    this.endMat?.dispose();
    this.tube = undefined;
    this.ends = undefined;
    this.tubeGeo = undefined;
    this.tubeMat = undefined;
    this.endGeo = undefined;
    this.endMat = undefined;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Coverage layer — the exploration CoverageGrid rendered as a floor heatmap via
// a CanvasTexture. Unknown cells read as a faint warm haze; observed cells ramp
// to phosphor green with observation count. One textured plane, so it stays
// cheap even at 120×120. Toggleable debug overlay.
// ─────────────────────────────────────────────────────────────────────────────
class CoverageLayer {
  group = new THREE.Group();
  private mesh?: THREE.Mesh;
  private tex?: THREE.CanvasTexture;
  private mat?: THREE.MeshBasicMaterial;
  private geo?: THREE.PlaneGeometry;

  set(p: CoveragePayload | null) {
    this.dispose();
    if (!p) {
      this.group.visible = false;
      return;
    }
    const { width: w, height: h } = p;
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d")!;
    const img = ctx.createImageData(w, h);

    let maxc = 1;
    for (let gz = 0; gz < h; gz++) {
      for (let gx = 0; gx < w; gx++) maxc = Math.max(maxc, p.cells[gz][gx]);
    }

    for (let gz = 0; gz < h; gz++) {
      // Canvas row 0 is sampled at v=0 (flipY disabled below). World +z grows
      // with gz, but the plane maps local +y → world -z, so flip rows here to
      // keep the heatmap aligned with the world grid.
      const row = h - 1 - gz;
      for (let gx = 0; gx < w; gx++) {
        const c = p.cells[gz][gx];
        const i = (row * w + gx) * 4;
        if (c <= 0) {
          img.data[i] = 44;
          img.data[i + 1] = 32;
          img.data[i + 2] = 22;
          img.data[i + 3] = 64; // faint unknown haze
        } else {
          const t = Math.min(1, c / maxc);
          img.data[i] = Math.round(40 + 60 * t);
          img.data[i + 1] = Math.round(120 + 130 * t);
          img.data[i + 2] = Math.round(108 + 90 * t);
          img.data[i + 3] = Math.round(95 + 130 * t);
        }
      }
    }
    ctx.putImageData(img, 0, 0);

    this.tex = new THREE.CanvasTexture(canvas);
    this.tex.flipY = false;
    this.tex.magFilter = THREE.NearestFilter;
    this.tex.minFilter = THREE.LinearFilter;

    const worldW = w * p.resolution;
    const worldH = h * p.resolution;
    this.geo = new THREE.PlaneGeometry(worldW, worldH);
    this.mat = new THREE.MeshBasicMaterial({
      map: this.tex,
      transparent: true,
      depthWrite: false,
      opacity: 0.82,
    });
    this.mesh = new THREE.Mesh(this.geo, this.mat);
    this.mesh.rotation.x = -Math.PI / 2;
    this.mesh.position.set(p.origin_x + worldW / 2, 0.004, p.origin_z + worldH / 2);
    this.group.add(this.mesh);
    this.group.visible = true;
  }

  private dispose() {
    if (this.mesh) this.group.remove(this.mesh);
    this.geo?.dispose();
    this.mat?.dispose();
    this.tex?.dispose();
    this.mesh = undefined;
    this.geo = undefined;
    this.mat = undefined;
    this.tex = undefined;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Occupancy layer — the planner's OccupancyGrid rendered as a red floor decal
// of blocked cells. Shown transiently when a plan fails so the user can see
// *why* the path was rejected (obstacle inflation, blocked goal/start). Free
// cells are fully transparent. Same CanvasTexture-plane trick as CoverageLayer.
// ─────────────────────────────────────────────────────────────────────────────
class OccupancyLayer {
  group = new THREE.Group();
  private mesh?: THREE.Mesh;
  private tex?: THREE.CanvasTexture;
  private mat?: THREE.MeshBasicMaterial;
  private geo?: THREE.PlaneGeometry;

  set(p: OccupancyPayload | null) {
    this.dispose();
    if (!p) {
      this.group.visible = false;
      return;
    }
    const { width: w, height: h } = p;
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d")!;
    const img = ctx.createImageData(w, h);

    for (let gz = 0; gz < h; gz++) {
      const row = h - 1 - gz; // flipped: plane local +y → world -z
      for (let gx = 0; gx < w; gx++) {
        const blocked = p.data[gz * w + gx] > 0;
        const i = (row * w + gx) * 4;
        img.data[i] = 235;
        img.data[i + 1] = 70;
        img.data[i + 2] = 70;
        img.data[i + 3] = blocked ? 150 : 0; // free cells fully transparent
      }
    }
    ctx.putImageData(img, 0, 0);

    this.tex = new THREE.CanvasTexture(canvas);
    this.tex.flipY = false;
    this.tex.magFilter = THREE.NearestFilter;
    this.tex.minFilter = THREE.LinearFilter;

    const worldW = w * p.resolution;
    const worldH = h * p.resolution;
    this.geo = new THREE.PlaneGeometry(worldW, worldH);
    this.mat = new THREE.MeshBasicMaterial({
      map: this.tex,
      transparent: true,
      depthWrite: false,
      opacity: 0.85,
    });
    this.mesh = new THREE.Mesh(this.geo, this.mat);
    this.mesh.rotation.x = -Math.PI / 2;
    this.mesh.position.set(p.origin[0] + worldW / 2, 0.006, p.origin[1] + worldH / 2);
    this.group.add(this.mesh);
    this.group.visible = true;
  }

  private dispose() {
    if (this.mesh) this.group.remove(this.mesh);
    this.geo?.dispose();
    this.mat?.dispose();
    this.tex?.dispose();
    this.mesh = undefined;
    this.geo = undefined;
    this.mat = undefined;
    this.tex = undefined;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Exploration goal marker — a colour-coded beacon at the planner's chosen
// viewpoint. Colour encodes the goal kind; a floating label shows kind + score.
// Pulses while active; retired on arrival.
// ─────────────────────────────────────────────────────────────────────────────
const GOAL_COLORS: Record<string, number> = {
  inspect_unknown: 0xffb45a,
  search_object: 0x74f7d0,
  verify_stale: 0xff8a8a,
  look_behind: 0xb18cff,
};

class ExplorationGoalMarker {
  group = new THREE.Group();
  private beam: THREE.Mesh;
  private ring: THREE.Mesh;
  private beamMat: THREE.MeshBasicMaterial;
  private ringMat: THREE.MeshBasicMaterial;
  private label?: THREE.Sprite;
  private active = false;

  constructor() {
    this.beamMat = new THREE.MeshBasicMaterial({ color: 0xffb45a, transparent: true, opacity: 0 });
    this.beam = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.018, 1.2, 12), this.beamMat);
    this.beam.position.y = 0.6;
    this.group.add(this.beam);

    this.ringMat = new THREE.MeshBasicMaterial({
      color: 0xffb45a,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0,
    });
    this.ring = new THREE.Mesh(new THREE.RingGeometry(0.18, 0.22, 48), this.ringMat);
    this.ring.rotation.x = -Math.PI / 2;
    this.ring.position.y = 0.006;
    this.group.add(this.ring);

    this.group.visible = false;
  }

  set(goal: ExplorationGoalView) {
    const color = GOAL_COLORS[goal.kind] ?? 0xffb45a;
    this.beamMat.color.setHex(color);
    this.ringMat.color.setHex(color);
    this.group.position.set(goal.position[0], 0, goal.position[2]);

    if (this.label) {
      this.group.remove(this.label);
      this.disposeLabel();
    }
    this.label = makeLabelSprite(`${goal.kind} · ${goal.score.toFixed(2)}`);
    this.label.position.set(0, 1.4, 0);
    this.group.add(this.label);

    this.group.visible = true;
    this.active = true;
  }

  clear() {
    this.active = false;
    this.group.visible = false;
  }

  updateAnim(tMs: number) {
    if (!this.active) return;
    const pulse = 0.45 + 0.35 * Math.sin((tMs / 1000) * 4.5);
    this.beamMat.opacity = 0.5 * pulse;
    this.ringMat.opacity = pulse;
    this.ring.scale.setScalar(1 + 0.15 * Math.sin((tMs / 1000) * 4.5));
  }

  private disposeLabel() {
    if (!this.label) return;
    const mat = this.label.material as THREE.SpriteMaterial;
    mat.map?.dispose();
    mat.dispose();
    this.label = undefined;
  }
}

function makeLabelSprite(text: string): THREE.Sprite {
  const canvas = document.createElement("canvas");
  const dpi = 2;
  canvas.width = 220 * dpi;
  canvas.height = 56 * dpi;
  const ctx = canvas.getContext("2d")!;
  ctx.scale(dpi, dpi);
  ctx.font = '500 14px "JetBrains Mono", monospace';
  const padX = 8;
  const padY = 6;
  const w = Math.min(220, Math.ceil(ctx.measureText(text).width) + padX * 2);
  ctx.clearRect(0, 0, canvas.width / dpi, canvas.height / dpi);
  // Background pill.
  ctx.fillStyle = "rgba(11, 17, 18, 0.85)";
  ctx.strokeStyle = "rgba(116, 247, 208, 0.6)";
  ctx.lineWidth = 1;
  roundRect(ctx, 1, 1, w - 2, 28 - 2, 3, true, true);
  ctx.fillStyle = "#e7e1c8";
  ctx.textBaseline = "middle";
  ctx.fillText(text, padX, padY + 8);

  const texture = new THREE.CanvasTexture(canvas);
  texture.minFilter = THREE.LinearFilter;
  const mat = new THREE.SpriteMaterial({ map: texture, transparent: true });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(0.55, 0.14, 1);
  return sprite;
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number, y: number, w: number, h: number, r: number,
  fill: boolean, stroke: boolean,
) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
  if (fill) ctx.fill();
  if (stroke) ctx.stroke();
}

// ─────────────────────────────────────────────────────────────────────────────
// Target marker — phosphor crosshair that lands at the pet's destination.
// ─────────────────────────────────────────────────────────────────────────────
class TargetMarker {
  group = new THREE.Group();
  private ring: THREE.Mesh;
  private innerRing: THREE.Mesh;
  private spawnAt = -Infinity;

  constructor() {
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x74f7d0,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.0,
    });
    this.ring = new THREE.Mesh(new THREE.RingGeometry(0.25, 0.27, 64), ringMat);
    this.ring.rotation.x = -Math.PI / 2;
    this.group.add(this.ring);

    const innerMat = new THREE.MeshBasicMaterial({
      color: 0xffb45a,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.0,
    });
    this.innerRing = new THREE.Mesh(new THREE.RingGeometry(0.04, 0.06, 32), innerMat);
    this.innerRing.rotation.x = -Math.PI / 2;
    this.group.add(this.innerRing);

    // Crosshair lines.
    const lineMat = new THREE.LineBasicMaterial({
      color: 0x74f7d0,
      transparent: true,
      opacity: 0.0,
    });
    const points = [
      new THREE.Vector3(-0.35, 0.001, 0),
      new THREE.Vector3(0.35, 0.001, 0),
      new THREE.Vector3(0, 0.001, -0.35),
      new THREE.Vector3(0, 0.001, 0.35),
    ];
    const geoH = new THREE.BufferGeometry().setFromPoints([points[0], points[1]]);
    const geoV = new THREE.BufferGeometry().setFromPoints([points[2], points[3]]);
    this.group.add(new THREE.Line(geoH, lineMat));
    this.group.add(new THREE.Line(geoV, lineMat));
    this.group.visible = false;
  }

  placeAt(x: number, y: number, z: number) {
    this.group.position.set(x, Math.max(0.002, y), z);
    this.group.visible = true;
    this.spawnAt = performance.now() / 1000;
  }

  update(_dt: number, tMs: number) {
    if (!this.group.visible) return;
    const t = tMs / 1000;
    const age = t - this.spawnAt;
    const pulse = 0.4 + 0.4 * Math.sin(t * 6);
    const fadeIn = Math.min(1, age / 0.3);
    const fadeOut = age > 3 ? Math.max(0, 1 - (age - 3) / 1.5) : 1;
    const alpha = fadeIn * fadeOut * pulse;
    this.group.traverse((obj) => {
      const m = (obj as THREE.Mesh).material as THREE.Material | undefined;
      if (m && "opacity" in m) (m as THREE.MeshBasicMaterial).opacity = alpha;
    });
    if (fadeOut <= 0) this.group.visible = false;
  }
}
