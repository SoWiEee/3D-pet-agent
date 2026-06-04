/**
 * Scene overlay layers for {@link PetScene} (spec §4–§9): world-object markers,
 * relation edges, the planned path, coverage + occupancy grids, the exploration
 * goal beacon and the target marker. Extracted from PetScene.ts so that file
 * stays focused on scene / camera / avatar orchestration (code-health split).
 */
import * as THREE from "three";

import type { CoveragePayload, OccupancyPayload } from "../composables/useWebSocket";
import { buildObjectModel, isBoxyLabel } from "./objectMeshes";

/** Argument for {@link PetScene.setExplorationGoal}. */
export interface ExplorationGoalView {
  position: [number, number, number];
  kind: string;
  score: number;
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

export class WorldObjectsLayer {
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
export class RelationEdgesLayer {
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
export class PlannedPathLayer {
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
export class CoverageLayer {
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
export class OccupancyLayer {
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

export class ExplorationGoalMarker {
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
export class TargetMarker {
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
