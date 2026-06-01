/**
 * Three.js scene for the 3D pet sandbox (spec §4).
 *
 * Stage aesthetic: dark obsidian floor with a softly glowing phosphor grid,
 * subtle fog, registration crosshairs at world origin. The cat itself is
 * matte ceramic-pearl — the one warm subject in a cold instrument room.
 */
import * as THREE from "three";
import { Tween, Easing, Group as TweenGroup } from "@tweenjs/tween.js";

export interface PetSceneOptions {
  el: HTMLElement;
}

export class PetScene {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  cat: Cat;
  targetMarker: TargetMarker;
  worldObjects!: WorldObjectsLayer;
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
    this.camera.lookAt(0, 0.4, 0);

    this.buildEnvironment();
    this.cat = new Cat();
    this.scene.add(this.cat.group);
    this.targetMarker = new TargetMarker();
    this.scene.add(this.targetMarker.group);
    this.worldObjects = new WorldObjectsLayer();
    this.scene.add(this.worldObjects.group);

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
    const start = this.cat.group.position.clone();
    const dist = start.distanceTo(new THREE.Vector3(x, y, z));
    const duration = Math.max(300, (dist / Math.max(0.2, speed)) * 1000);
    this.targetMarker.placeAt(x, y, z);
    this.cat.setAnimation("walk");
    // Face the destination.
    this.cat.faceTowards(x, z);

    const tween = new Tween(start, this.tweens)
      .to({ x, y, z }, duration)
      .easing(Easing.Quadratic.InOut)
      .onUpdate((v) => this.cat.group.position.set(v.x, v.y, v.z))
      .onComplete(() => {
        this.cat.setAnimation("idle");
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
    const cat = this.cat;

    // Mark the final goal.
    const goal = path[path.length - 1];
    this.targetMarker.placeAt(goal[0], goal[1], goal[2]);
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
    this.cat.lookAtPoint(x, y, z);
  }

  setEmotion(name: string) {
    this.cat.setEmotion(name);
  }

  setAnimation(name: string) {
    this.cat.setAnimation(name);
  }

  /** Phase 3: render lifted object centroids as phosphor markers. */
  setWorldObjects(markers: { center: [number, number, number]; label: string; depth?: number }[]) {
    this.worldObjects.update(markers);
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
    this.cat.update(dt, t);
    this.targetMarker.update(dt, t);
    this.renderer.render(this.scene, this.camera);
  };

  dispose() {
    cancelAnimationFrame(this.raf);
    this.resizeObserver?.disconnect();
    this.renderer.dispose();
    this.opts.el.removeChild(this.renderer.domElement);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Placeholder cat. Soft ceramic — capsule body, sphere head, cones for ears,
// curved tail. Has a subtle idle breathing + ear flicker.
// ─────────────────────────────────────────────────────────────────────────────
class Cat {
  group = new THREE.Group();
  private body: THREE.Mesh;
  private head: THREE.Group;
  private headPivot: THREE.Group;
  private earL: THREE.Mesh;
  private earR: THREE.Mesh;
  private tail: THREE.Mesh;
  private bodyMat: THREE.MeshStandardMaterial;
  private accentMat: THREE.MeshStandardMaterial;
  private emotion: string = "neutral";
  private animation: string = "idle";

  constructor() {
    this.bodyMat = new THREE.MeshStandardMaterial({
      color: 0xe7e1c8,
      roughness: 0.55,
      metalness: 0.04,
      emissive: 0x1a1612,
      emissiveIntensity: 0.4,
    });
    this.accentMat = new THREE.MeshStandardMaterial({
      color: 0x2a2a2e,
      roughness: 0.4,
      metalness: 0.1,
    });

    // Body — slightly elongated capsule.
    const bodyGeo = new THREE.CapsuleGeometry(0.22, 0.36, 8, 16);
    bodyGeo.rotateZ(Math.PI / 2);
    this.body = new THREE.Mesh(bodyGeo, this.bodyMat);
    this.body.position.set(0, 0.30, 0);
    this.body.castShadow = true;
    this.body.receiveShadow = true;
    this.group.add(this.body);

    // Head pivot allows look_at without rotating body.
    this.headPivot = new THREE.Group();
    this.headPivot.position.set(0.32, 0.40, 0);
    this.group.add(this.headPivot);

    this.head = new THREE.Group();
    this.headPivot.add(this.head);
    const headGeo = new THREE.SphereGeometry(0.18, 24, 18);
    const headMesh = new THREE.Mesh(headGeo, this.bodyMat);
    headMesh.castShadow = true;
    this.head.add(headMesh);

    // Eyes.
    const eyeGeo = new THREE.SphereGeometry(0.022, 12, 12);
    const eyeMat = new THREE.MeshStandardMaterial({
      color: 0x0a0a0a,
      roughness: 0.2,
      emissive: 0x101010,
    });
    const eyeL = new THREE.Mesh(eyeGeo, eyeMat);
    eyeL.position.set(0.14, 0.04, 0.08);
    this.head.add(eyeL);
    const eyeR = eyeL.clone();
    eyeR.position.z = -0.08;
    this.head.add(eyeR);

    // Nose — tiny phosphor dot, just enough character.
    const noseGeo = new THREE.SphereGeometry(0.014, 8, 8);
    const noseMat = new THREE.MeshBasicMaterial({ color: 0xff8a8a });
    const nose = new THREE.Mesh(noseGeo, noseMat);
    nose.position.set(0.18, -0.02, 0);
    this.head.add(nose);

    // Ears.
    const earGeo = new THREE.ConeGeometry(0.06, 0.12, 12);
    this.earL = new THREE.Mesh(earGeo, this.bodyMat);
    this.earL.position.set(0.05, 0.18, 0.09);
    this.earL.rotation.z = -0.2;
    this.head.add(this.earL);
    this.earR = this.earL.clone();
    this.earR.position.z = -0.09;
    this.head.add(this.earR);

    // Tail — gentle curve via cylinder.
    const tailGeo = new THREE.CylinderGeometry(0.04, 0.018, 0.55, 12);
    this.tail = new THREE.Mesh(tailGeo, this.bodyMat);
    this.tail.position.set(-0.34, 0.45, 0);
    this.tail.rotation.z = 0.4;
    this.tail.castShadow = true;
    this.group.add(this.tail);

    // Legs (4 short cylinders).
    const legGeo = new THREE.CylinderGeometry(0.05, 0.05, 0.22, 10);
    const legPositions: [number, number, number][] = [
      [0.18, 0.11, 0.13],
      [0.18, 0.11, -0.13],
      [-0.18, 0.11, 0.13],
      [-0.18, 0.11, -0.13],
    ];
    for (const [x, y, z] of legPositions) {
      const leg = new THREE.Mesh(legGeo, this.bodyMat);
      leg.position.set(x, y, z);
      leg.castShadow = true;
      this.group.add(leg);
    }

    // Pet faces +x by default.
    this.group.rotation.y = 0;
  }

  setAnimation(name: string) {
    this.animation = name;
  }
  setEmotion(name: string) {
    this.emotion = name;
    // Subtle color shift for emotion.
    const target = new THREE.Color(
      {
        happy: 0xfff2d4,
        curious: 0xe7e1c8,
        confused: 0xcfc8b2,
        scared: 0xc8d6d4,
        playful: 0xffe6c8,
        neutral: 0xe7e1c8,
      }[name] ?? 0xe7e1c8
    );
    this.bodyMat.color.lerp(target, 0.5);
  }
  faceTowards(x: number, z: number) {
    const dx = x - this.group.position.x;
    const dz = z - this.group.position.z;
    if (Math.abs(dx) + Math.abs(dz) < 1e-4) return;
    this.group.rotation.y = Math.atan2(dz, dx) * -1 + 0; // +x forward
  }
  lookAtPoint(x: number, y: number, z: number) {
    const worldPos = new THREE.Vector3();
    this.headPivot.getWorldPosition(worldPos);
    const dir = new THREE.Vector3(x - worldPos.x, y - worldPos.y, z - worldPos.z);
    const yaw = Math.atan2(-dir.z, dir.x);
    const pitch = Math.atan2(dir.y, Math.hypot(dir.x, dir.z));
    this.headPivot.rotation.y = yaw - this.group.rotation.y;
    this.headPivot.rotation.z = pitch * 0.5;
  }

  update(_dt: number, tMs: number) {
    const t = tMs / 1000;
    // Idle breathing — vertical bob on body.
    const bob = Math.sin(t * 1.5) * 0.012;
    this.body.position.y = 0.30 + bob;
    // Ear flicker.
    this.earL.rotation.x = Math.sin(t * 3.1) * 0.08;
    this.earR.rotation.x = Math.cos(t * 2.7) * 0.08;
    // Tail sway.
    const sway = this.animation === "walk" ? 0.35 : 0.12;
    this.tail.rotation.y = Math.sin(t * 2.5) * sway;
    // Walk: gentle pitch.
    if (this.animation === "walk") {
      this.body.rotation.y = Math.sin(t * 8) * 0.04;
    } else {
      this.body.rotation.y *= 0.92;
    }
    void this.emotion;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// World objects layer — phosphor dot + thin vertical stalk + class label for
// each lifted object centroid. Spec §6 acceptance: "debug view shows 3D
// centroids in the Three.js scene as small dots aligned with the camera view".
// ─────────────────────────────────────────────────────────────────────────────
class WorldObjectsLayer {
  group = new THREE.Group();
  private items: THREE.Group[] = [];
  private dotGeo = new THREE.SphereGeometry(0.025, 14, 12);
  private dotMat = new THREE.MeshBasicMaterial({
    color: 0x74f7d0,
    transparent: true,
    opacity: 0.95,
  });
  private stalkMat = new THREE.LineBasicMaterial({
    color: 0x2f6757,
    transparent: true,
    opacity: 0.55,
  });
  private haloGeo = new THREE.RingGeometry(0.05, 0.06, 32);
  private haloMat = new THREE.MeshBasicMaterial({
    color: 0x74f7d0,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.4,
  });

  update(markers: { center: [number, number, number]; label: string; depth?: number }[]) {
    // Reset.
    for (const g of this.items) {
      this.group.remove(g);
      g.traverse((obj) => {
        const m = (obj as THREE.Mesh).material as THREE.Material | undefined;
        if (m && (obj as THREE.Mesh).geometry !== this.dotGeo && (obj as THREE.Mesh).geometry !== this.haloGeo) {
          (obj as THREE.Mesh).geometry?.dispose?.();
        }
      });
    }
    this.items = [];

    // Build new markers. The lifter scales relative-depth into the same
    // world as the cat; for the camera at origin looking down -Z, lifted
    // points end up in front of the cat in graphics-world coordinates.
    for (const m of markers) {
      const [x, y, z] = m.center;
      const item = new THREE.Group();
      const dot = new THREE.Mesh(this.dotGeo, this.dotMat);
      dot.position.set(x, y, z);
      item.add(dot);

      // Vertical stalk down to the floor for readability.
      const stalkGeo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(x, y, z),
        new THREE.Vector3(x, 0.002, z),
      ]);
      item.add(new THREE.Line(stalkGeo, this.stalkMat));

      // Ground halo at the foot of the stalk.
      const halo = new THREE.Mesh(this.haloGeo, this.haloMat);
      halo.rotation.x = -Math.PI / 2;
      halo.position.set(x, 0.003, z);
      item.add(halo);

      // Labels live in the Readouts panel rather than as in-scene sprites —
      // multiple canvas-backed sprites in close proximity proved flaky in
      // headless renderers, and the panel is the canonical place to inspect
      // per-object metrics anyway.
      void m.label;

      this.group.add(item);
      this.items.push(item);
    }
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
