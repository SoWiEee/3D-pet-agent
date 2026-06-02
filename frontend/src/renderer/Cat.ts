/**
 * Procedural ceramic-pearl cat (spec §4).
 *
 * No skeletal rig — every motion is driven analytically from two inputs the
 * backend already sends: `animation` (idle/walk/run/sit/hide/look_at/…) and
 * `emotion` (happy/curious/confused/scared/playful/neutral). A small bag of
 * posture scalars (crouch, pitch, ear-back, head-tilt, tail amplitude, gait …)
 * is resolved from those two strings and then *eased* toward each frame, so the
 * cat transitions smoothly between states instead of snapping.
 *
 * Animation comes from:
 *  - a diagonal four-leg gait while walking/running (hip-pivoted swing),
 *  - genuine sit (rear down, front up, back legs tucked) and hide (crouched
 *    flat, ears pinned, tail tucked) postures,
 *  - per-emotion ear / tail / head-tilt / tremble cues layered on top.
 */
import * as THREE from "three";

/** Eased posture state. Each field is lerped toward its target every frame. */
interface Posture {
  crouch: number; // 0 = standing, 1 = belly to the floor
  pitch: number; // body z-rotation; + lifts the front (sitting)
  earBack: number; // -1 = perked forward, +1 = pinned back
  headTilt: number; // inquisitive head roll (radians)
  tailAmp: number; // tail sway amplitude (radians)
  tailSpeed: number; // tail sway angular speed
  tailTuck: number; // 0 = up/back, 1 = tucked under
  bobAmp: number; // breathing/idle vertical bob amplitude
  bobSpeed: number; // breathing speed
  gait: number; // 0 = still, 1 = walk, >1 = run (leg swing scale)
  gaitSpeed: number; // stride frequency
  tremble: number; // high-frequency fear jitter
  backLegTuck: number; // back legs folded forward (sitting)
}

const NEUTRAL: Posture = {
  crouch: 0,
  pitch: 0,
  earBack: 0,
  headTilt: 0,
  tailAmp: 0.12,
  tailSpeed: 2.5,
  tailTuck: 0,
  bobAmp: 0.012,
  bobSpeed: 1.5,
  gait: 0,
  gaitSpeed: 7,
  tremble: 0,
  backLegTuck: 0,
};

interface LegPart {
  group: THREE.Group;
  isFront: boolean;
  side: 1 | -1; // +1 = +z side, -1 = -z side
}

const EMOTION_COLOR: Record<string, number> = {
  happy: 0xfff2d4,
  curious: 0xe7e1c8,
  confused: 0xcfc8b2,
  scared: 0xc8d6d4,
  playful: 0xffe6c8,
  neutral: 0xe7e1c8,
};

export class Cat {
  group = new THREE.Group();
  private body: THREE.Mesh;
  private head: THREE.Group;
  private headPivot: THREE.Group;
  private earL: THREE.Mesh;
  private earR: THREE.Mesh;
  private tail: THREE.Mesh;
  private legs: LegPart[] = [];
  private bodyMat: THREE.MeshStandardMaterial;
  private accentMat: THREE.MeshStandardMaterial;
  private emotion = "neutral";
  private animation = "idle";
  // Live (eased) posture and its current target.
  private cur: Posture = { ...NEUTRAL };
  private target: Posture = { ...NEUTRAL };

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
    this.body.position.set(0, 0.3, 0);
    this.body.castShadow = true;
    this.body.receiveShadow = true;
    this.group.add(this.body);

    // Head pivot allows look_at without rotating body.
    this.headPivot = new THREE.Group();
    this.headPivot.position.set(0.32, 0.4, 0);
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

    // Ears — kept in small pivots so they can flatten back believably.
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

    // Legs — hip-pivoted so a z-rotation swings the foot fore/aft. The cylinder
    // geometry is shifted down by half its length so each leg group's origin
    // sits at the hip.
    const legLen = 0.22;
    const legGeo = new THREE.CylinderGeometry(0.05, 0.05, legLen, 10);
    legGeo.translate(0, -legLen / 2, 0);
    const hipY = 0.22;
    const legPositions: [number, number][] = [
      [0.18, 0.13],
      [0.18, -0.13],
      [-0.18, 0.13],
      [-0.18, -0.13],
    ];
    for (const [x, z] of legPositions) {
      const legGroup = new THREE.Group();
      legGroup.position.set(x, hipY, z);
      const mesh = new THREE.Mesh(legGeo, this.bodyMat);
      mesh.castShadow = true;
      legGroup.add(mesh);
      this.group.add(legGroup);
      this.legs.push({ group: legGroup, isFront: x > 0, side: z > 0 ? 1 : -1 });
    }

    // Pet faces +x by default.
    this.group.rotation.y = 0;
  }

  setAnimation(name: string) {
    this.animation = name;
    this.target = this.resolveTarget();
  }

  setEmotion(name: string) {
    this.emotion = name;
    this.bodyMat.color.lerp(new THREE.Color(EMOTION_COLOR[name] ?? 0xe7e1c8), 0.5);
    this.target = this.resolveTarget();
  }

  faceTowards(x: number, z: number) {
    const dx = x - this.group.position.x;
    const dz = z - this.group.position.z;
    if (Math.abs(dx) + Math.abs(dz) < 1e-4) return;
    this.group.rotation.y = Math.atan2(dz, dx) * -1; // +x forward
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

  /** Resolve the eased-toward posture from the current animation + emotion. */
  private resolveTarget(): Posture {
    const p: Posture = { ...NEUTRAL };

    // ── base posture from the animation/action ──
    switch (this.animation) {
      case "walk":
        p.gait = 1;
        p.gaitSpeed = 7;
        break;
      case "run":
        p.gait = 1.7;
        p.gaitSpeed = 12;
        p.crouch = 0.06;
        break;
      case "sit":
        p.crouch = 0.32;
        p.pitch = 0.3; // front lifts
        p.backLegTuck = 1;
        p.tailTuck = 0.2;
        break;
      case "hide":
        p.crouch = 0.85;
        p.earBack = 1;
        p.tailTuck = 1;
        p.bobAmp = 0.006;
        break;
      default:
        break; // idle / look_at / curious / confused use NEUTRAL base
    }

    // ── emotion overlay — ears, tail, head tilt, tremble ──
    switch (this.emotion) {
      case "happy":
        p.earBack = Math.min(p.earBack, -0.3);
        p.tailAmp = Math.max(p.tailAmp, 0.4);
        p.tailSpeed = 5;
        p.bobAmp = Math.max(p.bobAmp, 0.02);
        p.bobSpeed = 3.4;
        break;
      case "curious":
        p.earBack = Math.min(p.earBack, -0.5);
        p.headTilt = 0.32;
        break;
      case "confused":
        p.headTilt = -0.34;
        p.tailSpeed = 1.6;
        p.earBack = Math.max(p.earBack, 0.2);
        break;
      case "scared":
        p.earBack = 1;
        p.crouch = Math.max(p.crouch, 0.4);
        p.tailTuck = 1;
        p.tailSpeed = 6;
        p.tremble = 1;
        break;
      case "playful":
        p.tailAmp = Math.max(p.tailAmp, 0.45);
        p.tailSpeed = 6;
        p.bobAmp = Math.max(p.bobAmp, 0.026);
        p.bobSpeed = 4.2;
        break;
      default:
        break; // neutral
    }
    return p;
  }

  update(dt: number, tMs: number) {
    const t = tMs / 1000;
    this.ease(dt);
    const c = this.cur;

    // Crouch lowers the whole cat toward the floor; tremble adds fear jitter.
    const tremble = c.tremble * Math.sin(t * 34) * 0.004;
    this.group.position.y = -c.crouch * 0.16;
    this.group.position.x += tremble;

    // Breathing / idle bob, plus a synced footfall dip while moving.
    const footfall = c.gait > 0 ? Math.abs(Math.sin(t * c.gaitSpeed)) * 0.012 * c.gait : 0;
    this.body.position.y = 0.3 + Math.sin(t * c.bobSpeed) * c.bobAmp - footfall;
    // Sit/run pitch: lift the front of the body.
    this.body.rotation.z = c.pitch;

    // Diagonal four-leg gait: FL+BR swing together, FR+BL on the opposite phase.
    for (const leg of this.legs) {
      if (c.gait > 0.01) {
        const diagonalPhase = leg.isFront === leg.side > 0 ? 0 : Math.PI;
        const swing = Math.sin(t * c.gaitSpeed + diagonalPhase) * 0.42 * c.gait;
        leg.group.rotation.z = swing;
      } else if (!leg.isFront && c.backLegTuck > 0.01) {
        // Sitting: fold the back legs forward under the body.
        leg.group.rotation.z = c.backLegTuck * 1.1;
      } else {
        leg.group.rotation.z *= 0.85; // settle to neutral stand
      }
    }

    // Ears: base perk + emotion flatten + idle flicker.
    const flickerL = Math.sin(t * 3.1) * 0.06 * (1 - c.earBack);
    const flickerR = Math.cos(t * 2.7) * 0.06 * (1 - c.earBack);
    this.earL.rotation.x = c.earBack * 0.7 + flickerL;
    this.earR.rotation.x = c.earBack * 0.7 + flickerR;

    // Head tilt (emotion) on the inner head so it composes with look_at yaw.
    this.head.rotation.z = c.headTilt;

    // Tail: sway + tuck.
    this.tail.rotation.y = Math.sin(t * c.tailSpeed) * c.tailAmp;
    this.tail.rotation.z = 0.4 - c.tailTuck * 0.95;
  }

  /** Frame-rate-independent ease of every posture scalar toward its target. */
  private ease(dt: number) {
    const k = 1 - Math.exp(-dt * 6);
    const cur = this.cur as unknown as Record<string, number>;
    const tgt = this.target as unknown as Record<string, number>;
    for (const key of Object.keys(cur)) {
      cur[key] += (tgt[key] - cur[key]) * k;
    }
    void this.accentMat;
  }
}
