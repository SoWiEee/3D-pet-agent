/**
 * Stylized differential-drive robot avatar — spec §14.5 Stage E.
 *
 * Mirrors the `Cat` surface (`group` / `setAnimation` / `setEmotion` /
 * `faceTowards` / `lookAtPoint` / `update`) so `PetScene` can route the same
 * `move_follow_path` traversal to whichever avatar is active. Adds two
 * robot-specific behaviours:
 *
 *  - **Differential wheels**: `setDrive(v, ω)` (fed each frame by PetScene from
 *    the base's measured linear speed + yaw rate) rolls the left/right wheel
 *    groups at `v ∓ ω·track/2`, so outer wheels spin faster on a turn and inner
 *    wheels can counter-rotate on a pivot — the same unicycle model `control/`
 *    uses, not a faked spin.
 *  - **Arm pick**: `playPickSequence(tweens, onDone)` lowers the 2-segment arm,
 *    closes the gripper, attaches a carried block, and lifts (lift-and-hold).
 *
 * Local frame: forward = +X, up = +Y, left-right axle = ±Z (matches
 * `faceTowards`, which sets `rotation.y = atan2(dz, dx)·-1`).
 */

import * as THREE from "three";
import { Easing, Group as TweenGroup, Tween } from "@tweenjs/tween.js";

const WHEEL_RADIUS = 0.12;
const TRACK_WIDTH = 0.44; // axle length (wheel separation), metres

// Arm rest / extreme joint angles (radians, about the local Z axle).
const SHOULDER_REST = -0.5;
const SHOULDER_DOWN = 1.7; // added when reaching toward the floor
const SHOULDER_LIFT = 1.1; // subtracted when lifting the held object
const ELBOW_REST = 0.7;
const ELBOW_BEND = 0.5;
const ELBOW_RAISE = 0.6;

interface ArmPose {
  reach: number; // 0 = stowed, 1 = fingertip near floor
  grip: number; // 0 = open, 1 = closed (+ carried block faded in)
  lift: number; // 0 = low, 1 = raised with the object
}

export class Robot {
  group = new THREE.Group();

  private leftWheels: THREE.Mesh[] = [];
  private rightWheels: THREE.Mesh[] = [];
  private shoulder: THREE.Group;
  private elbow: THREE.Group;
  private fingerL: THREE.Mesh;
  private fingerR: THREE.Mesh;
  private carried: THREE.Mesh;
  private bodyMat: THREE.MeshStandardMaterial;

  private driveV = 0;
  private driveOmega = 0;
  private pose: ArmPose = { reach: 0, grip: 0, lift: 0 };
  private bobPhase = 0;

  constructor() {
    this.bodyMat = new THREE.MeshStandardMaterial({
      color: 0x9fb2c4,
      metalness: 0.55,
      roughness: 0.4,
    });
    const accentMat = new THREE.MeshStandardMaterial({
      color: 0xf2a33c,
      metalness: 0.3,
      roughness: 0.5,
    });
    const tyreMat = new THREE.MeshStandardMaterial({ color: 0x1c1f24, roughness: 0.85 });
    const hubMat = new THREE.MeshStandardMaterial({ color: 0xd8dde3, metalness: 0.7, roughness: 0.3 });

    // ── chassis ────────────────────────────────────────────────────────────
    const chassis = new THREE.Mesh(new THREE.BoxGeometry(0.62, 0.18, 0.42), this.bodyMat);
    chassis.position.y = 0.22;
    chassis.castShadow = true;
    this.group.add(chassis);

    // A sloped front + sensor strip so "forward" (+X) reads clearly.
    const visor = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.07, 0.34), accentMat);
    visor.position.set(0.3, 0.27, 0);
    this.group.add(visor);

    // ── wheels (axle along ±Z) ───────────────────────────────────────────
    const wheelGeo = new THREE.CylinderGeometry(WHEEL_RADIUS, WHEEL_RADIUS, 0.06, 20);
    wheelGeo.rotateX(Math.PI / 2); // bake axis Y → Z so rolling = rotation.z
    const spokeGeo = new THREE.BoxGeometry(WHEEL_RADIUS * 1.6, 0.02, 0.065);
    for (const sx of [0.22, -0.22]) {
      for (const sz of [0.24, -0.24]) {
        const wheel = new THREE.Mesh(wheelGeo, tyreMat);
        wheel.position.set(sx, WHEEL_RADIUS, sz);
        wheel.castShadow = true;
        // A spoke bar so the spin is visible.
        const spoke = new THREE.Mesh(spokeGeo, hubMat);
        wheel.add(spoke);
        this.group.add(wheel);
        (sz > 0 ? this.leftWheels : this.rightWheels).push(wheel);
      }
    }

    // ── 2-segment arm on top, shoulder near the back ──────────────────────
    this.shoulder = new THREE.Group();
    this.shoulder.position.set(-0.12, 0.31, 0);
    this.group.add(this.shoulder);
    const upper = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.06, 0.06), this.bodyMat);
    upper.position.set(0.17, 0, 0); // extends along the shoulder's local +X
    this.shoulder.add(upper);

    this.elbow = new THREE.Group();
    this.elbow.position.set(0.34, 0, 0);
    this.shoulder.add(this.elbow);
    const fore = new THREE.Mesh(new THREE.BoxGeometry(0.26, 0.05, 0.05), accentMat);
    fore.position.set(0.13, 0, 0);
    this.elbow.add(fore);

    // Gripper: a base + two fingers at the forearm tip.
    const grip = new THREE.Group();
    grip.position.set(0.26, 0, 0);
    this.elbow.add(grip);
    const palm = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.08, 0.1), hubMat);
    grip.add(palm);
    const fingerGeo = new THREE.BoxGeometry(0.08, 0.02, 0.03);
    this.fingerL = new THREE.Mesh(fingerGeo, hubMat);
    this.fingerL.position.set(0.05, 0, 0.04);
    this.fingerR = new THREE.Mesh(fingerGeo, hubMat);
    this.fingerR.position.set(0.05, 0, -0.04);
    grip.add(this.fingerL, this.fingerR);

    // The block the gripper carries once it grasps — hidden until grasp.
    this.carried = new THREE.Mesh(
      new THREE.BoxGeometry(0.08, 0.08, 0.08),
      new THREE.MeshStandardMaterial({
        color: 0xe7e1c8,
        emissive: 0x6a5a2a,
        emissiveIntensity: 0.4,
        transparent: true,
        opacity: 0,
      }),
    );
    this.carried.position.set(0.09, 0, 0);
    this.carried.visible = false;
    grip.add(this.carried);

    this.applyArmPose();
  }

  // ── Cat-compatible surface ───────────────────────────────────────────────
  setAnimation(_name: string) {
    /* drive state comes from setDrive; nothing posture-specific to do */
  }

  setEmotion(name: string) {
    const tint: Record<string, number> = { happy: 0xa6c8e0, curious: 0xb8c0a0, scared: 0xc4a0a0 };
    this.bodyMat.color.lerp(new THREE.Color(tint[name] ?? 0x9fb2c4), 0.4);
  }

  faceTowards(x: number, z: number) {
    const dx = x - this.group.position.x;
    const dz = z - this.group.position.z;
    if (Math.abs(dx) + Math.abs(dz) < 1e-4) return;
    this.group.rotation.y = Math.atan2(dz, dx) * -1;
  }

  lookAtPoint(_x: number, _y: number, _z: number) {
    /* no articulated head; the visor already marks forward */
  }

  /** Fed each frame by PetScene: base linear speed + yaw rate (world units). */
  setDrive(v: number, omega: number) {
    this.driveV = v;
    this.driveOmega = omega;
  }

  update(dt: number, _tMs: number) {
    // Differential wheel speeds: v_side = v ∓ ω·track/2, ω_wheel = v_side / r.
    const half = (this.driveOmega * TRACK_WIDTH) / 2;
    const dL = ((this.driveV - half) / WHEEL_RADIUS) * dt;
    const dR = ((this.driveV + half) / WHEEL_RADIUS) * dt;
    for (const w of this.leftWheels) w.rotation.z += dL;
    for (const w of this.rightWheels) w.rotation.z += dR;

    // Subtle idle bob so a parked robot still feels alive.
    this.bobPhase += dt * 2;
    this.group.position.y = Math.sin(this.bobPhase) * 0.004;
  }

  // ── arm pick (spec §14.5 Stage E) ────────────────────────────────────────
  /**
   * Lower the arm, close the gripper, attach the carried block, and lift.
   * Tweens are owned by PetScene (passed in) so they advance on the shared
   * render clock, exactly like `followPath`.
   */
  playPickSequence(tweens: TweenGroup, onDone?: () => void) {
    this.pose = { reach: 0, grip: 0, lift: 0 };
    this.carried.visible = true;

    const reach = new Tween(this.pose, tweens)
      .to({ reach: 1 }, 750)
      .easing(Easing.Quadratic.InOut)
      .onUpdate(() => this.applyArmPose());
    const grasp = new Tween(this.pose, tweens)
      .to({ grip: 1 }, 350)
      .easing(Easing.Quadratic.Out)
      .onUpdate(() => this.applyArmPose());
    const lift = new Tween(this.pose, tweens)
      .to({ lift: 1 }, 750)
      .easing(Easing.Quadratic.InOut)
      .onUpdate(() => this.applyArmPose())
      .onComplete(() => onDone?.());

    reach.chain(grasp);
    grasp.chain(lift);
    reach.start();
  }

  /** Reset the arm to stowed + drop the carried block (used on mode switch). */
  resetArm() {
    this.pose = { reach: 0, grip: 0, lift: 0 };
    this.carried.visible = false;
    this.applyArmPose();
  }

  private applyArmPose() {
    const p = this.pose;
    this.shoulder.rotation.z = SHOULDER_REST + p.reach * SHOULDER_DOWN - p.lift * SHOULDER_LIFT;
    this.elbow.rotation.z = ELBOW_REST + p.reach * ELBOW_BEND - p.lift * ELBOW_RAISE;
    // Fingers close inward along local Z as grip → 1.
    const open = 0.04 * (1 - p.grip) + 0.012;
    this.fingerL.position.z = open;
    this.fingerR.position.z = -open;
    const mat = this.carried.material as THREE.MeshStandardMaterial;
    mat.opacity = p.grip;
    this.carried.visible = p.grip > 0.02;
  }
}
