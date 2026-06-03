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
const TRACK_WIDTH = 0.48; // axle length (left-right wheel separation), metres
const WHEELBASE = 0.44; // front-rear axle separation, metres
const MAX_STEER = 0.55; // max front-wheel steering angle, radians (~31°)
const STEER_SIGN = -1; // calibrates steer direction to the body's yaw convention

// Arm joint angles (radians, pitch about local Z; +z swings the segment from
// forward toward up). The shoulder sits on TOP of the chassis. Tuned so the
// world pitches read: stowed → upper up-back (1.9) + forearm flat over the deck
// (0.1); reach → upper forward-down (−0.5) + forearm steep to the floor (−1.3);
// lift → raised with the held object. (forearm world pitch = shoulder + elbow.)
const SHOULDER_REST = 1.9; // stowed: upper arm up and slightly back
const SHOULDER_DOWN = 2.4; // reach: swing forward-down to the floor
const SHOULDER_LIFT = 1.5; // lift: raise back up with the object
const ELBOW_REST = -1.8; // stowed: fold the forearm flat over the deck
const ELBOW_BEND = 1.0; // reach: straighten the forearm toward the floor
const ELBOW_RAISE = 0.1;

interface ArmPose {
  reach: number; // 0 = stowed, 1 = fingertip near floor
  grip: number; // 0 = open, 1 = closed (+ carried block faded in)
  lift: number; // 0 = low, 1 = raised with the object
}

export class Robot {
  group = new THREE.Group();

  // Each wheel records its roll mesh + side (for differential speed). Front
  // wheels additionally hang under a steer pivot that yaws like car steering.
  private wheels: { mesh: THREE.Mesh; isLeft: boolean }[] = [];
  private frontSteers: THREE.Group[] = [];
  private steerAngle = 0;
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
    // Front wheels (+X) hang under a steer pivot so they yaw when turning, like
    // a car; rear wheels are fixed. All wheels roll via the mesh's local Z.
    const wheelGeo = new THREE.CylinderGeometry(WHEEL_RADIUS, WHEEL_RADIUS, 0.06, 20);
    wheelGeo.rotateX(Math.PI / 2); // bake axis Y → Z so rolling = rotation.z
    const spokeGeo = new THREE.BoxGeometry(WHEEL_RADIUS * 1.6, 0.02, 0.065);
    for (const sx of [0.22, -0.22]) {
      const isFront = sx > 0;
      for (const sz of [0.24, -0.24]) {
        const wheel = new THREE.Mesh(wheelGeo, tyreMat);
        wheel.castShadow = true;
        wheel.add(new THREE.Mesh(spokeGeo, hubMat)); // spoke makes spin visible
        if (isFront) {
          // Steer pivot at the wheel hub; the wheel rolls inside it.
          const steer = new THREE.Group();
          steer.position.set(sx, WHEEL_RADIUS, sz);
          steer.add(wheel);
          this.group.add(steer);
          this.frontSteers.push(steer);
        } else {
          wheel.position.set(sx, WHEEL_RADIUS, sz);
          this.group.add(wheel);
        }
        this.wheels.push({ mesh: wheel, isLeft: sz > 0 });
      }
    }

    // ── 2-segment arm mounted on TOP of the chassis (centred) ─────────────
    // A short mast lifts the shoulder above the deck so the folded arm clears
    // the body and the reach swings down past the wheels to the floor.
    const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.06, 0.12, 12), hubMat);
    mast.position.set(0, 0.37, 0);
    this.group.add(mast);
    this.shoulder = new THREE.Group();
    this.shoulder.position.set(0, 0.42, 0);
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
    for (const w of this.wheels) w.mesh.rotation.z += w.isLeft ? dL : dR;

    // Front-wheel steering (bicycle model): δ = atan(L·ω / v), clamped. A floor
    // on v keeps a near-stationary pivot from saturating instantly; we lerp the
    // angle so the wheels swing rather than snap.
    const vRef = Math.max(Math.abs(this.driveV), 0.18);
    let steer = Math.atan2(this.driveOmega * WHEELBASE, vRef) * STEER_SIGN;
    steer = Math.max(-MAX_STEER, Math.min(MAX_STEER, steer));
    this.steerAngle += (steer - this.steerAngle) * Math.min(1, dt * 9);
    for (const s of this.frontSteers) s.rotation.y = this.steerAngle;

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
    this.shoulder.rotation.z = SHOULDER_REST - p.reach * SHOULDER_DOWN + p.lift * SHOULDER_LIFT;
    this.elbow.rotation.z = ELBOW_REST + p.reach * ELBOW_BEND + p.lift * ELBOW_RAISE;
    // Fingers close inward along local Z as grip → 1.
    const open = 0.04 * (1 - p.grip) + 0.012;
    this.fingerL.position.z = open;
    this.fingerR.position.z = -open;
    const mat = this.carried.material as THREE.MeshStandardMaterial;
    mat.opacity = p.grip;
    this.carried.visible = p.grip > 0.02;
  }
}
