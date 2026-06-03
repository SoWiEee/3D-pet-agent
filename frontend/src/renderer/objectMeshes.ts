/**
 * Class-aware object models for the world-object layer.
 *
 * The backend only knows each object's class label + axis-aligned extent, so
 * we can't render a true mesh — but a cup should read as a cup, not a box.
 * This maps the common GroundingDINO classes to believable primitive shapes
 * (cylinders, spheres, compound pot+foliage, etc.) sized to the extent, while
 * genuinely box-like things (book, laptop, table) stay boxes.
 *
 * Every part shares the single translucent phosphor material passed in, so the
 * caller disposes exactly one material per marker; this module only owns the
 * geometries (returned for disposal).
 */
import * as THREE from "three";

export interface ObjectModel {
  /** Group centred on the object centroid (caller positions it in the world). */
  node: THREE.Object3D;
  /** Every geometry created here — the caller disposes these. */
  geometries: THREE.BufferGeometry[];
  /** Edge outline for box-like shapes (null for round/compound shapes, where
   *  wireframe edges read as noise rather than a crisp silhouette). */
  outline: THREE.BufferGeometry | null;
}

type Shape = "sphere" | "cup" | "bottle" | "bowl" | "plant" | "lamp" | "chair" | "box";

function classify(label: string): Shape {
  const l = label.toLowerCase();
  if (/\b(ball|orange|apple|egg|sphere|onion)\b/.test(l)) return "sphere";
  if (/\b(bottle|vase|can|flask)\b/.test(l)) return "bottle";
  if (/\b(cup|mug|glass|tumbler)\b/.test(l)) return "cup";
  if (/\bbowl\b/.test(l)) return "bowl";
  if (/\b(plant|flower|tree|cactus)\b/.test(l)) return "plant";
  if (/\b(lamp|light|torch)\b/.test(l)) return "lamp";
  if (/\b(chair|stool|sofa|couch|seat)\b/.test(l)) return "chair";
  return "box";
}

/** True when the class renders as a plain box (and thus gets a wireframe). */
export function isBoxyLabel(label: string): boolean {
  return classify(label) === "box";
}

/** Build a class-appropriate model centred on the centroid (local origin). */
export function buildObjectModel(
  label: string,
  ex: number,
  ey: number,
  ez: number,
  mat: THREE.Material,
): ObjectModel {
  const node = new THREE.Group();
  const geometries: THREE.BufferGeometry[] = [];
  let outline: THREE.BufferGeometry | null = null;

  const mesh = (geo: THREE.BufferGeometry, y = 0): THREE.Mesh => {
    geometries.push(geo);
    const m = new THREE.Mesh(geo, mat);
    m.position.y = y;
    node.add(m);
    return m;
  };

  switch (classify(label)) {
    case "sphere":
      mesh(new THREE.SphereGeometry(Math.min(ex, ey, ez) / 2, 24, 18));
      break;

    case "cup": {
      const r = Math.min(ex, ez) / 2;
      // Slight taper (narrower base) reads as a cup/mug.
      mesh(new THREE.CylinderGeometry(r, r * 0.82, ey, 24));
      break;
    }

    case "bottle": {
      const r = Math.min(ex, ez) / 2;
      const bodyH = ey * 0.7;
      const neckH = ey - bodyH;
      mesh(new THREE.CylinderGeometry(r, r, bodyH, 20), -neckH / 2);
      mesh(new THREE.CylinderGeometry(r * 0.4, r * 0.5, neckH, 16), bodyH / 2);
      break;
    }

    case "bowl": {
      // Lower hemisphere — an open bowl sitting on the floor.
      const r = Math.min(ex, ez) / 2;
      const g = new THREE.SphereGeometry(r, 24, 12, 0, Math.PI * 2, Math.PI / 2, Math.PI / 2);
      mesh(g, ey / 2 - 0.001);
      break;
    }

    case "plant": {
      // Tapered pot + spherical foliage.
      const r = Math.min(ex, ez) / 2;
      const potH = ey * 0.32;
      mesh(new THREE.CylinderGeometry(r * 0.62, r * 0.5, potH, 18), -ey / 2 + potH / 2);
      mesh(new THREE.SphereGeometry(r * 0.95, 18, 14), potH / 2);
      break;
    }

    case "lamp": {
      const r = Math.min(ex, ez) / 2;
      const baseH = ey * 0.1;
      const poleH = ey * 0.55;
      const shadeH = ey - baseH - poleH;
      mesh(new THREE.CylinderGeometry(r * 0.9, r * 0.95, baseH, 20), -ey / 2 + baseH / 2);
      mesh(new THREE.CylinderGeometry(0.012, 0.012, poleH, 8), -ey / 2 + baseH + poleH / 2);
      mesh(new THREE.ConeGeometry(r, shadeH, 22, 1, true), ey / 2 - shadeH / 2);
      break;
    }

    case "chair": {
      // Seat slab at ~45% height + a backrest filling the upper half along the
      // -Z edge. Legs would be sub-pixel at this scale; skip them.
      const seatH = Math.max(0.04, ey * 0.08);
      const seatY = -ey / 2 + ey * 0.45;
      mesh(new THREE.BoxGeometry(ex, seatH, ez), seatY);
      const backD = Math.max(0.03, ez * 0.12);
      const back = mesh(new THREE.BoxGeometry(ex, ey * 0.5, backD), ey * 0.25);
      back.position.z = -ez / 2 + backD / 2;
      break;
    }

    default: {
      const g = new THREE.BoxGeometry(ex, ey, ez);
      mesh(g);
      outline = new THREE.EdgesGeometry(g);
      break;
    }
  }

  return { node, geometries, outline };
}
