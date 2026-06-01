import { onBeforeUnmount, ref, shallowRef } from "vue";

export type Vec3 = { x: number; y: number; z: number };

export interface PetState {
  position: Vec3;
  look_at: Vec3 | null;
  animation: string;
  emotion: string;
  speed: number;
  speech: string | null;
  updated_at: number;
}

export type Waypoint = [number, number, number];

export type TrackingStatus = "tracked" | "occluded" | "stale" | "lost";

export interface WorldObjectMarker {
  object_id: string;
  class_label: string;
  center_3d_world: Waypoint;
  extent_3d?: Waypoint;
  median_depth?: number;
  depth_uncertainty?: number;
  confidence?: number;
  tracking_status?: TrackingStatus;
  last_seen_frame?: number;
}

export type RelationLabel =
  | "left_of"
  | "right_of"
  | "in_front_of"
  | "behind"
  | "above"
  | "below"
  | "near"
  | "far_from"
  | "on_surface"
  | "occluding"
  | "between";

export interface RelationEdge {
  subject: string;
  relation: RelationLabel;
  object: string;
  object_2?: string | null;
  score: number;
  evidence?: Record<string, unknown>;
}

export interface SceneGraphPayload {
  timestamp: number;
  frame_id: number;
  coordinate_frame: string;
  objects: string[];
  relations: RelationEdge[];
}

export interface PetAction {
  action:
    | "move_to"
    | "move_follow_path"
    | "look_at"
    | "play_animation"
    | "set_emotion"
    | "ask"
    | "state"
    | "world_update";
  target_position_3d?: Waypoint | null;
  path?: Waypoint[] | null;
  look_at_object_id?: string | null;
  animation?: string | null;
  emotion?: string | null;
  speed?: number | null;
  speech?: string | null;
  state?: PetState | null;
  world_objects?: WorldObjectMarker[] | null;
  scene_graph?: SceneGraphPayload | null;
  timestamp?: number;
}

export type ConnState = "connecting" | "open" | "closed" | "error";

export function usePetSocket(url: string) {
  const status = ref<ConnState>("closed");
  const lastAction = shallowRef<PetAction | null>(null);
  const petState = shallowRef<PetState | null>(null);
  const history = ref<PetAction[]>([]);
  let ws: WebSocket | null = null;
  let retry = 0;
  let timer: number | undefined;

  function connect() {
    status.value = "connecting";
    ws = new WebSocket(url);

    ws.onopen = () => {
      status.value = "open";
      retry = 0;
    };
    ws.onmessage = (ev) => {
      try {
        const action: PetAction = JSON.parse(ev.data);
        lastAction.value = action;
        if (action.state) petState.value = action.state;
        history.value.push(action);
        if (history.value.length > 60) history.value.splice(0, history.value.length - 60);
      } catch (e) {
        console.warn("bad ws message", e);
      }
    };
    ws.onerror = () => {
      status.value = "error";
    };
    ws.onclose = () => {
      status.value = "closed";
      retry += 1;
      const delay = Math.min(8000, 600 * retry);
      timer = window.setTimeout(connect, delay);
    };
  }

  function send(payload: PetAction) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(payload));
    }
  }

  connect();

  onBeforeUnmount(() => {
    if (timer) window.clearTimeout(timer);
    ws?.close();
  });

  return { status, lastAction, petState, history, send };
}
