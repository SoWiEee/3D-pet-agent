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

export type ExplorationGoalKind =
  | "inspect_unknown"
  | "search_object"
  | "verify_stale"
  | "look_behind";

export interface ExplorationGoalPayload {
  kind: ExplorationGoalKind | string;
  target_position_world: Waypoint;
  score: number;
  related_object_id?: string | null;
  explanation?: string;
}

/** Shape of `GET /exploration/coverage` (CoverageGrid.to_dict). */
export interface CoveragePayload {
  resolution: number;
  origin_x: number;
  origin_z: number;
  width: number;
  height: number;
  unobserved_ratio: number;
  cells: number[][]; // [height][width] observation counts
}

/** Shape of `GET /planning/occupancy` (OccupancyGrid.to_dict). */
export interface OccupancyPayload {
  resolution: number;
  origin: [number, number]; // [x, z]
  width: number;
  height: number;
  obstacle_ids: string[];
  data: number[]; // flattened row-major (height*width); 1 = blocked, 0 = free
}

/** One candidate's grounding score breakdown (POST /command). */
export interface GroundingBreakdown {
  object_id: string;
  class_label?: string;
  total: number;
  semantic: number;
  attribute: number;
  relation: number;
  visibility: number;
  feasibility: number;
}

export interface CommandIntentView {
  raw_text?: string;
  intent_type: string;
  target?: { class_label?: string | null; attributes?: string[] } | null;
  spatial_relation?: { relation?: string; anchor?: unknown } | null;
  confidence?: number;
}

/** Parsed POST /command response, surfaced in the reasoning panel. */
export interface CommandResult {
  parsed: boolean;
  intent?: CommandIntentView;
  status?: string;
  explanation?: string;
  candidate_breakdowns?: GroundingBreakdown[];
  weights?: Record<string, number>;
  goal_score?: number;
  planner_status?: string;
  candidates?: [string, number][];
  received_at: number;
  utterance: string;
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
    | "world_update"
    | "pick_object";
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
  exploration_goal?: ExplorationGoalPayload | null;
  // Stage E (§14.5): mobile-manipulator pick — the synthesised grasp + arm
  // primitive sequence for the robot avatar to animate.
  target_object_id?: string | null;
  grasp?: GraspPayload | null;
  manipulation_actions?: ManipulationActionPayload[] | null;
  timestamp?: number;
}

export interface GraspPayload {
  grasp_id: string;
  target_object_id: string;
  grasp_pose_world: { position: [number, number, number]; orientation: number[] };
  approach_vector_world: [number, number, number];
  gripper_width: number;
  confidence: number;
  explanation: string;
}

export interface ManipulationActionPayload {
  action: "reach" | "grasp" | "lift" | "place" | "retract";
  target_pose_world: { position: [number, number, number]; orientation: number[] };
  gripper: "open" | "closed";
  speed: number;
}

export type ConnState = "connecting" | "open" | "closed" | "error";

// Exponential backoff with full jitter. The server replays the pet state and
// the latest `world_update` on every fresh connection (see ws_pet), so a
// reconnect alone restores the scene — the client never has to re-request it.
const RECONNECT_BASE_MS = 500;
const RECONNECT_CAP_MS = 10000;

export function reconnectDelay(attempt: number, rng: () => number = Math.random): number {
  const exp = Math.min(RECONNECT_CAP_MS, RECONNECT_BASE_MS * 2 ** Math.max(0, attempt - 1));
  // Full jitter: random in [exp/2, exp] keeps a floor while spreading retries.
  return Math.round(exp / 2 + rng() * (exp / 2));
}

export function usePetSocket(url: string) {
  const status = ref<ConnState>("closed");
  const lastAction = shallowRef<PetAction | null>(null);
  const petState = shallowRef<PetState | null>(null);
  const history = ref<PetAction[]>([]);
  const reconnectAttempts = ref(0);
  let ws: WebSocket | null = null;
  let timer: number | undefined;
  let disposed = false;

  function scheduleReconnect() {
    if (disposed) return;
    reconnectAttempts.value += 1;
    const delay = reconnectDelay(reconnectAttempts.value);
    if (timer) window.clearTimeout(timer);
    timer = window.setTimeout(connect, delay);
  }

  function connect() {
    if (disposed) return;
    status.value = "connecting";
    ws = new WebSocket(url);

    ws.onopen = () => {
      status.value = "open";
      reconnectAttempts.value = 0;
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
      // onerror is always followed by onclose; let onclose own the retry so we
      // don't schedule two reconnects for one drop.
      status.value = "error";
    };
    ws.onclose = () => {
      status.value = "closed";
      scheduleReconnect();
    };
  }

  function send(payload: PetAction) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(payload));
    }
  }

  connect();

  onBeforeUnmount(() => {
    disposed = true;
    if (timer) window.clearTimeout(timer);
    if (ws) {
      // Drop handlers first so the impending close doesn't schedule a reconnect
      // after the component is gone.
      ws.onclose = null;
      ws.onerror = null;
      ws.close();
    }
  });

  return { status, lastAction, petState, history, reconnectAttempts, send };
}
