<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, shallowRef, watch } from "vue";
import { PetScene } from "./renderer/PetScene";
import {
  usePetSocket,
  type CommandResult,
  type CoveragePayload,
  type OccupancyPayload,
  type PetAction,
  type SceneGraphPayload,
  type WorldObjectMarker,
} from "./composables/useWebSocket";
import StatusBar from "./components/StatusBar.vue";
import SpatialInsightsModal from "./components/SpatialInsightsModal.vue";
import ReasoningModal from "./components/ReasoningModal.vue";
import EventStreamModal from "./components/EventStreamModal.vue";
import CommandBar from "./components/CommandBar.vue";
import RegistrationMarks from "./components/RegistrationMarks.vue";
import PetSpeech from "./components/PetSpeech.vue";
import EditorPanel from "./components/EditorPanel.vue";

const viewport = ref<HTMLDivElement | null>(null);
let scene: PetScene | null = null;

const wsUrl = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/pet`;
const { status, petState, history, lastAction, send } = usePetSocket(wsUrl);
const worldObjects = shallowRef<WorldObjectMarker[]>([]);
const sceneGraph = shallowRef<SceneGraphPayload | null>(null);

onMounted(() => {
  if (!viewport.value) return;
  scene = new PetScene({ el: viewport.value });
});

onBeforeUnmount(() => {
  if (occupancyTimer) window.clearTimeout(occupancyTimer);
  scene?.dispose();
});

// Drive Three.js from each PetAction broadcast.
watch(lastAction, (a: PetAction | null) => {
  if (!a || !scene) return;
  if (a.action === "move_to" && a.target_position_3d) {
    const [x, y, z] = a.target_position_3d;
    scene.moveTo(x, y, z, a.speed ?? 0.8);
  } else if (a.action === "move_follow_path" && a.path) {
    scene.followPath(a.path, a.speed ?? 0.35);
    if (a.exploration_goal) {
      const g = a.exploration_goal;
      scene.setExplorationGoal({
        position: g.target_position_world,
        kind: g.kind,
        score: g.score,
      });
      // An explore step likely changed coverage — refresh if the overlay is on.
      if (coverageOn.value) void refreshCoverage();
    } else {
      scene.clearExplorationGoal();
    }
  } else if (a.action === "look_at" && a.target_position_3d) {
    const [x, y, z] = a.target_position_3d;
    scene.lookAt(x, y, z);
  } else if (a.action === "play_animation" && a.animation) {
    scene.setAnimation(a.animation);
  } else if (a.action === "set_emotion" && a.emotion) {
    scene.setEmotion(a.emotion);
  } else if (a.action === "state" && a.state) {
    const p = a.state.position;
    scene.moveTo(p.x, p.y, p.z, 5.0);
    scene.setAnimation(a.state.animation);
    scene.setEmotion(a.state.emotion);
  } else if (a.action === "world_update" && a.world_objects) {
    worldObjects.value = a.world_objects;
    scene.setWorldObjects(
      a.world_objects.map((m) => ({
        center: m.center_3d_world,
        // Forward extent_3d so the renderer can draw a real-size box.
        // Falls back to a small cube when backend hasn't lifted extents.
        extent: m.extent_3d ?? [0.08, 0.08, 0.08],
        label: m.class_label,
        depth: m.median_depth,
        tracking_status: m.tracking_status,
      })),
    );
    if (a.scene_graph) {
      sceneGraph.value = a.scene_graph;
      const byId = new Map(a.world_objects.map((m) => [m.object_id, m.center_3d_world]));
      // Pair edges only — between-edges drawn in panel, not as a single line.
      const topPairs = a.scene_graph.relations
        .filter((r) => r.relation !== "between" && !r.object_2)
        .slice(0, 24);
      scene.setSceneGraph(
        topPairs
          .map((r) => {
            const from = byId.get(r.subject);
            const to = byId.get(r.object);
            if (!from || !to) return null;
            return { from, to, score: r.score };
          })
          .filter((e): e is NonNullable<typeof e> => e !== null),
      );
    } else {
      sceneGraph.value = null;
      scene.setSceneGraph([]);
    }
  }
});

function onCommand(payload: PetAction) {
  send(payload);
}

// ── grounding reasoning panel + path-failure overlay ──────────────────────
const lastReasoning = shallowRef<CommandResult | null>(null);
const PLAN_FAILURE_STATES = new Set(["plan_failed", "no_path", "goal_unreachable", "start_blocked"]);
let occupancyTimer: number | undefined;

async function showOccupancyOverlay() {
  try {
    const r = await fetch("/planning/occupancy");
    if (!r.ok) return;
    const p = (await r.json()) as OccupancyPayload;
    scene?.setOccupancy(p);
    // Auto-retire after 5 s — it's a transient "here's why" cue, not a layer.
    if (occupancyTimer) window.clearTimeout(occupancyTimer);
    occupancyTimer = window.setTimeout(() => scene?.setOccupancy(null), 5000);
  } catch (e) {
    console.warn("occupancy fetch failed", e);
  }
}

function onCommandResult(result: CommandResult) {
  lastReasoning.value = result;
  const failed =
    PLAN_FAILURE_STATES.has(result.status ?? "") ||
    (result.planner_status != null && result.planner_status !== "success");
  if (failed) void showOccupancyOverlay();
}

const speechText = computed(() => petState.value?.speech ?? null);

// Compact pose HUD overlaid bottom-left of the cat viewport.
const pose = computed(() => petState.value?.position ?? { x: 0, y: 0, z: 0 });
function fmtCoord(n: number) {
  const s = n.toFixed(2);
  return n >= 0 ? "+" + s : s;
}

// Map PetState.emotion to a single glyph. Falls back to a generic cat face
// for unknown / null emotions so the HUD never goes empty.
const EMOTION_EMOJI: Record<string, string> = {
  neutral: "😺",
  happy: "😸",
  curious: "🐱",
  confused: "😿",
  scared: "🙀",
  playful: "😻",
};
const emotionEmoji = computed(
  () => EMOTION_EMOJI[petState.value?.emotion ?? "neutral"] ?? "😺",
);

const insightsOpen = ref(false);
const eventsOpen = ref(false);
const reasoningOpen = ref(false);
function toggleInsights() {
  insightsOpen.value = !insightsOpen.value;
}
function toggleEvents() {
  eventsOpen.value = !eventsOpen.value;
}
function toggleReasoning() {
  reasoningOpen.value = !reasoningOpen.value;
}

// CoverageGrid debug overlay — pulled on demand from /exploration/coverage
// (it isn't pushed over the WS stream, so we fetch when the user toggles it on
// and after each exploration step).
const coverageOn = ref(false);
async function refreshCoverage() {
  try {
    const r = await fetch("/exploration/coverage");
    if (!r.ok) return;
    const p = (await r.json()) as CoveragePayload;
    scene?.setCoverage(coverageOn.value ? p : null);
  } catch (e) {
    console.warn("coverage fetch failed", e);
  }
}
function toggleCoverage() {
  coverageOn.value = !coverageOn.value;
  if (coverageOn.value) void refreshCoverage();
  else scene?.setCoverage(null);
}

// ── scene editor: click-to-place objects ──────────────────────────────────
// While editor mode is on, a left click on the floor POSTs an authored object
// to /editor/object; the server broadcasts world_update so the marker appears
// through the normal watch() path. Turning it off restores plain camera control.
const editorMode = ref(false);
const editorLabel = ref("cup");
const placedIds = shallowRef<string[]>([]);

async function placeObject(x: number, z: number) {
  try {
    const r = await fetch("/editor/object", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ class_label: editorLabel.value, x, z }),
    });
    if (!r.ok) return;
    const body = (await r.json()) as { object_id: string };
    placedIds.value = [...placedIds.value, body.object_id];
  } catch (e) {
    console.warn("place object failed", e);
  }
}

function toggleEditor() {
  editorMode.value = !editorMode.value;
  scene?.setEditorMode(editorMode.value, editorMode.value ? placeObject : undefined);
}

async function undoLastPlaced() {
  const id = placedIds.value[placedIds.value.length - 1];
  if (!id) return;
  try {
    const r = await fetch(`/editor/object/${id}`, { method: "DELETE" });
    if (r.ok || r.status === 404) placedIds.value = placedIds.value.slice(0, -1);
  } catch (e) {
    console.warn("undo failed", e);
  }
}

async function clearPlaced() {
  try {
    await fetch("/semantic/reset", { method: "POST" });
    placedIds.value = [];
  } catch (e) {
    console.warn("clear failed", e);
  }
}
</script>

<template>
  <main class="app">
    <StatusBar
      :conn="status"
      :state="petState"
      :perception-hz="2"
      @toggle-insights="toggleInsights"
      @toggle-events="toggleEvents"
      @toggle-reasoning="toggleReasoning"
    />

    <section class="stage">
      <div class="viewport-wrap">
        <div ref="viewport" class="viewport" />
        <RegistrationMarks />
        <div class="vp-label vp-label--tl">
          <span class="vp-label__id">VP/01</span>
          <span class="vp-label__name">tabletop · world frame</span>
        </div>
        <div class="vp-label vp-label--tr">
          <span class="vp-label__id">CAM</span>
          <span class="vp-label__name">(2.7, 2.1, 3.4) m · fov 38°</span>
        </div>
        <div class="vp-label vp-label--br">
          <span class="vp-label__id">GRID</span>
          <span class="vp-label__name">12 m · 0.5 m / div</span>
        </div>

        <!-- Coverage heatmap toggle (exploration debug overlay). -->
        <button
          class="vp-toggle"
          :class="{ 'vp-toggle--on': coverageOn }"
          type="button"
          @click="toggleCoverage"
        >
          ● 覆蓋圖
        </button>

        <!-- Scene editor toggle: switch into click-to-place object authoring. -->
        <button
          class="vp-toggle vp-toggle--editor"
          :class="{ 'vp-toggle--on': editorMode }"
          type="button"
          @click="toggleEditor"
        >
          ✎ 編輯場景
        </button>

        <EditorPanel
          v-if="editorMode"
          v-model:label="editorLabel"
          :placed-count="placedIds.length"
          @undo="undoLastPlaced"
          @clear="clearPlaced"
          @close="toggleEditor"
        />

        <!-- Compact bottom-left HUD: x / y / z · v · emoji. -->
        <div class="vp-hud">
          <span class="vp-hud__pair"><em>x</em><b>{{ fmtCoord(pose.x) }}</b></span>
          <span class="vp-hud__pair"><em>y</em><b>{{ fmtCoord(pose.y) }}</b></span>
          <span class="vp-hud__pair"><em>z</em><b>{{ fmtCoord(pose.z) }}</b></span>
          <span class="vp-hud__pair"><em>v</em><b>{{ (petState?.speed ?? 0).toFixed(2) }}</b></span>
          <span class="vp-hud__emoji" :title="petState?.emotion ?? 'neutral'">{{ emotionEmoji }}</span>
        </div>

        <PetSpeech :text="speechText" />
      </div>
    </section>

    <CommandBar @send="onCommand" @command-result="onCommandResult" />

    <SpatialInsightsModal
      :open="insightsOpen"
      :world-objects="worldObjects"
      :scene-graph="sceneGraph"
      @close="insightsOpen = false"
    />
    <EventStreamModal
      :open="eventsOpen"
      :history="history"
      @close="eventsOpen = false"
    />
    <ReasoningModal
      :open="reasoningOpen"
      :result="lastReasoning"
      @close="reasoningOpen = false"
    />
  </main>
</template>

<style scoped>
.app {
  display: grid;
  /* Rows: StatusBar · viewport (fills) · CommandBar.
     The standalone Readouts row is gone — pose is now overlaid on the
     viewport so the cat panel gets the full vertical space. */
  grid-template-rows: auto 1fr auto;
  height: 100vh;
  width: 100vw;
}
.stage {
  /* Full-width viewport now that the left ModulePanel column is gone. */
  display: grid;
  grid-template-columns: 1fr;
  min-height: 0;
  position: relative;
}
.viewport-wrap {
  position: relative;
  background: #07090a;
  overflow: hidden;
}
.viewport { position: absolute; inset: 0; }

.vp-label {
  position: absolute;
  display: inline-flex;
  gap: 8px;
  align-items: baseline;
  font-size: 10px;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--c-bone-dim);
  background: rgba(7, 9, 10, 0.55);
  padding: 4px 8px;
  border: 1px solid var(--c-line);
  z-index: 6;
}
.vp-label__id { color: var(--c-phosphor); font-family: var(--f-display); }
.vp-label--tl { top: 22px; left: 60px; }
.vp-label--tr { top: 22px; right: 60px; }
.vp-label--br { bottom: 22px; right: 60px; }

/* Compact pose HUD — bottom-left of the viewport. No card chrome. */
.vp-hud {
  position: absolute;
  bottom: 22px;
  left: 60px;
  z-index: 6;
  display: inline-flex;
  gap: 14px;
  align-items: baseline;
  font-size: 12px;
  letter-spacing: 0.02em;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.8);
}
.vp-hud__pair { display: inline-flex; gap: 4px; align-items: baseline; }
.vp-hud__pair em {
  font-style: normal;
  color: var(--c-bone-faint);
  font-size: 11px;
}
.vp-hud__pair b {
  color: var(--c-phosphor);
  font-variant-numeric: tabular-nums;
  font-weight: 500;
}
.vp-hud__emoji {
  font-size: 18px;
  line-height: 1;
  margin-left: 4px;
  filter: drop-shadow(0 1px 2px rgba(0, 0, 0, 0.6));
}

/* Coverage overlay toggle — sits under the CAM label, mirrors label chrome. */
.vp-toggle {
  position: absolute;
  top: 52px;
  right: 60px;
  z-index: 7;
  font-size: 10px;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--c-bone-dim);
  background: rgba(7, 9, 10, 0.55);
  padding: 4px 8px;
  border: 1px solid var(--c-line);
  cursor: pointer;
  transition: color 150ms ease, border-color 150ms ease, background 150ms ease;
}
/* Second toggle stacks directly under the coverage one. */
.vp-toggle--editor { top: 82px; }
.vp-toggle:hover { color: var(--c-bone); border-color: var(--c-phosphor); }
.vp-toggle--on {
  color: var(--c-phosphor);
  border-color: var(--c-phosphor);
  background: rgba(116, 247, 208, 0.08);
}
</style>
