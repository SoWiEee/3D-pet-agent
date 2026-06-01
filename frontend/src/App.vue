<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, shallowRef, watch } from "vue";
import { PetScene } from "./renderer/PetScene";
import {
  usePetSocket,
  type PetAction,
  type SceneGraphPayload,
  type WorldObjectMarker,
} from "./composables/useWebSocket";
import StatusBar from "./components/StatusBar.vue";
import ModulePanel from "./components/ModulePanel.vue";
import Readouts from "./components/Readouts.vue";
import CommandBar from "./components/CommandBar.vue";
import RegistrationMarks from "./components/RegistrationMarks.vue";
import PetSpeech from "./components/PetSpeech.vue";

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

const modules = computed(() => [
  { name: "detector · groundingdino", status: status.value === "open" ? "online" as const : "idle" as const, note: "open-vocab · text-conditioned" },
  { name: "segmenter · sam", status: status.value === "open" ? "online" as const : "idle" as const, note: "promptable · box → mask" },
  { name: "depth · anything v2", status: status.value === "open" ? "online" as const : "idle" as const, note: "monocular · depth-anything-v2-small" },
  { name: "tracker · iou+bytetrack", status: "off" as const, note: "phase 4" },
  { name: "scene graph", status: status.value === "open" ? "online" as const : "idle" as const, note: "phase 5 · 11 relations" },
  { name: "command parser", status: "off" as const, note: "phase 6 · using rule fallback" },
]);

function onCommand(payload: PetAction) {
  send(payload);
}

const speechText = computed(() => petState.value?.speech ?? null);
</script>

<template>
  <main class="app">
    <StatusBar :conn="status" :state="petState" :perception-hz="2" />

    <section class="stage">
      <ModulePanel :modules="modules" />

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
        <PetSpeech :text="speechText" />
      </div>
    </section>

    <Readouts
      :state="petState"
      :history="history"
      :world-objects="worldObjects"
      :scene-graph="sceneGraph"
    />
    <CommandBar @send="onCommand" />
  </main>
</template>

<style scoped>
.app {
  display: grid;
  grid-template-rows: auto 1fr auto auto;
  height: 100vh;
  width: 100vw;
}
.stage {
  display: grid;
  grid-template-columns: 280px 1fr;
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
</style>
