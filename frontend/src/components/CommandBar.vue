<script setup lang="ts">
import { ref } from "vue";
import type { PetAction } from "../composables/useWebSocket";

const emit = defineEmits<{ (e: "send", payload: PetAction): void }>();

const text = ref("");

function quick(action: PetAction) {
  emit("send", action);
}

function submit() {
  const v = text.value.trim().toLowerCase();
  if (!v) return;

  // "path x1 y1 z1 ; x2 y2 z2 ; ..." — exercises move_follow_path.
  const pathMatch = v.match(/^path\s+(.+)$/);
  if (pathMatch) {
    const waypoints = pathMatch[1].split(";").map((seg) => {
      const parts = seg.trim().split(/\s+/).map(Number);
      return parts.length === 3 ? (parts as [number, number, number]) : null;
    }).filter((w): w is [number, number, number] => w !== null);
    if (waypoints.length >= 2) {
      quick({ action: "move_follow_path", path: waypoints, speed: 0.4 });
      text.value = "";
      return;
    }
  }

  // Tiny command shim (full grounding lives in Phase 6).
  // Patterns: "move x y z" | "look x y z" | "anim <name>" | "emote <name>" | "say <text>"
  const m = v.match(/^(move|look)\s+(-?\d*\.?\d+)\s+(-?\d*\.?\d+)\s+(-?\d*\.?\d+)$/);
  if (m) {
    const [, kind, x, y, z] = m;
    quick({
      action: kind === "move" ? "move_to" : "look_at",
      target_position_3d: [+x, +y, +z],
    });
    text.value = "";
    return;
  }
  const a = v.match(/^anim\s+(\w+)$/);
  if (a) {
    quick({ action: "play_animation", animation: a[1] });
    text.value = "";
    return;
  }
  const e = v.match(/^emote\s+(\w+)$/);
  if (e) {
    quick({ action: "set_emotion", emotion: e[1] });
    text.value = "";
    return;
  }
  const s = v.match(/^say\s+(.+)$/);
  if (s) {
    quick({ action: "ask", speech: s[1] });
    text.value = "";
    return;
  }
  // Default: treat the typed line as speech.
  quick({ action: "ask", speech: text.value });
  text.value = "";
}
</script>

<template>
  <section class="cmd">
    <div class="cmd__chrome">
      <span class="cmd__label">cmd / ttyp0</span>
      <div class="cmd__quick">
        <button @click="quick({ action: 'move_to', target_position_3d: [0.6, 0, 0.8] })">P1 · cup</button>
        <button @click="quick({ action: 'move_to', target_position_3d: [-0.5, 0, 1.0] })">P2 · keyboard</button>
        <button
          @click="quick({
            action: 'move_follow_path',
            path: [
              [0.0, 0, 0.0],
              [0.3, 0, 0.4],
              [0.55, 0, 0.85],
              [0.55, 0, 1.20],
            ],
            speed: 0.45,
          })"
        >path · A*</button>
        <button @click="quick({ action: 'play_animation', animation: 'sit' })">sit</button>
        <button @click="quick({ action: 'play_animation', animation: 'hide' })">hide</button>
        <button @click="quick({ action: 'set_emotion', emotion: 'curious' })">curious</button>
      </div>
    </div>
    <form class="cmd__line" @submit.prevent="submit">
      <span class="cmd__caret">▸</span>
      <input
        v-model="text"
        type="text"
        autocomplete="off"
        spellcheck="false"
        placeholder="move 0.5 0 1.2  ·  path 0 0 0 ; 0.3 0 0.5 ; 0.6 0 1.0  ·  anim sit  ·  emote curious  ·  say hello"
      />
      <button type="submit">transmit</button>
    </form>
  </section>
</template>

<style scoped>
.cmd {
  background: linear-gradient(0deg, rgba(7,9,10,0.95), rgba(11,17,18,0.7));
  border-top: 1px solid var(--c-line);
}
.cmd__chrome {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 18px;
  border-bottom: 1px dashed var(--c-line);
}
.cmd__label {
  font-size: 10px;
  letter-spacing: 0.24em;
  text-transform: uppercase;
  color: var(--c-bone-faint);
}
.cmd__quick { display: flex; gap: 8px; flex-wrap: wrap; }
.cmd__quick button {
  font-size: 10px;
  padding: 4px 8px;
  letter-spacing: 0.16em;
}
.cmd__line {
  display: grid;
  grid-template-columns: 24px 1fr auto;
  align-items: center;
  gap: 8px;
  padding: 8px 18px 12px 18px;
}
.cmd__caret { color: var(--c-phosphor); font-size: 14px; }
.cmd__line input {
  width: 100%;
  background: transparent;
  border: 1px solid var(--c-line);
  padding: 9px 12px;
  letter-spacing: 0.06em;
  color: var(--c-bone);
}
.cmd__line input::placeholder { color: var(--c-bone-faint); }
.cmd__line input:focus { border-color: var(--c-phosphor); }
.cmd__line button { padding: 9px 14px; }
</style>
