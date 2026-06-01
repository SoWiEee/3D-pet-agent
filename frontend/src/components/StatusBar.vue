<script setup lang="ts">
import { computed } from "vue";
import type { ConnState, PetState } from "../composables/useWebSocket";

const props = defineProps<{
  conn: ConnState;
  state: PetState | null;
  perceptionHz?: number;
}>();

const now = computed(() => {
  const d = new Date();
  return d.toISOString().replace("T", " · ").slice(0, 19) + "Z";
});

const connLabel = computed(() => ({
  connecting: { text: "linking", cls: "tag--amber" },
  open: { text: "online", cls: "tag--phosphor" },
  closed: { text: "offline", cls: "tag--coral" },
  error: { text: "fault", cls: "tag--coral" },
}[props.conn]));
</script>

<template>
  <header class="bar">
    <div class="bar__left">
      <span class="mark">◑</span>
      <span class="brand">3d&nbsp;·&nbsp;pet&nbsp;·&nbsp;agent</span>
      <span class="subbrand">// perceptual&nbsp;console&nbsp;v0.1</span>
    </div>

    <div class="bar__mid">
      <span class="kv"><em>node</em><b>tabletop-01</b></span>
      <span class="kv"><em>backend</em><b>mainline · grounding+sam</b></span>
      <span class="kv"><em>perception&nbsp;hz</em><b>{{ (props.perceptionHz ?? 2).toFixed(1) }}</b></span>
    </div>

    <div class="bar__right">
      <span class="kv"><em>link</em>
        <span class="tag" :class="connLabel.cls">● {{ connLabel.text }}</span>
      </span>
      <span class="kv"><em>state</em><b>{{ props.state?.animation ?? "—" }}</b></span>
      <span class="kv"><em>utc</em><b>{{ now }}</b></span>
    </div>
  </header>
</template>

<style scoped>
.bar {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  align-items: center;
  gap: 24px;
  padding: 10px 18px;
  background: linear-gradient(180deg, rgba(11, 16, 17, 0.95), rgba(11, 16, 17, 0.65));
  border-bottom: 1px solid var(--c-line);
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--c-bone-dim);
  position: relative;
  z-index: 5;
}
.bar__left { display: flex; align-items: baseline; gap: 12px; }
.bar__mid { display: flex; align-items: center; gap: 22px; justify-content: center; }
.bar__right { display: flex; align-items: center; gap: 22px; justify-content: flex-end; }

.mark { color: var(--c-phosphor); font-size: 16px; line-height: 1; }
.brand {
  font-family: var(--f-display);
  color: var(--c-bone);
  letter-spacing: 0.2em;
  font-size: 13px;
}
.subbrand { color: var(--c-bone-faint); }

.kv { display: inline-flex; gap: 8px; align-items: center; }
.kv em { color: var(--c-bone-faint); font-style: normal; font-size: 10px; }
.kv b { color: var(--c-bone); font-weight: 500; }
</style>
