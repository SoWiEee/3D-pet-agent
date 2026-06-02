<script setup lang="ts">
import { computed } from "vue";
import type { ConnState, PetState } from "../composables/useWebSocket";

const props = defineProps<{
  conn: ConnState;
  state: PetState | null;
  perceptionHz?: number;
}>();
const emit = defineEmits<{
  (e: "toggle-insights"): void;
  (e: "toggle-events"): void;
  (e: "toggle-reasoning"): void;
}>();

const now = computed(() => {
  const d = new Date();
  return d.toISOString().replace("T", " · ").slice(0, 19) + "Z";
});

const connLabel = computed(() => ({
  connecting: { text: "連線中", cls: "tag--amber" },
  open: { text: "已連線", cls: "tag--phosphor" },
  closed: { text: "已斷線", cls: "tag--coral" },
  error: { text: "異常", cls: "tag--coral" },
}[props.conn]));
</script>

<template>
  <header class="bar">
    <div class="bar__left">
      <span class="mark">◑</span>
      <span class="brand">3D 寵物代理</span>
    </div>

    <div class="bar__right">
      <button class="insights-btn" type="button" @click="emit('toggle-insights')">
        顯示空間資訊
      </button>
      <button class="insights-btn" type="button" @click="emit('toggle-events')">
        活動紀錄
      </button>
      <button class="insights-btn" type="button" @click="emit('toggle-reasoning')">
        上一次推理
      </button>
      <span class="kv"><em>連線</em>
        <span class="tag" :class="connLabel.cls">● {{ connLabel.text }}</span>
      </span>
      <span class="kv"><em>狀態</em><b>{{ props.state?.animation ?? "—" }}</b></span>
      <span class="kv"><em>時間</em><b>{{ now }}</b></span>
    </div>
  </header>
</template>

<style scoped>
.bar {
  display: grid;
  grid-template-columns: auto 1fr;
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
.bar__right { display: flex; align-items: center; gap: 18px; justify-content: flex-end; }

.insights-btn {
  background: transparent;
  border: 1px solid var(--c-phosphor-dim);
  color: var(--c-phosphor);
  padding: 4px 12px;
  font-family: inherit;
  font-size: 11px;
  letter-spacing: 0.1em;
  cursor: pointer;
  transition: border-color 0.15s, color 0.15s, background 0.15s;
}
.insights-btn:hover {
  border-color: var(--c-phosphor);
  background: rgba(116, 247, 208, 0.08);
}

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
