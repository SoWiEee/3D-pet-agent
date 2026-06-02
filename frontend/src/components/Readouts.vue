<script setup lang="ts">
import { computed } from "vue";
import type { PetState } from "../composables/useWebSocket";

const props = defineProps<{
  state: PetState | null;
}>();

const pos = computed(() => props.state?.position ?? { x: 0, y: 0, z: 0 });

function fmt(n: number) {
  const s = n.toFixed(3);
  return n >= 0 ? "+" + s : s;
}
</script>

<template>
  <section class="readouts">
    <div class="card">
      <div class="card__head">
        <span class="card__num">A</span>
        <span class="card__title">位置（世界座標系）</span>
        <span class="card__hint">公尺 · 相對相機</span>
      </div>
      <div class="vec">
        <div class="vec__row"><em>x</em><b>{{ fmt(pos.x) }}</b></div>
        <div class="vec__row"><em>y</em><b>{{ fmt(pos.y) }}</b></div>
        <div class="vec__row"><em>z</em><b>{{ fmt(pos.z) }}</b></div>
      </div>
      <div class="meta">
        <span class="kv"><em>動畫</em><b>{{ props.state?.animation ?? "—" }}</b></span>
        <span class="kv"><em>情緒</em><b>{{ props.state?.emotion ?? "—" }}</b></span>
        <span class="kv"><em>速度</em><b>{{ (props.state?.speed ?? 0).toFixed(2) }}</b></span>
      </div>
    </div>
  </section>
</template>

<style scoped>
.readouts {
  display: grid;
  grid-template-columns: 360px;
  gap: 14px;
  padding: 14px 18px;
  border-top: 1px solid var(--c-line);
  background: linear-gradient(0deg, rgba(7,9,10,0.88), rgba(11,17,18,0.6));
}
.card {
  border: 1px solid var(--c-line);
  padding: 10px 12px;
  background: rgba(255,255,255,0.012);
}
.card__head {
  display: flex; align-items: baseline; gap: 10px;
  padding-bottom: 6px;
  border-bottom: 1px dashed var(--c-line);
  margin-bottom: 8px;
}
.card__num { color: var(--c-phosphor); font-size: 14px; }
.card__title { font-size: 12px; color: var(--c-bone-dim); }
.card__hint { margin-left: auto; color: var(--c-bone-faint); font-size: 11px; }

.vec { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }
.vec__row {
  display: flex; align-items: baseline; justify-content: space-between;
  border-bottom: 1px dotted var(--c-line);
  padding: 4px 6px;
}
.vec__row em { color: var(--c-bone-faint); font-style: normal; font-size: 11px; }
.vec__row b { color: var(--c-phosphor); font-variant-numeric: tabular-nums; font-weight: 500; }

.meta { display: flex; gap: 16px; margin-top: 10px; flex-wrap: wrap; }
.meta .kv { display: inline-flex; gap: 6px; align-items: baseline; font-size: 11px; }
.meta em { color: var(--c-bone-faint); font-style: normal; font-size: 11px; }
.meta b { color: var(--c-bone); font-weight: 500; }
</style>
