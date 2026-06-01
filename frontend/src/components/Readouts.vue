<script setup lang="ts">
import { computed } from "vue";
import type { PetState, PetAction, WorldObjectMarker } from "../composables/useWebSocket";

const props = defineProps<{
  state: PetState | null;
  history: PetAction[];
  worldObjects?: WorldObjectMarker[];
}>();

const pos = computed(() => props.state?.position ?? { x: 0, y: 0, z: 0 });
const recent = computed(() => props.history.slice(-9).reverse());
const objects = computed(() => props.worldObjects ?? []);

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
        <span class="card__title">pose · world frame</span>
        <span class="card__hint">m · camera-relative</span>
      </div>
      <div class="vec">
        <div class="vec__row"><em>x</em><b>{{ fmt(pos.x) }}</b></div>
        <div class="vec__row"><em>y</em><b>{{ fmt(pos.y) }}</b></div>
        <div class="vec__row"><em>z</em><b>{{ fmt(pos.z) }}</b></div>
      </div>
      <div class="meta">
        <span class="kv"><em>anim</em><b>{{ props.state?.animation ?? "—" }}</b></span>
        <span class="kv"><em>emote</em><b>{{ props.state?.emotion ?? "—" }}</b></span>
        <span class="kv"><em>v</em><b>{{ (props.state?.speed ?? 0).toFixed(2) }}</b></span>
      </div>
    </div>

    <div class="card card--wide">
      <div class="card__head">
        <span class="card__num">B</span>
        <span class="card__title">event stream</span>
        <span class="card__hint">last 9 · newest first</span>
      </div>
      <ul class="ev">
        <li v-for="(ev, i) in recent" :key="i" class="ev__row">
          <span class="ev__t">{{ ((ev.timestamp ?? 0) % 100000).toFixed(2) }}</span>
          <span class="ev__kind">{{ ev.action }}</span>
          <span class="ev__detail">
            <template v-if="ev.action === 'move_to' && ev.target_position_3d">
              → ({{ ev.target_position_3d.map((v) => v.toFixed(2)).join(", ") }})
            </template>
            <template v-else-if="ev.action === 'move_follow_path' && ev.path">
              ↝ {{ ev.path.length }} wp · end ({{ ev.path[ev.path.length - 1].map((v) => v.toFixed(2)).join(", ") }})
            </template>
            <template v-else-if="ev.action === 'play_animation'">{{ ev.animation }}</template>
            <template v-else-if="ev.action === 'set_emotion'">{{ ev.emotion }}</template>
            <template v-else-if="ev.action === 'ask'">"{{ ev.speech }}"</template>
            <template v-else-if="ev.action === 'state'">snapshot</template>
          </span>
        </li>
        <li v-if="recent.length === 0" class="ev__empty">— no events yet —</li>
      </ul>
    </div>

    <div class="card">
      <div class="card__head">
        <span class="card__num">C</span>
        <span class="card__title">world objects</span>
        <span class="card__hint">lifted · phase 3</span>
      </div>
      <ul class="objs">
        <li v-for="o in objects" :key="o.object_id" class="obj">
          <span class="obj__dot" />
          <span class="obj__label">{{ o.class_label }}</span>
          <span class="obj__xyz">
            ({{ o.center_3d_world.map((v) => v.toFixed(2)).join(", ") }})
          </span>
        </li>
        <li v-if="objects.length === 0" class="ev__empty">— no lifted objects —</li>
      </ul>
    </div>
  </section>
</template>

<style scoped>
.readouts {
  display: grid;
  grid-template-columns: 260px 1fr 260px;
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
.card--wide { min-width: 0; }
.card__head {
  display: flex; align-items: baseline; gap: 10px;
  padding-bottom: 6px;
  border-bottom: 1px dashed var(--c-line);
  margin-bottom: 8px;
}
.card__num {
  font-family: var(--f-display);
  color: var(--c-phosphor);
  font-size: 14px;
}
.card__title {
  font-size: 11px; letter-spacing: 0.22em; text-transform: uppercase;
  color: var(--c-bone-dim);
}
.card__hint { margin-left: auto; color: var(--c-bone-faint); font-size: 10px; letter-spacing: 0.1em; }

.vec { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }
.vec__row {
  display: flex; align-items: baseline; justify-content: space-between;
  border-bottom: 1px dotted var(--c-line);
  padding: 4px 6px;
}
.vec__row em { color: var(--c-bone-faint); font-style: normal; font-size: 10px; letter-spacing: 0.18em; text-transform: uppercase; }
.vec__row b { color: var(--c-phosphor); font-variant-numeric: tabular-nums; font-weight: 500; }

.meta { display: flex; gap: 16px; margin-top: 10px; flex-wrap: wrap; }
.meta .kv { display: inline-flex; gap: 6px; align-items: baseline; font-size: 11px; }
.meta em { color: var(--c-bone-faint); font-style: normal; font-size: 10px; text-transform: uppercase; letter-spacing: 0.16em; }
.meta b { color: var(--c-bone); font-weight: 500; }

.ev { list-style: none; margin: 0; padding: 0; max-height: 130px; overflow: auto; }
.ev__row {
  display: grid;
  grid-template-columns: 70px 110px 1fr;
  gap: 10px;
  padding: 3px 0;
  border-bottom: 1px dotted var(--c-line);
  font-size: 11px;
}
.ev__row:last-child { border-bottom: 0; }
.ev__t { color: var(--c-bone-faint); font-variant-numeric: tabular-nums; }
.ev__kind { color: var(--c-phosphor); text-transform: uppercase; letter-spacing: 0.16em; }
.ev__detail { color: var(--c-bone); }
.ev__empty { color: var(--c-bone-faint); padding: 12px 0; text-align: center; letter-spacing: 0.18em; text-transform: uppercase; font-size: 10px; }

.objs { list-style: none; margin: 0; padding: 0; max-height: 130px; overflow: auto; }
.obj {
  display: grid;
  grid-template-columns: 12px 1fr auto;
  align-items: center;
  gap: 8px;
  padding: 3px 0;
  border-bottom: 1px dotted var(--c-line);
  font-size: 11px;
}
.obj:last-child { border-bottom: 0; }
.obj__dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--c-phosphor);
  box-shadow: 0 0 6px var(--c-phosphor);
}
.obj__label { color: var(--c-bone); letter-spacing: 0.08em; }
.obj__xyz { color: var(--c-phosphor); font-variant-numeric: tabular-nums; }
</style>
