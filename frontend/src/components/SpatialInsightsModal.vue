<script setup lang="ts">
import { computed, onMounted, onBeforeUnmount, watch } from "vue";
import type {
  SceneGraphPayload,
  WorldObjectMarker,
} from "../composables/useWebSocket";

const props = defineProps<{
  open: boolean;
  worldObjects?: WorldObjectMarker[];
  sceneGraph?: SceneGraphPayload | null;
}>();
const emit = defineEmits<{ (e: "close"): void }>();

const objects = computed(() => props.worldObjects ?? []);
const relations = computed(() => (props.sceneGraph?.relations ?? []).slice(0, 16));

function close() {
  emit("close");
}

function onKey(e: KeyboardEvent) {
  if (e.key === "Escape") close();
}

onMounted(() => window.addEventListener("keydown", onKey));
onBeforeUnmount(() => window.removeEventListener("keydown", onKey));

watch(
  () => props.open,
  (v) => {
    document.body.style.overflow = v ? "hidden" : "";
  },
);
</script>

<template>
  <Transition name="modal">
    <div v-if="props.open" class="overlay" @click.self="close">
      <div class="modal" role="dialog" aria-modal="true" aria-labelledby="insights-title">
        <header class="modal__head">
          <h2 id="insights-title" class="modal__title">空間資訊</h2>
          <button class="modal__close" type="button" aria-label="關閉" @click="close">
            ✕
          </button>
        </header>

        <div class="modal__body">
          <section class="card">
            <div class="card__head">
              <span class="card__num">C</span>
              <span class="card__title">語意地圖</span>
              <span class="card__hint">已追蹤物件</span>
            </div>
            <ul class="objs">
              <li
                v-for="o in objects"
                :key="o.object_id"
                class="obj"
                :class="['obj--' + (o.tracking_status ?? 'tracked')]"
              >
                <span class="obj__dot" />
                <span class="obj__label">
                  <em>{{ o.object_id }}</em>
                  <b>{{ o.class_label }}</b>
                </span>
                <span class="obj__xyz">
                  ({{ o.center_3d_world.map((v) => v.toFixed(2)).join(", ") }})
                </span>
              </li>
              <li v-if="objects.length === 0" class="empty">— 尚無追蹤物件 —</li>
            </ul>
          </section>

          <section class="card">
            <div class="card__head">
              <span class="card__num">D</span>
              <span class="card__title">空間關係</span>
              <span class="card__hint">場景圖</span>
            </div>
            <ul class="rels">
              <li v-for="(r, i) in relations" :key="i" class="rel">
                <span class="rel__subj">{{ r.subject }}</span>
                <span class="rel__rel">{{ r.relation }}</span>
                <span class="rel__obj">
                  {{ r.object }}<template v-if="r.object_2"> + {{ r.object_2 }}</template>
                </span>
                <span class="rel__score">{{ r.score.toFixed(2) }}</span>
              </li>
              <li v-if="relations.length === 0" class="empty">— 尚無關係 —</li>
            </ul>
          </section>
        </div>
      </div>
    </div>
  </Transition>
</template>

<style scoped>
.overlay {
  position: fixed; inset: 0; z-index: 100;
  background: rgba(4, 6, 7, 0.72);
  backdrop-filter: blur(4px);
  display: grid; place-items: center; padding: 32px;
}
.modal {
  width: min(960px, 100%); max-height: 80vh;
  display: flex; flex-direction: column;
  background: var(--c-ink-1);
  border: 1px solid var(--c-line-strong);
  box-shadow: 0 24px 80px rgba(0, 0, 0, 0.6);
}
.modal__head {
  display: flex; align-items: center; justify-content: space-between;
  padding: 14px 18px; border-bottom: 1px solid var(--c-line);
}
.modal__title { margin: 0; font-size: 13px; font-weight: 400; letter-spacing: 0.2em; color: var(--c-bone); }
.modal__close {
  background: transparent; border: 1px solid var(--c-line); color: var(--c-bone-dim);
  width: 28px; height: 28px; cursor: pointer; font-size: 12px; line-height: 1;
}
.modal__close:hover { border-color: var(--c-phosphor); color: var(--c-phosphor); }
.modal__body {
  display: grid; grid-template-columns: 1fr 1fr; gap: 14px;
  padding: 16px 18px; overflow: auto; min-height: 0;
}

.card { border: 1px solid var(--c-line); padding: 12px 14px; background: rgba(255,255,255,0.012); min-width: 0; }
.card__head { display: flex; align-items: baseline; gap: 10px; padding-bottom: 6px; border-bottom: 1px dashed var(--c-line); margin-bottom: 8px; }
.card__num { color: var(--c-phosphor); font-size: 14px; }
.card__title { font-size: 12px; color: var(--c-bone-dim); }
.card__hint { margin-left: auto; color: var(--c-bone-faint); font-size: 11px; }

.objs, .rels { margin: 0; padding: 0; list-style: none; display: flex; flex-direction: column; gap: 4px; max-height: 50vh; overflow: auto; }
.obj { display: grid; grid-template-columns: 10px 1fr auto; align-items: center; gap: 8px; font-size: 11px; color: var(--c-bone-dim); }
.obj__dot { width: 6px; height: 6px; border-radius: 50%; background: var(--c-phosphor); }
.obj--occluded .obj__dot { background: var(--c-amber); }
.obj--stale .obj__dot { background: var(--c-bone-faint); }
.obj--lost .obj__dot { background: var(--c-coral); }
.obj__label em { color: var(--c-bone-faint); margin-right: 6px; font-style: normal; }
.obj__label b { color: var(--c-bone); font-weight: 500; }
.obj__xyz { color: var(--c-bone-faint); font-variant-numeric: tabular-nums; }

.rel { display: grid; grid-template-columns: 1fr auto 1fr auto; align-items: baseline; gap: 8px; font-size: 11px; color: var(--c-bone-dim); }
.rel__rel { color: var(--c-phosphor); font-size: 10px; letter-spacing: 0.08em; }
.rel__subj, .rel__obj { color: var(--c-bone); overflow: hidden; text-overflow: ellipsis; }
.rel__score { color: var(--c-bone-faint); font-variant-numeric: tabular-nums; }

.empty { color: var(--c-bone-faint); font-size: 11px; padding: 8px 0; }

.modal-enter-active, .modal-leave-active { transition: opacity 0.18s ease; }
.modal-enter-from, .modal-leave-to { opacity: 0; }
</style>
