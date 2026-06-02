<script setup lang="ts">
import { computed, onMounted, onBeforeUnmount, watch } from "vue";
import type { PetAction } from "../composables/useWebSocket";

const props = defineProps<{
  open: boolean;
  history: PetAction[];
}>();
const emit = defineEmits<{ (e: "close"): void }>();

const recent = computed(() => props.history.slice(-50).reverse());

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
      <div class="modal" role="dialog" aria-modal="true" aria-labelledby="events-title">
        <header class="modal__head">
          <h2 id="events-title" class="modal__title">活動紀錄</h2>
          <button class="modal__close" type="button" aria-label="關閉" @click="close">
            ✕
          </button>
        </header>

        <div class="modal__body">
          <section class="card">
            <div class="card__head">
              <span class="card__num">B</span>
              <span class="card__title">事件串流</span>
              <span class="card__hint">最近 50 筆 · 新到舊</span>
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
                    ↝ {{ ev.path.length }} 點 · 終點 ({{ ev.path[ev.path.length - 1].map((v) => v.toFixed(2)).join(", ") }})
                  </template>
                  <template v-else-if="ev.action === 'play_animation'">{{ ev.animation }}</template>
                  <template v-else-if="ev.action === 'set_emotion'">{{ ev.emotion }}</template>
                  <template v-else-if="ev.action === 'ask'">"{{ ev.speech }}"</template>
                  <template v-else-if="ev.action === 'state'">狀態快照</template>
                </span>
              </li>
              <li v-if="recent.length === 0" class="empty">— 尚無事件 —</li>
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
  width: min(720px, 100%); max-height: 80vh;
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
.modal__body { padding: 16px 18px; overflow: auto; min-height: 0; }

.card { border: 1px solid var(--c-line); padding: 12px 14px; background: rgba(255,255,255,0.012); }
.card__head { display: flex; align-items: baseline; gap: 10px; padding-bottom: 6px; border-bottom: 1px dashed var(--c-line); margin-bottom: 8px; }
.card__num { color: var(--c-phosphor); font-size: 14px; }
.card__title { font-size: 12px; color: var(--c-bone-dim); }
.card__hint { margin-left: auto; color: var(--c-bone-faint); font-size: 11px; }

.ev { list-style: none; margin: 0; padding: 0; max-height: 60vh; overflow: auto; }
.ev__row {
  display: grid; grid-template-columns: 70px 110px 1fr; gap: 10px;
  padding: 3px 0; border-bottom: 1px dotted var(--c-line); font-size: 11px;
}
.ev__row:last-child { border-bottom: 0; }
.ev__t { color: var(--c-bone-faint); font-variant-numeric: tabular-nums; }
.ev__kind { color: var(--c-phosphor); }
.ev__detail { color: var(--c-bone); }
.empty { color: var(--c-bone-faint); font-size: 11px; padding: 12px 0; text-align: center; }

.modal-enter-active, .modal-leave-active { transition: opacity 0.18s ease; }
.modal-enter-from, .modal-leave-to { opacity: 0; }
</style>
