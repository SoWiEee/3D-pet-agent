<script setup lang="ts">
import { onBeforeUnmount, ref, watch } from "vue";

// Speech queue (B5): rapid `runtime.ask` broadcasts used to overwrite each
// other — an explanation could be replaced by a clarification before it was
// readable. We queue distinct lines and show each for a minimum dwell, so
// every line gets its moment; the last one lingers a little longer.
const MIN_VISIBLE_MS = 1800; // each line stays at least this long
const LINGER_MS = 4500; // the final line lingers before fading out

const props = defineProps<{ text: string | null }>();

const shown = ref("");
const visible = ref(false);
const queue = ref<string[]>([]);
let lastEnqueued: string | null = null;
let timer: number | undefined;

watch(
  () => props.text,
  (v) => {
    // Ignore clears and unchanged re-broadcasts (speech persists in PetState).
    if (!v || v === lastEnqueued) return;
    lastEnqueued = v;
    queue.value = [...queue.value, v];
    if (!visible.value) advance();
  },
);

function advance() {
  if (timer) window.clearTimeout(timer);
  const next = queue.value[0];
  if (next === undefined) {
    visible.value = false;
    return;
  }
  queue.value = queue.value.slice(1);
  shown.value = next;
  // Re-trigger the rise transition for each new line.
  visible.value = false;
  requestAnimationFrame(() => (visible.value = true));
  // Dwell longer on the last line; otherwise move on to the next queued line.
  timer = window.setTimeout(advance, queue.value.length > 0 ? MIN_VISIBLE_MS : LINGER_MS);
}

onBeforeUnmount(() => {
  if (timer) window.clearTimeout(timer);
});
</script>

<template>
  <transition name="rise">
    <div v-if="visible && shown" class="speech">
      <span class="speech__rule" />
      <span class="speech__lead">she says</span>
      <p class="speech__body">&ldquo;{{ shown }}&rdquo;</p>
    </div>
  </transition>
</template>

<style scoped>
.speech {
  position: absolute;
  left: 50%;
  bottom: 14%;
  transform: translateX(-50%);
  max-width: 60%;
  text-align: center;
  pointer-events: none;
  z-index: 7;
}
.speech__rule {
  display: block;
  width: 28px;
  height: 1px;
  background: var(--c-phosphor);
  margin: 0 auto 8px auto;
  box-shadow: 0 0 8px var(--c-phosphor);
}
.speech__lead {
  font-size: 9px;
  letter-spacing: 0.32em;
  text-transform: uppercase;
  color: var(--c-bone-faint);
}
.speech__body {
  font-family: var(--f-serif);
  font-style: italic;
  font-size: 26px;
  line-height: 1.25;
  color: var(--c-bone);
  margin: 6px 0 0 0;
  text-shadow: 0 0 24px rgba(116, 247, 208, 0.25);
}
.rise-enter-active, .rise-leave-active { transition: opacity 350ms ease, transform 450ms cubic-bezier(.2,.7,.2,1); }
.rise-enter-from { opacity: 0; transform: translate(-50%, 14px); }
.rise-leave-to   { opacity: 0; transform: translate(-50%, -8px); }
</style>
