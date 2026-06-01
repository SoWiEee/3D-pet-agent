<script setup lang="ts">
import { computed, ref, watch } from "vue";

const props = defineProps<{ text: string | null }>();
const visible = ref(false);
let hideTimer: number | undefined;

watch(
  () => props.text,
  (v) => {
    if (!v) return;
    visible.value = false;
    requestAnimationFrame(() => (visible.value = true));
    if (hideTimer) window.clearTimeout(hideTimer);
    hideTimer = window.setTimeout(() => (visible.value = false), 4500);
  }
);

const shown = computed(() => props.text ?? "");
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
