<script setup lang="ts">
import { computed } from "vue";

// Class labels mirror the backend's `_EDITOR_DEFAULT_EXTENTS` so every option
// lands at a believable real-world size. zh display name + English value the
// grounding parser understands.
const CLASSES: { value: string; zh: string }[] = [
  { value: "cup", zh: "杯子" },
  { value: "bottle", zh: "瓶子" },
  { value: "bowl", zh: "碗" },
  { value: "book", zh: "書" },
  { value: "laptop", zh: "筆電" },
  { value: "keyboard", zh: "鍵盤" },
  { value: "ball", zh: "球" },
  { value: "lamp", zh: "檯燈" },
  { value: "potted plant", zh: "盆栽" },
  { value: "chair", zh: "椅子" },
  { value: "table", zh: "桌子" },
  { value: "box", zh: "箱子" },
];

const props = defineProps<{ label: string; placedCount: number }>();
const emit = defineEmits<{
  (e: "update:label", value: string): void;
  (e: "undo"): void;
  (e: "clear"): void;
  (e: "close"): void;
}>();

const hasPlaced = computed(() => props.placedCount > 0);
</script>

<template>
  <aside class="editor">
    <header class="editor__head">
      <span class="editor__id">EDIT</span>
      <span class="editor__name">場景編輯器</span>
      <button class="editor__x" type="button" title="關閉編輯模式" @click="emit('close')">✕</button>
    </header>

    <label class="editor__field">
      <span class="editor__lbl">物件類別</span>
      <select
        class="editor__select"
        :value="label"
        @change="emit('update:label', ($event.target as HTMLSelectElement).value)"
      >
        <option v-for="c in CLASSES" :key="c.value" :value="c.value">
          {{ c.zh }} · {{ c.value }}
        </option>
      </select>
    </label>

    <p class="editor__hint">
      <b>左鍵點地板</b> 放置 · 右鍵拖曳轉視角 · 滾輪縮放
    </p>

    <div class="editor__actions">
      <span class="editor__count">已放置 {{ placedCount }}</span>
      <button class="editor__btn" type="button" :disabled="!hasPlaced" @click="emit('undo')">
        復原
      </button>
      <button
        class="editor__btn editor__btn--danger"
        type="button"
        :disabled="!hasPlaced"
        @click="emit('clear')"
      >
        清空
      </button>
    </div>
  </aside>
</template>

<style scoped>
.editor {
  position: absolute;
  top: 62px;
  left: 60px;
  z-index: 8;
  width: 232px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 12px 14px 14px;
  background: rgba(7, 9, 10, 0.82);
  border: 1px solid var(--c-phosphor);
  box-shadow: 0 0 0 1px rgba(116, 247, 208, 0.12), 0 12px 40px rgba(0, 0, 0, 0.5);
  backdrop-filter: blur(6px);
}
.editor__head {
  display: flex;
  align-items: baseline;
  gap: 8px;
  font-size: 10px;
  letter-spacing: 0.2em;
  text-transform: uppercase;
}
.editor__id { color: var(--c-phosphor); font-family: var(--f-display); }
.editor__name { color: var(--c-bone-dim); }
.editor__x {
  margin-left: auto;
  color: var(--c-bone-faint);
  background: none;
  border: none;
  cursor: pointer;
  font-size: 12px;
  line-height: 1;
  padding: 2px;
  transition: color 150ms ease;
}
.editor__x:hover { color: var(--c-phosphor); }

.editor__field { display: flex; flex-direction: column; gap: 5px; }
.editor__lbl {
  font-size: 10px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--c-bone-faint);
}
.editor__select {
  appearance: none;
  width: 100%;
  padding: 7px 9px;
  font-size: 13px;
  color: var(--c-bone);
  background: rgba(12, 17, 18, 0.9);
  border: 1px solid var(--c-line);
  cursor: pointer;
  transition: border-color 150ms ease;
}
.editor__select:hover,
.editor__select:focus { border-color: var(--c-phosphor); outline: none; }

.editor__hint {
  margin: 0;
  font-size: 11px;
  line-height: 1.5;
  color: var(--c-bone-faint);
}
.editor__hint b { color: var(--c-bone-dim); font-weight: 500; }

.editor__actions { display: flex; align-items: center; gap: 8px; }
.editor__count {
  font-size: 11px;
  color: var(--c-bone-faint);
  font-variant-numeric: tabular-nums;
  margin-right: auto;
}
.editor__btn {
  font-size: 11px;
  letter-spacing: 0.08em;
  color: var(--c-bone-dim);
  background: rgba(12, 17, 18, 0.9);
  border: 1px solid var(--c-line);
  padding: 5px 11px;
  cursor: pointer;
  transition: color 150ms ease, border-color 150ms ease, background 150ms ease;
}
.editor__btn:hover:not(:disabled) { color: var(--c-phosphor); border-color: var(--c-phosphor); }
.editor__btn--danger:hover:not(:disabled) {
  color: #ff8a8a;
  border-color: #ff8a8a;
  background: rgba(255, 138, 138, 0.08);
}
.editor__btn:disabled { opacity: 0.35; cursor: not-allowed; }
</style>
