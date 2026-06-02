<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, watch } from "vue";
import type { CommandResult, GroundingBreakdown } from "../composables/useWebSocket";

const props = defineProps<{
  open: boolean;
  result: CommandResult | null;
}>();
const emit = defineEmits<{ (e: "close"): void }>();

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

const STATUS_LABEL: Record<string, { text: string; cls: string }> = {
  success: { text: "成功", cls: "st--ok" },
  clarification: { text: "待釐清", cls: "st--warn" },
  no_match: { text: "無匹配", cls: "st--bad" },
  empty_map: { text: "地圖為空", cls: "st--bad" },
  plan_failed: { text: "規劃失敗", cls: "st--bad" },
};

// Each grounding component + a colour for its stacked-bar segment. Labels are
// the resolver's weighted score components (spec §9.1).
const COMPONENTS: { key: keyof GroundingBreakdown; label: string; color: string }[] = [
  { key: "semantic", label: "語意", color: "var(--c-phosphor)" },
  { key: "attribute", label: "屬性", color: "var(--c-amber)" },
  { key: "relation", label: "關係", color: "#b18cff" },
  { key: "visibility", label: "可見", color: "#74c0f7" },
  { key: "feasibility", label: "可達", color: "var(--c-bone-dim)" },
];

const status = computed(() => {
  const s = props.result?.status ?? "";
  return STATUS_LABEL[s] ?? { text: s || "—", cls: "st--bad" };
});

const intent = computed(() => props.result?.intent ?? null);
const breakdowns = computed(() => props.result?.candidate_breakdowns ?? []);
const weights = computed(() => props.result?.weights ?? {});

// Stacked-bar segments per candidate: each segment width is the *weighted*
// contribution (weight × component score), so the total bar length equals the
// candidate's final score on a 0..1 track.
function segments(bd: GroundingBreakdown) {
  return COMPONENTS.map((c) => {
    const raw = Number(bd[c.key] ?? 0);
    const weighted = (weights.value[c.key] ?? 0) * raw;
    return { ...c, raw, weighted };
  });
}
</script>

<template>
  <Transition name="modal">
    <div v-if="props.open" class="overlay" @click.self="close">
      <div class="modal" role="dialog" aria-modal="true" aria-labelledby="reason-title">
        <header class="modal__head">
          <h2 id="reason-title" class="modal__title">上一次推理</h2>
          <button class="modal__close" type="button" aria-label="關閉" @click="close">✕</button>
        </header>

        <div class="modal__body">
          <p v-if="!props.result" class="empty">— 尚未下達任何指令 —</p>

          <template v-else>
            <!-- Utterance + status -->
            <section class="row">
              <span class="utterance">「{{ props.result.utterance }}」</span>
              <span class="status" :class="status.cls">{{ status.text }}</span>
            </section>

            <!-- Parsed intent -->
            <section class="card" v-if="intent">
              <div class="card__head"><span class="card__num">P</span><span class="card__title">解析意圖</span></div>
              <div class="kvs">
                <span class="kv"><em>類型</em><b>{{ intent.intent_type }}</b></span>
                <span class="kv"><em>目標</em><b>{{ intent.target?.class_label ?? "—" }}</b></span>
                <span class="kv"><em>關係</em><b>{{ intent.spatial_relation?.relation ?? "—" }}</b></span>
                <span class="kv"><em>信心</em><b>{{ (intent.confidence ?? 0).toFixed(2) }}</b></span>
              </div>
            </section>

            <!-- Explanation -->
            <section class="card" v-if="props.result.explanation">
              <div class="card__head"><span class="card__num">E</span><span class="card__title">說明</span></div>
              <p class="explanation">{{ props.result.explanation }}</p>
            </section>

            <!-- Candidate score breakdown -->
            <section class="card" v-if="breakdowns.length">
              <div class="card__head">
                <span class="card__num">S</span><span class="card__title">候選評分拆解</span>
                <span class="card__hint">加權貢獻 = 權重 × 分數</span>
              </div>
              <ul class="cands">
                <li v-for="bd in breakdowns" :key="bd.object_id" class="cand">
                  <div class="cand__top">
                    <b class="cand__id">{{ bd.object_id }}</b>
                    <span class="cand__cls">{{ bd.class_label }}</span>
                    <span class="cand__total">{{ bd.total.toFixed(3) }}</span>
                  </div>
                  <div class="bar">
                    <span
                      v-for="seg in segments(bd)"
                      :key="seg.key"
                      class="bar__seg"
                      :style="{ width: (seg.weighted * 100).toFixed(2) + '%', background: seg.color }"
                      :title="`${seg.label} ${seg.raw.toFixed(2)} × ${(weights[seg.key] ?? 0).toFixed(2)}`"
                    />
                  </div>
                </li>
              </ul>
              <div class="legend">
                <span v-for="c in COMPONENTS" :key="c.key" class="legend__item">
                  <span class="legend__sw" :style="{ background: c.color }" />
                  {{ c.label }} ·{{ (weights[c.key] ?? 0).toFixed(2) }}
                </span>
              </div>
            </section>

            <!-- Planner status -->
            <section class="row foot" v-if="props.result.planner_status || props.result.goal_score != null">
              <span v-if="props.result.goal_score != null" class="kv"><em>目標分</em><b>{{ props.result.goal_score.toFixed(3) }}</b></span>
              <span v-if="props.result.planner_status" class="kv"><em>規劃</em>
                <b :class="props.result.planner_status === 'success' ? 'ok' : 'bad'">{{ props.result.planner_status }}</b>
              </span>
            </section>
          </template>
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
  width: min(680px, 100%); max-height: 82vh;
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
.modal__body { display: flex; flex-direction: column; gap: 12px; padding: 16px 18px; overflow: auto; min-height: 0; }

.row { display: flex; align-items: center; gap: 12px; }
.row.foot { gap: 20px; }
.utterance { color: var(--c-bone); font-size: 13px; }
.status { margin-left: auto; font-size: 11px; padding: 2px 10px; border: 1px solid currentColor; letter-spacing: 0.1em; }
.st--ok { color: var(--c-phosphor); }
.st--warn { color: var(--c-amber); }
.st--bad { color: var(--c-coral); }

.card { border: 1px solid var(--c-line); padding: 12px 14px; background: rgba(255,255,255,0.012); }
.card__head { display: flex; align-items: baseline; gap: 10px; padding-bottom: 6px; border-bottom: 1px dashed var(--c-line); margin-bottom: 10px; }
.card__num { color: var(--c-phosphor); font-size: 14px; }
.card__title { font-size: 12px; color: var(--c-bone-dim); }
.card__hint { margin-left: auto; color: var(--c-bone-faint); font-size: 11px; }

.kvs { display: flex; flex-wrap: wrap; gap: 18px; }
.kv { display: inline-flex; gap: 6px; align-items: baseline; font-size: 12px; }
.kv em { color: var(--c-bone-faint); font-style: normal; font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; }
.kv b { color: var(--c-bone); font-weight: 500; font-variant-numeric: tabular-nums; }
.kv b.ok { color: var(--c-phosphor); }
.kv b.bad { color: var(--c-coral); }

.explanation { margin: 0; color: var(--c-bone-dim); font-size: 12px; line-height: 1.5; }

.cands { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 10px; }
.cand__top { display: flex; align-items: baseline; gap: 8px; font-size: 11px; margin-bottom: 4px; }
.cand__id { color: var(--c-bone); font-weight: 500; }
.cand__cls { color: var(--c-bone-faint); }
.cand__total { margin-left: auto; color: var(--c-phosphor); font-variant-numeric: tabular-nums; }
.bar { display: flex; height: 10px; background: rgba(255,255,255,0.04); border: 1px solid var(--c-line); overflow: hidden; }
.bar__seg { height: 100%; transition: width 0.2s ease; }

.legend { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 10px; font-size: 10px; color: var(--c-bone-faint); }
.legend__item { display: inline-flex; align-items: center; gap: 5px; }
.legend__sw { width: 9px; height: 9px; display: inline-block; }

.empty { color: var(--c-bone-faint); font-size: 12px; padding: 16px 0; text-align: center; }

.modal-enter-active, .modal-leave-active { transition: opacity 0.18s ease; }
.modal-enter-from, .modal-leave-to { opacity: 0; }
</style>
