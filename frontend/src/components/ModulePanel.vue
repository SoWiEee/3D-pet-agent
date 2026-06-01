<script setup lang="ts">
defineProps<{
  modules: { name: string; status: "online" | "idle" | "off"; note?: string }[];
}>();
</script>

<template>
  <aside class="panel">
    <div class="panel__head">
      <span class="panel__num">01</span>
      <span class="panel__title">perception · modules</span>
    </div>

    <ul class="modlist">
      <li v-for="m in modules" :key="m.name" class="mod">
        <div class="mod__row">
          <span class="mod__name">{{ m.name }}</span>
          <span
            class="dot"
            :class="{
              'dot--on': m.status === 'online',
              'dot--idle': m.status === 'idle',
              'dot--off': m.status === 'off',
            }"
          />
        </div>
        <div class="mod__note">{{ m.note ?? m.status }}</div>
      </li>
    </ul>

    <div class="panel__head">
      <span class="panel__num">02</span>
      <span class="panel__title">phase coverage</span>
    </div>

    <ol class="phases">
      <li class="phases__item phases__item--done">
        <span class="phases__id">P1</span>
        <span class="phases__name">pet runtime + sandbox</span>
      </li>
      <li class="phases__item phases__item--done">
        <span class="phases__id">P2</span>
        <span class="phases__name">interactive perception</span>
      </li>
      <li class="phases__item">
        <span class="phases__id">P3</span>
        <span class="phases__name">depth · 3d lifting</span>
      </li>
      <li class="phases__item">
        <span class="phases__id">P4</span>
        <span class="phases__name">tracking · memory</span>
      </li>
      <li class="phases__item">
        <span class="phases__id">P5</span>
        <span class="phases__name">scene graph · relations</span>
      </li>
      <li class="phases__item">
        <span class="phases__id">P6</span>
        <span class="phases__name">command grounding</span>
      </li>
    </ol>
  </aside>
</template>

<style scoped>
.panel {
  background: linear-gradient(180deg, rgba(11, 17, 18, 0.86), rgba(8, 13, 14, 0.92));
  border-right: 1px solid var(--c-line);
  padding: 18px 16px;
  display: flex;
  flex-direction: column;
  gap: 14px;
  overflow: auto;
}
.panel__head {
  display: flex;
  align-items: baseline;
  gap: 10px;
  padding-bottom: 8px;
  border-bottom: 1px dashed var(--c-line);
  margin-top: 6px;
}
.panel__num {
  font-family: var(--f-display);
  font-size: 16px;
  color: var(--c-phosphor);
  letter-spacing: 0.05em;
}
.panel__title {
  font-size: 11px;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--c-bone-dim);
}

.modlist { list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 10px; }
.mod {
  padding: 8px 10px;
  border: 1px solid var(--c-line);
  background: rgba(255,255,255,0.012);
}
.mod__row { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.mod__name {
  font-size: 12px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--c-bone);
}
.mod__note { color: var(--c-bone-faint); font-size: 10px; margin-top: 4px; letter-spacing: 0.08em; }

.dot {
  width: 8px; height: 8px; border-radius: 50%;
  box-shadow: 0 0 8px currentColor;
}
.dot--on   { color: var(--c-phosphor); background: var(--c-phosphor); }
.dot--idle { color: var(--c-amber);    background: var(--c-amber); }
.dot--off  { color: #4a4a4a;           background: #4a4a4a; box-shadow: none; }

.phases { list-style: none; padding: 0; margin: 0; }
.phases__item {
  display: grid;
  grid-template-columns: 32px 1fr;
  align-items: center;
  padding: 6px 0;
  border-bottom: 1px dotted var(--c-line);
  color: var(--c-bone-dim);
  letter-spacing: 0.12em;
  text-transform: uppercase;
  font-size: 11px;
}
.phases__item:last-child { border-bottom: 0; }
.phases__id {
  font-family: var(--f-display);
  color: var(--c-bone-faint);
}
.phases__item--done .phases__id { color: var(--c-phosphor); }
.phases__item--done .phases__name { color: var(--c-bone); }
.phases__item--done .phases__name::after {
  content: " · ack";
  color: var(--c-phosphor-dim);
  font-size: 9px;
}
</style>
