<script setup lang="ts"></script>

<template>
  <!-- Four corner crosshairs framing the 3D viewport. Pure SVG, decorative. -->
  <svg class="reg reg--tl" viewBox="0 0 40 40">
    <line x1="0" y1="0" x2="20" y2="0" />
    <line x1="0" y1="0" x2="0" y2="20" />
    <circle cx="0" cy="0" r="3" />
  </svg>
  <svg class="reg reg--tr" viewBox="0 0 40 40">
    <line x1="40" y1="0" x2="20" y2="0" />
    <line x1="40" y1="0" x2="40" y2="20" />
    <circle cx="40" cy="0" r="3" />
  </svg>
  <svg class="reg reg--bl" viewBox="0 0 40 40">
    <line x1="0" y1="40" x2="20" y2="40" />
    <line x1="0" y1="40" x2="0" y2="20" />
    <circle cx="0" cy="40" r="3" />
  </svg>
  <svg class="reg reg--br" viewBox="0 0 40 40">
    <line x1="40" y1="40" x2="20" y2="40" />
    <line x1="40" y1="40" x2="40" y2="20" />
    <circle cx="40" cy="40" r="3" />
  </svg>
</template>

<style scoped>
.reg {
  position: absolute;
  width: 40px;
  height: 40px;
  pointer-events: none;
  z-index: 6;
}
.reg line { stroke: var(--c-phosphor); stroke-width: 1; opacity: 0.7; }
.reg circle { fill: none; stroke: var(--c-phosphor); stroke-width: 1; opacity: 0.7; }
.reg--tl { top: 14px; left: 14px; }
.reg--tr { top: 14px; right: 14px; }
.reg--bl { bottom: 14px; left: 14px; }
.reg--br { bottom: 14px; right: 14px; }
</style>
