<script setup lang="ts">
defineProps<{ label: string }>()
</script>

<template>
  <div class="rounded border border-slate-200 bg-white p-3">
    <div class="text-xs text-slate-500">{{ label }}</div>
    <div class="mt-1 text-base font-semibold truncate">
      <slot />
    </div>
  </div>
</template>
