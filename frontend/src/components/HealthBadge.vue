<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { fetchHealth, type HealthResponse } from '@/api/client'

const state = ref<'loading' | 'ok' | 'error'>('loading')
const info = ref<HealthResponse | null>(null)
const message = ref<string>('')

onMounted(async () => {
  try {
    info.value = await fetchHealth()
    state.value = 'ok'
  } catch (err) {
    state.value = 'error'
    message.value = err instanceof Error ? err.message : String(err)
  }
})
</script>

<template>
  <div class="text-xs">
    <span v-if="state === 'loading'" class="text-slate-500">서버 확인 중…</span>
    <span
      v-else-if="state === 'ok' && info"
      class="inline-flex items-center gap-2 rounded-full bg-emerald-50 text-emerald-700 px-2 py-1"
    >
      <span class="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
      torch {{ info.torch_version }} ·
      {{ info.cuda_available ? info.cuda_device : 'CPU' }}
    </span>
    <span v-else class="rounded-full bg-rose-50 text-rose-700 px-2 py-1">
      서버 연결 실패: {{ message }}
    </span>
  </div>
</template>
