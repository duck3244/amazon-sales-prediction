<script setup lang="ts">
import { computed, ref } from 'vue'
import { compareModels, type CompareEntry } from '@/api/predict'
import MetricBarChart from '@/components/MetricBarChart.vue'

const entries = ref<CompareEntry[]>([])
const isLoading = ref(false)
const error = ref<string>('')
const hasRun = ref(false)

async function run() {
  error.value = ''
  isLoading.value = true
  try {
    const r = await compareModels()
    entries.value = r.entries
    hasRun.value = true
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    isLoading.value = false
  }
}

const okEntries = computed(() => entries.value.filter((e) => e.error == null))

const labels = computed(() => okEntries.value.map((e) => e.model_type))
const errSeries = computed(() => [
  { name: 'MAE', values: okEntries.value.map((e) => e.mae ?? 0) },
  { name: 'RMSE', values: okEntries.value.map((e) => e.rmse ?? 0) },
])
const r2Series = computed(() => [
  { name: 'R²', values: okEntries.value.map((e) => e.r2 ?? 0) },
])

const bestByR2 = computed(() => {
  if (okEntries.value.length === 0) return null
  return [...okEntries.value].sort((a, b) => (b.r2 ?? -Infinity) - (a.r2 ?? -Infinity))[0]
})
</script>

<template>
  <section class="space-y-6">
    <header class="space-y-1">
      <h2 class="text-2xl font-bold">모델 비교</h2>
      <p class="text-sm text-slate-600">
        학습된 모든 모델을 동일한 테스트셋에 대해 평가하여 한 번에 비교합니다.
      </p>
    </header>

    <div class="bg-white border border-slate-200 rounded-lg p-6 space-y-4">
      <div class="flex items-center gap-3">
        <button class="px-4 py-2 rounded bg-slate-900 text-white text-sm hover:bg-slate-700 disabled:opacity-50"
                :disabled="isLoading" @click="run">
          {{ isLoading ? '평가 중…' : '비교 실행' }}
        </button>
        <span v-if="error" class="text-sm text-rose-600">{{ error }}</span>
      </div>

      <div v-if="hasRun && entries.length === 0" class="text-sm text-slate-500">
        학습된 모델이 없습니다. 먼저 학습 페이지에서 모델을 만들어주세요.
      </div>

      <div v-if="entries.length" class="space-y-4">
        <div v-if="bestByR2" class="rounded bg-emerald-50 border border-emerald-200 px-3 py-2 text-sm text-emerald-800">
          🏆 R² 기준 최고 모델: <strong>{{ bestByR2.model_type }}</strong>
          (R² = {{ bestByR2.r2?.toFixed(4) }}, RMSE = {{ bestByR2.rmse?.toFixed(2) }})
        </div>

        <div class="bg-white border border-slate-200 rounded-lg overflow-hidden">
          <table class="w-full text-sm">
            <thead class="text-left text-slate-500 bg-slate-50">
              <tr>
                <th class="px-4 py-2 font-medium">모델</th>
                <th class="px-4 py-2 font-medium text-right">MAE</th>
                <th class="px-4 py-2 font-medium text-right">RMSE</th>
                <th class="px-4 py-2 font-medium text-right">R²</th>
                <th class="px-4 py-2 font-medium text-right">샘플</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-slate-100">
              <tr v-for="e in entries" :key="e.model_type">
                <td class="px-4 py-2 font-mono">{{ e.model_type }}</td>
                <template v-if="!e.error">
                  <td class="px-4 py-2 text-right">{{ e.mae?.toFixed(4) }}</td>
                  <td class="px-4 py-2 text-right">{{ e.rmse?.toFixed(4) }}</td>
                  <td class="px-4 py-2 text-right">{{ e.r2?.toFixed(4) }}</td>
                  <td class="px-4 py-2 text-right">{{ e.n_samples?.toLocaleString() }}</td>
                </template>
                <template v-else>
                  <td colspan="4" class="px-4 py-2 text-rose-600">{{ e.error }}</td>
                </template>
              </tr>
            </tbody>
          </table>
        </div>

        <div v-if="okEntries.length" class="grid md:grid-cols-2 gap-4">
          <div class="bg-white border border-slate-200 rounded-lg p-4">
            <h3 class="text-sm font-medium mb-2">오차 (낮을수록 좋음)</h3>
            <MetricBarChart :labels="labels" :series="errSeries" />
          </div>
          <div class="bg-white border border-slate-200 rounded-lg p-4">
            <h3 class="text-sm font-medium mb-2">R² (1에 가까울수록 좋음)</h3>
            <MetricBarChart :labels="labels" :series="r2Series" />
          </div>
        </div>
      </div>
    </div>
  </section>
</template>
