<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import {
  evaluateBatch,
  listModels,
  type EvaluationResponse,
  type TrainedModelInfo,
} from '@/api/predict'
import StatCard from '@/components/StatCard.vue'
import ScatterChart from '@/components/ScatterChart.vue'

const models = ref<TrainedModelInfo[]>([])
const selected = ref<string>('basic')
const result = ref<EvaluationResponse | null>(null)
const isLoading = ref(false)
const error = ref<string>('')

const r2Color = computed(() => {
  if (!result.value) return ''
  const r2 = result.value.r2
  if (r2 >= 0.8) return 'text-emerald-600'
  if (r2 >= 0.4) return 'text-amber-600'
  return 'text-rose-600'
})

async function refreshModels() {
  models.value = await listModels()
  if (models.value.length && !models.value.find((m) => m.model_type === selected.value)) {
    selected.value = models.value[0].model_type
  }
}

async function onEvaluate() {
  error.value = ''
  isLoading.value = true
  try {
    result.value = await evaluateBatch(selected.value)
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    isLoading.value = false
  }
}

onMounted(refreshModels)
</script>

<template>
  <section class="space-y-6">
    <header class="space-y-1">
      <h2 class="text-2xl font-bold">평가</h2>
      <p class="text-sm text-slate-600">
        학습된 모델을 테스트셋(<code class="font-mono">X_test/y_test</code>)에 대해 평가합니다.
      </p>
    </header>

    <div class="bg-white border border-slate-200 rounded-lg p-6 space-y-4">
      <div class="flex items-center gap-3">
        <label class="text-sm">
          <span class="block text-slate-700 mb-1">모델 선택</span>
          <select
            v-model="selected"
            class="rounded border-slate-300 text-sm min-w-[12rem]"
            :disabled="isLoading || models.length === 0"
          >
            <option v-for="m in models" :key="m.model_type" :value="m.model_type">
              {{ m.model_type }}
              {{ m.test_loss != null ? ` · test loss ${m.test_loss.toFixed(3)}` : '' }}
            </option>
            <option v-if="models.length === 0" :value="selected">학습된 모델 없음</option>
          </select>
        </label>
        <button
          class="self-end px-4 py-2 rounded bg-slate-900 text-white text-sm hover:bg-slate-700 disabled:opacity-50"
          :disabled="isLoading || models.length === 0"
          @click="onEvaluate"
        >
          {{ isLoading ? '평가 중…' : '평가 실행' }}
        </button>
        <button
          class="self-end px-3 py-2 rounded bg-slate-100 text-sm hover:bg-slate-200"
          @click="refreshModels"
        >
          모델 목록 새로고침
        </button>
      </div>
      <p v-if="error" class="text-sm text-rose-600">{{ error }}</p>
    </div>

    <div v-if="result" class="space-y-4">
      <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatCard label="MAE">{{ result.mae.toFixed(4) }}</StatCard>
        <StatCard label="RMSE">{{ result.rmse.toFixed(4) }}</StatCard>
        <StatCard label="R²">
          <span :class="r2Color">{{ result.r2.toFixed(4) }}</span>
        </StatCard>
        <StatCard label="샘플 수">{{ result.n_samples.toLocaleString() }}</StatCard>
      </div>

      <div class="bg-white border border-slate-200 rounded-lg p-4">
        <h3 class="text-sm font-medium mb-2 text-slate-700">예측 vs 실제 (상위 50 샘플)</h3>
        <ScatterChart :actuals="result.actuals_sample" :predictions="result.predictions_sample" />
      </div>
    </div>
  </section>
</template>
