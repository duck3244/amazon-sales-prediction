<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import {
  listModels,
  predictBatch,
  predictSingle,
  type BatchPredictResult,
  type TrainedModelInfo,
} from '@/api/predict'
import StatCard from '@/components/StatCard.vue'

const models = ref<TrainedModelInfo[]>([])
const selected = ref<string>('basic')
const tab = ref<'single' | 'batch'>('single')

// 단건
const features = ref<string>('')
const prediction = ref<number | null>(null)

// 배치
const batchFile = ref<HTMLInputElement | null>(null)
const batchResult = ref<BatchPredictResult | null>(null)

const isLoading = ref(false)
const error = ref<string>('')

const expectedDim = computed(
  () => models.value.find((m) => m.model_type === selected.value)?.input_dim ?? 0,
)
const parsedFeatures = computed<number[]>(() => {
  if (!features.value.trim()) return []
  return features.value
    .split(/[\s,]+/)
    .map((s) => s.trim())
    .filter(Boolean)
    .map((s) => Number(s))
})
const isValid = computed(
  () =>
    parsedFeatures.value.length === expectedDim.value &&
    parsedFeatures.value.every((n) => Number.isFinite(n)),
)

function fillZeros() {
  features.value = Array(expectedDim.value).fill(0).join(', ')
}

async function refreshModels() {
  models.value = await listModels()
  if (models.value.length && !models.value.find((m) => m.model_type === selected.value)) {
    selected.value = models.value[0].model_type
  }
}

async function onSingle() {
  error.value = ''
  prediction.value = null
  isLoading.value = true
  try {
    const r = await predictSingle(selected.value, parsedFeatures.value)
    prediction.value = r.prediction
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    isLoading.value = false
  }
}

async function onBatch(e: Event) {
  const input = e.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file) return
  error.value = ''
  batchResult.value = null
  isLoading.value = true
  try {
    batchResult.value = await predictBatch(selected.value, file)
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    isLoading.value = false
    if (batchFile.value) batchFile.value.value = ''
  }
}

function downloadBatch() {
  if (!batchResult.value) return
  const url = URL.createObjectURL(batchResult.value.blob)
  const a = document.createElement('a')
  a.href = url
  a.download = batchResult.value.filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

onMounted(refreshModels)
</script>

<template>
  <section class="space-y-6">
    <header class="space-y-1">
      <h2 class="text-2xl font-bold">예측</h2>
      <p class="text-sm text-slate-600">
        단건 예측은 전처리된 특성 벡터를 직접 입력합니다. 배치 예측은 raw CSV를 올리면
        학습 시 저장된 전처리기로 변환 후 예측합니다.
      </p>
    </header>

    <div class="bg-white border border-slate-200 rounded-lg p-6 space-y-4">
      <div class="flex flex-wrap items-end gap-3">
        <label class="text-sm">
          <span class="block text-slate-700 mb-1">모델</span>
          <select v-model="selected" class="rounded border-slate-300 text-sm min-w-[10rem]"
                  :disabled="models.length === 0">
            <option v-for="m in models" :key="m.model_type" :value="m.model_type">
              {{ m.model_type }} (input={{ m.input_dim }})
            </option>
            <option v-if="models.length === 0" :value="selected">학습된 모델 없음</option>
          </select>
        </label>
        <button class="px-3 py-2 rounded bg-slate-100 text-sm hover:bg-slate-200"
                type="button" @click="refreshModels">
          모델 새로고침
        </button>
        <div class="flex gap-1 ml-auto">
          <button
            type="button"
            class="px-3 py-1.5 rounded text-sm"
            :class="tab === 'single' ? 'bg-slate-900 text-white' : 'bg-slate-100 hover:bg-slate-200'"
            @click="tab = 'single'"
          >단건</button>
          <button
            type="button"
            class="px-3 py-1.5 rounded text-sm"
            :class="tab === 'batch' ? 'bg-slate-900 text-white' : 'bg-slate-100 hover:bg-slate-200'"
            @click="tab = 'batch'"
          >배치 (CSV)</button>
        </div>
      </div>

      <!-- 단건 -->
      <div v-if="tab === 'single'" class="space-y-3">
        <div class="flex gap-2">
          <button class="px-3 py-2 rounded bg-slate-100 text-sm hover:bg-slate-200"
                  type="button" @click="fillZeros">
            0으로 채우기
          </button>
          <span class="self-center text-xs text-slate-500">
            입력 dimensions = {{ expectedDim || '?' }}
          </span>
        </div>
        <label class="block text-sm">
          <span class="block text-slate-700 mb-1">특성 벡터 ({{ expectedDim }}개 필요)</span>
          <textarea v-model="features" rows="3"
                    class="w-full font-mono text-sm rounded border-slate-300"
                    placeholder="예: 0.12, -1.04, 0.0, 1.5, ..."></textarea>
          <span class="text-xs"
                :class="isValid ? 'text-emerald-600' : 'text-slate-500'">
            현재 입력 길이: {{ parsedFeatures.length }} / {{ expectedDim }}
            {{ isValid ? '· 유효' : '' }}
          </span>
        </label>
        <button class="px-4 py-2 rounded bg-slate-900 text-white text-sm hover:bg-slate-700 disabled:opacity-50"
                :disabled="!isValid || isLoading" @click="onSingle">
          {{ isLoading ? '예측 중…' : '예측하기' }}
        </button>
        <div v-if="prediction !== null" class="text-base">
          예측값:
          <span class="font-mono font-semibold">{{ prediction.toFixed(4) }}</span>
        </div>
      </div>

      <!-- 배치 -->
      <div v-if="tab === 'batch'" class="space-y-3">
        <p class="text-sm text-slate-600">
          학습에 사용한 동일한 컬럼 구조의 CSV를 업로드하세요. 결과는
          <code class="font-mono">Predicted</code> 컬럼이 추가된 CSV로 다운로드됩니다.
        </p>
        <input
          ref="batchFile"
          type="file"
          accept=".csv,text/csv"
          :disabled="isLoading"
          class="block text-sm file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:bg-slate-900 file:text-white hover:file:bg-slate-700 disabled:opacity-50"
          @change="onBatch"
        />
        <p v-if="isLoading" class="text-sm text-slate-500">예측 중…</p>

        <div v-if="batchResult" class="space-y-3">
          <div class="grid grid-cols-2 md:grid-cols-3 gap-3">
            <StatCard label="예측 행 수">{{ batchResult.rows.toLocaleString() }}</StatCard>
            <StatCard label="소요 시간">{{ batchResult.elapsedSeconds.toFixed(2) }}s</StatCard>
            <StatCard label="모델">{{ batchResult.model }}</StatCard>
          </div>
          <button class="px-4 py-2 rounded bg-emerald-600 text-white text-sm hover:bg-emerald-500"
                  type="button" @click="downloadBatch">
            CSV 다운로드 ({{ batchResult.filename }})
          </button>
        </div>
      </div>

      <p v-if="error" class="text-sm text-rose-600">{{ error }}</p>
    </div>
  </section>
</template>
