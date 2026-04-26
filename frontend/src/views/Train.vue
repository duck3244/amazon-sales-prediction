<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import {
  getActiveJob,
  getJob,
  getJobLog,
  runPreprocess,
  startTraining,
  type TrainRequest,
} from '@/api/train'
import { useTrainingStore } from '@/stores/training'
import LossChart from '@/components/LossChart.vue'
import StatCard from '@/components/StatCard.vue'

const store = useTrainingStore()

const form = reactive<TrainRequest>({
  model_type: 'basic',
  epochs: 30,
  batch_size: 256,
  lr: 0.001,
  weight_decay: 0.00001,
  dropout: 0.3,
  hidden_dims: [256, 128, 64],
  patience: 15,
  seed: 42,
  deterministic: false,
})

const error = ref<string>('')
const preprocessInfo = ref<string>('')
const isPreprocessing = ref(false)

const logLines = ref<string[]>([])
const logOffset = ref(0)
const logEl = ref<HTMLElement | null>(null)

let pollTimer: number | null = null

const isRunning = computed(() => store.status?.status === 'running')
const progressPct = computed(() => {
  const s = store.status
  if (!s || !s.total_epochs) return 0
  return Math.round((s.current_epoch / s.total_epochs) * 100)
})

async function pollOnce() {
  if (!store.jobId) return
  try {
    const [s, log] = await Promise.all([
      getJob(store.jobId),
      getJobLog(store.jobId, logOffset.value),
    ])
    store.setStatus(s)
    if (log.lines.length) {
      logLines.value = [...logLines.value, ...log.lines]
      logOffset.value = log.next_offset
      // 자동 스크롤
      requestAnimationFrame(() => {
        if (logEl.value) logEl.value.scrollTop = logEl.value.scrollHeight
      })
    }
    if (s.status === 'completed' || s.status === 'failed') {
      stopPolling()
    }
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
    stopPolling()
  }
}

function resetLog() {
  logLines.value = []
  logOffset.value = 0
}

function startPolling() {
  stopPolling()
  pollOnce()
  pollTimer = window.setInterval(pollOnce, 1500)
}

function stopPolling() {
  if (pollTimer !== null) {
    window.clearInterval(pollTimer)
    pollTimer = null
  }
}

async function onPreprocess() {
  error.value = ''
  preprocessInfo.value = ''
  isPreprocessing.value = true
  try {
    const r = await runPreprocess()
    preprocessInfo.value = `Train ${r.train.toLocaleString()} / Val ${r.val.toLocaleString()} / Test ${r.test.toLocaleString()} · 특성 ${r.n_features}개`
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    isPreprocessing.value = false
  }
}

async function onSubmit() {
  error.value = ''
  resetLog()
  try {
    const r = await startTraining(form)
    store.setJob(r.job_id)
    startPolling()
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  }
}

onMounted(async () => {
  // 페이지 새로고침 후 진행 중 잡 복원
  if (store.jobId) {
    try {
      const s = await getJob(store.jobId)
      store.setStatus(s)
      if (s.status === 'running') startPolling()
    } catch {
      store.clear()
    }
  } else {
    // 메모리는 비어있어도 서버에 활성 잡이 있을 수 있음
    const active = await getActiveJob().catch(() => null)
    if (active) {
      store.setJob(active.job_id)
      store.setStatus(active)
      if (active.status === 'running') startPolling()
    }
  }
})

onBeforeUnmount(stopPolling)
</script>

<template>
  <section class="space-y-6">
    <header class="space-y-1">
      <h2 class="text-2xl font-bold">모델 학습</h2>
      <p class="text-sm text-slate-600">
        먼저 전처리를 실행한 뒤 모델을 학습합니다. 진행 중에는 실시간 loss 곡선을 확인할 수 있습니다.
      </p>
    </header>

    <div class="bg-white border border-slate-200 rounded-lg p-6 space-y-4">
      <div class="flex items-center gap-3">
        <button
          class="px-3 py-1.5 rounded bg-slate-200 text-sm hover:bg-slate-300 disabled:opacity-50"
          :disabled="isPreprocessing || isRunning"
          @click="onPreprocess"
        >
          전처리 실행
        </button>
        <span v-if="isPreprocessing" class="text-sm text-slate-500">전처리 중…</span>
        <span v-if="preprocessInfo" class="text-sm text-emerald-700">{{ preprocessInfo }}</span>
      </div>

      <form class="grid grid-cols-2 md:grid-cols-3 gap-4" @submit.prevent="onSubmit">
        <label class="text-sm">
          <span class="block text-slate-700 mb-1">모델 타입</span>
          <select v-model="form.model_type" class="w-full rounded border-slate-300 text-sm">
            <option value="basic">basic</option>
            <option value="advanced">advanced</option>
            <option value="attention">attention</option>
          </select>
        </label>
        <label class="text-sm">
          <span class="block text-slate-700 mb-1">epochs</span>
          <input v-model.number="form.epochs" type="number" min="1" max="500"
                 class="w-full rounded border-slate-300 text-sm" />
        </label>
        <label class="text-sm">
          <span class="block text-slate-700 mb-1">batch size</span>
          <input v-model.number="form.batch_size" type="number" min="1" max="4096"
                 class="w-full rounded border-slate-300 text-sm" />
        </label>
        <label class="text-sm">
          <span class="block text-slate-700 mb-1">learning rate</span>
          <input v-model.number="form.lr" type="number" step="0.0001" min="0.00001" max="1"
                 class="w-full rounded border-slate-300 text-sm" />
        </label>
        <label class="text-sm">
          <span class="block text-slate-700 mb-1">dropout</span>
          <input v-model.number="form.dropout" type="number" step="0.05" min="0" max="0.9"
                 class="w-full rounded border-slate-300 text-sm" />
        </label>
        <label class="text-sm">
          <span class="block text-slate-700 mb-1">patience</span>
          <input v-model.number="form.patience" type="number" min="1" max="200"
                 class="w-full rounded border-slate-300 text-sm" />
        </label>

        <div class="col-span-full flex items-center gap-3">
          <button
            type="submit"
            class="px-4 py-2 rounded bg-slate-900 text-white text-sm hover:bg-slate-700 disabled:opacity-50"
            :disabled="isRunning"
          >
            {{ isRunning ? '학습 진행 중…' : '학습 시작' }}
          </button>
          <span v-if="error" class="text-sm text-rose-600">{{ error }}</span>
        </div>
      </form>
    </div>

    <div v-if="store.status" class="space-y-4">
      <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatCard label="상태">{{ store.status.status }}</StatCard>
        <StatCard label="진행">
          {{ store.status.current_epoch }}/{{ store.status.total_epochs }} ({{ progressPct }}%)
        </StatCard>
        <StatCard label="best val loss">
          {{ store.status.best_val_loss != null ? store.status.best_val_loss.toFixed(4) : '-' }}
        </StatCard>
        <StatCard label="test loss">
          {{ store.status.test_loss != null ? store.status.test_loss.toFixed(4) : '-' }}
        </StatCard>
      </div>

      <div v-if="store.status.history.length" class="bg-white border border-slate-200 rounded-lg p-4">
        <LossChart :history="store.status.history" />
      </div>

      <div v-if="logLines.length" class="bg-white border border-slate-200 rounded-lg overflow-hidden">
        <header class="px-4 py-2 bg-slate-50 border-b border-slate-200 text-sm font-medium">
          학습 로그 ({{ logLines.length }}줄)
        </header>
        <div
          ref="logEl"
          class="px-4 py-3 font-mono text-xs leading-relaxed text-slate-700 max-h-72 overflow-auto whitespace-pre-wrap"
        >
          <div v-for="(line, idx) in logLines" :key="idx">{{ line }}</div>
        </div>
      </div>

      <div v-if="store.status.error"
           class="bg-rose-50 border border-rose-200 rounded p-3 text-sm text-rose-700 whitespace-pre-wrap font-mono">{{ store.status.error }}</div>
    </div>
  </section>
</template>
