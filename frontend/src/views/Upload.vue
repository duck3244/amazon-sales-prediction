<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { fetchSummary, uploadDataset } from '@/api/data'
import { useDatasetStore } from '@/stores/dataset'
import StatCard from '@/components/StatCard.vue'

const store = useDatasetStore()

const fileInput = ref<HTMLInputElement | null>(null)
const isUploading = ref(false)
const error = ref<string>('')

const formattedSize = computed(() => {
  const bytes = store.sizeBytes
  if (!bytes) return ''
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`
})

const sortedMissing = computed(() => {
  const cols = store.summary?.columns ?? []
  return [...cols]
    .filter((c) => c.missing > 0)
    .sort((a, b) => b.missing_pct - a.missing_pct)
})

onMounted(async () => {
  if (!store.summary) {
    try {
      const summary = await fetchSummary()
      store.setUploaded({ filename: '(서버에 이미 존재)', sizeBytes: 0, summary })
    } catch {
      // 서버에 데이터셋이 없으면 무시 — 사용자가 업로드하도록 유도
    }
  }
})

async function handleFile(e: Event) {
  const target = e.target as HTMLInputElement
  const file = target.files?.[0]
  if (!file) return

  error.value = ''
  isUploading.value = true
  try {
    const res = await uploadDataset(file)
    store.setUploaded({
      filename: res.filename,
      sizeBytes: res.size_bytes,
      summary: res.summary,
    })
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    isUploading.value = false
    if (fileInput.value) fileInput.value.value = ''
  }
}
</script>

<template>
  <section class="space-y-6">
    <header class="space-y-1">
      <h2 class="text-2xl font-bold">데이터 업로드</h2>
      <p class="text-sm text-slate-600">
        Amazon Sale Report 형식의 CSV를 업로드합니다. (최대 100MB)
      </p>
    </header>

    <div class="bg-white border border-slate-200 rounded-lg p-6">
      <label class="flex flex-col gap-3">
        <span class="text-sm font-medium text-slate-700">CSV 파일 선택</span>
        <input
          ref="fileInput"
          type="file"
          accept=".csv,text/csv"
          :disabled="isUploading"
          class="block text-sm file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:bg-slate-900 file:text-white hover:file:bg-slate-700 disabled:opacity-50"
          @change="handleFile"
        />
      </label>
      <p v-if="isUploading" class="mt-3 text-sm text-slate-500">업로드/파싱 중…</p>
      <p v-if="error" class="mt-3 text-sm text-rose-600">{{ error }}</p>
    </div>

    <div v-if="store.summary" class="space-y-4">
      <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatCard label="파일">{{ store.filename }}</StatCard>
        <StatCard label="크기">{{ formattedSize || '-' }}</StatCard>
        <StatCard label="행 수">{{ store.summary.rows.toLocaleString() }}</StatCard>
        <StatCard label="컬럼 수">{{ store.summary.cols }}</StatCard>
      </div>

      <div
        v-if="!store.summary.has_target"
        class="bg-amber-50 border border-amber-200 rounded p-3 text-sm text-amber-800"
      >
        ⚠ 기본 타겟 컬럼 <code class="font-mono">Amount</code>가 보이지 않습니다.
        학습 단계에서 다른 타겟을 지정해야 할 수 있습니다.
      </div>

      <div class="bg-white border border-slate-200 rounded-lg overflow-hidden">
        <header class="px-4 py-3 bg-slate-50 border-b border-slate-200 text-sm font-medium">
          결측치가 있는 컬럼
        </header>
        <div v-if="sortedMissing.length === 0" class="px-4 py-6 text-sm text-slate-500">
          결측치가 있는 컬럼이 없습니다.
        </div>
        <table v-else class="w-full text-sm">
          <thead class="text-left text-slate-500 bg-slate-50">
            <tr>
              <th class="px-4 py-2 font-medium">컬럼</th>
              <th class="px-4 py-2 font-medium">dtype</th>
              <th class="px-4 py-2 font-medium text-right">결측 개수</th>
              <th class="px-4 py-2 font-medium text-right">비율</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-slate-100">
            <tr v-for="col in sortedMissing" :key="col.name">
              <td class="px-4 py-2 font-mono">{{ col.name }}</td>
              <td class="px-4 py-2 text-slate-500">{{ col.dtype }}</td>
              <td class="px-4 py-2 text-right">{{ col.missing.toLocaleString() }}</td>
              <td class="px-4 py-2 text-right">{{ col.missing_pct.toFixed(2) }}%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>
</template>
