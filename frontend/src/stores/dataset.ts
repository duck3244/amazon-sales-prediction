import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { DatasetSummary } from '@/api/data'

export const useDatasetStore = defineStore(
  'dataset',
  () => {
    const summary = ref<DatasetSummary | null>(null)
    const filename = ref<string | null>(null)
    const sizeBytes = ref<number | null>(null)

    function setUploaded(payload: {
      filename: string
      sizeBytes: number
      summary: DatasetSummary
    }) {
      summary.value = payload.summary
      filename.value = payload.filename
      sizeBytes.value = payload.sizeBytes
    }

    function clear() {
      summary.value = null
      filename.value = null
      sizeBytes.value = null
    }

    return { summary, filename, sizeBytes, setUploaded, clear }
  },
  {
    persist: true,
  },
)
