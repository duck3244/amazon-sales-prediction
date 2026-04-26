import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { TrainStatusResponse } from '@/api/train'

export const useTrainingStore = defineStore(
  'training',
  () => {
    const jobId = ref<string | null>(null)
    const status = ref<TrainStatusResponse | null>(null)

    function setJob(id: string) {
      jobId.value = id
    }

    function setStatus(s: TrainStatusResponse) {
      status.value = s
    }

    function clear() {
      jobId.value = null
      status.value = null
    }

    return { jobId, status, setJob, setStatus, clear }
  },
  {
    persist: { paths: ['jobId'] }, // status는 매번 서버에서 새로 받기
  },
)
